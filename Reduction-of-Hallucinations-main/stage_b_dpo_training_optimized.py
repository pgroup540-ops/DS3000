"""
Stage B DPO Training - Memory Optimized for 12GB GPU
====================================================

Optimizations for limited VRAM:
1. 8-bit quantization for both models (~50% memory reduction)
2. Reference model on CPU, active model on GPU
3. Gradient checkpointing
4. Batch size 1 with gradient accumulation

This allows training on RTX 5070 (12GB) by:
- Reference model: CPU (no VRAM used)
- Active model: GPU with 8-bit (~7GB)
- Activations + gradients: ~4GB
- Total: ~11GB (fits in 12GB!)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from dpo_dataset import DPODataset, DPODataCollator, create_dpo_dataloaders


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for memory-optimized DPO training."""
    
    # SFT Model (reference model)
    sft_model_path: str = "./models/sft_specialist/final_model"
    
    # Data
    train_data_path: str = "phase2_data/dpo/train_dpo.jsonl"
    val_data_path: str = "phase2_data/dpo/val_dpo.jsonl"
    
    # Training
    num_epochs: int = 2
    batch_size: int = 1  # Small for memory
    gradient_accumulation_steps: int = 4  # Effective batch size = 4
    learning_rate: float = 5e-6
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # DPO Specific
    beta: float = 0.1
    
    # LoRA Configuration
    use_lora: bool = True
    lora_r: int = 8  # Reduced from 16 for memory
    lora_alpha: int = 16  # Reduced from 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Model Configuration
    max_length: int = 512
    
    # Output
    output_dir: str = "./models/dpo_hallucination_resistant"
    logging_steps: int = 10
    
    # Memory Optimization
    use_8bit: bool = True  # CRITICAL for 12GB GPU
    use_gradient_checkpointing: bool = True
    reference_on_cpu: bool = True  # Put reference model on CPU
    
    # Other
    seed: int = 42


class MemoryOptimizedDPOTrainer:
    """Memory-optimized DPO trainer for 12GB GPU."""
    
    def __init__(self, config: DPOConfig):
        self.config = config
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        
        logger.info(f"GPU device: {self.gpu_device}")
        logger.info(f"Reference model device: {'CPU' if config.reference_on_cpu else 'GPU'}")
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {config.sft_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load models with memory optimization
        logger.info("Loading reference model (on CPU to save GPU memory)")
        self.reference_model = self._load_reference_model()
        
        logger.info("Loading active model (on GPU with 8-bit quantization)")
        self.model = self._load_active_model()
        
        self._log_model_info()
        self._log_memory_usage()
    
    def _load_reference_model(self):
        """Load reference model on CPU."""
        logger.info("Loading reference model to CPU...")
        
        try:
            # Load as LoRA model first
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.config.sft_model_path,
                torch_dtype=torch.float16,
                device_map="cpu",  # Force CPU
            )
            # Merge LoRA weights for reference
            model = model.merge_and_unload()
            logger.info("Loaded and merged LoRA model to CPU")
        except:
            logger.info("Loading as regular model to CPU")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.sft_model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        model.eval()  # Set to eval mode
        return model
    
    def _load_active_model(self):
        """Load active model on GPU with 8-bit quantization."""
        logger.info("Loading active model to GPU with 8-bit quantization...")
        
        # 8-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
        )
        
        # Load model and ensure we have trainable LoRA layers
        logger.info("Loading model with 8-bit quantization")
        
        # Load the PEFT model (which has LoRA from Stage A)
        model = AutoPeftModelForCausalLM.from_pretrained(
            self.config.sft_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            is_trainable=True,  # Make it trainable
        )
        logger.info("Loaded PEFT model with existing LoRA layers")
        logger.info("Model will be trained with Stage A LoRA weights as initialization")
        
        # Explicitly set all LoRA parameters to require gradients
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        
        # Set model to training mode
        model.train()
        logger.info("Set model to training mode with LoRA parameters requiring gradients")
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        return model
    
    def _log_model_info(self):
        """Log model information."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Active model total params: {total:,}")
        logger.info(f"Active model trainable params: {trainable:,}")
        logger.info(f"Trainable: {100 * trainable / total:.2f}%")
    
    def _log_memory_usage(self):
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def dpo_loss(
        self,
        model_logps: torch.Tensor,
        ref_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss."""
        # Extract probabilities
        model_chosen_logps = model_logps[:, 0]
        model_rejected_logps = model_logps[:, 1]
        ref_chosen_logps = ref_logps[:, 0]
        ref_rejected_logps = ref_logps[:, 1]
        
        # DPO objective
        log_odds = (
            (model_chosen_logps - model_rejected_logps) - 
            (ref_chosen_logps - ref_rejected_logps)
        )
        
        loss = -F.logsigmoid(self.config.beta * log_odds).mean()
        
        # Calculate metrics
        with torch.no_grad():
            chosen_preference = (model_chosen_logps > model_rejected_logps).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'chosen_preference': chosen_preference.item(),
            'avg_log_odds': log_odds.mean().item(),
        }
        
        return loss, metrics
    
    def get_batch_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model,
        device,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Compute log probabilities for a batch.
        
        Args:
            requires_grad: If True, computes with gradients (for active model training)
                          If False, uses no_grad context (for reference model)
        """
        # Move inputs to model's device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits
        
        # Shift for next token prediction
        logits = logits[:, :-1, :].contiguous()
        input_ids_for_loss = input_ids[:, 1:].contiguous()
        attention_mask_for_loss = attention_mask[:, 1:].contiguous()
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        indices = input_ids_for_loss.unsqueeze(-1)
        selected_log_probs = torch.gather(log_probs, -1, indices).squeeze(-1)
        
        # Average over tokens (only non-padded)
        selected_log_probs = selected_log_probs * attention_mask_for_loss
        batch_logps = selected_log_probs.sum(dim=1) / attention_mask_for_loss.sum(dim=1).clamp(min=1)
        
        # Move back to GPU if needed
        return batch_logps.to(self.gpu_device)
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        logger.info("Creating dataloaders")
        
        train_loader, val_loader = create_dpo_dataloaders(
            train_jsonl=self.config.train_data_path,
            val_jsonl=self.config.val_data_path,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
        )
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Main training loop."""
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )
        
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting memory-optimized DPO training for {self.config.num_epochs} epochs")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"Beta (KL weight): {self.config.beta}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        
        training_stats = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'chosen_preference': [],
        }
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")
            
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_preference = self._validate(val_loader)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Chosen Preference: {val_preference:.2%}")
            
            training_stats['epoch'].append(epoch + 1)
            training_stats['train_loss'].append(train_loss)
            training_stats['val_loss'].append(val_loss)
            training_stats['chosen_preference'].append(val_preference)
            
            # Save checkpoint
            checkpoint_path = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch + 1}"
            self._save_checkpoint(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            self._log_memory_usage()
        
        logger.info(f"\n{'='*60}")
        logger.info("DPO Training completed!")
        logger.info(f"{'='*60}\n")
        
        final_path = Path(self.config.output_dir) / "final_model"
        self._save_checkpoint(final_path)
        logger.info(f"Saved final model to {final_path}")
        
        stats_path = Path(self.config.output_dir) / "dpo_training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        logger.info(f"Saved training stats to {stats_path}")
        
        return training_stats
    
    def _train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        total_preference = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # Get logps from reference model (on CPU)
            ref_device = self.cpu_device if self.config.reference_on_cpu else self.gpu_device
            
            # Get logps from reference model (no gradients needed)
            with torch.no_grad():
                ref_chosen_logps = self.get_batch_logps(
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask'],
                    self.reference_model,
                    ref_device,
                    requires_grad=False
                )
                ref_rejected_logps = self.get_batch_logps(
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask'],
                    self.reference_model,
                    ref_device,
                    requires_grad=False
                )
            
            # Get logps from active model (WITH gradients for training)
            model_chosen_logps = self.get_batch_logps(
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                self.model,
                self.gpu_device,
                requires_grad=True  # Enable gradients for active model
            )
            model_rejected_logps = self.get_batch_logps(
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                self.model,
                self.gpu_device,
                requires_grad=True  # Enable gradients for active model
            )
            
            # Stack logps
            model_logps = torch.stack([model_chosen_logps, model_rejected_logps], dim=1)
            ref_logps = torch.stack([ref_chosen_logps, ref_rejected_logps], dim=1)
            
            # Compute DPO loss
            loss, metrics = self.dpo_loss(model_logps, ref_logps)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_preference += metrics['chosen_preference']
            
            # Backward
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({
                'loss': loss.item() * self.config.gradient_accumulation_steps,
                'pref': f"{metrics['chosen_preference']:.2%}"
            })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        total_preference = 0.0
        
        ref_device = self.cpu_device if self.config.reference_on_cpu else self.gpu_device
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Get logps from reference model
                ref_chosen_logps = self.get_batch_logps(
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask'],
                    self.reference_model,
                    ref_device
                )
                ref_rejected_logps = self.get_batch_logps(
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask'],
                    self.reference_model,
                    ref_device
                )
                
                # Get logps from active model
                model_chosen_logps = self.get_batch_logps(
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask'],
                    self.model,
                    self.gpu_device
                )
                model_rejected_logps = self.get_batch_logps(
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask'],
                    self.model,
                    self.gpu_device
                )
                
                model_logps = torch.stack([model_chosen_logps, model_rejected_logps], dim=1)
                ref_logps = torch.stack([ref_chosen_logps, ref_rejected_logps], dim=1)
                
                loss, metrics = self.dpo_loss(model_logps, ref_logps)
                
                total_loss += loss.item()
                total_preference += metrics['chosen_preference']
        
        avg_loss = total_loss / len(val_loader)
        avg_preference = total_preference / len(val_loader)
        
        return avg_loss, avg_preference
    
    def _save_checkpoint(self, save_path: Path):
        """Save model checkpoint."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        config_dict = {
            'sft_model': self.config.sft_model_path,
            'max_length': self.config.max_length,
            'beta': self.config.beta,
            'lora_r': self.config.lora_r,
        }
        config_path = save_path / "dpo_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Memory-Optimized Stage B: DPO Training")
    
    parser.add_argument(
        "--sft_model_path",
        default="./models/sft_specialist/final_model",
        help="Path to SFT model from Stage A"
    )
    
    parser.add_argument(
        "--train_data_path",
        default="phase2_data/dpo/train_dpo.jsonl",
        help="Path to training DPO data"
    )
    parser.add_argument(
        "--val_data_path",
        default="phase2_data/dpo/val_dpo.jsonl",
        help="Path to validation DPO data"
    )
    
    parser.add_argument(
        "--num_epochs", type=int, default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--beta", type=float, default=0.1,
        help="KL-divergence penalty weight"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./models/dpo_hallucination_resistant",
        help="Output directory for trained model"
    )
    
    args = parser.parse_args()
    
    config = DPOConfig(
        sft_model_path=args.sft_model_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        output_dir=args.output_dir,
    )
    
    logger.info("Memory-Optimized DPO Training Configuration:")
    logger.info(json.dumps(vars(config), indent=2, default=str))
    
    trainer = MemoryOptimizedDPOTrainer(config)
    train_loader, val_loader = trainer.create_dataloaders()
    stats = trainer.train(train_loader, val_loader)
    
    logger.info("\nDPO Training completed successfully!")
    logger.info(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
