"""
Stage B: Direct Preference Optimization (DPO) Training
=====================================================

Turn the SFT specialist into a Hallucination-Resistant Expert.

Input: dpo_train_data.jsonl (Triplets: Prompt + Chosen + Rejected)
Goal: Teach the model to statistically prefer factual responses over hallucinations
Mechanism: DPO Loss with KL-Divergence penalty to prevent model drift

Key Difference from Stage A:
- TWO models loaded in memory:
  * Active Model: Being trained (weights change)
  * Reference Model: Frozen copy from Stage A (weights frozen)
- Loss compares probability of chosen vs rejected responses
- KL penalty prevents active model from deviating from reference model's style

Key Parameters:
- Learning Rate: 5e-6 or 1e-6 (much lower than SFT!)
- Beta: 0.1 (KL-divergence penalty weight)
- Epochs: 1-3 (similar to SFT)

Usage:
    python stage_b_dpo_training.py \
        --sft_model_path "./models/sft_specialist/final_model" \
        --num_epochs 2 \
        --learning_rate 5e-6 \
        --beta 0.1
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
)
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, TaskType

from dpo_dataset import DPODataset, DPODataCollator, create_dpo_dataloaders


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    
    # SFT Model (reference model)
    sft_model_path: str = "./models/sft_specialist/final_model"
    
    # Data
    train_data_path: str = "phase2_data/dpo/train_dpo.jsonl"
    val_data_path: str = "phase2_data/dpo/val_dpo.jsonl"
    
    # Training
    num_epochs: int = 2
    batch_size: int = 4  # Smaller batch size due to dual model
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6  # Much lower than SFT!
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # DPO Specific
    beta: float = 0.1  # KL-divergence penalty weight
    label_smoothing: float = 0.0  # Optional label smoothing
    
    # LoRA Configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Model Configuration
    max_length: int = 512
    torch_dtype: str = "float16"
    
    # Output
    output_dir: str = "./models/dpo_hallucination_resistant"
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # Hardware
    use_8bit: bool = False
    use_gradient_checkpointing: bool = True
    device_map: str = "auto"
    
    # Other
    seed: int = 42


class DPOTrainer:
    """Trainer for Direct Preference Optimization."""
    
    def __init__(self, config: DPOConfig, force_cpu: bool = False):
        self.config = config
        if force_cpu:
            self.device = torch.device("cpu")
            logger.info("Forcing CPU device as requested")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {config.sft_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load models
        logger.info("Loading reference and active models")
        self.reference_model = self._load_model(config.sft_model_path, freeze=True)
        self.model = self._load_model(config.sft_model_path, freeze=False)
        
        # Model already has LoRA from Stage A - just enable training
        logger.info("Model already has LoRA adapters from Stage A")
        logger.info("Enabling training mode on existing LoRA parameters")
        
        # Ensure LoRA parameters are trainable
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        
        # Set model to training mode
        self.model.train()
        
        self.reference_model.to(self.device)
        self.model.to(self.device)
        
        self._log_model_info()
    
    def _load_model(self, model_path: str, freeze: bool = False):
        """Load model from SFT checkpoint."""
        torch_dtype = getattr(torch, self.config.torch_dtype.replace("torch.", ""))
        
        # Use device_map string for loading
        device_map_str = "cpu" if str(self.device) == "cpu" else "auto"
        
        try:
            # Try to load as LoRA model
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map_str,
                load_in_8bit=self.config.use_8bit,
            )
            # Merge if we want the base model
            if not self.config.use_lora:
                model = model.merge_and_unload()
            logger.info("Loaded as LoRA model")
        except:
            logger.info("Loading as regular model")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map_str,
                load_in_8bit=self.config.use_8bit,
            )
        
        # Disable gradient checkpointing on CPU - it breaks gradient flow with LoRA
        if self.config.use_gradient_checkpointing and str(self.device) != "cpu":
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        elif str(self.device) == "cpu":
            logger.info("Gradient checkpointing disabled on CPU to preserve gradient flow")
        
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        
        return model
    
    def _apply_lora(self, model):
        """Apply LoRA to model."""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.lora_target_modules,
        )
        
        model = get_peft_model(model, lora_config)
        return model
    
    def _log_model_info(self):
        """Log model information."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Active model total params: {total:,}")
        logger.info(f"Active model trainable params: {trainable:,}")
        logger.info(f"Trainable: {100 * trainable / total:.2f}%")
    
    def dpo_loss(
        self,
        model_logps: torch.Tensor,  # Log probabilities from active model
        ref_logps: torch.Tensor,    # Log probabilities from reference model
        chosen: torch.Tensor = None, # Not used in simplifed version
        rejected: torch.Tensor = None,  # Not used in simplified version
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.
        
        DPO Loss = -log(sigmoid(beta * (log_chosen - log_rejected_active - log_chosen_ref + log_rejected_ref)))
        
        This encourages:
        - Active model to assign HIGH probability to chosen
        - Active model to assign LOW probability to rejected
        - While staying close to reference model (KL penalty)
        """
        # Extract probabilities
        model_chosen_logps = model_logps[:, 0]
        model_rejected_logps = model_logps[:, 1]
        ref_chosen_logps = ref_logps[:, 0]
        ref_rejected_logps = ref_logps[:, 1]
        
        # DPO objective: maximize the difference while minimizing KL
        # log odds ratio
        log_odds = (
            (model_chosen_logps - model_rejected_logps) - 
            (ref_chosen_logps - ref_rejected_logps)
        )
        
        # Sigmoid of log odds (preference probability)
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
    ) -> torch.Tensor:
        """
        Compute log probabilities for a batch of sequences.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            model: The model to compute logps with
        
        Returns:
            Log probabilities for each sequence
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits
        
        # Shift for next token prediction
        # We want log prob of token at position i given tokens up to i-1
        logits = logits[:, :-1, :].contiguous()
        input_ids_for_loss = input_ids[:, 1:].contiguous()
        attention_mask_for_loss = attention_mask[:, 1:].contiguous()
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        batch_size, seq_len = input_ids_for_loss.shape
        indices = input_ids_for_loss.unsqueeze(-1)
        selected_log_probs = torch.gather(log_probs, -1, indices).squeeze(-1)
        
        # Average over tokens (only non-padded)
        selected_log_probs = selected_log_probs * attention_mask_for_loss
        batch_logps = selected_log_probs.sum(dim=1) / attention_mask_for_loss.sum(dim=1).clamp(min=1)
        
        return batch_logps
    
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
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )
        
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting DPO training for {self.config.num_epochs} epochs")
        logger.info(f"Beta (KL weight): {self.config.beta}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        
        training_stats = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'chosen_preference': [],
            'learning_rate': [],
        }
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\
{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")
            
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_preference = self._validate(val_loader)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Chosen Preference: {val_preference:.2%}")
            logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            
            training_stats['epoch'].append(epoch + 1)
            training_stats['train_loss'].append(train_loss)
            training_stats['val_loss'].append(val_loss)
            training_stats['chosen_preference'].append(val_preference)
            training_stats['learning_rate'].append(scheduler.get_last_lr()[0])
            
            # Save checkpoint
            checkpoint_path = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch + 1}"
            self._save_checkpoint(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        logger.info(f"\
{'='*60}")
        logger.info("DPO Training completed!")
        logger.info(f"{'='*60}\
")
        
        final_path = Path(self.config.output_dir) / "final_model"
        self._save_checkpoint(final_path)
        logger.info(f"Saved final model to {final_path}")
        
        stats_path = Path(self.config.output_dir) / "dpo_training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        logger.info(f"Saved training stats to {stats_path}")
        
        return training_stats
    
    def _train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_preference = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            logger.info(f"Starting step {step + 1}/{len(train_loader)}")
            sys.stdout.flush()
            chosen_input_ids = batch['chosen_input_ids'].to(self.device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
            rejected_input_ids = batch['rejected_input_ids'].to(self.device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
            logger.info("  Computing reference model logps (chosen)...")
            sys.stdout.flush()
            
            # Get logps from both models
            with torch.no_grad():
                ref_chosen_logps = self.get_batch_logps(
                    chosen_input_ids,
                    chosen_attention_mask,
                    self.reference_model
                )
                logger.info("  Computing reference model logps (rejected)...")
                sys.stdout.flush()
                ref_rejected_logps = self.get_batch_logps(
                    rejected_input_ids,
                    rejected_attention_mask,
                    self.reference_model
                )
            
            logger.info("  Computing active model logps (chosen)...")
            sys.stdout.flush()
            model_chosen_logps = self.get_batch_logps(
                chosen_input_ids,
                chosen_attention_mask,
                self.model
            )
            logger.info("  Computing active model logps (rejected)...")
            sys.stdout.flush()
            model_rejected_logps = self.get_batch_logps(
                rejected_input_ids,
                rejected_attention_mask,
                self.model
            )
            
            logger.info("  Computing DPO loss...")
            sys.stdout.flush()
            # Stack logps
            model_logps = torch.stack([model_chosen_logps, model_rejected_logps], dim=1)
            ref_logps = torch.stack([ref_chosen_logps, ref_rejected_logps], dim=1)
            
            # Compute DPO loss
            loss, metrics = self.dpo_loss(model_logps, ref_logps)
            logger.info(f"  Loss: {loss.item():.4f}, Preference: {metrics['chosen_preference']:.2%}")
            sys.stdout.flush()
            
            total_loss += loss.item()
            total_preference += metrics['chosen_preference']
            
            logger.info("  Running backward pass...")
            sys.stdout.flush()
            # Backward
            loss.backward()
            
            logger.info("  Clipping gradients and optimizer step...")
            sys.stdout.flush()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            logger.info(f"  Step {step + 1} completed!")
            sys.stdout.flush()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'pref': f"{metrics['chosen_preference']:.2%}"
            })
            
            if (step + 1) % self.config.logging_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {step + 1}: Loss = {loss.item():.4f}, "
                    f"Pref = {metrics['chosen_preference']:.2%}, LR = {current_lr:.2e}"
                )
        
        avg_loss = total_loss / len(train_loader)
        avg_preference = total_preference / len(train_loader)
        logger.info(f"Epoch avg preference: {avg_preference:.2%}")
        
        return avg_loss
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        total_preference = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                
                # Get logps
                ref_chosen_logps = self.get_batch_logps(
                    chosen_input_ids, chosen_attention_mask, self.reference_model
                )
                ref_rejected_logps = self.get_batch_logps(
                    rejected_input_ids, rejected_attention_mask, self.reference_model
                )
                
                model_chosen_logps = self.get_batch_logps(
                    chosen_input_ids, chosen_attention_mask, self.model
                )
                model_rejected_logps = self.get_batch_logps(
                    rejected_input_ids, rejected_attention_mask, self.model
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
            'lora_r': self.config.lora_r if self.config.use_lora else None,
        }
        config_path = save_path / "dpo_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Stage B: DPO Training")
    
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
        "--batch_size", type=int, default=4,
        help="Batch size (smaller than SFT due to dual model)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-6,
        help="Learning rate (much lower than SFT!)"
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
    parser.add_argument(
        "--device", default="cuda",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    config = DPOConfig(
        sft_model_path=args.sft_model_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        output_dir=args.output_dir,
    )
    
    logger.info("DPO Training Configuration:")
    logger.info(json.dumps(vars(config), indent=2, default=str))
    
    # Force CPU if requested
    force_cpu = (args.device.lower() == "cpu")
    
    trainer = DPOTrainer(config, force_cpu=force_cpu)
    train_loader, val_loader = trainer.create_dataloaders()
    stats = trainer.train(train_loader, val_loader)
    
    logger.info("\nDPO Training completed successfully!")
    logger.info(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
