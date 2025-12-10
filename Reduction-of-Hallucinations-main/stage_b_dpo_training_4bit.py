"""
Stage B: 4-bit QLoRA DPO Training (Optimized for 12GB GPU)
===========================================================

This script implements Direct Preference Optimization with aggressive
4-bit quantization to fit both reference and active models in 12GB VRAM.

Key Optimizations:
- 4-bit NormalFloat quantization (reduces model size by 75%)
- Double quantization for further compression
- QLoRA: Only train small adapter layers (few MB)
- Gradient checkpointing for activation memory savings
- Sequential batch processing to minimize peak memory

Expected Memory Usage:
- Reference Model (4-bit): ~3.5GB
- Active Model (4-bit): ~3.5GB
- LoRA adapters: ~50MB
- Activations + gradients: ~2-3GB
- Total: ~9-10GB (fits in 12GB!)

Usage:
    python stage_b_dpo_training_4bit.py \
        --sft_model_path "models/sft_specialist_fast_fp16/final_model" \
        --train_data_path "phase2_data/dpo/train_dpo.jsonl" \
        --val_data_path "phase2_data/dpo/val_dpo.jsonl" \
        --num_epochs 2 \
        --batch_size 1 \
        --learning_rate 5e-6
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
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
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from dpo_dataset import create_dpo_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DPOTrainer4Bit:
    """4-bit Quantized DPO Trainer optimized for 12GB GPU."""
    
    def __init__(
        self,
        sft_model_path: str,
        train_data_path: str,
        val_data_path: str,
        output_dir: str = "./models/dpo_hallucination_resistant_4bit",
        num_epochs: int = 2,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 5e-6,
        warmup_steps: int = 100,
        beta: float = 0.1,
        max_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        save_steps: int = 100,
        eval_steps: int = 50,
        logging_steps: int = 10,
        seed: int = 42,
    ):
        self.sft_model_path = sft_model_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.beta = beta
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.seed = seed
        
        # Setup
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type != "cuda":
            raise RuntimeError("4-bit quantization requires CUDA GPU. Use CPU version for CPU training.")
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.save_config()
        
    def save_config(self):
        """Save training configuration."""
        config = {
            "sft_model_path": self.sft_model_path,
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path,
            "output_dir": self.output_dir,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "beta": self.beta,
            "max_length": self.max_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "quantization": "4-bit NormalFloat with double quantization",
            "seed": self.seed,
        }
        
        config_path = Path(self.output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        logger.info(f"Training Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
    
    def create_4bit_config(self):
        """Create 4-bit quantization configuration."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4 - best for inference & training
            bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
            bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
        )
        return bnb_config
    
    def load_models(self):
        """Load reference and active models with 4-bit quantization."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = self.create_4bit_config()
        
        logger.info("Loading REFERENCE model (frozen, 4-bit)...")
        try:
            # Try loading as PEFT model first
            self.reference_model = AutoPeftModelForCausalLM.from_pretrained(
                self.sft_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("✓ Reference model loaded as PEFT model")
        except:
            # Fallback to base model
            logger.info("Loading as base model with LoRA...")
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.sft_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()
        
        logger.info("Loading ACTIVE model (trainable, 4-bit)...")
        try:
            # Try loading as PEFT model
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.sft_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("✓ Active model loaded as PEFT model")
            
            # Check if LoRA already exists
            if hasattr(self.model, 'peft_config'):
                logger.info("✓ Model already has LoRA from Stage A - will continue training")
                # Prepare for training
                self.model = prepare_model_for_kbit_training(self.model)
            else:
                logger.info("Adding new LoRA adapters...")
                self.model = self._add_lora_adapters(self.model)
                
        except:
            # Fallback: load base model and add LoRA
            logger.info("Loading as base model and adding LoRA...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.sft_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = self._add_lora_adapters(self.model)
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled")
        
        # Set to training mode
        self.model.train()
        
        self._log_model_info()
        self._log_memory_usage("After loading models")
    
    def _add_lora_adapters(self, model):
        """Add LoRA adapters to model."""
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        logger.info("✓ LoRA adapters added")
        return model
    
    def _log_model_info(self):
        """Log model parameter information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model Parameters:")
        logger.info(f"  Total: {total_params:,}")
        logger.info(f"  Trainable: {trainable_params:,}")
        logger.info(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    def _log_memory_usage(self, label: str = ""):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory {label}:")
            logger.info(f"  Allocated: {allocated:.2f} GB")
            logger.info(f"  Reserved: {reserved:.2f} GB")
    
    def get_batch_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for a batch."""
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # Mixed precision
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs.logits
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Get log probs of actual tokens
        per_token_logps = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.unsqueeze(2)
        ).squeeze(2)
        
        # Mask padding tokens
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        per_token_logps = per_token_logps * mask
        
        # Sum log probs per sequence
        sequence_logps = per_token_logps.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return sequence_logps
    
    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute DPO loss."""
        # Log ratios
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        
        # DPO loss
        logits = policy_logratios - reference_logratios
        losses = -F.logsigmoid(self.beta * logits)
        loss = losses.mean()
        
        # Metrics
        chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "reward_accuracy": reward_accuracy.item(),
            "chosen_rewards_mean": chosen_rewards.mean().item(),
            "rejected_rewards_mean": rejected_rewards.mean().item(),
        }
        
        return loss, metrics
    
    def train_step(self, batch: Dict) -> Dict:
        """Execute one training step."""
        # Move batch to device
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)
        
        # Compute reference model log probs (no gradients)
        with torch.no_grad():
            reference_chosen_logps = self.get_batch_logps(
                self.reference_model,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_labels,
            )
            reference_rejected_logps = self.get_batch_logps(
                self.reference_model,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_labels,
            )
        
        # Compute policy model log probs (with gradients)
        policy_chosen_logps = self.get_batch_logps(
            self.model,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_labels,
        )
        policy_rejected_logps = self.get_batch_logps(
            self.model,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_labels,
        )
        
        # Compute loss
        loss, metrics = self.compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        
        return loss, metrics
    
    def train(self):
        """Main training loop."""
        logger.info("=" * 70)
        logger.info("Starting 4-bit QLoRA DPO Training")
        logger.info("=" * 70)
        
        # Load models
        self.load_models()
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_dataloader, val_dataloader = create_dpo_dataloaders(
            train_path=self.train_data_path,
            val_path=self.val_data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        
        logger.info(f"Training batches: {len(train_dataloader)}")
        logger.info(f"Validation batches: {len(val_dataloader)}")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        # Setup scheduler
        total_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {self.warmup_steps}")
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info("=" * 70)
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info("=" * 70)
            
            self.model.train()
            epoch_metrics = []
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                loss, metrics = self.train_step(batch)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Log metrics
                    if global_step % self.logging_steps == 0:
                        logger.info(f"Step {global_step}: {metrics}")
                    
                    # Save checkpoint
                    if global_step % self.save_steps == 0:
                        self.save_checkpoint(global_step)
                    
                    # Evaluation
                    if global_step % self.eval_steps == 0:
                        val_loss = self.evaluate(val_dataloader)
                        logger.info(f"Validation loss: {val_loss:.4f}")
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_checkpoint(global_step, is_best=True)
                        
                        self.model.train()
                
                epoch_metrics.append(metrics)
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['reward_accuracy']:.2%}",
                })
            
            # Epoch summary
            avg_metrics = {
                key: np.mean([m[key] for m in epoch_metrics])
                for key in epoch_metrics[0].keys()
            }
            logger.info(f"Epoch {epoch + 1} average metrics: {avg_metrics}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")
        
        # Save final model
        self.save_final_model()
        
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)
        logger.info(f"Final model saved to: {self.output_dir}/final_model")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    def evaluate(self, dataloader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                loss, _ = self.train_step(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, step, is_best=False):
        """Save model checkpoint."""
        if is_best:
            checkpoint_dir = Path(self.output_dir) / "best_model"
        else:
            checkpoint_dir = Path(self.output_dir) / f"checkpoint-{step}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"✓ Checkpoint saved to {checkpoint_dir}")
    
    def save_final_model(self):
        """Save final trained model."""
        final_dir = Path(self.output_dir) / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        logger.info(f"✓ Final model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="4-bit QLoRA DPO Training for Stage B")
    
    parser.add_argument("--sft_model_path", type=str, required=True,
                        help="Path to Stage A SFT model")
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to training DPO data (JSONL)")
    parser.add_argument("--val_data_path", type=str, required=True,
                        help="Path to validation DPO data (JSONL)")
    parser.add_argument("--output_dir", type=str, 
                        default="./models/dpo_hallucination_resistant_4bit",
                        help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Evaluate every N steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = DPOTrainer4Bit(
        sft_model_path=args.sft_model_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
    )
    
    # Train
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
