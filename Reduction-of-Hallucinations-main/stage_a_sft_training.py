"""
Stage A: Supervised Fine-Tuning (SFT) Training
===============================================

Transform a generic base model into a Medical Specialist using LoRA fine-tuning.

Input: sft_train_data.csv (Prompt + Truth pairs)
Goal: Teach the model domain-specific medical knowledge and terminology
Mechanism: Standard "Next Token Prediction" (Cross-Entropy Loss)

Key Parameters:
- Learning Rate: 2e-4 to 1e-4 (standard LoRA rates)
- Epochs: 1-3 (avoid overtraining)
- LoRA Config: r=16 or r=32 for sufficient capacity
- Loss: Cross-entropy on next token prediction

Usage:
    python stage_a_sft_training.py \\
        --model_name "meta-llama/Llama-2-7b-hf" \\
        --output_dir "./models/sft_specialist" \\
        --num_epochs 2 \\
        --learning_rate 2e-4 \\
        --lora_r 16
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

if not torch.cuda.is_available():
    raise SystemError("❌ No GPU detected! This script requires a CUDA-enabled GPU.")

device = torch.device("cuda")
print("🔥 Using GPU:", torch.cuda.get_device_name(0))

from huggingface_hub import login
# Set your token via environment variable: export HF_TOKEN=your_token_here
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)


# Transformers imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
)

# PEFT (Parameter Efficient Fine-Tuning) imports for LoRA
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

# Import our custom SFT dataset
from sft_dataset import SFTDataset, SFTDataCollator, create_sft_dataloaders


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"

    # Data
    train_data_path: str = "phase1_data_medhal/sft/train_set_processed.csv"
    val_data_path: str = "phase1_data_medhal/sft/validation_set_processed.csv"

    # Training
    num_epochs: int = 2
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # LoRA Configuration
    lora_r: int = 16  # Rank of LoRA matrices
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Model Configuration
    max_length: int = 256  # reduced from 512 for faster training
    torch_dtype: str = "float16"  # or "bfloat16" or "float32"

    # Output
    output_dir: str = "./models/sft_specialist"
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    save_total_limit: int = 3

    # Hardware
    use_8bit: bool = False  # Use 8-bit quantization
    use_gradient_checkpointing: bool = True
    device_map: str = "auto"

    # Other
    seed: int = 42
    use_evidence: bool = False


class SFTTrainer:
    """Custom trainer for Supervised Fine-Tuning with LoRA."""

    def __init__(self, config: SFTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Load tokenizer
        logger.info(f"Loading tokenizer from {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        # Load model
        logger.info(f"Loading model from {config.model_name}")
        self.model = self._load_model()

        # Apply LoRA
        logger.info("Applying LoRA configuration")
        self.model = self._apply_lora()

        # NOTE:
        # The model is loaded with `device_map=self.config.device_map` ("auto" by default),
        # which uses Accelerate to place shards on devices and may leave some parameters
        # on the `meta` device. Calling `self.model.to(self.device)` afterwards can cause
        # "Cannot copy out of meta tensor" errors. We therefore rely on the device map
        # and DO NOT call `.to(self.device)` here.

        # Log model info
        self._log_model_info()

    def _load_model(self):
        """Load the base model with appropriate settings.

        We avoid `device_map="auto"` / meta device sharding here and instead
        load the full model onto a single device (GPU or CPU), to keep
        parameters and gradients on the same device and prevent meta-tensor
        errors during backward.
        """
        torch_dtype = getattr(torch, self.config.torch_dtype.replace("torch.", ""))

        kwargs = {
            "torch_dtype": torch_dtype,
        }

        # Add 8-bit quantization if requested
        if self.config.use_8bit:
            kwargs["load_in_8bit"] = True

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **kwargs
        )

        # For 8-bit models, bitsandbytes already places the model on the
        # correct device and dtype; calling `.to(...)` is not supported.
        if not self.config.use_8bit:
            model = model.to(self.device)

        # Enable gradient checkpointing for memory efficiency
        if self.config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

        return model

    def _apply_lora(self):
        """Apply LoRA configuration to the model."""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.lora_target_modules,
        )

        model = get_peft_model(self.model, lora_config)
        logger.info(f"Applied LoRA with r={self.config.lora_r}, alpha={self.config.lora_alpha}")

        return model

    def _log_model_info(self):
        """Log information about the model and trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        logger.info("Creating dataloaders")

        train_loader, val_loader = create_sft_dataloaders(
            train_csv=self.config.train_data_path,
            val_csv=self.config.val_data_path,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            use_evidence=self.config.use_evidence,
        )

        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")

        return train_loader, val_loader

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Main training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader

        Returns:
            Dictionary with training metrics
        """

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Training loop
        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        training_stats = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")

            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)

            # Validation phase
            val_loss = self._validate(val_loader)

            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

            training_stats['epoch'].append(epoch + 1)
            training_stats['train_loss'].append(train_loss)
            training_stats['val_loss'].append(val_loss)
            training_stats['learning_rate'].append(scheduler.get_last_lr()[0])

            # Save checkpoint
            checkpoint_path = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch + 1}"
            self._save_checkpoint(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        logger.info(f"\n{'='*60}")
        logger.info("Training completed!")
        logger.info(f"{'='*60}\n")

        # Save final model
        final_path = Path(self.config.output_dir) / "final_model"
        self._save_checkpoint(final_path)
        logger.info(f"Saved final model to {final_path}")

        # Save training stats
        stats_path = Path(self.config.output_dir) / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        logger.info(f"Saved training stats to {stats_path}")

        return training_stats

    def _train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc="Training")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

            if (step + 1) % self.config.logging_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"Step {step + 1}: Loss = {loss.item():.4f}, LR = {current_lr:.2e}")

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def _validate(self, val_loader: DataLoader) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def _save_checkpoint(self, save_path: Path):
        """Save model checkpoint."""
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save config
        config_dict = {
            'model_name': self.config.model_name,
            'max_length': self.config.max_length,
            'lora_r': self.config.lora_r,
            'lora_alpha': self.config.lora_alpha,
            'lora_dropout': self.config.lora_dropout,
        }
        config_path = save_path / "sft_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Stage A: SFT Training")

    # Model
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf",
                       help="Base model name/path")

    # Data
    parser.add_argument("--train_data_path", default="phase1_data/sft/train_set_processed.csv",
                       help="Path to training data CSV")
    parser.add_argument("--val_data_path", default="phase1_data/sft/validation_set_processed.csv",
                       help="Path to validation data CSV")

    # Training
    parser.add_argument("--num_epochs", type=int, default=2,
                       help="Number of training epochs (1-3 recommended)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate (2e-4 or 1e-4 typical)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank (16 or 32)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")

    # Output
    parser.add_argument("--output_dir", default="./models/sft_specialist",
                       help="Output directory for trained model")

    # Hardware
    parser.add_argument("--use_8bit", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Create config
    config = SFTConfig(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output_dir,
        use_8bit=args.use_8bit,
    )

    # Log configuration
    logger.info("SFT Training Configuration:")
    logger.info(json.dumps(vars(config), indent=2, default=str))

    # Initialize trainer
    trainer = SFTTrainer(config)

    # Create dataloaders
    train_loader, val_loader = trainer.create_dataloaders()

    # Train
    stats = trainer.train(train_loader, val_loader)

    logger.info("\nTraining completed successfully!")
    logger.info(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
