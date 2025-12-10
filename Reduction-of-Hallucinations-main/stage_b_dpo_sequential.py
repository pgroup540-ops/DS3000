"""
Stage B: Sequential DPO Training with Cached Reference Logits
==============================================================

This script implements DPO in two phases to fit in 12GB GPU:

Phase 1: Cache Reference Logits
- Load reference model (4-bit)
- Compute logits for all training data
- Save to disk cache
- Unload model

Phase 2: Train with Cached Logits
- Load active model (4-bit)
- Load cached reference logits from disk
- Compute DPO loss and train
- Only ONE model in memory at a time!

Memory Usage:
- Phase 1: ~6GB (reference model only)
- Phase 2: ~6GB (active model only)
- Both phases fit in 12GB GPU!

Usage:
    python stage_b_dpo_sequential.py \
        --sft_model_path "models/sft_specialist_fast_fp16/final_model" \
        --train_data_path "phase2_data/dpo/train_dpo.jsonl" \
        --val_data_path "phase2_data/dpo/val_dpo.jsonl" \
        --cache_dir "./cache/reference_logits" \
        --num_epochs 2
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pickle

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
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
)

from dpo_dataset import create_dpo_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SequentialDPOTrainer:
    """Sequential DPO trainer that caches reference logits to avoid dual-model memory usage."""
    
    def __init__(
        self,
        sft_model_path: str,
        train_data_path: str,
        val_data_path: str,
        cache_dir: str = "./cache/reference_logits",
        output_dir: str = "./models/dpo_hallucination_resistant_sequential",
        num_epochs: int = 2,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 5e-6,
        warmup_steps: int = 100,
        beta: float = 0.1,
        max_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 32,
        save_steps: int = 100,
        eval_steps: int = 50,
        seed: int = 42,
    ):
        self.sft_model_path = sft_model_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.beta = beta
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.seed = seed
        
        # Setup
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if not torch.cuda.is_available():
            raise RuntimeError("This script requires CUDA GPU.")
        
        self.device = torch.device("cuda")
        logger.info(f"Using device: {self.device}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache paths
        self.train_cache_dir = self.cache_dir / "train"
        self.val_cache_dir = self.cache_dir / "val"
        self.train_cache_dir.mkdir(parents=True, exist_ok=True)
        self.val_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def create_4bit_config(self):
        """Create 4-bit quantization configuration."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    def log_memory(self, label=""):
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory {label}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def load_tokenizer(self):
        """Load tokenizer."""
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.sft_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def load_reference_model(self):
        """Load reference model with 4-bit quantization."""
        logger.info("Loading reference model (4-bit, frozen)...")
        bnb_config = self.create_4bit_config()
        
        model = AutoPeftModelForCausalLM.from_pretrained(
            self.sft_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        
        self.log_memory("after loading reference model")
        return model
    
    def load_active_model(self):
        """Load active model with 4-bit quantization for training."""
        logger.info("Loading active model (4-bit, trainable)...")
        bnb_config = self.create_4bit_config()
        
        try:
            # Try to load as PEFT model (has existing LoRA)
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.sft_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("✓ Loaded as PEFT model with existing LoRA adapters")
            
            # Prepare for k-bit training
            model = prepare_model_for_kbit_training(model)
            
            # Check if we have trainable params
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_params == 0:
                logger.warning("No trainable parameters found in PEFT model, adding LoRA...")
                # Need to add LoRA if model was merged
                from peft import LoraConfig, get_peft_model, TaskType
                lora_config = LoraConfig(
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, lora_config)
                logger.info("✓ Added LoRA adapters")
                
        except Exception as e:
            logger.info(f"Could not load as PEFT model ({e}), loading base model...")
            # Load base model and add LoRA
            model = AutoModelForCausalLM.from_pretrained(
                self.sft_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
            
            # Add LoRA adapters
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            logger.info("✓ Added LoRA adapters to base model")
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        model.train()
        
        # Log parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        if trainable_params == 0:
            raise RuntimeError("Model has no trainable parameters! Cannot train.")
        
        self.log_memory("after loading active model")
        return model
    
    def get_batch_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for a batch."""
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs.logits
        
        # Shift for next-token prediction
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
        
        # Mask padding
        mask = (shift_labels != -100).float()
        per_token_logps = per_token_logps * mask
        
        # Average over sequence
        sequence_logps = per_token_logps.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return sequence_logps
    
    def cache_reference_logits(self, dataloader, cache_dir: Path, split_name: str):
        """Phase 1: Compute and cache reference model logits."""
        logger.info("=" * 70)
        logger.info(f"PHASE 1: Caching Reference Logits ({split_name})")
        logger.info("=" * 70)
        
        # Check if cache exists
        cache_complete_flag = cache_dir / ".complete"
        if cache_complete_flag.exists():
            logger.info(f"✓ Cache already exists for {split_name}, skipping...")
            return
        
        # Load reference model
        tokenizer = self.load_tokenizer()
        reference_model = self.load_reference_model()
        
        logger.info(f"Processing {len(dataloader)} batches...")
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Caching {split_name}")):
                    # Move to device
                    chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                    chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                    # Create labels from input_ids (standard for causal LM)
                    chosen_labels = chosen_input_ids.clone()
                    
                    rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                    rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                    # Create labels from input_ids (standard for causal LM)
                    rejected_labels = rejected_input_ids.clone()
                    
                    # Compute logits
                    chosen_logps = self.get_batch_logps(
                        reference_model,
                        chosen_input_ids,
                        chosen_attention_mask,
                        chosen_labels,
                    )
                    
                    rejected_logps = self.get_batch_logps(
                        reference_model,
                        rejected_input_ids,
                        rejected_attention_mask,
                        rejected_labels,
                    )
                    
                    # Save to cache (move to CPU to save memory)
                    cache_data = {
                        'chosen_logps': chosen_logps.cpu(),
                        'rejected_logps': rejected_logps.cpu(),
                    }
                    
                    cache_file = cache_dir / f"batch_{batch_idx:05d}.pkl"
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                    
                    # Log progress
                    if (batch_idx + 1) % 100 == 0:
                        logger.info(f"Cached {batch_idx + 1}/{len(dataloader)} batches")
            
            # Mark cache as complete
            cache_complete_flag.touch()
            logger.info(f"✓ Cache complete for {split_name}")
            
        finally:
            # Clean up model
            del reference_model
            torch.cuda.empty_cache()
            self.log_memory("after cleaning up reference model")
    
    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute DPO loss."""
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = policy_logratios - reference_logratios
        losses = -F.logsigmoid(self.beta * logits)
        loss = losses.mean()
        
        # Metrics
        chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return loss, {
            "loss": loss.item(),
            "reward_accuracy": reward_accuracy.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
        }
    
    def train_with_cached_logits(self, dataloader, cache_dir: Path, split_name: str, is_training: bool = True):
        """Phase 2: Train using cached reference logits."""
        if is_training:
            logger.info("=" * 70)
            logger.info(f"PHASE 2: Training with Cached Logits")
            logger.info("=" * 70)
        
        # Load active model (only if training, reuse if evaluating)
        if not hasattr(self, 'model'):
            self.tokenizer = self.load_tokenizer()
            self.model = self.load_active_model()
        
        if not is_training:
            self.model.eval()
            total_loss = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
                    # Load cached reference logits
                    cache_file = cache_dir / f"batch_{batch_idx:05d}.pkl"
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    ref_chosen_logps = cache_data['chosen_logps'].to(self.device)
                    ref_rejected_logps = cache_data['rejected_logps'].to(self.device)
                    
                    # Move batch to device
                    chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                    chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                    chosen_labels = chosen_input_ids.clone()
                    
                    rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                    rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                    rejected_labels = rejected_input_ids.clone()
                    
                    # Compute policy logits
                    policy_chosen_logps = self.get_batch_logps(
                        self.model, chosen_input_ids, chosen_attention_mask, chosen_labels
                    )
                    policy_rejected_logps = self.get_batch_logps(
                        self.model, rejected_input_ids, rejected_attention_mask, rejected_labels
                    )
                    
                    # Compute loss
                    loss, _ = self.compute_dpo_loss(
                        policy_chosen_logps, policy_rejected_logps,
                        ref_chosen_logps, ref_rejected_logps
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            return total_loss / num_batches
        
        # Training mode
        self.model.train()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        # Setup scheduler
        total_steps = len(dataloader) * self.num_epochs // self.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info("=" * 70)
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info("=" * 70)
            
            epoch_metrics = []
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Load cached reference logits
                cache_file = cache_dir / f"batch_{batch_idx:05d}.pkl"
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                ref_chosen_logps = cache_data['chosen_logps'].to(self.device)
                ref_rejected_logps = cache_data['rejected_logps'].to(self.device)
                
                # Move batch to device
                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                chosen_labels = chosen_input_ids.clone()
                
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                rejected_labels = rejected_input_ids.clone()
                
                # Compute policy logits
                policy_chosen_logps = self.get_batch_logps(
                    self.model, chosen_input_ids, chosen_attention_mask, chosen_labels
                )
                policy_rejected_logps = self.get_batch_logps(
                    self.model, rejected_input_ids, rejected_attention_mask, rejected_labels
                )
                
                # Compute DPO loss
                loss, metrics = self.compute_dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                
                # Backward
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                # Update weights
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if global_step % 10 == 0:
                        logger.info(f"Step {global_step}: loss={metrics['loss']:.4f}, acc={metrics['reward_accuracy']:.2%}")
                    
                    # Save checkpoint
                    if global_step % self.save_steps == 0:
                        self.save_checkpoint(global_step)
                    
                    # Evaluation
                    if global_step % self.eval_steps == 0 and hasattr(self, 'val_dataloader'):
                        val_loss = self.train_with_cached_logits(
                            self.val_dataloader, self.val_cache_dir, "validation", is_training=False
                        )
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
            avg_metrics = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0].keys()}
            logger.info(f"Epoch {epoch + 1} avg: {avg_metrics}")
            
            self.save_checkpoint(f"epoch_{epoch + 1}")
        
        # Save final
        self.save_final_model()
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info("=" * 70)
    
    def save_checkpoint(self, step, is_best=False):
        """Save model checkpoint."""
        if is_best:
            checkpoint_dir = self.output_dir / "best_model"
        else:
            checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"✓ Saved checkpoint to {checkpoint_dir}")
    
    def save_final_model(self):
        """Save final model."""
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"✓ Final model saved to {final_dir}")
    
    def train(self):
        """Main training pipeline."""
        logger.info("=" * 70)
        logger.info("Sequential DPO Training")
        logger.info("=" * 70)
        logger.info(f"Model: {self.sft_model_path}")
        logger.info(f"Cache dir: {self.cache_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info("=" * 70)
        
        # Load data
        tokenizer = self.load_tokenizer()
        train_dataloader, val_dataloader = create_dpo_dataloaders(
            self.train_data_path,
            self.val_data_path,
            tokenizer,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        
        self.val_dataloader = val_dataloader  # Save for evaluation
        
        logger.info(f"Train batches: {len(train_dataloader)}")
        logger.info(f"Val batches: {len(val_dataloader)}")
        
        # Phase 1: Cache reference logits
        self.cache_reference_logits(train_dataloader, self.train_cache_dir, "train")
        self.cache_reference_logits(val_dataloader, self.val_cache_dir, "validation")
        
        # Phase 2: Train with cached logits
        self.train_with_cached_logits(train_dataloader, self.train_cache_dir, "train", is_training=True)


def main():
    parser = argparse.ArgumentParser(description="Sequential DPO Training with Cached Reference Logits")
    
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache/reference_logits")
    parser.add_argument("--output_dir", type=str, default="./models/dpo_hallucination_resistant_sequential")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    trainer = SequentialDPOTrainer(
        sft_model_path=args.sft_model_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        max_length=args.max_length,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
    )
    
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
