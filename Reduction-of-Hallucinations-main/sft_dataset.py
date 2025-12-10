"""
SFT Dataset Loader
Converts CSV training data into prompt-response pairs suitable for supervised fine-tuning.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, List
import json


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning Dataset.
    
    Converts clinical note + model summary pairs into input-target sequences.
    Input: clinical_note (prompt)
    Target: model_summary (the response to learn)
    """
    
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 512,
        use_evidence: bool = False,
        only_factual: bool = True
    ):
        """
        Args:
            csv_path: Path to the SFT training CSV file
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            use_evidence: If True, use evidence-annotated summaries when available
            only_factual: If True, only include factual examples (label='factual')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_evidence = use_evidence
        self.only_factual = only_factual
        
        # Load data
        self.data = pd.read_csv(csv_path)
        
        # Filter to only factual examples if requested
        if self.only_factual:
            self.data = self.data[
                (self.data['label'] == 'factual') | 
                (self.data['label'] == 'factual_with_evidence')
            ].reset_index(drop=True)
        
        # Use evidence summaries if available and requested
        if self.use_evidence:
            # Prefer evidence-annotated versions
            self.data['model_summary'] = self.data.apply(
                lambda row: row['model_summary'] 
                if pd.isna(row['model_summary']) or len(str(row['model_summary'])) == 0
                else row['model_summary'],
                axis=1
            )
        
        print(f"Loaded {len(self.data)} examples from {csv_path}")
        if self.only_factual:
            print(f"  Filtered to {len(self.data)} factual examples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a tokenized prompt-response pair.
        
        Format:
        "Clinical Note: <clinical_note>\n\nSummary: <model_summary>"
        """
        row = self.data.iloc[idx]
        
        clinical_note = str(row['clinical_note']).strip()
        model_summary = str(row['model_summary']).strip()
        
        # Construct the prompt-response format
        # For SFT, we want the model to learn to produce the summary given the clinical note
        prompt = f"Clinical Note: {clinical_note}\n\nSummary:"
        full_text = f"{prompt} {model_summary}"
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels: we want to train on the full sequence
        # In standard SFT, we compute loss on the entire sequence
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # Labels are the same as input_ids (the model learns to predict the next token)
        # You can optionally mask the prompt part if you want to only train on the response
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'clinical_note': clinical_note,
            'model_summary': model_summary
        }


class SFTDataCollator:
    """Custom data collator for SFT training."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch and pad sequences."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def create_sft_dataloaders(
    train_csv: str,
    val_csv: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 0,
    use_evidence: bool = False
):
    """
    Create train and validation dataloaders for SFT.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        tokenizer: Hugging Face tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        use_evidence: Whether to use evidence-annotated summaries
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = SFTDataset(
        train_csv,
        tokenizer,
        max_length=max_length,
        use_evidence=use_evidence,
        only_factual=True
    )
    
    val_dataset = SFTDataset(
        val_csv,
        tokenizer,
        max_length=max_length,
        use_evidence=use_evidence,
        only_factual=True
    )
    
    collator = SFTDataCollator(tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Simple test
    from transformers import AutoTokenizer
    
    # Test with a small model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = SFTDataset(
        "phase1_data/sft/train_set_processed.csv",
        tokenizer,
        max_length=256
    )
    
    print(f"\nDataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"\nClinical note: {sample['clinical_note'][:100]}...")
    print(f"Model summary: {sample['model_summary']}")
