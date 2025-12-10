"""
DPO Dataset Loader
Converts JSONL triplet data into (prompt, chosen, rejected) format suitable for DPO training.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import pandas as pd


class DPODataset(Dataset):
    """
    Direct Preference Optimization Dataset.
    
    Processes JSONL data with triplets: (prompt, chosen, rejected)
    Where chosen is the factually correct response and rejected is the hallucinated response.
    """
    
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with triplets
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for full examples
            max_prompt_length: Maximum length for prompts alone
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        
        # Load data
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} DPO triplets from {jsonl_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a DPO training example with prompt, chosen, and rejected responses.
        """
        example = self.data[idx]
        
        # Extract triplet
        prompt = str(example.get('prompt', '')).strip()
        chosen = str(example.get('chosen', '')).strip()
        rejected = str(example.get('rejected', '')).strip()
        
        # Tokenize prompt
        prompt_encodings = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize chosen (full sequence: prompt + chosen)
        chosen_text = f"{prompt} {chosen}"
        chosen_encodings = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize rejected (full sequence: prompt + rejected)
        rejected_text = f"{prompt} {rejected}"
        rejected_encodings = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'prompt_input_ids': prompt_encodings['input_ids'].squeeze(),
            'prompt_attention_mask': prompt_encodings['attention_mask'].squeeze(),
            'chosen_input_ids': chosen_encodings['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_encodings['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_encodings['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_encodings['attention_mask'].squeeze(),
        }


class DPODataCollator:
    """Custom data collator for DPO training."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch and pad sequences."""
        
        # Stack chosen and rejected sequences
        chosen_input_ids = torch.stack([item['chosen_input_ids'] for item in batch])
        chosen_attention_mask = torch.stack([item['chosen_attention_mask'] for item in batch])
        
        rejected_input_ids = torch.stack([item['rejected_input_ids'] for item in batch])
        rejected_attention_mask = torch.stack([item['rejected_attention_mask'] for item in batch])
        
        return {
            'chosen_input_ids': chosen_input_ids,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_input_ids': rejected_input_ids,
            'rejected_attention_mask': rejected_attention_mask,
        }


def create_dpo_dataloaders(
    train_jsonl: str,
    val_jsonl: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for DPO.
    
    Args:
        train_jsonl: Path to training JSONL file
        val_jsonl: Path to validation JSONL file
        tokenizer: Hugging Face tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    
    # Create datasets
    train_dataset = DPODataset(
        train_jsonl,
        tokenizer,
        max_length=max_length,
    )
    
    val_dataset = DPODataset(
        val_jsonl,
        tokenizer,
        max_length=max_length,
    )
    
    collator = DPODataCollator(tokenizer, max_length)
    
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


def convert_adversarial_to_dpo(
    adversarial_csv: str,
    output_jsonl: str,
):
    """
    Convert adversarial data from Phase 1 to DPO JSONL format.
    
    Args:
        adversarial_csv: Path to CSV with adversarial examples
        output_jsonl: Path to output JSONL file
    """
    df = pd.read_csv(adversarial_csv)
    
    triplets = []
    
    for idx, row in df.iterrows():
        # Only use factual and adversarial pairs
        if row.get('label') == 'factual' or pd.isna(row.get('label')):
            prompt = str(row.get('clinical_note', '')).strip()
            
            # Try to find corresponding adversarial example
            # For now, create entry if we have factual summary
            if pd.notna(row.get('model_summary')):
                chosen = str(row.get('model_summary', '')).strip()
                
                # Try to get rejected from adversarial or hallucinated version
                rejected = str(row.get('rejected_summary', row.get('hallucinated_summary', ''))).strip()
                
                if prompt and chosen and rejected:
                    triplets.append({
                        'prompt': prompt,
                        'chosen': chosen,
                        'rejected': rejected,
                    })
    
    # Write to JSONL
    with open(output_jsonl, 'w') as f:
        for triplet in triplets:
            f.write(json.dumps(triplet) + '\n')
    
    print(f"Converted {len(triplets)} triplets to {output_jsonl}")


if __name__ == "__main__":
    # Simple test
    from transformers import AutoTokenizer
    
    # Create sample JSONL file for testing
    sample_data = [
        {
            "prompt": "Clinical Note: Patient reports fever of 38.5°C and cough. Tested positive for influenza A.\n\nSummary:",
            "chosen": "The patient tested positive for influenza A and experienced fever.",
            "rejected": "The patient tested negative for influenza and recovered without symptoms."
        },
        {
            "prompt": "Clinical Note: Patient recovering from COVID-19 infection. No respiratory distress.\n\nSummary:",
            "chosen": "The patient is recovering from COVID-19 without respiratory distress.",
            "rejected": "The patient is suffering from severe respiratory distress due to COVID-19."
        }
    ]
    
    # Write sample data
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
        sample_path = f.name
    
    # Test dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = DPODataset(sample_path, tokenizer, max_length=256)
    
    print(f"\nDataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Chosen input IDs shape: {sample['chosen_input_ids'].shape}")
    print(f"Rejected input IDs shape: {sample['rejected_input_ids'].shape}")
    print(f"\nPrompt: {sample['prompt'][:80]}...")
    print(f"Chosen: {sample['chosen']}")
    print(f"Rejected: {sample['rejected']}")
    
    # Cleanup
    import os
    os.remove(sample_path)
