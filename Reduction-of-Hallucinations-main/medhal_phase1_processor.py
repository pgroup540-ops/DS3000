"""
MedHal Phase 1 Data Engineering Script

This script processes the medhal_preprocessed.csv dataset and transforms it into
the phase1 format required for SFT (Supervised Fine-Tuning) training.

Input format (medhal_preprocessed.csv):
  - id: Unique identifier
  - inner_id: Inner identifier
  - full_text: The medical text (contains clinical note and/or summary)
  - label: 0 = hallucinated, 1 = factual
  - synthetic: Whether the example is synthetic

Output format (phase1_data/sft/):
  - id: Unique identifier
  - clinical_note: The input clinical text
  - model_summary: The output summary/response
  - label: "factual" or "hallucinated"
  - hallucination_type: Type of hallucination (if applicable)
  - evidence_stats: Evidence statistics (empty for now)
"""

import pandas as pd
import numpy as np
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List
import sys


class MedHalPhase1Processor:
    """Process MedHal dataset for Phase 1 training."""
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        Initialize the processor.
        
        Args:
            train_ratio: Ratio of data for training set (default: 0.7)
            val_ratio: Ratio of data for validation set (default: 0.15)
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        
        self.stats = {
            "total_records": 0,
            "factual_records": 0,
            "hallucinated_records": 0,
            "train_records": 0,
            "val_records": 0,
            "test_records": 0,
            "skipped_records": 0,
        }
    
    def detect_hallucination_type(self, text: str, is_hallucinated: bool) -> str:
        """
        Detect the type of hallucination based on text patterns.
        
        Args:
            text: The full text
            is_hallucinated: Whether the example is labeled as hallucinated
            
        Returns:
            Hallucination type string
        """
        if not is_hallucinated:
            return ""
        
        text_lower = text.lower()
        
        # Check for common hallucination patterns
        if "however, according to" in text_lower or "but according to" in text_lower:
            return "contradiction"
        elif "not mentioned" in text_lower or "no evidence" in text_lower:
            return "fabrication"
        elif "incorrect" in text_lower or "wrong" in text_lower:
            return "error"
        else:
            return "hallucination"
    
    def parse_text_with_separator(self, text: str) -> Tuple[str, str]:
        """
        Parse text that contains explicit separators between note and summary.
        
        Args:
            text: Input text that might contain separators
            
        Returns:
            Tuple of (clinical_note, model_summary)
        """
        # Common separator patterns
        separators = [
            r"however,?\s+according to[^:]*:\s*",
            r"but\s+according to[^:]*:\s*",
            r"###\s*",
            r"---+\s*",
            r"Summary:\s*",
            r"Answer:\s*",
            r"Response:\s*",
        ]
        
        for sep_pattern in separators:
            match = re.search(sep_pattern, text, re.IGNORECASE)
            if match:
                split_pos = match.end()
                clinical_note = text[:match.start()].strip()
                model_summary = text[split_pos:].strip()
                
                if clinical_note and model_summary:
                    return clinical_note, model_summary
        
        return None, None
    
    def create_note_summary_pair(self, text: str, label: int) -> Tuple[str, str]:
        """
        Create clinical_note and model_summary from the full_text.
        
        For hallucinated examples (label=0), we try to extract both the correct
        statement and the hallucinated statement.
        
        For factual examples (label=1), we use the text as clinical_note and
        create a simple summary.
        
        Args:
            text: The full_text from the dataset
            label: 0 = hallucinated, 1 = factual
            
        Returns:
            Tuple of (clinical_note, model_summary)
        """
        text = text.strip()
        
        # Try to parse with separators first
        clinical_note, model_summary = self.parse_text_with_separator(text)
        
        if clinical_note and model_summary:
            return clinical_note, model_summary
        
        # If no separator found, use different strategies based on label
        if label == 1:  # Factual
            # For factual examples, treat the text as a clinical fact/note
            # and create a simple restatement as summary
            sentences = text.split('. ')
            if len(sentences) > 1:
                # Use first part as note, rest as summary
                clinical_note = sentences[0] + '.'
                model_summary = ' '.join(sentences[1:]).strip()
                if not model_summary.endswith('.'):
                    model_summary += '.'
            else:
                # Single sentence - use as both (this is a teaching example)
                clinical_note = text
                model_summary = text
            
            return clinical_note, model_summary
        
        else:  # Hallucinated (label=0)
            # For hallucinated examples without clear separator,
            # treat entire text as the clinical note, and we'll need
            # to generate the hallucinated version through augmentation
            # For now, we'll use the text as the note and mark it for review
            clinical_note = text
            model_summary = text  # Will be replaced by augmentation
            
            return clinical_note, model_summary
    
    def process_record(self, row: pd.Series) -> Dict:
        """
        Process a single record from the MedHal dataset.
        
        Args:
            row: A pandas Series representing one row
            
        Returns:
            Processed record dictionary
        """
        # Extract basic fields
        record_id = row['id']
        full_text = row['full_text']
        label_numeric = row['label']
        synthetic = row['synthetic']
        
        # Convert label to string format
        is_factual = (label_numeric == 1)
        label_str = "factual" if is_factual else "hallucinated"
        
        # Parse text to extract clinical_note and model_summary
        clinical_note, model_summary = self.create_note_summary_pair(
            full_text, 
            label_numeric
        )
        
        # Detect hallucination type
        hallucination_type = self.detect_hallucination_type(
            full_text, 
            not is_factual
        )
        
        # Create processed record
        processed = {
            "id": record_id,
            "clinical_note": clinical_note,
            "model_summary": model_summary,
            "label": label_str,
            "hallucination_type": hallucination_type,
            "evidence_stats": "",  # Empty for now, can be filled by evidence_annotator
        }
        
        return processed
    
    def split_dataset(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.
        
        Stratified split to maintain label distribution.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Shuffle the dataset
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Separate by label for stratified split
        factual_df = df_shuffled[df_shuffled['label'] == 'factual']
        hallucinated_df = df_shuffled[df_shuffled['label'] == 'hallucinated']
        
        # Calculate split sizes for each label
        def split_by_label(label_df):
            n = len(label_df)
            train_size = int(n * self.train_ratio)
            val_size = int(n * self.val_ratio)
            
            train = label_df[:train_size]
            val = label_df[train_size:train_size + val_size]
            test = label_df[train_size + val_size:]
            
            return train, val, test
        
        factual_train, factual_val, factual_test = split_by_label(factual_df)
        hall_train, hall_val, hall_test = split_by_label(hallucinated_df)
        
        # Combine and shuffle each split
        train_df = pd.concat([factual_train, hall_train]).sample(frac=1, random_state=42)
        val_df = pd.concat([factual_val, hall_val]).sample(frac=1, random_state=42)
        test_df = pd.concat([factual_test, hall_test]).sample(frac=1, random_state=42)
        
        return train_df, val_df, test_df
    
    def process_dataset(
        self,
        input_path: str,
        output_dir: str,
        max_records: int = None
    ):
        """
        Process the entire MedHal dataset.
        
        Args:
            input_path: Path to medhal_preprocessed.csv
            output_dir: Output directory for processed files
            max_records: Maximum number of records to process (for testing)
        """
        print("=" * 70)
        print("MedHal Phase 1 Data Engineering")
        print("=" * 70)
        print(f"Input: {input_path}")
        print(f"Output: {output_dir}")
        print()
        
        # Read the dataset
        print("Loading dataset...")
        if max_records:
            df = pd.read_csv(input_path, nrows=max_records)
            print(f"  Loaded {len(df)} records (limited to {max_records})")
        else:
            df = pd.read_csv(input_path)
            print(f"  Loaded {len(df)} records")
        
        self.stats["total_records"] = len(df)
        
        # Show label distribution
        print("\nLabel distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            label_name = "Factual" if label == 1 else "Hallucinated"
            print(f"  {label_name} (label={label}): {count}")
        
        # Process all records
        print("\nProcessing records...")
        processed_records = []
        
        for idx, row in df.iterrows():
            try:
                processed = self.process_record(row)
                processed_records.append(processed)
                
                # Update stats
                if processed['label'] == 'factual':
                    self.stats["factual_records"] += 1
                else:
                    self.stats["hallucinated_records"] += 1
                
                # Progress indicator
                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} records...")
                    
            except Exception as e:
                print(f"  Warning: Skipped record {row['id']}: {e}")
                self.stats["skipped_records"] += 1
                continue
        
        print(f"  Completed: {len(processed_records)} records processed")
        
        # Create dataframe
        processed_df = pd.DataFrame(processed_records)
        
        # Split into train/val/test
        print("\nSplitting dataset...")
        train_df, val_df, test_df = self.split_dataset(processed_df)
        
        self.stats["train_records"] = len(train_df)
        self.stats["val_records"] = len(val_df)
        self.stats["test_records"] = len(test_df)
        
        print(f"  Training set: {len(train_df)} records")
        print(f"  Validation set: {len(val_df)} records")
        print(f"  Test set: {len(test_df)} records")
        
        # Create output directories
        output_path = Path(output_dir)
        sft_dir = output_path / "sft"
        sft_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the splits
        print("\nSaving processed data...")
        
        train_path = sft_dir / "train_set_processed.csv"
        val_path = sft_dir / "validation_set_processed.csv"
        test_path = sft_dir / "test_set_processed.csv"
        
        train_df.to_csv(train_path, index=False)
        print(f"  ✓ Saved training set: {train_path}")
        
        val_df.to_csv(val_path, index=False)
        print(f"  ✓ Saved validation set: {val_path}")
        
        test_df.to_csv(test_path, index=False)
        print(f"  ✓ Saved test set: {test_path}")
        
        # Print summary
        self.print_summary()
        
        # Show sample output
        print("\n" + "=" * 70)
        print("Sample Output (first training example):")
        print("=" * 70)
        sample = train_df.iloc[0]
        print(f"ID: {sample['id']}")
        print(f"Label: {sample['label']}")
        print(f"Hallucination Type: {sample['hallucination_type']}")
        print(f"\nClinical Note:\n  {sample['clinical_note'][:200]}...")
        print(f"\nModel Summary:\n  {sample['model_summary'][:200]}...")
        print("=" * 70)
    
    def print_summary(self):
        """Print processing summary statistics."""
        print("\n" + "=" * 70)
        print("Processing Summary")
        print("=" * 70)
        print(f"Total records processed: {self.stats['total_records']}")
        print(f"  Factual records: {self.stats['factual_records']}")
        print(f"  Hallucinated records: {self.stats['hallucinated_records']}")
        print(f"  Skipped records: {self.stats['skipped_records']}")
        print(f"\nDataset splits:")
        print(f"  Training: {self.stats['train_records']} ({self.train_ratio*100:.0f}%)")
        print(f"  Validation: {self.stats['val_records']} ({self.val_ratio*100:.0f}%)")
        print(f"  Test: {self.stats['test_records']} ({self.test_ratio*100:.0f}%)")
        print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process MedHal dataset for Phase 1 training"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="./sets/medhal_preprocessed.csv",
        help="Path to medhal_preprocessed.csv (default: ./sets/medhal_preprocessed.csv)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./phase1_data_medhal",
        help="Output directory for processed data (default: ./phase1_data_medhal)"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio of data for training set (default: 0.7)"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio of data for validation set (default: 0.15)"
    )
    
    # Set default to 10,000 for max_records
    parser.add_argument(
        "--max-records",
        type=int,
        default=10000,
        help="Maximum number of records to process (default: 10000)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        print("Error: train_ratio + val_ratio must be < 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Create processor
    processor = MedHalPhase1Processor(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    # Process dataset
    try:
        processor.process_dataset(
            args.input,
            args.output_dir,
            args.max_records  # This limits the number of records processed
        )
        print("\n✓ Phase 1 data engineering completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Review the processed data in: {args.output_dir}")
        print(f"  2. Run Stage A SFT training with:")
        print(f"     python stage_a_sft_training.py \\")
        print(f"       --data_dir {args.output_dir}/sft")
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
