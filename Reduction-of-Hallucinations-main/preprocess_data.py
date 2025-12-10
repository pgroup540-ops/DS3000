"""
Main Data Preprocessing Pipeline
Combines normalization, PHI redaction, and augmentation modules.
"""

import pandas as pd
import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

from text_normalizer import TextNormalizer
from phi_redactor import PHIRedactor
from adversarial_augmenter import AdversarialAugmenter
from evidence_annotator import EvidenceAnnotator
from dpo_triplet_generator import DPOTripletGenerator


class DataPreprocessor:
    """Main preprocessing pipeline for clinical text data."""
    
    def __init__(
        self,
        normalize: bool = True,
        redact_phi: bool = False,
        generate_adversarial: bool = False,
        generate_evidence: bool = False,
        generate_dpo_triplets: bool = False,
        adversarial_ratio: float = 0.5,
        phi_mask_style: str = "category",
        output_format: str = "pairs"
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            normalize: Whether to normalize text
            redact_phi: Whether to redact PHI
            generate_adversarial: Whether to generate adversarial examples
            generate_evidence: Whether to annotate with evidence
            generate_dpo_triplets: Whether to generate DPO triplets instead of separate rows
            adversarial_ratio: Ratio of adversarial examples to generate (0-1)
            phi_mask_style: PHI masking style ("category", "hash", "generic")
            output_format: Output format ("pairs" for SFT or "triplets" for DPO)
        """
        self.normalize = normalize
        self.redact_phi = redact_phi
        self.generate_adversarial = generate_adversarial
        self.generate_evidence = generate_evidence
        self.generate_dpo_triplets = generate_dpo_triplets
        self.adversarial_ratio = adversarial_ratio
        self.output_format = output_format
        
        # Initialize modules
        self.normalizer = TextNormalizer() if normalize else None
        self.phi_redactor = PHIRedactor(mask_style=phi_mask_style) if redact_phi else None
        self.augmenter = AdversarialAugmenter() if generate_adversarial else None
        self.evidence_annotator = EvidenceAnnotator() if generate_evidence else None
        self.triplet_generator = DPOTripletGenerator() if generate_dpo_triplets else None
        
        self.stats = {
            "total_records": 0,
            "normalized": 0,
            "phi_redacted": 0,
            "adversarial_generated": 0,
            "evidence_annotated": 0,
            "dpo_triplets_generated": 0,
            "invalid_triplets": 0,
        }
    
    def process_text(self, text: str) -> str:
        """
        Apply text normalization and PHI redaction to a single text.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize
        if self.normalize and self.normalizer:
            text = self.normalizer.normalize(text)
            self.stats["normalized"] += 1
        
        # Redact PHI
        if self.redact_phi and self.phi_redactor:
            text, _ = self.phi_redactor.redact_all(text)
            self.stats["phi_redacted"] += 1
        
        return text
    
    def process_record(self, record: Dict) -> Dict:
        """
        Process a single data record.
        
        Args:
            record: Dictionary containing clinical_note and model_summary
            
        Returns:
            Processed record
        """
        processed = record.copy()
        
        # Process clinical note
        if "clinical_note" in processed:
            processed["clinical_note"] = self.process_text(processed["clinical_note"])
        
        # Process summary
        if "model_summary" in processed:
            processed["model_summary"] = self.process_text(processed["model_summary"])
        
        return processed
    
    def generate_dpo_triplets_batch(
        self,
        df: pd.DataFrame,
        split: str = "train"
    ) -> List[Dict]:
        """
        Generate DPO triplets from the dataset.
        
        Args:
            df: Input dataframe
            split: Dataset split (only augment train set)
            
        Returns:
            List of DPO triplets
        """
        triplets = []
        
        # Only generate triplets for training data
        if split != "train":
            return triplets
        
        if self.generate_dpo_triplets and self.triplet_generator:
            factual_examples = df[df["label"] == "factual"]
            num_to_generate = int(len(factual_examples) * self.adversarial_ratio)
            
            for _, row in factual_examples.head(num_to_generate).iterrows():
                record_triplets = self.triplet_generator.generate_triplet(
                    record_id=row["id"],
                    clinical_note=row["clinical_note"],
                    factual_summary=row["model_summary"],
                    adversarial_strategies=["entity_swap", "negation_invert", "fabrication"]
                )
                
                for triplet in record_triplets:
                    # Validate triplet
                    is_valid, error = self.triplet_generator.validate_triplet(triplet)
                    
                    if is_valid:
                        triplets.append(triplet)
                        self.stats["dpo_triplets_generated"] += 1
                    else:
                        self.stats["invalid_triplets"] += 1
        
        return triplets
    
    def generate_augmented_examples(
        self,
        df: pd.DataFrame,
        split: str = "train"
    ) -> List[Dict]:
        """
        Generate augmented examples from the dataset.
        
        Args:
            df: Input dataframe
            split: Dataset split (only augment train set)
            
        Returns:
            List of augmented examples
        """
        augmented_examples = []
        
        # Only generate augmentations for training data
        if split != "train":
            return augmented_examples
        
        # Generate adversarial negatives from factual examples (independent rows for SFT)
        if self.generate_adversarial and self.augmenter and not self.generate_dpo_triplets:
            factual_examples = df[df["label"] == "factual"]
            num_to_generate = int(len(factual_examples) * self.adversarial_ratio)
            
            for _, row in factual_examples.head(num_to_generate).iterrows():
                strategies = ["entity_swap", "negation_invert", "fabrication"]
                for strategy in strategies:
                    adversarial = self.augmenter.generate_adversarial_negative(
                        row["clinical_note"],
                        row["model_summary"],
                        strategy=strategy
                    )
                    
                    augmented_examples.append({
                        "id": f"{row['id']}_adv_{strategy}",
                        "clinical_note": adversarial["clinical_note"],
                        "model_summary": adversarial["model_summary"],
                        "label": adversarial["label"],
                        "hallucination_type": adversarial["hallucination_type"],
                    })
                    
                    self.stats["adversarial_generated"] += 1
        
        # Generate evidence-annotated positives from factual examples
        if self.generate_evidence and self.evidence_annotator:
            factual_examples = df[df["label"] == "factual"]
            
            for _, row in factual_examples.iterrows():
                evidence_example = self.evidence_annotator.generate_evidence_augmented_positive(
                    row["clinical_note"],
                    row["model_summary"]
                )
                
                augmented_examples.append({
                    "id": f"{row['id']}_evidence",
                    "clinical_note": evidence_example["clinical_note"],
                    "model_summary": evidence_example["model_summary"],
                    "label": evidence_example["label"],
                    "hallucination_type": "",
                    "evidence_stats": json.dumps(evidence_example["statistics"])
                })
                
                self.stats["evidence_annotated"] += 1
        
        return augmented_examples
    
    def process_dataset(
        self,
        input_path: str,
        output_path: str,
        split: str = "train"
    ):
        """
        Process an entire dataset file.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output CSV file
            split: Dataset split name
        """
        print(f"\nProcessing {split} set: {input_path}")
        
        # Read data
        df = pd.read_csv(input_path)
        self.stats["total_records"] += len(df)
        
        print(f"  Loaded {len(df)} records")
        
        # Process each record
        processed_records = []
        for _, row in df.iterrows():
            processed = self.process_record(row.to_dict())
            processed_records.append(processed)
        
        # Create processed dataframe
        processed_df = pd.DataFrame(processed_records)
        
        # Generate augmented examples based on output format
        if self.output_format == "triplets" and self.generate_dpo_triplets:
            # Generate DPO triplets
            dpo_triplets = self.generate_dpo_triplets_batch(df, split)
            
            if dpo_triplets:
                # Convert triplets to rows
                triplet_rows = [self.triplet_generator.triplet_to_row(t) for t in dpo_triplets]
                triplets_df = pd.DataFrame(triplet_rows)
                processed_df = pd.concat([processed_df, triplets_df], ignore_index=True)
                print(f"  Generated {len(dpo_triplets)} DPO triplets")
        else:
            # Generate regular augmented examples for SFT
            augmented_examples = self.generate_augmented_examples(df, split)
            
            if augmented_examples:
                augmented_df = pd.DataFrame(augmented_examples)
                processed_df = pd.concat([processed_df, augmented_df], ignore_index=True)
                print(f"  Generated {len(augmented_examples)} augmented examples")

        # Save processed data
        if self.output_format == "triplets":
            # DPO/training libraries prefer JSONL for nested text data
            jsonl_path = output_path.replace(".csv", ".jsonl")
            processed_df.to_json(jsonl_path, orient="records", lines=True)
            print(f"  Saved {len(processed_df)} triplets to {jsonl_path}")
        else:
            # SFT data is usually fine as CSV
            processed_df.to_csv(output_path, index=False)
            print(f"  Saved {len(processed_df)} records to {output_path}")
    
    def process_all_splits(
        self,
        data_dir: str,
        output_dir: str,
        splits: List[str] = None
    ):
        """
        Process all dataset splits.
        
        Args:
            data_dir: Directory containing input CSV files
            output_dir: Directory for output CSV files
            splits: List of split names (default: train, validation, test)
        """
        if splits is None:
            splits = ["train", "validation", "test"]
        
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("="*60)
        print("Starting Data Preprocessing Pipeline")
        print("="*60)
        print(f"Configuration:")
        print(f"  Normalize text: {self.normalize}")
        print(f"  Redact PHI: {self.redact_phi}")
        print(f"  Generate adversarial: {self.generate_adversarial}")
        print(f"  Generate DPO triplets: {self.generate_dpo_triplets}")
        print(f"  Generate evidence: {self.generate_evidence}")
        print(f"  Output format: {self.output_format}")
        
        for split in splits:
            input_file = data_path / f"{split}_set.csv"
            output_file = output_path / f"{split}_set_processed.csv"
            
            if input_file.exists():
                self.process_dataset(str(input_file), str(output_file), split)
            else:
                print(f"\n⚠ Warning: {input_file} not found, skipping")
        
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary statistics."""
        print("\n" + "="*60)
        print("Processing Summary")
        print("="*60)
        for key, value in self.stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print("="*60)


def main():
    """Main entry point for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess clinical text data with normalization, PHI redaction, and augmentation"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="Directory containing input CSV files (default: current directory)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed",
        help="Directory for output CSV files (default: ./processed)"
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Apply text normalization (default: True)"
    )
    
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable text normalization"
    )
    
    parser.add_argument(
        "--redact-phi",
        action="store_true",
        help="Redact PHI from text (default: False)"
    )
    
    parser.add_argument(
        "--phi-mask-style",
        type=str,
        choices=["category", "hash", "generic"],
        default="category",
        help="PHI masking style (default: category)"
    )
    
    parser.add_argument(
        "--generate-adversarial",
        action="store_true",
        help="Generate adversarial negative examples (default: False)"
    )
    
    parser.add_argument(
        "--adversarial-ratio",
        type=float,
        default=0.5,
        help="Ratio of adversarial examples to generate (default: 0.5)"
    )
    
    parser.add_argument(
        "--generate-evidence",
        action="store_true",
        help="Generate evidence-annotated examples (default: False)"
    )
    
    parser.add_argument(
        "--generate-dpo-triplets",
        action="store_true",
        help="Generate DPO triplets (prompt, chosen, rejected) instead of separate examples (default: False)"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["pairs", "triplets"],
        default="pairs",
        help="Output format: 'pairs' for SFT data or 'triplets' for DPO data (default: pairs)"
    )
    
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to process (default: train validation test)"
    )
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        normalize=args.normalize,
        redact_phi=args.redact_phi,
        generate_adversarial=args.generate_adversarial,
        generate_evidence=args.generate_evidence,
        generate_dpo_triplets=args.generate_dpo_triplets,
        adversarial_ratio=args.adversarial_ratio,
        phi_mask_style=args.phi_mask_style,
        output_format=args.output_format
    )
    
    # Process all splits
    try:
        preprocessor.process_all_splits(
            args.input_dir,
            args.output_dir,
            args.splits
        )
        print("\n✓ Preprocessing completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
