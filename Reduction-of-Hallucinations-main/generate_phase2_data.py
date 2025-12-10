"""
Generate Phase 2 DPO Training Data from Phase 1 Outputs
=========================================================

This script bridges Phase 1 (SFT) and Phase 2 (DPO) by:
1. Reading Phase 1 processed data (SFT format)
2. Generating DPO triplets (prompt, chosen, rejected)
3. Outputting JSONL files for DPO training

Usage:
    python generate_phase2_data.py
    
    # Or with custom paths:
    python generate_phase2_data.py \
        --phase1_dir "phase1_data/sft" \
        --phase2_dir "phase2_data/dpo" \
        --adversarial_ratio 1.0
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
import sys

from dpo_triplet_generator import DPOTripletGenerator


def ensure_directories(phase2_dir: str):
    """Create necessary directories if they don't exist."""
    phase2_path = Path(phase2_dir)
    phase2_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {phase2_path}")


def load_phase1_data(phase1_file: str) -> pd.DataFrame:
    """Load Phase 1 processed data."""
    if not Path(phase1_file).exists():
        raise FileNotFoundError(
            f"Phase 1 data not found: {phase1_file}\n"
            f"Please run Phase 1 preprocessing first."
        )
    
    df = pd.read_csv(phase1_file)
    print(f"✓ Loaded {len(df)} records from {phase1_file}")
    return df


def generate_dpo_triplets_from_phase1(
    df: pd.DataFrame,
    adversarial_ratio: float = 1.0,
    strategies: List[str] = None
) -> List[Dict]:
    """
    Generate DPO triplets from Phase 1 data.
    
    Args:
        df: Phase 1 dataframe
        adversarial_ratio: Ratio of factual examples to convert (0-1)
        strategies: Adversarial strategies to use
        
    Returns:
        List of DPO triplets
    """
    if strategies is None:
        strategies = ["entity_swap", "negation_invert", "fabrication"]
    
    generator = DPOTripletGenerator()
    
    # Filter to factual examples only
    factual_df = df[df['label'] == 'factual'].copy()
    print(f"✓ Found {len(factual_df)} factual examples")
    
    # Determine how many to process
    num_to_process = int(len(factual_df) * adversarial_ratio)
    factual_df = factual_df.head(num_to_process)
    
    print(f"✓ Generating DPO triplets for {num_to_process} examples...")
    print(f"  Strategies: {strategies}")
    
    all_triplets = []
    
    for idx, row in factual_df.iterrows():
        # Generate triplets for this example
        triplets = generator.generate_triplet(
            record_id=str(row.get('id', f'record_{idx}')),
            clinical_note=str(row['clinical_note']),
            factual_summary=str(row['model_summary']),
            adversarial_strategies=strategies
        )
        
        # Validate and add
        for triplet in triplets:
            is_valid, error = generator.validate_triplet(triplet)
            if is_valid:
                all_triplets.append(triplet)
            else:
                print(f"  Warning: Skipped invalid triplet: {error}")
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{num_to_process} examples...")
    
    print(f"✓ Generated {len(all_triplets)} valid DPO triplets")
    return all_triplets


def save_dpo_jsonl(triplets: List[Dict], output_path: str):
    """Save DPO triplets to JSONL file."""
    # Convert to JSONL format (only keep required fields)
    with open(output_path, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            jsonl_entry = {
                'prompt': triplet['prompt'],
                'chosen': triplet['chosen'],
                'rejected': triplet['rejected'],
            }
            f.write(json.dumps(jsonl_entry) + '\n')
    
    print(f"✓ Saved {len(triplets)} triplets to {output_path}")


def generate_phase2_data(
    phase1_dir: str = "phase1_data/sft",
    phase2_dir: str = "phase2_data/dpo",
    adversarial_ratio: float = 1.0,
    strategies: List[str] = None
):
    """
    Main function to generate Phase 2 DPO data from Phase 1 outputs.
    
    Args:
        phase1_dir: Directory containing Phase 1 processed data
        phase2_dir: Directory to save Phase 2 DPO data
        adversarial_ratio: Ratio of factual examples to convert
        strategies: Adversarial strategies to use
    """
    print("=" * 70)
    print("Phase 2 DPO Data Generation")
    print("=" * 70)
    print(f"Phase 1 directory: {phase1_dir}")
    print(f"Phase 2 directory: {phase2_dir}")
    print(f"Adversarial ratio: {adversarial_ratio}")
    print()
    
    # Ensure output directory exists
    ensure_directories(phase2_dir)
    
    # Process train set
    print("\n" + "-" * 70)
    print("Processing Training Set")
    print("-" * 70)
    train_file = Path(phase1_dir) / "train_set_processed.csv"
    train_df = load_phase1_data(str(train_file))
    
    train_triplets = generate_dpo_triplets_from_phase1(
        train_df,
        adversarial_ratio=adversarial_ratio,
        strategies=strategies
    )
    
    train_output = Path(phase2_dir) / "train_dpo.jsonl"
    save_dpo_jsonl(train_triplets, str(train_output))
    
    # Process validation set
    print("\n" + "-" * 70)
    print("Processing Validation Set")
    print("-" * 70)
    val_file = Path(phase1_dir) / "validation_set_processed.csv"
    val_df = load_phase1_data(str(val_file))
    
    val_triplets = generate_dpo_triplets_from_phase1(
        val_df,
        adversarial_ratio=adversarial_ratio,
        strategies=strategies
    )
    
    val_output = Path(phase2_dir) / "val_dpo.jsonl"
    save_dpo_jsonl(val_triplets, str(val_output))
    
    # Summary
    print("\n" + "=" * 70)
    print("Phase 2 Data Generation Complete!")
    print("=" * 70)
    print(f"Training triplets: {len(train_triplets)}")
    print(f"Validation triplets: {len(val_triplets)}")
    print(f"Total triplets: {len(train_triplets) + len(val_triplets)}")
    print()
    print("Output files:")
    print(f"  - {train_output}")
    print(f"  - {val_output}")
    print()
    print("Next step:")
    print("  Run Phase 2 (DPO) training with:")
    print(f"    python stage_b_dpo_training.py \\")
    print(f"        --train_data_path \"{train_output}\" \\")
    print(f"        --val_data_path \"{val_output}\"")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Phase 2 DPO data from Phase 1 outputs"
    )
    
    parser.add_argument(
        "--phase1_dir",
        type=str,
        default="phase1_data/sft",
        help="Directory containing Phase 1 processed data (default: phase1_data/sft)"
    )
    
    parser.add_argument(
        "--phase2_dir",
        type=str,
        default="phase2_data/dpo",
        help="Output directory for Phase 2 DPO data (default: phase2_data/dpo)"
    )
    
    parser.add_argument(
        "--adversarial_ratio",
        type=float,
        default=1.0,
        help="Ratio of factual examples to convert to DPO triplets (0-1, default: 1.0)"
    )
    
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["entity_swap", "negation_invert", "fabrication"],
        help="Adversarial strategies to use (default: entity_swap negation_invert fabrication)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 < args.adversarial_ratio <= 1.0:
        print("Error: adversarial_ratio must be between 0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Check if Phase 1 directory exists
    phase1_path = Path(args.phase1_dir)
    if not phase1_path.exists():
        print(f"Error: Phase 1 directory not found: {args.phase1_dir}", file=sys.stderr)
        print("\nPlease run Phase 1 preprocessing first:", file=sys.stderr)
        print("  python preprocess_data.py --generate-adversarial", file=sys.stderr)
        sys.exit(1)
    
    # Generate Phase 2 data
    try:
        generate_phase2_data(
            phase1_dir=args.phase1_dir,
            phase2_dir=args.phase2_dir,
            adversarial_ratio=args.adversarial_ratio,
            strategies=args.strategies
        )
    except Exception as e:
        print(f"\n✗ Error during Phase 2 data generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
