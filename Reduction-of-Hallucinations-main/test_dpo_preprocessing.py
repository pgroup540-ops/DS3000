"""
Test script for DPO triplet preprocessing pipeline
"""

import pandas as pd
import tempfile
import os
from preprocess_data import DataPreprocessor

# Create sample test data
def create_sample_data(output_path: str, num_records: int = 5):
    """Create sample CSV data for testing."""
    clinical_notes = [
        "Patient reports mild chest pain for 2 days. ECG normal. No history of hypertension.",
        "Patient presents with persistent cough for 1 week. Chest X-ray shows pneumonia. Started antibiotics.",
        "Type 2 diabetes mellitus patient. HbA1c: 7.2%. Continue metformin 500mg BID.",
        "40-year-old female with migraine. Prescribed sumatriptan. No previous history.",
        "Post-operative day 3 from knee surgery. Incision healing well. Pain controlled with ibuprofen.",
    ]
    
    model_summaries = [
        "The patient has chest pain with normal ECG findings and no hypertension history.",
        "Patient has pneumonia diagnosed by chest X-ray and is on antibiotics.",
        "Patient is a type 2 diabetic with well-controlled glucose levels.",
        "40-year-old female presenting with migraine, treated with sumatriptan.",
        "Post-operative recovery is progressing well with good incision healing.",
    ]
    
    data = {
        "id": [f"test_{i:03d}" for i in range(num_records)],
        "clinical_note": clinical_notes[:num_records],
        "model_summary": model_summaries[:num_records],
        "label": ["factual"] * num_records,
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return output_path

# Test Run 1: SFT with Evidence
def test_run1_sft_with_evidence():
    print("\n" + "="*80)
    print("RUN 1: SFT with Evidence (Pairs)")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "train_set.csv")
        output_file = os.path.join(tmpdir, "train_set_processed.csv")
        
        create_sample_data(input_file)
        
        preprocessor = DataPreprocessor(
            normalize=True,
            redact_phi=True,
            generate_adversarial=False,
            generate_evidence=True,
            generate_dpo_triplets=False,
            output_format="pairs"
        )
        
        preprocessor.process_dataset(input_file, output_file, split="train")
        
        result_df = pd.read_csv(output_file)
        print(f"\nOutput shape: {result_df.shape}")
        print(f"Columns: {list(result_df.columns)}")
        print(f"\nFirst row:")
        print(result_df.iloc[0])
        print(f"\nProcessing stats:")
        for key, value in preprocessor.stats.items():
            print(f"  {key}: {value}")

# Test Run 2: DPO with Triplets
def test_run2_dpo_triplets():
    print("\n" + "="*80)
    print("RUN 2: DPO with Triplets")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "train_set.csv")
        output_file = os.path.join(tmpdir, "train_set_processed.csv")
        
        create_sample_data(input_file, num_records=3)  # Smaller sample
        
        preprocessor = DataPreprocessor(
            normalize=True,
            redact_phi=True,
            generate_adversarial=False,
            generate_evidence=True,
            generate_dpo_triplets=True,
            output_format="triplets",
            adversarial_ratio=1.0  # Generate triplets for all factual examples
        )
        
        preprocessor.process_dataset(input_file, output_file, split="train")
        
        result_df = pd.read_csv(output_file)
        print(f"\nOutput shape: {result_df.shape}")
        print(f"Columns: {list(result_df.columns)}")
        
        # Show DPO triplets
        dpo_rows = result_df[result_df["data_format"] == "dpo_triplet"]
        print(f"\nDPO Triplets: {len(dpo_rows)}")
        
        if len(dpo_rows) > 0:
            print(f"\nFirst triplet:")
            first_triplet = dpo_rows.iloc[0]
            print(f"  ID: {first_triplet['id']}")
            print(f"  Strategy: {first_triplet['strategy_used']}")
            print(f"  Prompt: {first_triplet['prompt'][:60]}...")
            print(f"  Chosen: {first_triplet['chosen'][:60]}...")
            print(f"  Rejected: {first_triplet['rejected'][:60]}...")
            print(f"  Hallucination Type: {first_triplet['hallucination_type']}")
        
        print(f"\nProcessing stats:")
        for key, value in preprocessor.stats.items():
            print(f"  {key}: {value}")

# Test Run 3: Evaluation (no augmentation)
def test_run3_evaluation():
    print("\n" + "="*80)
    print("RUN 3: Evaluation (No Augmentation)")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "test_set.csv")
        output_file = os.path.join(tmpdir, "test_set_processed.csv")
        
        create_sample_data(input_file)
        
        preprocessor = DataPreprocessor(
            normalize=True,
            redact_phi=True,
            generate_adversarial=False,
            generate_evidence=False,
            generate_dpo_triplets=False,
            output_format="pairs"
        )
        
        preprocessor.process_dataset(input_file, output_file, split="test")
        
        result_df = pd.read_csv(output_file)
        print(f"\nOutput shape: {result_df.shape}")
        print(f"Columns: {list(result_df.columns)}")
        print(f"\nFirst row:")
        print(result_df.iloc[0])
        print(f"\nProcessing stats:")
        for key, value in preprocessor.stats.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    test_run1_sft_with_evidence()
    test_run2_dpo_triplets()
    test_run3_evaluation()
    
    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("="*80)
