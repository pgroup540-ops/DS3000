"""
Evaluate Stage B DPO Model (Checkpoint-1300)
============================================

Evaluates the partially trained DPO model with comprehensive metrics.
"""

import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(checkpoint_path, base_model_path, device="cuda"):
    """Load DPO checkpoint with base model."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    logger.info(f"Base model: {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    try:
        # Try loading as PEFT model with base model
        logger.info("Loading as PEFT model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        
        # Load LoRA adapters from checkpoint
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            local_files_only=True,
        )
        logger.info("✓ Loaded PEFT model with checkpoint adapters")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    model.eval()
    
    return model, tokenizer


def generate_summary(model, tokenizer, clinical_note, max_length=512, device="cuda"):
    """Generate summary for a clinical note."""
    prompt = f"Summarize the following clinical note:\n\n{clinical_note}\n\nSummary:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract summary
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated summary (after "Summary:")
    if "Summary:" in full_text:
        summary = full_text.split("Summary:")[-1].strip()
    else:
        summary = full_text[len(prompt):].strip()
    
    return summary


def evaluate_model(
    checkpoint_path,
    base_model_path,
    test_data_path,
    output_dir,
    max_examples=50,
    device="cuda"
):
    """Evaluate the DPO model."""
    
    logger.info("=" * 70)
    logger.info("Stage B DPO Model Evaluation")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Test data: {test_data_path}")
    logger.info(f"Max examples: {max_examples}")
    logger.info("")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, base_model_path, device)
    
    # Load test data
    logger.info("Loading test data...")
    df = pd.read_csv(test_data_path)
    df = df.head(max_examples)
    logger.info(f"✓ Loaded {len(df)} test examples")
    logger.info("")
    
    # Generate summaries
    logger.info("Generating summaries...")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        clinical_note = row['clinical_note']
        reference_summary = row['model_summary']
        label = row['label']
        
        # Generate prediction
        predicted_summary = generate_summary(model, tokenizer, clinical_note, device=device)
        
        results.append({
            'id': row.get('id', idx),
            'clinical_note': clinical_note,
            'reference_summary': reference_summary,
            'predicted_summary': predicted_summary,
            'label': label,
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_csv = Path(output_dir) / "evaluation_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"✓ Results saved to {results_csv}")
    
    # Compute basic statistics
    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Total examples: {len(results_df)}")
    logger.info(f"Factual examples: {len(results_df[results_df['label'] == 'factual'])}")
    logger.info(f"Hallucinated examples: {len(results_df[results_df['label'] == 'hallucinated'])}")
    logger.info("")
    
    # Save summary
    summary = {
        'checkpoint': checkpoint_path,
        'base_model': base_model_path,
        'num_examples': len(results_df),
        'factual_count': len(results_df[results_df['label'] == 'factual']),
        'hallucinated_count': len(results_df[results_df['label'] == 'hallucinated']),
    }
    
    summary_path = Path(output_dir) / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Summary saved to {summary_path}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluation complete!")
    logger.info("=" * 70)
    logger.info(f"Results: {results_csv}")
    logger.info("")
    logger.info("Next step: Run manual assessment")
    logger.info(f"  python manual_assessment_tool.py --results_csv \"{results_csv}\"")
    logger.info("=" * 70)
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Stage B DPO Model")
    parser.add_argument("--checkpoint_path", type=str, 
                       default="models/dpo_hallucination_resistant_sequential/checkpoint-1300",
                       help="Path to DPO checkpoint")
    parser.add_argument("--base_model_path", type=str,
                       default="models/sft_specialist_fast_fp16/final_model",
                       help="Path to base Stage A model")
    parser.add_argument("--test_data", type=str,
                       default="phase1_data_medhal/sft/test_set_processed.csv",
                       help="Path to test data CSV")
    parser.add_argument("--output_dir", type=str,
                       default="evaluation_results_stage_b_checkpoint1300",
                       help="Output directory for results")
    parser.add_argument("--max_examples", type=int, default=50,
                       help="Maximum number of examples to evaluate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint_path,
        base_model_path=args.base_model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        device=args.device,
    )
