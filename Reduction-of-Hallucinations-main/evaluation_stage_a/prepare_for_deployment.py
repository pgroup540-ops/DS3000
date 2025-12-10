"""
Prepare Stage A Model for Deployment
=====================================

Merge LoRA weights and optimize the model for production deployment.

Usage:
    python prepare_for_deployment.py \
        --model_path ./models/sft_specialist/final_model \
        --output_dir ./models/sft_specialist_merged
"""

import argparse
import logging
import torch
from pathlib import Path
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_lora_weights(model_path: str, output_dir: str):
    """
    Merge LoRA adapter weights into base model.
    
    This creates a single merged model without adapters,
    which is faster for inference and easier to deploy.
    
    Args:
        model_path: Path to model with LoRA adapters
        output_dir: Path to save merged model
    """
    logger.info("="*60)
    logger.info("MERGING LORA WEIGHTS INTO BASE MODEL")
    logger.info("="*60)
    
    # Load LoRA model
    logger.info(f"\nLoading LoRA model from {model_path}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu"  # Load on CPU for merging
    )
    logger.info("✓ Model loaded")
    
    # Merge LoRA weights
    logger.info("\nMerging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()
    logger.info("✓ LoRA weights merged")
    
    # Save merged model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving merged model to {output_dir}")
    merged_model.save_pretrained(output_dir)
    logger.info("✓ Model saved")
    
    # Save tokenizer
    logger.info("\nSaving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_dir)
    logger.info("✓ Tokenizer saved")
    
    logger.info("\n" + "="*60)
    logger.info("MERGE COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nMerged model saved to: {output_dir}")
    logger.info("\nThis merged model:")
    logger.info("  - Contains no LoRA adapters (faster inference)")
    logger.info("  - Is easier to deploy and share")
    logger.info("  - Can be loaded with standard transformers.AutoModel")
    logger.info("\nNext: Use this merged model for deployment\n")


def test_merged_model(model_path: str):
    """
    Test the merged model with a sample inference.
    
    Args:
        model_path: Path to merged model
    """
    logger.info("="*60)
    logger.info("TESTING MERGED MODEL")
    logger.info("="*60)
    
    from transformers import AutoModelForCausalLM
    
    logger.info(f"\nLoading merged model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("✓ Model loaded")
    
    # Test inference
    test_prompt = "Clinical Note: Patient reports fever of 38.5°C and cough.\n\nSummary:"
    
    logger.info(f"\nTest prompt:\n{test_prompt}")
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    logger.info("\nGenerating summary...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = generated_text.replace(test_prompt, "").strip()
    
    logger.info(f"\nGenerated summary:\n{summary}")
    
    logger.info("\n" + "="*60)
    logger.info("TEST PASSED!")
    logger.info("="*60)
    logger.info("\nMerged model is working correctly.\n")


def get_model_info(model_path: str):
    """
    Display information about the model.
    
    Args:
        model_path: Path to model
    """
    import json
    
    logger.info("="*60)
    logger.info("MODEL INFORMATION")
    logger.info("="*60)
    
    # Check for adapter config (LoRA model)
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        logger.info("\n✓ This is a LoRA model (has adapter_config.json)")
        with open(adapter_config_path) as f:
            config = json.load(f)
        logger.info(f"  - LoRA rank (r): {config.get('r', 'N/A')}")
        logger.info(f"  - LoRA alpha: {config.get('lora_alpha', 'N/A')}")
        logger.info(f"  - Target modules: {config.get('target_modules', 'N/A')}")
        logger.info("\n→ Recommendation: Merge LoRA for deployment")
    else:
        logger.info("\n✓ This is a merged/base model (no LoRA adapters)")
        logger.info("→ Ready for deployment as-is")
    
    # Check model size
    model_files = list(Path(model_path).glob("*.bin")) + list(Path(model_path).glob("*.safetensors"))
    if model_files:
        total_size = sum(f.stat().st_size for f in model_files)
        logger.info(f"\nModel size: {total_size / (1024**3):.2f} GB")
    
    logger.info("")


def main():
    """Main preparation function."""
    parser = argparse.ArgumentParser(description="Prepare Stage A Model for Deployment")
    
    parser.add_argument(
        "--model_path",
        default="./models/sft_specialist/final_model",
        help="Path to Stage A model"
    )
    parser.add_argument(
        "--output_dir",
        default="./models/sft_specialist_merged",
        help="Path to save merged model"
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Skip merge (if already merged)"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only test the model"
    )
    parser.add_argument(
        "--info_only",
        action="store_true",
        help="Only show model info"
    )
    
    args = parser.parse_args()
    
    # Show model info
    if args.info_only:
        get_model_info(args.model_path)
        return
    
    # Test only
    if args.test_only:
        test_merged_model(args.model_path)
        return
    
    # Full preparation workflow
    logger.info("\n" + "="*60)
    logger.info("STAGE A MODEL PREPARATION FOR DEPLOYMENT")
    logger.info("="*60 + "\n")
    
    # Step 1: Show info
    get_model_info(args.model_path)
    
    # Step 2: Merge if not skipped
    if not args.skip_merge:
        merge_lora_weights(args.model_path, args.output_dir)
        test_model_path = args.output_dir
    else:
        logger.info("Skipping merge (--skip_merge flag set)")
        test_model_path = args.model_path
    
    # Step 3: Test merged model
    test_merged_model(test_model_path)
    
    # Final summary
    logger.info("="*60)
    logger.info("PREPARATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nDeployment-ready model: {args.output_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Deploy API: python deploy_api.py --model_path " + args.output_dir)
    logger.info("  2. Or use directly: python sft_inference.py --model_path " + args.output_dir)
    logger.info("")


if __name__ == "__main__":
    main()
