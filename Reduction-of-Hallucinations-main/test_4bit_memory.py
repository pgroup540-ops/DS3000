"""
Memory Test for 4-bit QLoRA DPO Training
=========================================

This script tests if both reference and active models can fit
in your 12GB GPU with 4-bit quantization before starting full training.

Usage:
    python test_4bit_memory.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_bytes(bytes_value):
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.2f} TB"


def test_4bit_loading():
    """Test loading both models with 4-bit quantization."""
    
    print("=" * 70)
    print("4-bit QLoRA Memory Test")
    print("=" * 70)
    print()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: No CUDA GPU detected!")
        print("This script requires a CUDA-capable GPU.")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {format_bytes(total_memory)}")
    print()
    
    # Model path
    model_path = "models/sft_specialist_fast_fp16/final_model"
    
    print(f"Model path: {model_path}")
    print()
    
    # Create 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("Quantization config:")
    print("  - 4-bit NormalFloat (nf4)")
    print("  - Double quantization: Yes")
    print("  - Compute dtype: bfloat16")
    print()
    
    try:
        # Load tokenizer
        print("-" * 70)
        print("Step 1: Loading tokenizer...")
        print("-" * 70)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")
        print()
        
        # Initial memory
        torch.cuda.reset_peak_memory_stats()
        initial_allocated = torch.cuda.memory_allocated()
        print(f"Initial GPU memory: {format_bytes(initial_allocated)}")
        print()
        
        # Load reference model
        print("-" * 70)
        print("Step 2: Loading REFERENCE model (4-bit, frozen)...")
        print("-" * 70)
        reference_model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Freeze reference
        for param in reference_model.parameters():
            param.requires_grad = False
        reference_model.eval()
        
        ref_allocated = torch.cuda.memory_allocated()
        ref_reserved = torch.cuda.memory_reserved()
        
        print(f"✓ Reference model loaded")
        print(f"  Allocated: {format_bytes(ref_allocated)}")
        print(f"  Reserved: {format_bytes(ref_reserved)}")
        print()
        
        # Load active model
        print("-" * 70)
        print("Step 3: Loading ACTIVE model (4-bit, trainable)...")
        print("-" * 70)
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare for training
        model = prepare_model_for_kbit_training(model)
        model.train()
        
        active_allocated = torch.cuda.memory_allocated()
        active_reserved = torch.cuda.memory_reserved()
        
        print(f"✓ Active model loaded")
        print(f"  Allocated: {format_bytes(active_allocated)}")
        print(f"  Reserved: {format_bytes(active_reserved)}")
        print()
        
        # Calculate memory usage
        print("-" * 70)
        print("Step 4: Memory Analysis")
        print("-" * 70)
        
        total_allocated = active_allocated
        total_reserved = active_reserved
        free_memory = total_memory - active_reserved
        usage_percent = (active_reserved / total_memory) * 100
        
        print(f"Total allocated: {format_bytes(total_allocated)}")
        print(f"Total reserved: {format_bytes(total_reserved)}")
        print(f"Free memory: {format_bytes(free_memory)}")
        print(f"Usage: {usage_percent:.1f}%")
        print()
        
        # Safety check
        print("-" * 70)
        print("Step 5: Safety Check")
        print("-" * 70)
        
        # Need at least 2GB free for activations and gradients during training
        safety_buffer = 2 * 1024 * 1024 * 1024  # 2GB
        
        if free_memory > safety_buffer:
            print(f"✅ SUCCESS! Models fit with {format_bytes(free_memory)} free")
            print(f"   (Need at least {format_bytes(safety_buffer)} for training)")
            print()
            print("=" * 70)
            print("🎉 Your 12GB GPU can run 4-bit QLoRA DPO training!")
            print("=" * 70)
            print()
            print("You can now run: .\\run_stage_b_4bit.ps1")
            return True
        else:
            print(f"⚠️  WARNING: Only {format_bytes(free_memory)} free")
            print(f"   Need at least {format_bytes(safety_buffer)} for safe training")
            print()
            print("=" * 70)
            print("Memory might be too tight. Consider:")
            print("1. Reducing max_length to 384 or 256")
            print("2. Using cloud GPU with more VRAM")
            print("=" * 70)
            return False
            
    except torch.cuda.OutOfMemoryError as e:
        print()
        print("=" * 70)
        print("❌ OUT OF MEMORY ERROR")
        print("=" * 70)
        print()
        print("The models don't fit in your GPU even with 4-bit quantization.")
        print()
        print("Options:")
        print("1. Try with shorter max_length (384 or 256 tokens)")
        print("2. Use cloud GPU (Lambda Labs: $3-5 for full training)")
        print("3. Try CPU training (very slow: 20+ days)")
        print()
        return False
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ ERROR: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_4bit_loading()
    exit(0 if success else 1)
