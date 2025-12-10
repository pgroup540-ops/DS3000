# Phase 2 Execution Guide: Complete Next Steps

**Project**: Reduction of Hallucinations in Medical LLMs  
**Current Status**: ✅ Phase 1 Complete | ⚠️ Phase 2 Ready for Execution  
**Last Updated**: 2025-11-21

---

## Table of Contents

1. [Quick Overview](#quick-overview)
2. [Step 1: Install Dependencies](#step-1-install-dependencies)
3. [Step 2: Stage A - SFT Training](#step-2-stage-a---supervised-fine-tuning-sft)
4. [Step 3: Verify Stage A](#step-3-verify-stage-a-output)
5. [Step 4: Stage B - DPO Training](#step-4-stage-b---direct-preference-optimization-dpo)
6. [Step 5: Evaluate Results](#step-5-evaluate-results)
7. [Mac-Specific Considerations](#mac-specific-considerations)
8. [Troubleshooting](#troubleshooting)
9. [Expected Timelines](#expected-timelines)

---

## Quick Overview

Your project has completed Phase 1 (data preparation) and is ready for Phase 2 (model training).

**Phase 2 consists of two sequential stages**:

1. **Stage A (SFT)**: Train base model on factual medical summaries
   - Input: 11 factual pairs
   - Output: Medical Specialist model
   - Time: 1-3 hours
   - GPU: 20GB+ VRAM (or CPU on Mac)

2. **Stage B (DPO)**: Fine-tune Stage A model on hard negatives
   - Input: 13 hard negative triplets
   - Output: Hallucination-Resistant Expert
   - Time: 2-4 hours
   - GPU: 24-32GB VRAM (or CPU on Mac)

**Total Expected Time**: 3-7 hours (GPU) or 6-12 hours (CPU)

---

## Step 1: Install Dependencies

### Objective
Install all required Python packages for training.

### Commands

```bash
# Navigate to project directory (if not already there)
cd path/to/Reduction-of-Hallucinations

# Install dependencies
pip install -r requirements_training.txt
```

### What Gets Installed
- `torch>=2.0.0` - PyTorch deep learning framework
- `transformers>=4.35.0` - Hugging Face model library
- `peft>=0.6.0` - LoRA (Parameter-Efficient Fine-Tuning)
- `accelerate>=0.24.0` - Multi-GPU support utilities
- `tqdm>=4.65.0` - Progress bars
- `bitsandbytes>=0.40.0` - 8-bit quantization (optional)

### Verification

```bash
# Verify installation
python -c "
import torch
import transformers
import peft
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ PEFT: {peft.__version__}')
print('✅ All dependencies installed!')
"
```

### Troubleshooting Installation

**Issue**: `ModuleNotFoundError: No module named 'torch'`
```bash
# Solution: Install PyTorch first
pip install torch torchvision torchaudio
```

**Issue**: Slow installation
```bash
# Use cached packages
pip install --no-cache-dir -r requirements_training.txt
```

**Issue**: Permission errors
```bash
# Use user install
pip install --user -r requirements_training.txt
```

---

## Step 2: Stage A - Supervised Fine-Tuning (SFT)

### Objective
Transform base model into a Medical Specialist by training on factual summaries.

### Before You Start

**Checklist**:
- [ ] Dependencies installed (completed Step 1)
- [ ] GPU has 20GB+ VRAM available (or ready to use CPU)
- [ ] Training data exists at `phase1_data/sft/train_set_processed.csv`
- [ ] Validation data exists at `phase1_data/sft/validation_set_processed.csv`

**Verify data files**:
```bash
ls -la phase1_data/sft/
# Should show:
# - train_set_processed.csv (12 lines)
# - validation_set_processed.csv (2 lines)
```

### Execution

#### Option A: Standard Training (GPU Recommended)
```bash
python stage_a_sft_training.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --train_data_path "phase1_data/sft/train_set_processed.csv" \
    --val_data_path "phase1_data/sft/validation_set_processed.csv" \
    --num_epochs 2 \
    --learning_rate 2e-4 \
    --batch_size 8 \
    --lora_r 16 \
    --output_dir "./models/sft_specialist"
```

#### Option B: Mac/CPU Training (Slower but Works)
```bash
python stage_a_sft_training.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --train_data_path "phase1_data/sft/train_set_processed.csv" \
    --val_data_path "phase1_data/sft/validation_set_processed.csv" \
    --num_epochs 1 \
    --learning_rate 2e-4 \
    --batch_size 2 \
    --lora_r 8 \
    --device cpu \
    --output_dir "./models/sft_specialist"
```

#### Option C: Memory-Constrained (Limited GPU Memory)
```bash
python stage_a_sft_training.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --num_epochs 2 \
    --batch_size 4 \
    --lora_r 8 \
    --use_8bit
```

### What Happens During Training

The script will:

1. **Download base model** (~7GB, first time only)
   ```
   Loading tokenizer from meta-llama/Llama-2-7b-hf
   Loading model from meta-llama/Llama-2-7b-hf
   Applied LoRA with r=16, alpha=32
   ```

2. **Load training data**
   ```
   Loaded 11 examples from phase1_data/sft/train_set_processed.csv
   Filtered to 11 factual examples
   Creating dataloaders
   Train batches: 2
   Val batches: 1
   ```

3. **Train for 2 epochs**
   ```
   Epoch 1/2
   ============================================================
   Training: 100%|████████| 2/2 [00:XX<00:00, XXs/it]
   Train Loss: 2.1453
   Val Loss: 2.0876
   Learning Rate: 2.00e-04
   
   Epoch 2/2
   ============================================================
   Training: 100%|████████| 2/2 [00:XX<00:00, XXs/it]
   Train Loss: 1.8234
   Val Loss: 1.7956
   Learning Rate: 1.50e-04
   
   Training completed!
   Saved final model to ./models/sft_specialist/final_model
   ```

### Expected Output

After successful training, you should have:

```
./models/sft_specialist/
├── final_model/
│   ├── adapter_config.json      # LoRA configuration
│   ├── adapter_model.bin        # Trained LoRA weights
│   ├── tokenizer.json           # Tokenizer config
│   └── sft_config.json          # Training config
├── checkpoint_epoch_1/
├── checkpoint_epoch_2/
└── training_stats.json          # Loss metrics
```

### Success Criteria

- [ ] Loss decreases from epoch 1 to epoch 2
- [ ] No NaN or Inf values in loss
- [ ] Model checkpoint created at `./models/sft_specialist/final_model/`
- [ ] Training completes without errors

### Expected Training Time

| Hardware | Time | Notes |
|----------|------|-------|
| A100 GPU | 30-60 min | Fastest |
| V100 GPU | 60-120 min | Typical datacenter |
| RTX 3090 | 90-180 min | Consumer GPU |
| M1/M2 Mac (MPS) | 2-4 hours | Mac with GPU acceleration |
| M1/M2 Mac (CPU) | 6-12 hours | Slower but works |

---

## Step 3: Verify Stage A Output

### Objective
Confirm Stage A training was successful before proceeding to Stage B.

### Verification Checklist

```bash
# 1. Check model files exist
ls -la ./models/sft_specialist/final_model/
# Should show: adapter_config.json, adapter_model.bin, tokenizer.json

# 2. Check file sizes are reasonable
du -h ./models/sft_specialist/final_model/*
# adapter_model.bin should be ~100-200MB
```

### Quick Inference Test

Test that the model can generate medical summaries:

```bash
# Single example
python sft_inference.py \
    --model_path "./models/sft_specialist/final_model" \
    --clinical_note "Patient reports fever of 38.5°C and cough. Tested positive for influenza A."
```

**Expected output**:
```
============================================================
Clinical Note:
Patient reports fever of 38.5°C and cough. Tested positive for influenza A.

------------------------------------------------------------
Generated Summary:
The patient tested positive for influenza A and experienced fever.
============================================================
```

### What to Look For

✅ **Good signs**:
- Model loads successfully
- Summary is medically coherent
- Language is professional and relevant

❌ **Bad signs**:
- Model fails to load
- Summary is gibberish
- Complete hallucination (unrelated to input)

### If Verification Fails

**Problem**: Model doesn't load
```bash
# Check PyTorch is working
python -c "import torch; print(torch.cuda.is_available())"

# Try loading model directly
python -c "from peft import AutoPeftModelForCausalLM; m = AutoPeftModelForCausalLM.from_pretrained('./models/sft_specialist/final_model')"
```

**Problem**: Generated text is garbage
- This might be normal early training. Proceed to Stage B.
- Stage B will improve quality significantly.

---

## Step 4: Stage B - Direct Preference Optimization (DPO)

### Objective
Transform Medical Specialist into Hallucination-Resistant Expert using hard negatives.

### Prerequisites

**Checklist**:
- [ ] Stage A training completed
- [ ] Model checkpoint verified (Step 3)
- [ ] DPO training data exists at `phase1_data/dpo/train_set_processed.jsonl`
- [ ] GPU has 24-32GB VRAM available (or ready to use CPU)

**Verify DPO data**:
```bash
# Check file exists and has records
wc -l phase1_data/dpo/train_set_processed.jsonl
# Should show at least 13 lines
```

### Execution

#### Option A: Standard Training (GPU Recommended)
```bash
python stage_b_dpo_training.py \
    --sft_model_path "./models/sft_specialist/final_model" \
    --train_data_path "phase1_data/dpo/train_set_processed.jsonl" \
    --val_data_path "phase1_data/dpo/train_set_processed.jsonl" \
    --num_epochs 2 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --batch_size 4 \
    --output_dir "./models/dpo_hallucination_resistant"
```

#### Option B: Mac/CPU Training
```bash
python stage_b_dpo_training.py \
    --sft_model_path "./models/sft_specialist/final_model" \
    --train_data_path "phase1_data/dpo/train_set_processed.jsonl" \
    --val_data_path "phase1_data/dpo/train_set_processed.jsonl" \
    --num_epochs 1 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --batch_size 1 \
    --device cpu \
    --output_dir "./models/dpo_hallucination_resistant"
```

#### Option C: Memory-Constrained
```bash
python stage_b_dpo_training.py \
    --sft_model_path "./models/sft_specialist/final_model" \
    --num_epochs 2 \
    --learning_rate 5e-6 \
    --batch_size 2 \
    --lora_r 8 \
    --use_8bit
```

### What Happens During Training

The script will:

1. **Load dual models**
   ```
   Loading reference and active models
   Loading tokenizer from ./models/sft_specialist/final_model
   Reference model: 7B params (frozen)
   Active model: 7B params with LoRA (trainable)
   ```

2. **Load DPO training data**
   ```
   Loaded 13 DPO triplets from phase1_data/dpo/train_set_processed.jsonl
   Creating dataloaders
   Train batches: 4
   Val batches: 4
   ```

3. **Train for 2 epochs with DPO loss**
   ```
   Epoch 1/2
   ============================================================
   Train Loss: 0.68
   Val Loss: 0.65
   Chosen Preference: 55%  ← Model learning to prefer factual
   Learning Rate: 5.00e-06
   
   Epoch 2/2
   ============================================================
   Train Loss: 0.45
   Val Loss: 0.42
   Chosen Preference: 78%  ← Strong preference established
   Learning Rate: 2.50e-06
   
   DPO Training completed!
   Saved final model to ./models/dpo_hallucination_resistant/final_model
   ```

### Understanding DPO Metrics

**Chosen Preference** (0-100%):
- 50% = Random (no learning happening)
- 60-70% = Early learning
- 70-80% = Good learning
- 80-90% = Strong learning
- 90%+ = Excellent (watch for overfitting)

**Expected progression**:
```
Epoch 1: 55% → Loss: 0.68
Epoch 2: 78% → Loss: 0.45
```

**Loss** (should decrease):
```
Epoch 1: Train: 0.68, Val: 0.65
Epoch 2: Train: 0.45, Val: 0.42
```

### Expected Output

After successful training:

```
./models/dpo_hallucination_resistant/
├── final_model/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── tokenizer.json
│   └── dpo_config.json
├── checkpoint_epoch_1/
├── checkpoint_epoch_2/
└── dpo_training_stats.json
```

### Success Criteria

- [ ] Chosen Preference increases over epochs (50% → 70%+)
- [ ] Loss decreases smoothly
- [ ] No NaN or Inf values
- [ ] Model checkpoint created at `./models/dpo_hallucination_resistant/final_model/`
- [ ] Training completes without errors

### Expected Training Time

| Hardware | Time |
|----------|------|
| A100 GPU | 1-2 hours |
| V100 GPU | 2-3 hours |
| RTX 3090 | 3-4 hours |
| M1/M2 Mac (CPU) | 12-20 hours |

---

## Step 5: Evaluate Results

### Objective
Compare SFT vs DPO models to verify hallucination reduction.

### 5A: Basic Comparison

```bash
# Test on same clinical note with both models

echo "Testing SFT model (Stage A):"
python sft_inference.py \
    --model_path "./models/sft_specialist/final_model" \
    --clinical_note "Patient on 50mg aspirin daily. No side effects."

echo ""
echo "Testing DPO model (Stage B):"
python sft_inference.py \
    --model_path "./models/dpo_hallucination_resistant/final_model" \
    --clinical_note "Patient on 50mg aspirin daily. No side effects."
```

**Expected observation**:
- SFT model: May say "Patient on 500mg aspirin" (hallucination)
- DPO model: Correctly says "Patient on 50mg aspirin" (factual)

### 5B: Batch Testing

Create a test file with multiple examples:

```bash
# Create test file
cat > test_examples.txt << 'EOF'
Patient reports fever of 38.5°C and cough. Tested positive for influenza A.
Patient recovering from COVID-19 infection. No respiratory distress.
Patient has type 2 diabetes. HbA1c is 8.1%. No signs of neuropathy.
EOF

# Test SFT model
echo "=== SFT Model Results ==="
python sft_inference.py \
    --model_path "./models/sft_specialist/final_model" \
    --input_file test_examples.txt

# Test DPO model
echo ""
echo "=== DPO Model Results ==="
python sft_inference.py \
    --model_path "./models/dpo_hallucination_resistant/final_model" \
    --input_file test_examples.txt
```

### 5C: Qualitative Evaluation

Compare outputs on these dimensions:

| Dimension | SFT | DPO | Notes |
|-----------|-----|-----|-------|
| Medical accuracy | ? | Should be ✅ higher |
| Factual grounding | ? | Should be ✅ better |
| Hallucination rate | Baseline | Should be ✅ lower |
| Citation accuracy | ? | Should be ✅ improved |

### 5D: Quantitative Metrics (Optional)

If you have a test set with ground truth:

```python
from sft_inference import SFTInference

# Load both models
sft = SFTInference("./models/sft_specialist/final_model")
dpo = SFTInference("./models/dpo_hallucination_resistant/final_model")

# Compare outputs
hallucinations_sft = 0
hallucinations_dpo = 0

for clinical_note in test_notes:
    sft_result = sft.generate_summary(clinical_note)
    dpo_result = dpo.generate_summary(clinical_note)
    
    # Evaluate against ground truth
    if is_hallucinated(sft_result):
        hallucinations_sft += 1
    if is_hallucinated(dpo_result):
        hallucinations_dpo += 1

print(f"SFT hallucination rate: {hallucinations_sft/len(test_notes)*100:.1f}%")
print(f"DPO hallucination rate: {hallucinations_dpo/len(test_notes)*100:.1f}%")
print(f"Improvement: {(hallucinations_sft-hallucinations_dpo)/hallucinations_sft*100:.1f}%")
```

### Expected Improvement

**Before DPO (SFT only)**:
- Hallucination rate: ~30-40%
- Medical knowledge: Excellent ✅
- Factual grounding: Moderate

**After DPO**:
- Hallucination rate: ~5-15% (75%+ reduction)
- Medical knowledge: Maintained ✅
- Factual grounding: Excellent ✅

---

## Mac-Specific Considerations

### Hardware Options on Mac

#### Option 1: CPU Training (Recommended for First Run)

**Pros**:
- Works on any Mac
- No special setup needed
- Stable and reliable

**Cons**:
- Slower (3-7x slower than GPU)
- Takes 6-12 hours total

**Command**:
```bash
python stage_a_sft_training.py \
    --device cpu \
    --batch_size 2 \
    --num_epochs 1
```

#### Option 2: MPS (Metal Performance Shaders)

**Pros**:
- Much faster than CPU (2-3x)
- Uses Mac GPU acceleration
- Native support on M1/M2

**Cons**:
- Only works on Apple Silicon (M1/M2/M3)
- Experimental support
- Sometimes unstable

**Command**:
```bash
python stage_a_sft_training.py \
    --device mps \
    --batch_size 4
```

#### Option 3: Cloud GPU (Fastest)

**Pros**:
- Very fast (1-3 hours total)
- No local resource constraints
- Professional infrastructure

**Cons**:
- Costs money ($5-20 per training)
- Requires cloud account setup

**Providers**:
- Google Colab (free tier available)
- Lambda Labs
- Paperspace

### Recommended Mac Setup

**For M1/M2 Mac**:

1. **First attempt**: Use CPU with small batch size
   ```bash
   --device cpu --batch_size 1 --num_epochs 1
   ```

2. **If time permits**: Try MPS for speed boost
   ```bash
   --device mps --batch_size 2
   ```

3. **If need speed**: Use cloud GPU for full training

**Monitor Resources**:
```bash
# Watch memory usage
watch -n 1 'ps aux | grep python'

# Check disk space (need ~30GB for models)
df -h
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "CUDA out of memory"

**Cause**: GPU doesn't have enough memory

**Solutions** (try in order):
```bash
# 1. Reduce batch size
--batch_size 2

# 2. Use smaller LoRA rank
--lora_r 8

# 3. Use 8-bit quantization
--use_8bit

# 4. Use CPU
--device cpu
```

---

#### Issue: "Model download stuck or very slow"

**Cause**: Network issue or model server slow

**Solutions**:
```bash
# 1. Use smaller model
--model_name "microsoft/phi-2"  # 2.7B instead of 7B

# 2. Download to cache manually
huggingface-cli download meta-llama/Llama-2-7b-hf

# 3. Use local model if available
--model_name "./local_model_path"
```

---

#### Issue: "Loss diverges (NaN or Inf)"

**Cause**: Learning rate too high or data issue

**Solutions**:
```bash
# For SFT:
--learning_rate 1e-4  # Lower from 2e-4

# For DPO:
--learning_rate 1e-6  # Even lower from 5e-6
--beta 0.05          # Lower from 0.1
```

---

#### Issue: "Generated text is gibberish"

**Cause**: Model undertrained or LR too high

**Solutions**:
```bash
# For Stage A:
--num_epochs 3
--learning_rate 1e-4

# Then proceed to Stage B - it will improve significantly
```

---

#### Issue: "Module not found" errors

**Cause**: Dependencies not installed

**Solution**:
```bash
pip install -r requirements_training.txt
python -c "import torch; import transformers; import peft"
```

---

#### Issue: "FileNotFoundError: training data not found"

**Cause**: Wrong file paths

**Verify**:
```bash
ls phase1_data/sft/train_set_processed.csv
ls phase1_data/dpo/train_set_processed.jsonl
```

---

#### Issue: "Authentication error with Hugging Face"

**Cause**: Need Hugging Face credentials for gated models

**Solution**:
```bash
huggingface-cli login
# Then enter your token from huggingface.co

# Or use non-gated model:
--model_name "mistralai/Mistral-7B"
```

---

### Getting Help

If you encounter other issues:

1. Check logs: Look for error messages in terminal output
2. Review documentation:
   - `STAGE_A_SFT_GUIDE.md` for Stage A issues
   - `STAGE_B_DPO_GUIDE.md` for Stage B issues
3. Common issues:
   - Memory: Reduce batch_size
   - Speed: Use smaller model or GPU
   - Quality: Check training data format

---

## Expected Timelines

### Full Timeline (Typical GPU)

```
Step 1: Install dependencies
  Time: 5-10 minutes
  Total elapsed: 5-10 min

Step 2: Stage A SFT Training
  Time: 1-3 hours
  Total elapsed: 1-3 hours

Step 3: Verify Stage A
  Time: 5-10 minutes
  Total elapsed: 1.1-3.2 hours

Step 4: Stage B DPO Training
  Time: 2-4 hours
  Total elapsed: 3.1-7.2 hours

Step 5: Evaluate Results
  Time: 10-30 minutes
  Total elapsed: 3.2-7.5 hours
```

### Full Timeline (Mac CPU)

```
Step 1: Install dependencies
  Time: 10-20 minutes
  Total elapsed: 10-20 min

Step 2: Stage A SFT Training
  Time: 6-12 hours
  Total elapsed: 6-12 hours

Step 3: Verify Stage A
  Time: 5-10 minutes
  Total elapsed: 6-12 hours

Step 4: Stage B DPO Training
  Time: 12-20 hours
  Total elapsed: 18-32 hours

Step 5: Evaluate Results
  Time: 10-30 minutes
  Total elapsed: 18-32 hours
```

### By Hardware

| Hardware | Total Time | Per Stage |
|----------|-----------|-----------|
| A100 GPU | 2-3 hours | Stage A: 30-60 min, Stage B: 60-90 min |
| V100 GPU | 3-5 hours | Stage A: 60-120 min, Stage B: 120-180 min |
| RTX 3090 | 4-6 hours | Stage A: 90-180 min, Stage B: 180-240 min |
| M1/M2 MPS | 4-8 hours | Stage A: 120-240 min, Stage B: 240-360 min |
| M1/M2 CPU | 18-32 hours | Stage A: 360-720 min, Stage B: 720-1200 min |

---

## Quick Command Reference

### Install & Verify
```bash
pip install -r requirements_training.txt
python -c "import torch; import transformers; import peft; print('✅ Ready')"
```

### Stage A - SFT
```bash
python stage_a_sft_training.py --num_epochs 2 --learning_rate 2e-4 --batch_size 8
```

### Stage B - DPO
```bash
python stage_b_dpo_training.py --num_epochs 2 --learning_rate 5e-6 --beta 0.1 --batch_size 4
```

### Test Model
```bash
python sft_inference.py --model_path "./models/sft_specialist/final_model" --clinical_note "Patient has fever."
```

---

## Success Checklist

After completing all steps:

- [ ] Dependencies installed successfully
- [ ] Stage A training completed without errors
- [ ] Loss decreased from epoch 1 to epoch 2
- [ ] SFT model checkpoint created
- [ ] Stage A verification passed
- [ ] Stage B training completed without errors
- [ ] Chosen Preference reached 70%+
- [ ] DPO model checkpoint created
- [ ] DPO model generates coherent medical text
- [ ] Comparison shows hallucination reduction
- [ ] Both models work with `sft_inference.py`

---

## Next Actions (In Order)

1. **Now**: Run Step 1 (Install dependencies)
2. **After installation**: Run Step 2 (Stage A training)
3. **After Stage A**: Run Step 3 (Verify output)
4. **After verification**: Run Step 4 (Stage B training)
5. **After Stage B**: Run Step 5 (Evaluate results)

---

## Support Documents

- **Detailed SFT info**: `STAGE_A_SFT_GUIDE.md`
- **Detailed DPO info**: `STAGE_B_DPO_GUIDE.md`
- **Completion audit**: `PHASE_COMPLETION_AUDIT.md`
- **Checklists**: `COMPLETION_CHECKLIST.txt`

---

**Start with Step 1 and proceed sequentially. Good luck! 🚀**
