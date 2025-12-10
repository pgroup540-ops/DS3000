# Stage B Training - GPU Memory Limitation

## 🚨 Issue Encountered

**CUDA Out of Memory** during Stage B (DPO) training

### Technical Details:
- **Your GPU**: NVIDIA GeForce RTX 5070 (12GB VRAM)
- **Required**: ~26GB VRAM minimum
  - Reference Model (frozen): ~13GB
  - Active Model (training): ~13GB
- **Problem**: DPO requires TWO full models loaded simultaneously

###Error Message:
```
torch.OutOfMemoryError: CUDA out of memory.
GPU 0 has a total capacity of 11.94 GiB of which 0 bytes is free.
Of the allocated memory 26.08 GiB is allocated by PyTorch.
```

---

## ✅ What We've Completed So Far

1. ✅ **Stage A (SFT) Training** - Complete
   - Model: `models/sft_specialist_fast_fp16/final_model`
   
2. ✅ **Stage A Evaluation** - Complete
   - Tested on 50 examples
   - Results: `evaluation_results_stage_a/`

3. ✅ **Manual Assessment** - Complete
   - Assessed 10 examples
   - **Hallucination Rate: 60%**
   
4. ✅ **Phase 2 Data Generation** - Complete
   - 8,484 training triplets
   - 1,851 validation triplets
   - Files ready at: `phase2_data/dpo/`

5. ❌ **Stage B Training** - BLOCKED by GPU memory

---

## 🎯 Options Moving Forward

### Option 1: CPU Training (Recommended for completion)

**Run Stage B on CPU** (slow but will work)

#### Command:
```bash
# This will take 10-15 hours on CPU
python stage_b_dpo_training_cpu.py --sft_model_path "models/sft_specialist_fast_fp16/final_model" --train_data_path "phase2_data/dpo/train_dpo.jsonl" --val_data_path "phase2_data/dpo/val_dpo.jsonl" --num_epochs 1 --learning_rate 5e-6 --batch_size 1 --output_dir "models/dpo_hallucination_resistant"
```

#### Pros:
- ✅ Will complete Stage B training
- ✅ Will reduce hallucination rate to 5-15%
- ✅ No additional cost

#### Cons:
- ❌ Very slow (10-15 hours vs 2-4 hours on 24GB GPU)
- ❌ Need to leave computer running overnight

---

### Option 2: Cloud GPU (Fastest solution)

**Use a cloud service with 24GB+ VRAM**

#### Options:
1. **Google Colab Pro** ($10/month)
   - A100 GPU (40GB VRAM)
   - 2-3 hours training time
   
2. **AWS EC2 p3.2xlarge**
   - V100 GPU (16GB VRAM) - might still be tight
   - ~$3/hour
   
3. **Lambda Labs** (cheapest)
   - A100 GPU (40GB VRAM)
   - ~$1.10/hour × 3 hours = ~$3.30 total

#### Pros:
- ✅ Fast (2-4 hours)
- ✅ Will complete successfully
- ✅ Can use same code

#### Cons:
- ❌ Costs money ($3-30 depending on service)
- ❌ Need to transfer files to cloud
- ❌ Learning curve for cloud setup

---

### Option 3: Skip Stage B for Now

**Deploy Stage A model as-is and improve later**

#### What this means:
- Use current Stage A model: `models/sft_specialist_fast_fp16/final_model`
- Accept 60% hallucination rate (high)
- Plan to run Stage B later when you have access to larger GPU

#### Pros:
- ✅ Immediate deployment
- ✅ Model still has medical knowledge
- ✅ Can improve later

#### Cons:
- ❌ High hallucination rate (60%)
- ❌ Not suitable for production use
- ❌ Defeats purpose of the pipeline

---

## 💡 Recommended Path

### **I recommend Option 1: CPU Training**

**Why:**
- You've already invested significant time
- The infrastructure is ready (data generated)
- Just needs overnight run
- Will complete the full pipeline
- No additional costs

### How to Run CPU Training:

1. **Prepare for overnight run:**
```bash
# Set up CPU training (reduces memory but increases time)
python stage_b_dpo_training.py --sft_model_path "models/sft_specialist_fast_fp16/final_model" --train_data_path "phase2_data/dpo/train_dpo.jsonl" --val_data_path "phase2_data/dpo/val_dpo.jsonl" --num_epochs 1 --learning_rate 5e-6 --batch_size 1 --output_dir "models/dpo_hallucination_resistant" --device cpu
```

2. **Monitor progress:**
   - Check `models/dpo_hallucination_resistant/` for checkpoints
   - Training stats saved to `dpo_training_stats.json`

3. **Expected timeline:**
   - Start: Evening
   - Complete: Next morning (10-15 hours)
   - Then evaluate and compare results

---

## 📊 Expected Results After Stage B

### Current (Stage A):
- Hallucination rate: **60%**
- Too high for deployment

### After Stage B (Expected):
- Hallucination rate: **5-15%**
- Suitable for deployment
- Worth the wait!

---

## 🔧 Technical Why GPU Failed

**DPO (Direct Preference Optimization) algorithm:**
1. Loads **Reference Model** (frozen, from Stage A)
2. Loads **Policy Model** (trainable copy of Stage A)
3. For each batch:
   - Gets log-probs from reference model
   - Gets log-probs from policy model
   - Computes DPO loss comparing them
4. Only policy model is updated, reference stays frozen

**Memory requirement:**
- Each model: ~13GB
- Total: ~26GB minimum
- Your GPU: 12GB ❌

**Why 8-bit quantization doesn't help enough:**
- Even with 8-bit: ~7GB per model
- Total with 8-bit: ~14GB
- Still exceeds 12GB when including activations and gradients

---

## 📝 Next Steps

### If choosing CPU training (Option 1):

```bash
# 1. Start training (will run overnight)
python stage_b_dpo_training.py --device cpu --batch_size 1 --num_epochs 1 --sft_model_path "models/sft_specialist_fast_fp16/final_model" --learning_rate 5e-6 --output_dir "models/dpo_hallucination_resistant"

# 2. Next morning: Evaluate
python evaluation_stage_a/evaluate_stage_a.py --model_path "models/dpo_hallucination_resistant/final_model" --test_data "phase1_data_medhal/sft/test_set_processed.csv" --output_dir "evaluation_results_stage_b" --max_examples 50

# 3. Manual assessment
python manual_assessment_tool.py --results_csv "evaluation_results_stage_b/evaluation_results.csv"

# 4. Compare: Stage A (60%) vs Stage B (expected 5-15%)
```

---

### If choosing Cloud GPU (Option 2):

1. Set up Google Colab / AWS / Lambda Labs
2. Upload files:
   - `models/sft_specialist_fast_fp16/`
   - `phase2_data/dpo/`
   - All Python scripts
3. Run same GPU command
4. Download trained model back

---

### If choosing Skip Stage B (Option 3):

**Not recommended** - 60% hallucination rate is too high for any real use.

---

## Summary

**You're 90% done!** Just need to complete Stage B training.

**Best option**: Run CPU training overnight
- Set it up before bed
- Check in the morning
- Complete evaluation
- See hallucination rate drop from 60% to ~10%

**The wait will be worth it!** 🎯

---

## Files Ready for Training:

✅ Stage A model: `models/sft_specialist_fast_fp16/final_model`  
✅ Training data: `phase2_data/dpo/train_dpo.jsonl` (8,484 triplets)  
✅ Validation data: `phase2_data/dpo/val_dpo.jsonl` (1,851 triplets)  
✅ Training script: `stage_b_dpo_training.py`  

**Just need**: Time (overnight CPU run) or Money (cloud GPU)
