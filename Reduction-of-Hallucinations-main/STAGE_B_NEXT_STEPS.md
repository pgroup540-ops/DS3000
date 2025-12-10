# Stage B (DPO) Training - Next Steps

## 🎯 Current Situation

Your manual assessment revealed:
- **Hallucination Rate: > 15%**
- **Decision: Proceed to Stage B (DPO Training)**

This is **normal and expected**! Stage A taught the model medical knowledge, but now Stage B will teach it to prefer factual outputs.

---

## 📊 What Stage B Does

**Direct Preference Optimization (DPO)** trains the model to:
- ✅ Prefer factual summaries over hallucinated ones
- ✅ Learn from paired examples (good vs bad)
- ✅ Reduce hallucination rate from ~30-60% → 5-15%

**How it works:**
- For each clinical note, the model sees:
  - ✅ **Chosen**: Factual, accurate summary
  - ❌ **Rejected**: Hallucinated summary (with errors)
- The model learns to increase probability of chosen, decrease rejected

---

## 🚀 Stage B Workflow (3 Steps)

### **Step 1: Generate DPO Training Data** (5-10 minutes)

This creates preference pairs from your Phase 1 data.

```bash
python generate_phase2_data.py \
    --phase1_dir "phase1_data_medhal/sft" \
    --phase2_dir "phase2_data/dpo" \
    --adversarial_ratio 1.0
```

**What this does:**
- Reads Phase 1 factual summaries
- Generates adversarial (hallucinated) versions
- Creates triplets: (prompt, chosen, rejected)
- Outputs: `train_dpo.jsonl` and `val_dpo.jsonl`

**Expected output:**
```
✓ Training triplets: 8000-15000
✓ Validation triplets: 1000-2000
```

---

### **Step 2: Train Stage B Model** (2-4 hours on GPU)

```bash
python stage_b_dpo_training.py \
    --sft_model_path "models/sft_specialist_fast_fp16/final_model" \
    --train_data_path "phase2_data/dpo/train_dpo.jsonl" \
    --val_data_path "phase2_data/dpo/val_dpo.jsonl" \
    --num_epochs 2 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --batch_size 4 \
    --output_dir "models/dpo_hallucination_resistant"
```

**Key Parameters:**
- `learning_rate: 5e-6` ← **100x lower than Stage A!** (Critical!)
- `beta: 0.1` ← Controls preference strength
- `batch_size: 4` ← Smaller (DPO uses two models)
- `num_epochs: 2` ← Usually 1-3 is enough

**What this does:**
- Loads your Stage A model as starting point
- Trains on preference pairs
- Learns to prefer factual outputs
- Saves to `models/dpo_hallucination_resistant/`

---

### **Step 3: Evaluate Stage B Model** (10-20 minutes)

```bash
python evaluation_stage_a/evaluate_stage_a.py \
    --model_path "models/dpo_hallucination_resistant/final_model" \
    --test_data "phase1_data_medhal/sft/test_set_processed.csv" \
    --output_dir "evaluation_results_stage_b" \
    --max_examples 50 \
    --device cuda
```

Then manually assess again:
```bash
python manual_assessment_tool.py \
    --results_csv "evaluation_results_stage_b/evaluation_results.csv"
```

**Expected result:**
- Hallucination rate should drop to **5-15%** (from 30-60%)
- If still high, can run Stage B for 1 more epoch

---

## ⚙️ Important Parameters Explained

### Learning Rate (CRITICAL!)

**Stage A (SFT)**: `2e-4` (0.0002)
**Stage B (DPO)**: `5e-6` (0.000005) ← **100x smaller!**

**Why so small?**
- Stage A already learned medical knowledge
- Stage B only adjusts preferences slightly
- Too high = destroys Stage A knowledge
- Too low = no improvement

### Beta (Preference Strength)

- `beta = 0.1` (default, recommended)
- Higher beta = stronger preference enforcement
- Lower beta = gentler adjustment

Range: 0.05 - 0.5
- Start with 0.1
- If not improving, try 0.2

### Batch Size

- Stage A: 8-16
- Stage B: 2-4 ← **Smaller!**

**Why?**
- DPO maintains reference model + policy model
- Uses ~2x memory of Stage A
- Reduce batch size to fit in GPU memory

---

## 📋 Quick Command Reference

```bash
# Step 1: Generate DPO data (5-10 min)
python generate_phase2_data.py

# Step 2: Train Stage B (2-4 hours)
python stage_b_dpo_training.py \
    --sft_model_path "models/sft_specialist_fast_fp16/final_model" \
    --learning_rate 5e-6 \
    --num_epochs 2

# Step 3: Evaluate Stage B
python evaluation_stage_a/evaluate_stage_a.py \
    --model_path "models/dpo_hallucination_resistant/final_model" \
    --output_dir "evaluation_results_stage_b"

# Step 4: Manual assessment
python manual_assessment_tool.py \
    --results_csv "evaluation_results_stage_b/evaluation_results.csv"
```

---

## 🎯 Expected Results

### After Stage A (Current):
- ❌ Hallucination rate: 30-60%
- ✅ Learned medical knowledge
- ❌ Adds unsupported details

### After Stage B (Expected):
- ✅ Hallucination rate: 5-15%
- ✅ Maintains medical knowledge
- ✅ Prefers factual outputs
- ✅ More conservative (fewer details)

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python stage_b_dpo_training.py --batch_size 2

# Or use gradient accumulation
python stage_b_dpo_training.py --batch_size 2 --gradient_accumulation_steps 2
```

### "Model not converging"
- Check learning rate (should be 5e-6, not 5e-4!)
- Try increasing beta to 0.2
- Run for 1 more epoch

### "Hallucination rate still high after Stage B"
- Train for 1 more epoch (total 3)
- Try beta = 0.2 (stronger preference)
- Verify learning rate is correct (5e-6)

---

## 💡 Why Stage B Works

### The Problem:
Stage A (SFT) learns from all training examples equally, including:
- Short accurate summaries
- Long detailed summaries

The model learns: "More details = better summary"
→ Sometimes adds unsupported details = hallucinations

### The Solution:
Stage B (DPO) explicitly teaches:
- ✅ This summary is GOOD (factual)
- ❌ This summary is BAD (hallucinated)

The model learns: "Factual accuracy > Adding details"
→ Reduces hallucinations significantly

---

## 📊 Timeline

| Task | Time | 
|------|------|
| Generate DPO data | 5-10 min |
| Train Stage B | 2-4 hours (GPU) |
| Inference evaluation | 10-20 min |
| Manual assessment | 30-60 min |
| **Total** | **3-5 hours** |

---

## ✅ Checklist

Before starting Stage B:
- [ ] Manual assessment shows hallucination rate > 15%
- [ ] Stage A model exists at `models/sft_specialist_fast_fp16/final_model`
- [ ] Phase 1 data exists at `phase1_data_medhal/sft/`
- [ ] GPU available (recommended)

---

## 🚦 Ready to Start?

Run the first command:
```bash
python generate_phase2_data.py
```

This will create the DPO training data, then you can proceed to Stage B training!

---

## 📞 Need Help?

- Check training logs for errors
- Monitor GPU memory usage
- Review Guides/STAGE_B_DPO_GUIDE.md for more details
- If model performance degrades, learning rate is likely too high

---

**Next Command:**
```bash
python generate_phase2_data.py
```

Let's reduce those hallucinations! 🎯
