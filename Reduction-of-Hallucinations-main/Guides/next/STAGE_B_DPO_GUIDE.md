# Stage B: Direct Preference Optimization (DPO) Implementation Guide

## Overview

Stage B transforms your Medical Specialist into a **Hallucination-Resistant Expert** through Direct Preference Optimization.

### What Happens in Stage B

- **Input**: `dpo_train_data.jsonl` containing triplets of (Clinical Note, Factual Summary, Hallucinated Summary)
- **Goal**: Teach the model to statistically prefer factual responses over hallucinated ones
- **Mechanism**: DPO Loss with KL-Divergence penalty
- **Output**: A model that generates medical summaries with dramatically reduced hallucinations

### Why After Stage A?

Stage A gives the model medical knowledge. Stage B teaches it to **prefer truth over plausible-sounding lies**. The hard negatives (very similar to truth but factually wrong) teach fine-grained distinctions.

---

## Key Concepts

### The Two Models

Unlike RLHF which uses a separate reward model, DPO requires TWO copies of your model:

1. **Reference Model** (Frozen)
   - Loaded from Stage A checkpoint
   - Weights NEVER change during training
   - Used to compute baseline probabilities
   - Prevents the active model from "drifting"

2. **Active Model** (Being Trained)
   - Initialized from Stage A checkpoint
   - Weights CHANGE during training
   - Learning to prefer chosen over rejected
   - Has LoRA adapters on top (optional)

### The DPO Loss Function

```
DPO Loss = -log(sigmoid(β * (log_p_chosen - log_p_rejected - log_p_ref_chosen + log_p_ref_rejected)))
```

Where:
- `log_p_chosen`: Active model's probability of chosen (factual) response
- `log_p_rejected`: Active model's probability of rejected (hallucinated) response
- `log_p_ref_*`: Reference model's probabilities (baseline)
- `β` (beta): Weight of the KL-divergence penalty

This loss function:
1. **Increases** probability of chosen response in active model
2. **Decreases** probability of rejected response in active model
3. **Penalizes** divergence from reference model via KL term
4. Creates fine-grained medical distinctions (e.g., "50mg" good vs "500mg" bad)

---

## Installation & Setup

### 1. Ensure Stage A is Complete

```bash
# Verify the SFT model exists
ls ./models/sft_specialist/final_model/
```

Should contain:
- `adapter_config.json`
- `adapter_model.bin`
- `tokenizer.json`

### 2. Prepare DPO Data

Create JSONL file at `phase2_data/dpo/train_dpo.jsonl`:

```json
{"prompt": "Clinical Note: Patient reports fever of 38.5°C and cough. Tested positive for influenza A.\n\nSummary:", "chosen": "The patient tested positive for influenza A and experienced fever.", "rejected": "The patient tested negative for influenza and recovered completely."}
{"prompt": "Clinical Note: Patient recovering from COVID-19 infection. No respiratory distress.\n\nSummary:", "chosen": "The patient is recovering from COVID-19 without respiratory distress.", "rejected": "The patient is suffering from severe respiratory distress due to COVID-19."}
```

Each line must be valid JSON with:
- `prompt`: Clinical note + "Summary:" prefix
- `chosen`: Factually correct summary
- `rejected`: Hallucinated (hard negative) summary

### 3. Verify Dependencies

All dependencies already installed from Stage A:

```bash
python -c "import torch; import peft; import transformers; print('All dependencies ready')"
```

---

## Technical Specifications

### Learning Rate: 5e-6 or 1e-6

**CRITICAL**: DPO learning rates are **100x lower** than SFT!

| Learning Rate | Use Case |
|--------------|----------|
| 1e-6 | Conservative (recommended for first attempt) |
| 5e-6 | Standard (balanced) |
| 1e-5 | Aggressive (risk of degradation) |
| > 1e-5 | NOT RECOMMENDED (model collapse risk) |

**Why so low?**
- DPO is training two models simultaneously
- Frozen reference model means no gradient flow through it
- Active model only has gradients through changed weights
- High LR causes unstable training and model degradation

### Beta: 0.1 (Default)

Controls the strength of the KL-divergence penalty.

| Beta | Effect |
|------|--------|
| 0.05 | Weaker preference learning, more model drift allowed |
| 0.1 | Balanced (RECOMMENDED) |
| 0.5 | Stronger preference learning, less drift |
| 1.0 | Very conservative, strong KL penalty |

### Batch Size: 4 (vs 8 for SFT)

Smaller batch size because:
- TWO models loaded in GPU memory
- Computing log probabilities for both
- Reference model is frozen but still takes memory
- Batch size 4 typically uses 24-32GB VRAM

| Hardware | Max Batch Size |
|----------|--------|
| A100 (40GB) | 8 |
| V100 (32GB) | 4 |
| RTX 3090 (24GB) | 2 |
| M1/M2 Mac | 1 |

### Epochs: 1-3 (Like SFT)

| Epochs | Duration | Quality |
|--------|----------|---------|
| 1 | Fast | Good baseline hallucination reduction |
| 2 | Medium | Best quality (RECOMMENDED) |
| 3 | Long | Risk of overfitting to DPO data |

### Max Length: 512 tokens

- Same as Stage A
- Typical medical notes: 300-400 tokens
- Summaries: 100-150 tokens

---

## Quick Start: Training

### Option 1: Standard Training (Recommended)

```bash
python stage_b_dpo_training.py \
    --sft_model_path "./models/sft_specialist/final_model" \
    --num_epochs 2 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --batch_size 4
```

### Option 2: Conservative Training (First Attempt)

```bash
python stage_b_dpo_training.py \
    --sft_model_path "./models/sft_specialist/final_model" \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --beta 0.1 \
    --batch_size 4
```

### Option 3: Aggressive Training (Limited Resources)

```bash
python stage_b_dpo_training.py \
    --sft_model_path "./models/sft_specialist/final_model" \
    --num_epochs 2 \
    --learning_rate 5e-6 \
    --batch_size 2 \
    --lora_r 8
```

---

## Understanding DPO Training Output

### Expected Metrics

**Chosen Preference**: Should approach 1.0 (100%)

```
Epoch 1: Chosen Preference: 55%
Epoch 2: Chosen Preference: 78%
```

This shows the model is learning to prefer chosen over rejected responses.

**Loss**: Should decrease smoothly

```
Epoch 1: Train Loss: 0.68 → Val Loss: 0.65
Epoch 2: Train Loss: 0.45 → Val Loss: 0.42
```

Similar to SFT, validation loss should track training loss.

### What "Preference" Means

```
Chosen Preference = Percentage of examples where:
    log_p(model|chosen) > log_p(model|rejected)
```

- 50%: Random (no learning)
- 60-70%: Early learning
- 80%+: Strong preference for factual responses
- 95%+: Excellent learning (may be overfitting)

### Model Checkpoints

```
./models/dpo_hallucination_resistant/
├── final_model/                    # Use this
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── tokenizer.json
│   └── dpo_config.json
├── checkpoint_epoch_1/
├── checkpoint_epoch_2/
└── dpo_training_stats.json
```

---

## DPO Data Format

### Creating DPO Triplets

Your Phase 1 adversarial data likely already has:
- Factual summaries
- Hallucinated summaries

Convert to DPO format:

```python
import json

triplets = []

# Pair each factual summary with its hallucination
for factual_note, factual_summary in factual_pairs:
    # Find corresponding hallucination
    hallucinated_summary = get_hard_negative(factual_note)
    
    prompt = f"Clinical Note: {factual_note}\n\nSummary:"
    
    triplet = {
        "prompt": prompt,
        "chosen": factual_summary,
        "rejected": hallucinated_summary,
    }
    triplets.append(triplet)

# Write to JSONL
with open("dpo_train.jsonl", "w") as f:
    for t in triplets:
        f.write(json.dumps(t) + "\n")
```

### Quality Triplets

**Good triplet**:
```json
{
  "prompt": "Clinical Note: Patient on 50mg aspirin daily. No side effects.\n\nSummary:",
  "chosen": "Patient is on a 50mg daily aspirin regimen without adverse effects.",
  "rejected": "Patient is on a 500mg daily aspirin regimen with severe side effects."
}
```

**Bad triplet** (too easy to distinguish):
```json
{
  "prompt": "...",
  "chosen": "Patient has high fever.",
  "rejected": "Patient is healthy and has no symptoms."
}
```

---

## Hyperparameter Tuning

### When to Adjust Beta

| Symptom | Cause | Solution |
|---------|-------|----------|
| Chosen preference stuck at 50-60% | Too high KL penalty | Decrease beta to 0.05 |
| Chosen preference reaches 99%+ | Underfitting KL | Increase beta to 0.5 |
| Model collapses/gibberish | KL penalty too weak | Increase beta |

### When to Adjust Learning Rate

| Problem | Cause | Solution |
|---------|-------|----------|
| Preference not increasing | LR too low | Increase to 1e-5 (cautious!) |
| Loss diverges/NaN | LR too high | Decrease to 1e-7 |
| Oscillating loss | Unstable training | Reduce to 1e-6 |
| Very slow convergence | Conservative LR | Increase to 5e-6 |

### When to Adjust Batch Size

| Constraint | Action |
|-----------|--------|
| GPU OOM | Reduce to 2 |
| Very slow per-step time | Increase to 8 (if memory allows) |
| Noisy gradients | Increase batch size |

---

## Troubleshooting

### Problem: "CUDA out of memory"

```bash
# Solution 1: Reduce batch size
python stage_b_dpo_training.py --batch_size 2

# Solution 2: Reduce LoRA rank
python stage_b_dpo_training.py --lora_r 8

# Solution 3: Use CPU (very slow)
python stage_b_dpo_training.py --device cpu
```

### Problem: Chosen Preference Not Increasing

```bash
# The model isn't learning to prefer chosen
# Try lower learning rate
python stage_b_dpo_training.py --learning_rate 1e-6

# Or higher beta (stronger preference signal)
python stage_b_dpo_training.py --beta 0.5
```

### Problem: Model Generates Gibberish

```bash
# Model drifted too far from reference
# Increase beta (stronger KL penalty)
python stage_b_dpo_training.py --beta 1.0

# Or reduce learning rate
python stage_b_dpo_training.py --learning_rate 1e-7
```

### Problem: Training Very Slow

```bash
# Check GPU utilization (should be >80%)
# If low, increase batch size if memory allows
python stage_b_dpo_training.py --batch_size 8

# Or disable gradient checkpointing (uses more memory, faster)
# (would need code modification)
```

---

## Expected Performance

### Training Speed

| Hardware | Time per Epoch |
|----------|--------|
| A100 (40GB) | 30-50 min |
| V100 (32GB) | 60-90 min |
| RTX 3090 (24GB) | 90-120 min |
| M1/M2 Mac (CPU) | 6-10 hours |

### Quality Improvement

**Before DPO (After SFT)**:
- Hallucination rate: ~30-40%
- Medical knowledge: Excellent ✓
- Preference for truth: Moderate

**After DPO (2 epochs)**:
- Hallucination rate: ~5-15%
- Medical knowledge: Maintained ✓
- Preference for truth: Excellent ✓

---

## Advanced: The DPO Loss Explained

### Step-by-Step

1. **Get reference model probabilities** (frozen):
   ```
   log_ref_chosen = reference_model.logprob(chosen)
   log_ref_rejected = reference_model.logprob(rejected)
   ```

2. **Get active model probabilities** (trainable):
   ```
   log_model_chosen = active_model.logprob(chosen)
   log_model_rejected = active_model.logprob(rejected)
   ```

3. **Compute log odds ratio**:
   ```
   log_odds = (log_model_chosen - log_model_rejected) - 
              (log_ref_chosen - log_ref_rejected)
   ```

4. **Apply sigmoid and DPO loss**:
   ```
   loss = -log(sigmoid(β * log_odds))
   ```

### Why This Works

- High `log_odds`: Model prefers chosen → sigmoid(β*large) → low loss ✓
- Low `log_odds`: Model prefers rejected → sigmoid(β*small) → high loss ✗
- The subtraction of reference log odds ensures **relative preference learning**
- KL term (implicit in β) prevents model from abandoning reference behavior

---

## Next Steps After DPO

### 1. Evaluate on Test Set

```python
from sft_inference import SFTInference

# Load DPO model
inference = SFTInference(
    model_path="./models/dpo_hallucination_resistant/final_model"
)

# Compare with SFT model
# Measure hallucination reduction
```

### 2. Merge LoRA Weights

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "./models/dpo_hallucination_resistant/final_model"
)
merged = model.merge_and_unload()
merged.save_pretrained("./models/dpo_merged")
```

### 3. Quantize for Deployment

```python
# 8-bit or 4-bit quantization for inference
# Reduces model size 4x or 8x
```

### 4. A/B Testing

Compare outputs:
- Stage A model (SFT) vs Stage B model (DPO)
- Measure hallucination metrics
- Human evaluation

---

## Common Questions

**Q: Can I skip Stage A and just do DPO?**
A: No. The model needs baseline medical knowledge first.

**Q: Why is learning rate so much lower in DPO?**
A: DPO trains with two models and implicit KL penalty. High LR causes instability.

**Q: What if my GPU can't fit two models?**
A: Reduce batch size or use 8-bit quantization for one model.

**Q: How do I know if my hard negatives are good?**
A: Good negatives should:
- Be very similar to truth (same format, mostly correct)
- Differ in ONE critical aspect (wrong dosage, wrong condition, etc.)
- NOT be obviously wrong

**Q: Can I use different base models for reference and active?**
A: Not recommended. Both should start from same SFT checkpoint.

---

## References

- **DPO Paper**: https://arxiv.org/abs/2305.18290
- **LoRA Paper**: https://arxiv.org/abs/2106.09714
- **PEFT Library**: https://huggingface.co/docs/peft

---

## Summary

**Stage B (DPO) teaches your model to prefer truth over hallucinations through**:
1. Loading TWO copies of your SFT model (reference + active)
2. Computing log probabilities for factual and hallucinated responses
3. Using DPO loss to maximize preferred (factual) probability
4. Minimizing KL divergence to prevent model drift
5. Resulting in dramatically reduced hallucinations

**Key Metrics**:
- Learning Rate: 5e-6 (100x lower than SFT)
- Chosen Preference: Should increase from 50% → 80%+
- Beta: 0.1 (KL penalty strength)
- Batch Size: 4 (vs 8 for SFT)

**Training Time**: 2-4 hours on GPU depending on hardware

**Result**: Hallucination-resistant medical specialist ready for production
