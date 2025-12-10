# Stage A: Supervised Fine-Tuning (SFT) Implementation Guide

## Overview

Stage A transforms a generic base model (like Llama-2, Llama-3, or Mistral) into a **Medical Specialist** through supervised fine-tuning with LoRA.

### What Happens in Stage A

- **Input**: `sft_train_data.csv` containing pairs of (Clinical Note, Medical Summary)
- **Goal**: Teach the model medical domain knowledge, terminology, and the specific output format (with citations)
- **Mechanism**: Standard "Next Token Prediction" using Cross-Entropy Loss
- **Output**: A model that can coherently discuss medicine in your preferred format

### Why It Comes First

You cannot teach a model to avoid hallucinations (Stage B) if it doesn't even know how to talk about medicine. SFT establishes the baseline capability.

---

## Installation & Setup

### 1. Install Dependencies

```bash
# Install training dependencies
pip install -r requirements_training.txt

# Or install individually for M1/M2 Macs with special handling:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft accelerate tqdm
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

---

## Key Technical Specifications

### Learning Rate: 2e-4 or 1e-4
- Standard LoRA training rate
- Lower than full fine-tuning (5e-4 to 1e-3)
- Prevents catastrophic forgetting

### Epochs: 1-3
- **1 epoch**: Quick baseline (minimum recommended)
- **2 epochs**: Standard choice (recommended)
- **3 epochs**: Maximum to avoid overfitting
- Too many epochs → model overfits to training data

### LoRA Configuration

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `r` (rank) | 16-32 | 16 = minimal capacity; 32 = higher capacity for complex medical facts |
| `alpha` | 32 | Scaling factor (typically 2x the rank) |
| `dropout` | 0.05 | 5% dropout on LoRA layers (prevents overfitting) |
| `target_modules` | `["q_proj", "v_proj"]` | Query and Value projections in attention heads |

### Sequence Length: 512 tokens
- Typical clinical notes: 200-400 tokens
- Summaries: 100-150 tokens
- Total: ~300 tokens on average

---

## Training Data Format

Your training CSV should have these columns:

```
id,clinical_note,model_summary,label,hallucination_type
1,Patient had fever of 38.5°C and cough. Tested positive for influenza A.,The patient tested positive for influenza A and experienced fever.,factual,
2,Patient recovering from COVID-19 infection. No respiratory distress.,The patient is recovering from COVID-19 without respiratory distress.,factual,
```

**Key Points**:
- Only **factual** examples are used during SFT
- Hallucinated examples are **ignored** (saved for Stage B DPO)
- Use evidence-annotated summaries if available (e.g., `model_summary` with inline citations)

---

## Quick Start: Training

### Option 1: Default Training (Recommended)

```bash
cd /Users/joshiin/Projects/Reduction\ of\ Hallucinations

python stage_a_sft_training.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --output_dir "./models/sft_specialist" \
    --num_epochs 2 \
    --learning_rate 2e-4 \
    --batch_size 8 \
    --lora_r 16
```

### Option 2: Memory-Efficient Training (M1/M2 Mac or Limited GPU)

```bash
python stage_a_sft_training.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --output_dir "./models/sft_specialist" \
    --num_epochs 2 \
    --learning_rate 2e-4 \
    --batch_size 4 \
    --lora_r 8 \
    --device cpu
```

### Option 3: With Smaller Model (Faster Training)

```bash
python stage_a_sft_training.py \
    --model_name "mistralai/Mistral-7B" \
    --output_dir "./models/sft_specialist" \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 8
```

---

## Understanding Training Output

### Training Progress

```
Epoch 1/2
============================================================
Train Loss: 2.1453
Val Loss: 2.0876
Learning Rate: 2.00e-04
Saved checkpoint to ./models/sft_specialist/checkpoint_epoch_1

Epoch 2/2
============================================================
Train Loss: 1.8234
Val Loss: 1.7956
Learning Rate: 1.50e-04
Saved checkpoint to ./models/sft_specialist/checkpoint_epoch_2

Training completed!
Saved final model to ./models/sft_specialist/final_model
```

### Expected Loss Curves

- **Good training**: Loss decreases smoothly (e.g., 3.0 → 2.5 → 2.2)
- **Overfitting**: Train loss ↓ but val loss ↑ (consider fewer epochs)
- **Underfitting**: Loss barely changes (consider more epochs or larger `lora_r`)
- **Diverging loss**: NaN or Inf (reduce learning rate or batch size)

### Model Checkpoints

```
./models/sft_specialist/
├── final_model/                    # Final trained model
│   ├── adapter_config.json        # LoRA configuration
│   ├── adapter_model.bin          # LoRA weights
│   ├── tokenizer.json
│   └── sft_config.json
├── checkpoint_epoch_1/            # Intermediate checkpoint
├── checkpoint_epoch_2/
└── training_stats.json            # Loss metrics
```

---

## Using the Fine-Tuned Model

### Option 1: Direct Inference (Simple)

```bash
python sft_inference.py \
    --model_path "./models/sft_specialist/final_model" \
    --clinical_note "Patient reports mild chest pain. ECG normal. No fever."
```

### Option 2: Batch Inference (Multiple Notes)

Create a file `clinical_notes.txt`:
```
Patient reports fever of 38.5°C and cough.
Patient recovering from knee surgery. Pain level 3/10.
Child with mild asthma. Using inhaler daily.
```

Then run:
```bash
python sft_inference.py \
    --model_path "./models/sft_specialist/final_model" \
    --input_file clinical_notes.txt
```

### Option 3: Python Code

```python
from sft_inference import SFTInference

# Initialize
inference = SFTInference(
    model_path="./models/sft_specialist/final_model",
    device="cuda"
)

# Generate summary
result = inference.generate_summary(
    clinical_note="Patient had fever of 38.5°C and cough. Tested positive for influenza A.",
    max_new_tokens=150,
    temperature=0.7
)

print(result['generated_summary'])
```

---

## Expected Outputs from Stage A

### Example 1: Basic Summary
```
Input:  Patient reports fever of 38.5°C and cough. Tested positive for influenza A.
Output: The patient tested positive for influenza A and has fever.
```

### Example 2: Complex Case
```
Input:  Patient recovering from knee surgery. Pain level 3/10. Using crutches.
        No signs of infection. Mobility improving.
Output: Patient is recovering from knee surgery with mild pain (3/10) and is using crutches.
        Mobility is improving with no infection signs.
```

### Example 3: With Evidence (if trained on evidence data)
```
Input:  Patient had fever of 38.5°C and cough. Tested positive for influenza A.
        Recent contact with infected family member.
Output: The patient tested positive for influenza A and experienced fever. 
        [Evidence: S1,2; Conf: 0.85]
```

---

## Hyperparameter Tuning Guide

### When to Adjust `lora_r`

| Scenario | Adjustment | Reasoning |
|----------|------------|-----------|
| Model underfits (loss plateaus) | Increase to 32 | More capacity needed |
| OOM errors | Decrease to 8 | Reduce memory usage |
| Training is very slow | Decrease to 8 | Faster training |
| Overfitting observed | Decrease to 16 | Regularize |

### When to Adjust Learning Rate

| Problem | Solution | Why |
|---------|----------|-----|
| Loss diverges (NaN) | Reduce to 1e-4 | Too aggressive updates |
| Loss stagnates | Increase to 5e-4 | Too conservative |
| Oscillating loss | Reduce to 1e-5 | Unstable training |
| Very slow convergence | Increase to 5e-4 | Accelerate |

### When to Adjust Batch Size

| Constraint | Action | Trade-off |
|------------|--------|-----------|
| GPU OOM | Reduce to 4 or 2 | Slower training |
| Noisy gradients | Increase to 16 | Needs more GPU memory |
| Very fast training | Increase to 16 | Better gradient estimates |
| Limited GPU | Reduce to 4 | Acceptable |

---

## Troubleshooting

### Problem: "CUDA out of memory"
```bash
# Solution 1: Reduce batch size
python stage_a_sft_training.py ... --batch_size 4

# Solution 2: Use 8-bit quantization
python stage_a_sft_training.py ... --use_8bit

# Solution 3: Use smaller LoRA rank
python stage_a_sft_training.py ... --lora_r 8
```

### Problem: Loss doesn't decrease (underfitting)
```bash
# Increase training capacity
python stage_a_sft_training.py ... --num_epochs 3 --lora_r 32

# Or increase learning rate slightly
python stage_a_sft_training.py ... --learning_rate 5e-4
```

### Problem: Training stops abruptly
```bash
# Likely data issue. Check CSV format:
# 1. Ensure no missing values in critical columns
# 2. Verify clinical_note and model_summary are non-empty
# 3. Check that label='factual' for training examples

# Test dataset:
python sft_dataset.py
```

### Problem: Model won't load for inference
```bash
# Try with device='cpu'
python sft_inference.py \
    --model_path "./models/sft_specialist/final_model" \
    --device cpu

# Check if model files exist
ls ./models/sft_specialist/final_model/
```

---

## Next Steps: Preparing for Stage B

After Stage A completes successfully:

1. **Evaluate the model** on test data
2. **Compare outputs** with baselines to verify improvement
3. **Prepare DPO data** using hard negatives
4. **Move to Stage B**: Direct Preference Optimization

### Evaluation Questions to Ask

- ✓ Does the model understand medical terminology?
- ✓ Does it follow your output format (citations, structure)?
- ✓ Are summaries accurate reflections of clinical notes?
- ✓ Is there evidence of hallucinations?

**If Yes to all but last**: Ready for Stage B
**If No to some**: Adjust Stage A training and retrain

---

## Advanced Topics

### Merging LoRA Weights

To convert a LoRA model to a standalone model:

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load LoRA model
model = AutoPeftModelForCausalLM.from_pretrained(
    "./models/sft_specialist/final_model"
)

# Merge LoRA weights into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./models/sft_specialist/merged_model")
```

### Using with Hugging Face Trainer

If you prefer the official Hugging Face Trainer API:

```bash
python -c "
from transformers import Trainer, TrainingArguments
from sft_dataset import create_sft_dataloaders

# Create dataset
train_loader, val_loader = create_sft_dataloaders(
    'phase1_data/sft/train_set_processed.csv',
    'phase1_data/sft/validation_set_processed.csv',
    tokenizer
)

# Setup training arguments
args = TrainingArguments(
    output_dir='./models/sft_specialist',
    num_train_epochs=2,
    learning_rate=2e-4,
    per_device_train_batch_size=8,
)

# Train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
trainer.train()
"
```

---

## References

- **LoRA Paper**: https://arxiv.org/abs/2106.09714
- **Supervised Fine-Tuning**: Standard approach in NLP
- **Medical LLMs**: Relevant to specialized domain adaptation
- **PEFT Library**: https://github.com/huggingface/peft

---

## Summary

**Stage A transforms a generic model into a Medical Specialist through**:
1. Loading a base model (Llama-2, Llama-3, Mistral, etc.)
2. Applying LoRA for parameter-efficient fine-tuning
3. Training on factual (prompt, response) pairs
4. Using standard cross-entropy loss for next-token prediction
5. Achieving domain expertise in medical text summarization

**Key Metrics**:
- Train Loss: Should decrease smoothly (3.0 → 2.0 → 1.5)
- Val Loss: Should follow similar trend
- Epoch Time: 30-60 min depending on hardware

**Next Phase**: Stage B (DPO) uses this specialized model + hard negatives to create hallucination resistance
