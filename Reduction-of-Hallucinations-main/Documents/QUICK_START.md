# Quick Start: DPO Preprocessing Pipeline

## Quick Commands

### Run 1: SFT with Evidence
```bash
python preprocess_data.py \
  --input-dir ./data \
  --output-dir ./processed/run1_sft \
  --normalize --redact-phi --generate-evidence \
  --output-format pairs \
  --splits train validation test
```
**Output**: Pairs `(clinical_note, summary_with_evidence)`

---

### Run 2: DPO with Triplets
```bash
python preprocess_data.py \
  --input-dir ./data \
  --output-dir ./processed/run2_dpo \
  --normalize --redact-phi --generate-dpo-triplets \
  --output-format triplets \
  --adversarial-ratio 0.5 \
  --splits train
```
**Output**: Triplets `(prompt, chosen, rejected)` grouped by strategy

---

### Run 3: Evaluation (No Augmentation)
```bash
python preprocess_data.py \
  --input-dir ./data \
  --output-dir ./processed/run3_eval \
  --normalize --redact-phi \
  --output-format pairs \
  --splits validation test
```
**Output**: Clean pairs `(clinical_note, summary)` - no augmentation

---

## Key Differences

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| **Goal** | Teach facts | Reduce hallucinations | Evaluate |
| **Evidence** | ✓ | ✓ | ✗ |
| **Hard Negatives** | ✗ | ✓ | ✗ |
| **Output Format** | Pairs | **Triplets** | Pairs |
| **Data Split** | Train | Train | Test |
| **Output per Example** | 2 rows | 3 rows | 1 row |

---

## Understanding the Outputs

### Run 1 Output CSV
```csv
id,clinical_note,model_summary,label,hallucination_type,evidence_stats
test_001,Patient reports...,The patient has...,factual,
test_001_evidence,Patient reports...,The patient has... [Evidence: S1; Conf: 0.95],factual,
```
→ Use for **SFT training**

### Run 2 Output CSV
```csv
id,prompt,chosen,rejected,data_format,hallucination_type,strategy_used
test_001_dpo_0,Patient reports...,The patient has...,The patient exhibits...,dpo_triplet,adversarial_entity_swap,entity_swap
test_001_dpo_1,Patient reports...,The patient has...,The patient has no...,dpo_triplet,adversarial_negation_inverted,negation_invert
test_001_dpo_2,Patient reports...,The patient has...,The patient has... Patient is on aspirin.,dpo_triplet,adversarial_fabrication,fabrication
```
→ Use for **DPO training**
- Each row is a complete training example
- `prompt` = clinical note (same across all 3 strategies)
- `chosen` = correct summary (preferred)
- `rejected` = hallucinated summary (dispreferred)

### Run 3 Output CSV
```csv
id,clinical_note,model_summary,label
test_001,Patient reports...,The patient has...,factual
test_002,Patient presents...,Patient has pneumonia...,factual
```
→ Use for **Model evaluation** (clean, no augmentation)

---

## CSV Column Reference

### Run 1 & 3 (Pairs Format)
- `id`: Unique identifier
- `clinical_note`: Input medical text (prompt)
- `model_summary`: Expected output (target)
- `label`: factual/hallucinated
- `hallucination_type`: Type of hallucination (if any)
- `evidence_stats`: JSON with evidence metrics (Run 1 only)

### Run 2 (Triplets Format)
- `id`: Unique identifier
- `prompt`: Input medical text (same across all 3 strategies)
- `chosen`: Correct summary (preferred)
- `rejected`: Hallucinated summary (dispreferred)
- `data_format`: Always "dpo_triplet"
- `hallucination_type`: Type of hallucination created
- `strategy_used`: Which strategy created the rejection (entity_swap/negation_invert/fabrication)
- `modifications`: JSON array of modifications applied

---

## Critical Structural Fix

**Before (Wrong for DPO)**:
```
Row 1: (clinical_note="...", summary="...", label="factual")
Row 2: (clinical_note="...", summary="..._adversarial", label="hallucinated")
→ No explicit relationship between rows
```

**After (Correct for DPO)**:
```
Row 1: (prompt="...", chosen="...", rejected="..._adversarial_v1")
Row 2: (prompt="...", chosen="...", rejected="..._adversarial_v2")
Row 3: (prompt="...", chosen="...", rejected="..._adversarial_v3")
→ Complete triplet in each row ✓
```

---

## Validation

Each triplet is validated for:
- ✓ Non-empty `prompt`, `chosen`, `rejected`
- ✓ `chosen ≠ rejected` (must differ)
- ✓ All fields are strings

Stats example (3 examples, 3 strategies each = 9 attempts):
```
DPO Triplets Generated: 7 ✓
Invalid Triplets: 2     ← Skipped if chosen == rejected
```

---

## Integration with Training

### Using Run 1 for SFT
```python
import pandas as pd
from datasets import Dataset

df = pd.read_csv("processed/run1_sft/train_set_processed.csv")
dataset = Dataset.from_pandas(df[['clinical_note', 'model_summary']])
# sft_trainer.train(dataset)
```

### Using Run 2 for DPO
```python
import pandas as pd
from datasets import Dataset

df = pd.read_csv("processed/run2_dpo/train_set_processed.csv")
# Filter for valid triplets
valid = df[df["data_format"] == "dpo_triplet"]
dataset = Dataset.from_pandas(valid[['prompt', 'chosen', 'rejected']])
# dpo_trainer.train(dataset)
```

### Using Run 3 for Evaluation
```python
import pandas as pd
from datasets import Dataset

df = pd.read_csv("processed/run3_eval/test_set_processed.csv")
dataset = Dataset.from_pandas(df[['clinical_note', 'model_summary']])
# model.evaluate(dataset)
```

---

## Troubleshooting

**Q: Run 2 generates fewer triplets than expected?**
- Some triplets fail validation (e.g., entity_swap didn't change summary)
- Check `invalid_triplets` count in stats
- Try `--adversarial-ratio 1.0` to test with all examples

**Q: How many triplets per example in Run 2?**
- 3 strategies × 1 attempt = 3 potential triplets per factual example
- Invalid ones are filtered out during validation

**Q: Can I customize the strategies?**
- Yes, edit `adversarial_strategies` list in `generate_dpo_triplets_batch()`
- Options: `entity_swap`, `negation_invert`, `fabrication`, `multiple`

**Q: Why is evidence included in Run 2?**
- It grounds the model in source text, making hallucinations more obvious
- Helps DPO learn to align with evidence during contrast learning
