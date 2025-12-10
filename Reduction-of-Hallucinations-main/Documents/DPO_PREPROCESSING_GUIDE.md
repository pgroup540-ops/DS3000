# DPO Triplet Preprocessing Guide

This guide explains how to run the preprocessing pipeline for all three data runs: SFT, DPO, and Evaluation.

## Overview

The refactored pipeline now supports two output formats:

1. **Pairs Format** (for SFT): Each row contains `(clinical_note, model_summary)` 
2. **Triplets Format** (for DPO): Each row contains `(prompt, chosen, rejected)` as a single unit

## Architecture Changes

### New Module: `dpo_triplet_generator.py`

This module is responsible for creating DPO-compatible triplets:

- **`DPOTripletGenerator`**: Main class that generates `(prompt, chosen, rejected)` triplets
- **Features**:
  - Generates triplets from factual examples paired with adversarial negatives
  - Supports multiple augmentation strategies: `entity_swap`, `negation_invert`, `fabrication`
  - Validates triplets to ensure `chosen ≠ rejected`
  - Tracks invalid triplets for quality assurance

### Key Updates to `preprocess_data.py`

- New parameters:
  - `generate_dpo_triplets`: Whether to generate DPO triplets
  - `output_format`: "pairs" for SFT or "triplets" for DPO
  
- New method: `generate_dpo_triplets_batch()` - generates triplets while maintaining prompt-chosen-rejected relationships

## Usage: Three Run Configuration

### Run 1: SFT with Evidence (Pairs Format)

**Purpose**: Teach the model what medical facts are correct

**Configuration**:
- Normalize: ON
- PHI Mask: ON  
- Evidence: ON
- Hard Negatives: OFF
- Output Format: pairs

**Command**:
```bash
python preprocess_data.py \
  --input-dir ./data \
  --output-dir ./processed/run1_sft \
  --normalize \
  --redact-phi \
  --generate-evidence \
  --no-generate-dpo-triplets \
  --output-format pairs \
  --splits train validation test
```

**Output Structure**:
```csv
id,clinical_note,model_summary,label,hallucination_type,evidence_stats
test_001,Patient reports...,The patient has...,factual,,{"total_summary_sentences": 3, "supported_sentences": 3, ...}
test_001_evidence,Patient reports...,"The patient has... [Evidence: S1; Conf: 0.95]",...
```

**Use case**: Fine-tune with SFT on original + evidence-annotated pairs to teach factual grounding

---

### Run 2: DPO with Triplets (Triplet Format)

**Purpose**: Teach the model what NOT to say (hallucination reduction)

**Configuration**:
- Normalize: ON
- PHI Mask: ON
- Evidence: ON
- Hard Negatives: ON ← Key difference from Run 1
- Output Format: triplets

**Command**:
```bash
python preprocess_data.py \
  --input-dir ./data \
  --output-dir ./processed/run2_dpo \
  --normalize \
  --redact-phi \
  --generate-evidence \
  --generate-dpo-triplets \
  --output-format triplets \
  --adversarial-ratio 0.5 \
  --splits train
```

**Output Structure**:
```csv
id,prompt,chosen,rejected,data_format,hallucination_type,strategy_used,modifications
test_001_dpo_0,"Patient reports...","The patient has...","The patient exhibits...","dpo_triplet","adversarial_entity_swap","entity_swap","[\"entity_swap\"]"
test_001_dpo_1,"Patient reports...","The patient has...","The patient has chest pain without...","dpo_triplet","adversarial_negation_inverted","negation_invert","[\"negation_inverted\"]"
test_001_dpo_2,"Patient reports...","The patient has...","The patient has... Patient is on aspirin.","dpo_triplet","adversarial_fabrication_medication","fabrication","[\"fabrication_medication\"]"
```

**Structure Details**:
- **prompt**: Clinical note (same for all triplets from same source)
- **chosen**: Correct/factual summary (the preferred response)
- **rejected**: Adversarial summary (the dispreferred response)
- **data_format**: Always "dpo_triplet" to identify format
- **strategy_used**: Which augmentation created the rejection
- **modifications**: JSON array of modifications applied

**Multiple triplets per source**: Each factual example generates 3 triplets (one per strategy)

**Use case**: Train with DPO algorithm to maximize likelihood of correct summaries while minimizing hallucinated ones

---

### Run 3: Evaluation (Pairs Format, No Augmentation)

**Purpose**: Blind test without help (evaluate final model)

**Configuration**:
- Normalize: ON
- PHI Mask: ON
- Evidence: OFF ← Removed to test model independently
- Hard Negatives: OFF
- Output Format: pairs

**Command**:
```bash
python preprocess_data.py \
  --input-dir ./data \
  --output-dir ./processed/run3_eval \
  --normalize \
  --redact-phi \
  --no-generate-evidence \
  --no-generate-dpo-triplets \
  --output-format pairs \
  --splits validation test
```

**Output Structure**:
```csv
id,clinical_note,model_summary,label
test_001,Patient reports...,The patient has...,factual
test_002,Patient presents...,Patient has pneumonia...,factual
```

**Use case**: Evaluate the model's ability to generate correct summaries without evidence annotations

---

## Data Flow Comparison

### Run 1 (SFT): 
```
Input:
  5 factual examples

Processing:
  - Normalize & redact
  - Add evidence annotations

Output:
  5 originals + 5 evidence-annotated = 10 pairs
  Each with (prompt, truth) structure
  ↓ Used for SFT
```

### Run 2 (DPO):
```
Input:
  3 factual examples

Processing:
  - Normalize & redact
  - Generate 3 strategies × 3 examples = 9 adversarial versions
  - Validate triplets
  - Group as (prompt, chosen, rejected)

Output:
  3 examples × 3 strategies = 9 triplets → 7 valid
  Each with (prompt, chosen, rejected) structure
  ↓ Used for DPO training
```

### Run 3 (Eval):
```
Input:
  5 test examples

Processing:
  - Normalize & redact only

Output:
  5 pairs (no augmentation)
  Each with (prompt, truth) structure
  ↓ Used for evaluation
```

---

## Quality Metrics

The preprocessing pipeline tracks:

- **total_records**: Original examples loaded
- **normalized**: Text normalized
- **phi_redacted**: PHI redacted
- **evidence_annotated**: Evidence annotations added
- **dpo_triplets_generated**: Valid triplets created
- **invalid_triplets**: Triplets rejected (e.g., chosen == rejected)

Example stats for Run 2:
```
Total Records: 3
DPO Triplets Generated: 7
Invalid Triplets: 2  ← These are skipped
```

---

## Triplet Validation

Triplets are validated to ensure:

1. ✓ All required fields present: `prompt`, `chosen`, `rejected`
2. ✓ No field is empty
3. ✓ `chosen ≠ rejected` (they must differ)
4. ✓ All fields are strings

Failed validations are counted and logged but don't stop processing.

---

## Advanced Usage

### Custom Adversarial Strategies

Modify which strategies to use for hard negatives:

```python
# In preprocess_data.py, modify generate_dpo_triplets_batch:
adversarial_strategies = ["entity_swap", "negation_invert", "fabrication", "multiple"]
```

### Adjusting Triplet Generation Ratio

Control what percentage of factual examples get triplets:

```bash
--adversarial-ratio 1.0  # Generate triplets for ALL factual examples
--adversarial-ratio 0.5  # Generate triplets for 50% of factual examples
```

### Output Format Inspection

Load and inspect generated triplets:

```python
import pandas as pd

# Run 2 output
dpo_df = pd.read_csv("processed/run2_dpo/train_set_processed.csv")
triplets = dpo_df[dpo_df["data_format"] == "dpo_triplet"]

# Inspect one triplet
triplet = triplets.iloc[0]
print(f"Strategy: {triplet['strategy_used']}")
print(f"Prompt: {triplet['prompt'][:100]}...")
print(f"Chosen: {triplet['chosen'][:100]}...")
print(f"Rejected: {triplet['rejected'][:100]}...")
```

---

## Training Integration

### For Run 1 (SFT):
```python
from datasets import Dataset

pairs = pd.read_csv("processed/run1_sft/train_set_processed.csv")
dataset = Dataset.from_pandas(pairs[['clinical_note', 'model_summary']])

# Fine-tune with SFT trainer
```

### For Run 2 (DPO):
```python
from datasets import Dataset

triplets = pd.read_csv("processed/run2_dpo/train_set_processed.csv")
# Filter for valid triplets
valid_triplets = triplets[triplets["data_format"] == "dpo_triplet"]

dpo_dataset = Dataset.from_pandas(valid_triplets[['prompt', 'chosen', 'rejected']])

# Train with DPO trainer
# dpo_trainer.train(dpo_dataset)
```

### For Run 3 (Eval):
```python
from datasets import Dataset

test_data = pd.read_csv("processed/run3_eval/test_set_processed.csv")
eval_dataset = Dataset.from_pandas(test_data[['clinical_note', 'model_summary']])

# Evaluate model
# predictions = model.generate(eval_dataset)
```

---

## Troubleshooting

**Issue**: "InvalidArgumentError: Chosen and rejected outputs are identical"

**Cause**: Adversarial augmentation failed to modify the text

**Solution**: 
- Check that summaries contain entities that can be swapped (medications, conditions, etc.)
- Lower the similarity threshold in `evidence_annotator.py`
- Increase the pool of adversarial strategies

**Issue**: Run 2 generates fewer triplets than expected

**Cause**: Some generated triplets fail validation

**Check stats**: Look for `invalid_triplets` count in processing summary

**Solution**:
- Review failed triplets in logs
- Adjust augmentation strategies if needed
- Consider using `adversarial_ratio=1.0` to see all generated candidates

---

## Summary Table

| Aspect | Run 1 (SFT) | Run 2 (DPO) | Run 3 (Eval) |
|--------|-----------|-----------|------------|
| **Purpose** | Teach facts | Reduce hallucinations | Blind test |
| **Output Format** | Pairs | Triplets | Pairs |
| **Hard Negatives** | No | Yes | No |
| **Evidence Annotations** | Yes | Yes | No |
| **Data Split** | 85% train | 85% train | 15% test |
| **Augmentation** | Evidence only | Hard negatives | None |
| **Rows per Example** | 2 (original + evidence) | 3 (3 strategies) | 1 (original) |
| **Training Algorithm** | SFT | DPO | Evaluation |
