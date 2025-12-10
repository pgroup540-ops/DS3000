# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Common Use Cases

### 1. Basic Preprocessing (Normalization Only)

For quick text cleaning and standardization:

```bash
python preprocess_data.py
```

**Output:** Normalized text files in `./processed/`

---

### 2. Training Data Augmentation

For generating diverse training examples with adversarial negatives and evidence:

```bash
python preprocess_data.py \
    --output-dir ./augmented_train \
    --generate-adversarial \
    --generate-evidence \
    --adversarial-ratio 0.5
```

**Result:**
- Original 8 training examples → 15+ examples
- Adversarial negatives for robustness
- Evidence-annotated positives for interpretability

---

### 3. HIPAA-Compliant Processing

For de-identification and compliance:

```bash
python preprocess_data.py \
    --output-dir ./compliant \
    --redact-phi \
    --phi-mask-style hash
```

**Result:** All PHI (dates, names, IDs) are redacted

---

### 4. Full Pipeline (All Features)

For comprehensive preprocessing:

```bash
python preprocess_data.py \
    --output-dir ./processed_complete \
    --redact-phi \
    --generate-adversarial \
    --generate-evidence \
    --adversarial-ratio 0.3
```

---

## Verify Your Results

### Check processed files
```bash
ls -lh processed/
```

### View sample output
```bash
head -20 processed/train_set_processed.csv
```

### Count augmented examples
```bash
wc -l processed/train_set_processed.csv
wc -l train_set.csv
```

---

## Individual Module Testing

### Test Text Normalization
```bash
python text_normalizer.py
```

### Test PHI Redaction
```bash
python phi_redactor.py
```

### Test Adversarial Augmentation
```bash
python adversarial_augmenter.py
```

### Test Evidence Annotation
```bash
python evidence_annotator.py
```

---

## Expected Results

### Your Dataset Stats
- **Training set**: 8 records → 15+ with augmentation
- **Validation set**: 1 record
- **Test set**: 1 record

### Augmentation Breakdown
With `--adversarial-ratio 0.3` and `--generate-evidence`:
- **Factual examples in train**: 4
- **Adversarial negatives**: 3 (entity_swap, negation_invert, fabrication)
- **Evidence-annotated**: 4 (one per factual example)

---

## Next Steps

1. **Examine the output**: Check `processed/train_set_processed.csv`
2. **Adjust parameters**: Modify `--adversarial-ratio` based on your needs
3. **Integrate with model**: Use processed data for training your hallucination detector
4. **Iterate**: Re-run with different settings as needed

---

## Troubleshooting

### Issue: Import errors
**Solution**: Ensure all modules are in the same directory
```bash
ls *.py
# Should show: preprocess_data.py, text_normalizer.py, phi_redactor.py, etc.
```

### Issue: No augmented examples generated
**Check**: Augmentation only applies to train set with factual examples
```bash
# Verify factual examples exist
grep "factual" train_set.csv
```

### Issue: Memory errors with large datasets
**Solution**: Process splits individually
```bash
python preprocess_data.py --splits train
python preprocess_data.py --splits validation
python preprocess_data.py --splits test
```

---

## Pro Tips

1. **Start Simple**: Begin with normalization only, then add features
2. **Inspect Output**: Always check a few processed examples manually
3. **Version Control**: Keep different processed versions in separate directories
4. **Reproducibility**: Note your command and parameters for each run

---

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test individual modules
python text_normalizer.py
python phi_redactor.py

# 3. Run basic preprocessing
python preprocess_data.py --output-dir ./v1_basic

# 4. Run with full augmentation for training
python preprocess_data.py \
    --output-dir ./v2_augmented \
    --generate-adversarial \
    --generate-evidence \
    --adversarial-ratio 0.5

# 5. Compare results
wc -l train_set.csv v1_basic/train_set_processed.csv v2_augmented/train_set_processed.csv

# 6. Use augmented data for model training
# (your training code here)
```

---

## Support

For detailed documentation, see `README.md`

For issues or questions, review the module docstrings:
```python
python -c "from text_normalizer import TextNormalizer; help(TextNormalizer)"
```
