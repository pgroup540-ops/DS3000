# Clinical Text Preprocessing Pipeline

A comprehensive data preprocessing and augmentation pipeline for clinical text data, designed to reduce hallucinations in medical text summarization models.

## Features

### 1. **Text Normalization** (`text_normalizer.py`)
- Unicode normalization (BOM removal, NFC form)
- Whitespace normalization
- Punctuation standardization
- Medical abbreviation expansion
- Unit standardization (temperature, medication dosages, etc.)
- Number format normalization
- Tokenization support

### 2. **PHI Redaction** (`phi_redactor.py`)
HIPAA-compliant Protected Health Information (PHI) detection and masking:
- Dates (multiple formats)
- Ages over 89
- Phone numbers and email addresses
- Medical record numbers (MRN) and SSNs
- Names with titles
- Street addresses and ZIP codes
- Location references

**Masking Styles:**
- `category`: Replace with `[CATEGORY]` (e.g., `[NAME]`, `[DATE]`)
- `hash`: Replace with hash of original value
- `generic`: Replace with generic placeholders (e.g., `PATIENT001`)

### 3. **Adversarial Data Augmentation** (`adversarial_augmenter.py`)
Generate hard negative examples through:
- **Entity Swaps**: Replace medications, conditions, test results, measurements
- **Negation Inversion**: Convert positive findings to negative and vice versa
- **Information Fabrication**: Add hallucinated medical information

### 4. **Evidence Annotation** (`evidence_annotator.py`)
Generate summaries with inline evidence pointers:
- Sentence-level evidence mapping
- Confidence scoring for evidence links
- Multiple citation formats (inline, superscript, footnote)
- Support ratio calculation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process all datasets with normalization only:
```bash
python preprocess_data.py
```

### Advanced Usage

#### With All Features Enabled
```bash
python preprocess_data.py \
    --input-dir . \
    --output-dir ./processed \
    --redact-phi \
    --generate-adversarial \
    --generate-evidence
```

#### Normalize and Redact PHI
```bash
python preprocess_data.py \
    --redact-phi \
    --phi-mask-style category
```

#### Generate Adversarial Examples
```bash
python preprocess_data.py \
    --generate-adversarial \
    --adversarial-ratio 0.5
```

#### Generate Evidence-Annotated Examples
```bash
python preprocess_data.py \
    --generate-evidence
```

#### Disable Normalization
```bash
python preprocess_data.py \
    --no-normalize \
    --redact-phi
```

### Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-dir` | str | `.` | Directory containing input CSV files |
| `--output-dir` | str | `./processed` | Directory for output CSV files |
| `--normalize` | flag | `True` | Apply text normalization |
| `--no-normalize` | flag | - | Disable text normalization |
| `--redact-phi` | flag | `False` | Redact PHI from text |
| `--phi-mask-style` | str | `category` | PHI masking style (`category`, `hash`, `generic`) |
| `--generate-adversarial` | flag | `False` | Generate adversarial negative examples |
| `--adversarial-ratio` | float | `0.5` | Ratio of adversarial examples to generate |
| `--generate-evidence` | flag | `False` | Generate evidence-annotated examples |
| `--splits` | list | `train validation test` | Dataset splits to process |

## Input Format

The pipeline expects CSV files with the following columns:
- `id`: Unique identifier
- `clinical_note`: Original clinical text
- `model_summary`: Model-generated summary
- `label`: `factual` or `hallucinated`
- `hallucination_type`: Type of hallucination (if applicable)

Expected filenames: `train_set.csv`, `validation_set.csv`, `test_set.csv`

## Output Format

Processed files are saved with the same structure, with additional augmented examples appended (if enabled).

### Output Files
- `train_set_processed.csv`
- `validation_set_processed.csv`
- `test_set_processed.csv`

### Augmented Example IDs
- Adversarial: `{original_id}_adv_{strategy}`
- Evidence: `{original_id}_evidence`

## Module Usage Examples

### Text Normalization
```python
from text_normalizer import TextNormalizer

normalizer = TextNormalizer()
text = "Patient   reports fever of 38.5 deg C and cough."
normalized = normalizer.normalize(text)
# Output: "Patient reports fever of 38.5°C and cough."
```

### PHI Redaction
```python
from phi_redactor import PHIRedactor

redactor = PHIRedactor(mask_style="category")
text = "Patient John Doe, MRN 123456, was seen on 03/15/2024."
redacted, stats = redactor.redact_all(text)
# Output: "Patient [NAME], MRN [ID], was seen on [DATE]."
```

### Adversarial Augmentation
```python
from adversarial_augmenter import AdversarialAugmenter

augmenter = AdversarialAugmenter()
clinical_note = "Patient reports mild chest pain. ECG normal."
summary = "Patient has chest pain with normal ECG."

adversarial = augmenter.generate_adversarial_negative(
    clinical_note,
    summary,
    strategy="negation_invert"
)
# Generates a contradictory summary
```

### Evidence Annotation
```python
from evidence_annotator import EvidenceAnnotator

annotator = EvidenceAnnotator(similarity_threshold=0.3)
clinical_note = "Patient reports mild chest pain for 2 days. ECG normal."
summary = "Patient has chest pain with normal ECG findings."

result = annotator.annotate_with_evidence(clinical_note, summary)
# Output includes evidence pointers: "[Evidence: S1,S2; Conf: 0.85]"
```

## Architecture

```
preprocess_data.py (Main Pipeline)
    ├── text_normalizer.py (Text Normalization)
    ├── phi_redactor.py (PHI Redaction)
    ├── adversarial_augmenter.py (Adversarial Augmentation)
    └── evidence_annotator.py (Evidence Annotation)
```

## Processing Statistics

The pipeline provides detailed statistics on completion:
- Total records processed
- Number of normalized texts
- Number of PHI redactions
- Adversarial examples generated
- Evidence-annotated examples created

## Best Practices

1. **For Training Data**: Enable all augmentation features
   ```bash
   python preprocess_data.py --generate-adversarial --generate-evidence
   ```

2. **For Production/Inference**: Use normalization and PHI redaction only
   ```bash
   python preprocess_data.py --redact-phi
   ```

3. **For Compliance-Critical Applications**: Always enable PHI redaction
   ```bash
   python preprocess_data.py --redact-phi --phi-mask-style hash
   ```

4. **For Research/Development**: Start with normalization only, then iterate
   ```bash
   python preprocess_data.py
   ```

## Testing Individual Modules

Each module includes a test section that can be run independently:

```bash
python text_normalizer.py
python phi_redactor.py
python adversarial_augmenter.py
python evidence_annotator.py
```

## Future Enhancements

Consider integrating:
- **spaCy/scispaCy**: Advanced NER for better entity extraction
- **Medical ontologies**: UMLS, SNOMED CT for entity normalization
- **Transformer models**: For semantic similarity in evidence annotation
- **Active learning**: For iterative augmentation refinement

## License

This pipeline is designed for research and educational purposes.

## Contributing

When contributing, ensure:
1. All new features include documentation
2. Test functions are provided for each module
3. Code follows existing patterns and style
4. PHI handling remains HIPAA-compliant
