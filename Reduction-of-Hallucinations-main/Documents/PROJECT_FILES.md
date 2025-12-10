# Complete Project File Index

## Core Data Processing Modules

### `text_normalizer.py`
**Purpose**: Text standardization and canonicalization
- Handles unicode normalization
- Whitespace & punctuation standardization
- Medical abbreviation expansion (pt→patient, w/→with)
- Unit standardization (deg C → °C)
- Number format normalization (European to US decimals)
- **Used in**: All preprocessing runs (SFT, DPO, Eval)

### `phi_redactor.py`
**Purpose**: HIPAA compliance via Protected Health Information masking
- Detects: dates, ages >89, phone numbers, emails, SSNs, MRNs, names, addresses, ZIP codes, locations
- Masking styles: category-based `[NAME]`, hash-based, or generic `PATIENT001`
- Returns: redacted text + statistics
- **Used in**: All preprocessing runs (required for clinical data)

### `evidence_annotator.py`
**Purpose**: Evidence grounding for transparency
- Annotates summaries with pointers to supporting sentences from source
- Similarity matching: sequence matching + keyword overlap
- Produces evidence maps with confidence scores
- Citation formats: inline, superscript, footnote
- **Used in**: Run 1 (SFT) and Run 2 (DPO) - evidence helps model learn factual grounding

### `adversarial_augmenter.py`
**Purpose**: Hard negative generation for hallucination detection
- Entity extraction: medications, conditions, procedures, test results, measurements
- Three strategies:
  - `entity_swap`: Replace medical entities with alternatives
  - `negation_invert`: Change denies→has or vice versa
  - `fabrication`: Add false medical facts
- Produces hallucinated summaries with modification tracking
- **Used in**: Run 2 (DPO) - creates `rejected` outputs

### `dpo_triplet_generator.py` ⭐ **[NEW - Core Fix]**
**Purpose**: DPO-compatible triplet generation
- Key class: `DPOTripletGenerator`
- Generates `(prompt, chosen, rejected)` triplets from factual examples + adversarial negatives
- Validates triplets: non-empty fields, chosen ≠ rejected, all strings
- Tracks valid vs invalid triplets
- Supports multiple adversarial strategies per example
- **Used in**: Run 2 (DPO) - **Fixes the critical structural gap**

## Main Preprocessing Pipeline

### `preprocess_data.py` ⭐ **[REFACTORED]**
**Purpose**: Orchestrates entire preprocessing workflow
- **Main class**: `DataPreprocessor`
- Integrates: normalization, PHI redaction, evidence annotation, adversarial augmentation, triplet generation
- **New features**:
  - Dual output format support: "pairs" (SFT) or "triplets" (DPO)
  - Conditional routing based on training mode
  - DPO triplet validation
  - Configuration-driven pipeline
- **CLI arguments**: 20+ configurable options
- Produces: CSV files ready for training
- **Used in**: All three preprocessing runs

## Testing & Validation

### `test_dpo_preprocessing.py` ⭐ **[NEW]**
**Purpose**: Validates entire preprocessing pipeline
- Tests all three run configurations:
  - **Run 1**: SFT with Evidence (pairs)
  - **Run 2**: DPO with Triplets (NEW structure)
  - **Run 3**: Evaluation (clean)
- Generates sample clinical data
- Verifies output structure and statistics
- All tests pass ✓

**Test Results**:
```
Run 1: 10 records (5 originals + 5 evidence-augmented)
Run 2: 7 valid triplets, 2 invalid (correctly filtered)
Run 3: 5 clean records (no augmentation)
```

## Documentation

### `QUICK_START.md` ⭐ **[START HERE]**
**Purpose**: Quickest way to get running
- CLI commands for all three runs
- Output format examples
- Key differences table
- Integration code snippets
- Troubleshooting Q&A
- ~200 lines, highly condensed

### `DPO_PREPROCESSING_GUIDE.md` ⭐ **[COMPREHENSIVE]**
**Purpose**: Full documentation of DPO approach
- Architecture overview
- Detailed Run 1/2/3 configurations
- Data flow comparisons
- Quality metrics and tracking
- Validation rules
- Advanced usage patterns
- Training integration examples
- ~350 lines, complete reference

### `REFACTORING_SUMMARY.md` ⭐ **[TECHNICAL]**
**Purpose**: Explains what changed and why
- Problem identified (triplets vs separate rows)
- Solution architecture
- Key changes in detail
- Data flow examples
- Backwards compatibility notes
- Files modified/created
- ~265 lines, technical deep-dive

### `README.md`
**Purpose**: Project overview (original)
- Project context
- Module descriptions
- Basic usage
- Existing documentation

### `PROJECT_SUMMARY.md`
**Purpose**: Original project summary
- Existing functionality overview

## Three-Run Workflow Map

```
INPUT: Raw clinical data (85% train + 15% test)

RUN 1: SFT Training Data
├─ Command: --normalize --redact-phi --generate-evidence --output-format pairs
├─ Modules Used: TextNormalizer → PHIRedactor → EvidenceAnnotator
├─ Output: pairs (clinical_note, summary_with_evidence)
├─ Example: 5 inputs → 10 records (5 + 5 evidence)
└─ Purpose: Teach model what facts are correct

RUN 2: DPO Training Data ⭐ [FIXED STRUCTURE]
├─ Command: --normalize --redact-phi --generate-dpo-triplets --output-format triplets
├─ Modules Used: TextNormalizer → PHIRedactor → AdversarialAugmenter → DPOTripletGenerator
├─ Output: triplets (prompt, chosen, rejected)
├─ Example: 3 inputs × 3 strategies → 9 attempts → 7 valid triplets
├─ Each row contains: (clinical_note, correct_summary, hallucinated_summary)
└─ Purpose: Teach model what NOT to say

RUN 3: Evaluation Data
├─ Command: --normalize --redact-phi --output-format pairs (no augmentation)
├─ Modules Used: TextNormalizer → PHIRedactor
├─ Output: pairs (clinical_note, summary) - clean
├─ Example: 5 inputs → 5 records
└─ Purpose: Blind test without help
```

## CSV Output Formats

### Run 1 Output (SFT Pairs)
```csv
id,clinical_note,model_summary,label,hallucination_type,evidence_stats
test_001,Patient reports...,The patient has...,factual,,
test_001_evidence,Patient reports...,The patient has... [Evidence: S1; Conf: 0.95],factual,
```

### Run 2 Output (DPO Triplets) - **NEW STRUCTURE**
```csv
id,prompt,chosen,rejected,data_format,hallucination_type,strategy_used,modifications
test_001_dpo_0,Patient reports...,The patient has...,The patient exhibits...,dpo_triplet,adversarial_entity_swap,entity_swap,"[""entity_swap""]"
test_001_dpo_1,Patient reports...,The patient has...,The patient has no...,dpo_triplet,adversarial_negation_inverted,negation_invert,"[""negation_inverted""]"
test_001_dpo_2,Patient reports...,The patient has...,The patient has... Patient is on aspirin.,dpo_triplet,adversarial_fabrication,fabrication,"[""fabrication_medication""]"
```

### Run 3 Output (Evaluation Pairs)
```csv
id,clinical_note,model_summary,label
test_001,Patient reports...,The patient has...,factual
test_002,Patient presents...,Patient has pneumonia...,factual
```

## Module Dependency Graph

```
preprocess_data.py (Main orchestrator)
├── text_normalizer.py
│   └── [Applied first to all data]
├── phi_redactor.py
│   └── [Applied second to all data]
├── evidence_annotator.py
│   └── [Applied to Run 1 & 2 (SFT & DPO)]
├── adversarial_augmenter.py
│   └── [Used by dpo_triplet_generator]
└── dpo_triplet_generator.py ⭐ [NEW]
    ├── Uses: adversarial_augmenter.py
    └── [Applied to Run 2 (DPO) only]
```

## Configuration by Run

| Parameter | Run 1 | Run 2 | Run 3 |
|-----------|-------|-------|-------|
| `normalize` | True | True | True |
| `redact_phi` | True | True | True |
| `generate_adversarial` | False | False | False |
| `generate_evidence` | True | True | False |
| `generate_dpo_triplets` | False | True | False |
| `output_format` | pairs | triplets | pairs |
| `adversarial_ratio` | N/A | 0.5 | N/A |
| `splits` | train,val,test | train | val,test |

## Key Concepts

### DPO Triplet Structure (The Fix)
**Before**: Hard negatives as independent rows
```
Row 1: (clinical_note, summary, "factual")
Row 2: (clinical_note, summary_bad, "hallucinated")
→ No explicit relationship ❌
```

**After**: Complete triplets in single rows
```
Row 1: (prompt, chosen, rejected_v1)
Row 2: (prompt, chosen, rejected_v2)
Row 3: (prompt, chosen, rejected_v3)
→ Each row is a complete training example ✓
```

### Validation Rules
Each DPO triplet must satisfy:
1. ✓ All fields non-empty
2. ✓ `chosen ≠ rejected` (substantively different)
3. ✓ All fields are strings

Invalid triplets are tracked separately and excluded.

### Strategies (Adversarial Augmentation)
- **entity_swap**: Change medication/condition/procedure names
- **negation_invert**: Flip positive/negative statements
- **fabrication**: Add false medical information

Each strategy generates one triplet per factual example.

## Usage Patterns

### For SFT Training (Run 1 Output)
```python
df = pd.read_csv("processed/run1_sft/train_set_processed.csv")
dataset = Dataset.from_pandas(df[['clinical_note', 'model_summary']])
# sft_trainer.train(dataset)
```

### For DPO Training (Run 2 Output)
```python
df = pd.read_csv("processed/run2_dpo/train_set_processed.csv")
valid = df[df["data_format"] == "dpo_triplet"]
dataset = Dataset.from_pandas(valid[['prompt', 'chosen', 'rejected']])
# dpo_trainer.train(dataset)
```

### For Evaluation (Run 3 Output)
```python
df = pd.read_csv("processed/run3_eval/test_set_processed.csv")
dataset = Dataset.from_pandas(df[['clinical_note', 'model_summary']])
# results = model.evaluate(dataset)
```

## Summary

### Files Status
- **Core Modules**: 5 (all functional)
- **Main Pipeline**: 1 (refactored)
- **Testing**: 1 (new, all passing)
- **Documentation**: 3 (comprehensive)
- **Total**: 13 files

### Key Improvements
✓ Fixed DPO triplet structure (was the critical gap)
✓ Added automatic triplet validation
✓ Maintained backwards compatibility with SFT
✓ Comprehensive documentation
✓ End-to-end test coverage

### Next Steps
1. Read `QUICK_START.md` for commands
2. Read `DPO_PREPROCESSING_GUIDE.md` for details
3. Prepare data (85% train / 15% test)
4. Run: Run 1 → Run 2 → Run 3
5. Train: SFT (Run 1) → DPO (Run 2)
6. Evaluate: Run 3 output
