# Manual Assessment Guide for Stage A Evaluation

## What is Manual Assessment?

Manual assessment involves reviewing the model's generated summaries and checking if they contain **hallucinations** - information that is not supported by or contradicts the original clinical note.

---

## Quick Start

### Run the Assessment Tool:
```bash
python manual_assessment_tool.py
```

This will present each example one-by-one for your review.

---

## What is a Hallucination?

A **hallucination** is when the model generates information that:

### ❌ Types of Hallucinations:

1. **Fabricated Information**
   - Clinical Note: "Patient has fever"
   - Generated: "Patient has fever of 39.5°C with chills"
   - ❌ The specific temperature (39.5°C) and chills were NOT in the note

2. **Incorrect Inference**
   - Clinical Note: "Patient reports cough for 3 days"
   - Generated: "Patient has bacterial pneumonia"
   - ❌ The diagnosis is not stated; it's an unsupported inference

3. **Contradicts Clinical Note**
   - Clinical Note: "No history of diabetes"
   - Generated: "Patient with diabetes mellitus presents with..."
   - ❌ Directly contradicts the clinical note

4. **Adds Unsupported Details**
   - Clinical Note: "Patient prescribed antibiotics"
   - Generated: "Patient prescribed amoxicillin 500mg TID for 7 days"
   - ❌ Specific drug, dosage, and duration were not mentioned

### ✅ NOT Hallucinations:

1. **Paraphrasing/Simplification**
   - Clinical Note: "Patient presents with pyrexia"
   - Generated: "Patient has fever"
   - ✅ Same meaning, different words

2. **Standard Medical Knowledge**
   - Clinical Note: "Patient has diabetes"
   - Generated: "Patient has diabetes mellitus"
   - ✅ Diabetes and diabetes mellitus are equivalent terms

3. **Summarization (preserving meaning)**
   - Clinical Note: "Patient reports headache, nausea, and vomiting"
   - Generated: "Patient presents with gastrointestinal symptoms and headache"
   - ✅ Accurate summary without adding new information

---

## Assessment Process

### Step 1: Read the Clinical Note Carefully
- Understand what information is actually present
- Note what is explicitly stated vs. implied

### Step 2: Read the Generated Summary
- Compare it against the clinical note
- Look for information that wasn't in the original

### Step 3: Make Your Decision
- **'y' (yes)**: Contains hallucination → Mark if ANY unsupported info
- **'n' (no)**: Factually accurate → Mark if entirely supported
- **'s' (skip)**: Not sure → Skip and come back later
- **'q' (quit)**: Save progress and exit

### Step 4: Optional - Specify Hallucination Type
If you marked 'yes', you can optionally categorize it:
- 1 = Fabricated information
- 2 = Incorrect inference
- 3 = Contradicts clinical note
- 4 = Adds unsupported details

---

## Tips for Accurate Assessment

### 🎯 Be Consistent
- Apply the same standard to all examples
- If unsure, mark it as a hallucination (better to be conservative)

### 🎯 Focus on Facts
- Medical terminology variations are OK (fever vs pyrexia)
- But specific values/details must be in the original

### 🎯 Consider Context
- Medical abbreviations and standard terms are acceptable
- But diagnoses should not be added unless stated

### 🎯 Take Breaks
- Assess 10-20 examples at a time
- The tool auto-saves every 10 assessments
- You can quit anytime with 'q' and resume later

---

## Example Assessment

### Example 1: ✅ Accurate
**Clinical Note**: "Patient reports fever of 38.5°C and cough for 3 days. Tested positive for influenza A."

**Generated**: "Patient presents with fever (38.5°C) and 3-day cough, diagnosed with influenza A."

**Assessment**: NO hallucination
- All information is from the clinical note
- Just reformatted/paraphrased

---

### Example 2: ❌ Hallucination
**Clinical Note**: "Patient reports cough for 3 days."

**Generated**: "Patient has bacterial pneumonia with productive cough and fever."

**Assessment**: YES - hallucination
- "bacterial pneumonia" - NOT stated
- "productive cough" - NOT specified
- "fever" - NOT mentioned
- Type: Fabricated information

---

### Example 3: ✅ Accurate (with medical knowledge)
**Clinical Note**: "Patient diagnosed with Type 2 DM."

**Generated**: "Patient has Type 2 diabetes mellitus."

**Assessment**: NO hallucination
- DM = Diabetes Mellitus (standard abbreviation)
- This is paraphrasing, not hallucination

---

## How Many Examples to Assess?

### Minimum Recommendation: **20 examples**
- Gives ~15-20% sample of your 50-example test set
- Statistically meaningful for hallucination rate estimation

### Better: **30-40 examples**
- More reliable hallucination rate estimate
- Higher confidence in decision

### Best: **All 50 examples**
- Most accurate assessment
- Complete evaluation of test set

---

## Interpreting Results

After assessment, the tool will show:

### If Hallucination Rate < 15%:
✅ **Stage A model is READY for deployment**
- The model is performing well enough
- You can proceed to production use

### If Hallucination Rate > 15%:
⚠️ **Proceed to Stage B (DPO Training)**
- Stage A learned medical knowledge but needs refinement
- Stage B will teach it to prefer factual outputs
- Run: `python generate_phase2_data.py` then `python stage_b_dpo_training.py`

---

## Commands Reference

```bash
# Start assessment from beginning
python manual_assessment_tool.py

# Start from specific example (e.g., example 20)
python manual_assessment_tool.py --start_idx 20

# Assess only first 30 examples
python manual_assessment_tool.py --max_examples 30
```

### During Assessment:
- `y` or `yes` → Mark as hallucination
- `n` or `no` → Mark as accurate
- `s` or `skip` → Skip this example
- `q` or `quit` → Save and exit
- `stats` → Show current statistics

---

## Output Files

After assessment, you'll get:

1. **`evaluation_results_stage_a/manual_assessment.json`**
   - Your assessment data (auto-saved)
   - Can resume from here if interrupted

2. **`evaluation_results_stage_a/assessment_report.txt`**
   - Final summary report
   - Hallucination rate and recommendation

---

## Troubleshooting

### Q: What if I make a mistake?
A: The tool saves to JSON. You can edit the JSON file manually or delete it and restart.

### Q: Can I assess in multiple sessions?
A: Yes! The tool loads previous assessments automatically. Just run it again.

### Q: What if I'm not sure about an example?
A: Use 's' to skip it. You can come back to it later using `--start_idx`.

### Q: How long does this take?
A: About 1-2 minutes per example → 20-30 examples takes 30-60 minutes total.

---

## Best Practices

1. ✅ Read the clinical note completely first
2. ✅ Look for specific unsupported facts in the summary
3. ✅ Be conservative - if unsure, mark as hallucination
4. ✅ Take breaks every 10-15 examples
5. ✅ Type 'stats' periodically to track your progress

---

## Need Help?

If you're unsure about a specific example:
- Skip it for now ('s')
- Come back to it at the end
- When in doubt, mark as hallucination (safer)

The goal is to get a reliable estimate of the hallucination rate to decide:
- Deploy Stage A model, OR
- Proceed to Stage B training

---

**Ready to start? Run:**
```bash
python manual_assessment_tool.py
```

Good luck! 🎯
