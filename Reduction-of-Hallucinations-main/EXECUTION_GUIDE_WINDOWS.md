# Execution Guide - Running on Windows Computer

**📍 Use this guide when you're at your friend's house to run the complete pipeline on their Windows machine.**

---

## 📋 Pre-Flight Checklist

Before you start, make sure you have:
- [ ] GitHub repository link or ZIP file
- [ ] Your training data (CSV files)
- [ ] Access to your friend's Windows computer
- [ ] Internet connection (for downloading models)
- [ ] GPU with 16GB+ VRAM (recommended) or patience for CPU training

---

## 🚀 Step 1: Initial Setup (15 minutes)

### 1A. Download the Project

**Option A: Using Git**
```cmd
cd C:\Users\YourFriend\Documents
git clone https://github.com/YOUR_USERNAME/Reduction-of-Hallucinations.git
cd Reduction-of-Hallucinations
```

**Option B: From ZIP File**
1. Extract the ZIP file to `C:\Users\YourFriend\Documents\Reduction-of-Hallucinations`
2. Open Command Prompt
3. Navigate to the folder:
```cmd
cd C:\Users\YourFriend\Documents\Reduction-of-Hallucinations
```

### 1B. Verify Python Installation

```cmd
python --version
```

Expected output: `Python 3.8.x` or higher

If not installed:
1. Go to https://www.python.org/downloads/
2. Download Python 3.8+
3. **Check "Add Python to PATH"** during installation
4. Restart Command Prompt

### 1C. Install Dependencies

```cmd
pip install pandas scikit-learn torch transformers peft tqdm openpyxl
```

**This will take 5-10 minutes.** Wait for it to complete.

### 1D. Setup Directory Structure

```cmd
python setup_directories.py
```

Expected output:
```
======================================================================
Setting Up Directory Structure
======================================================================
...
✓ All critical directories verified!
```

### 1E. Verify GPU (Optional but Recommended)

```cmd
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

- If `True`: You can use GPU (faster training)
- If `False`: Training will use CPU (slower but works)

---

## 📊 Step 2: Prepare Your Data (5 minutes)

### 2A. Copy Your Data Files

Place your CSV files in the `Sets\` folder:

```cmd
REM If your data is on a USB drive (D:)
copy D:\your_data\train_set.csv Sets\
copy D:\your_data\validation_set.csv Sets\
copy D:\your_data\test_set.csv Sets\
```

### 2B. Verify Data Format

Your CSV files must have these columns:
- `id` - Unique identifier
- `clinical_note` - The clinical text
- `model_summary` - The summary
- `label` - "factual" or "hallucinated"

Quick check:
```cmd
python -c "import pandas as pd; df = pd.read_csv('Sets/train_set.csv'); print('Columns:', list(df.columns)); print('Rows:', len(df))"
```

---

## 🔄 Step 3: Phase 1 - Data Processing (15 minutes)

### 3A. Preprocess Data

```cmd
python preprocess_data.py ^
    --input-dir "Sets" ^
    --output-dir "phase1_data/sft" ^
    --generate-adversarial ^
    --adversarial-ratio 0.5
```

**What this does:**
- Normalizes text
- Generates adversarial examples (hard negatives)
- Creates training-ready CSV files

**Expected time:** 10-15 minutes depending on data size

**Expected output:**
```
Processing train set: Sets/train_set.csv
  Loaded X records
  Generated X augmented examples
  Saved X records to phase1_data/sft/train_set_processed.csv
```

### 3B. Verify Phase 1 Output

```cmd
dir phase1_data\sft
```

You should see:
- `train_set_processed.csv`
- `validation_set_processed.csv`
- `test_set_processed.csv` (if you have test data)

---

## 🎓 Step 4: Phase 1 - SFT Training (2-4 hours)

### 4A. Start SFT Training

**For GPU (Recommended):**
```cmd
python stage_a_sft_training.py ^
    --model_name "meta-llama/Llama-2-7b-hf" ^
    --train_data_path "phase1_data/sft/train_set_processed.csv" ^
    --val_data_path "phase1_data/sft/validation_set_processed.csv" ^
    --num_epochs 2 ^
    --batch_size 8 ^
    --learning_rate 2e-4 ^
    --output_dir "models/sft_specialist"
```

**For CPU (If no GPU):**
```cmd
python stage_a_sft_training.py ^
    --model_name "meta-llama/Llama-2-7b-hf" ^
    --train_data_path "phase1_data/sft/train_set_processed.csv" ^
    --val_data_path "phase1_data/sft/validation_set_processed.csv" ^
    --num_epochs 1 ^
    --batch_size 2 ^
    --learning_rate 2e-4 ^
    --output_dir "models/sft_specialist" ^
    --device cpu
```

**Expected time:**
- GPU (RTX 3090): 2-3 hours
- CPU: 10-15 hours (let it run overnight!)

### 4B. Monitor Training Progress

You'll see output like:
```
Epoch 1/2:
============================================================
Step 10: Loss = 2.45, LR = 2.00e-04
Step 20: Loss = 2.12, LR = 2.00e-04
...
Train Loss: 1.85
Val Loss: 1.92
```

**Good signs:**
- Loss decreasing over time
- No "CUDA out of memory" errors
- No crashes

**If you see "CUDA out of memory":**
```cmd
REM Stop training (Ctrl+C) and restart with smaller batch size
python stage_a_sft_training.py ^
    --batch_size 4 ^
    [... other args ...]
```

### 4C. Verify SFT Output

```cmd
dir models\sft_specialist\final_model
```

You should see:
- `adapter_config.json`
- `adapter_model.bin` (or `adapter_model.safetensors`)
- `tokenizer.json`
- `tokenizer_config.json`

**If files are present:** ✅ Phase 1 complete! Move to Phase 2.

---

## 🎯 Step 5: Phase 2 - Generate DPO Data (10 minutes)

### 5A. Generate DPO Triplets

```cmd
python generate_phase2_data.py ^
    --phase1_dir "phase1_data/sft" ^
    --phase2_dir "phase2_data/dpo" ^
    --adversarial_ratio 1.0
```

**What this does:**
- Reads Phase 1 processed data
- Creates triplets: (prompt, chosen, rejected)
- Saves JSONL files for DPO training

**Expected output:**
```
======================================================================
Phase 2 DPO Data Generation
======================================================================
✓ Found X factual examples
✓ Generating DPO triplets for X examples...
✓ Generated X valid DPO triplets
✓ Saved X triplets to phase2_data/dpo/train_dpo.jsonl
```

### 5B. Verify Phase 2 Data

```cmd
dir phase2_data\dpo
```

You should see:
- `train_dpo.jsonl`
- `val_dpo.jsonl`

Quick check:
```cmd
python -c "with open('phase2_data/dpo/train_dpo.jsonl') as f: print('Lines:', sum(1 for _ in f))"
```

---

## 🚀 Step 6: Phase 2 - DPO Training (2-4 hours)

### 6A. Start DPO Training

**For GPU (Recommended):**
```cmd
python stage_b_dpo_training.py ^
    --sft_model_path "models/sft_specialist/final_model" ^
    --train_data_path "phase2_data/dpo/train_dpo.jsonl" ^
    --val_data_path "phase2_data/dpo/val_dpo.jsonl" ^
    --num_epochs 2 ^
    --batch_size 4 ^
    --learning_rate 5e-6 ^
    --beta 0.1 ^
    --output_dir "models/dpo_hallucination_resistant"
```

**For CPU (If no GPU):**
```cmd
python stage_b_dpo_training.py ^
    --sft_model_path "models/sft_specialist/final_model" ^
    --train_data_path "phase2_data/dpo/train_dpo.jsonl" ^
    --val_data_path "phase2_data/dpo/val_dpo.jsonl" ^
    --num_epochs 1 ^
    --batch_size 1 ^
    --learning_rate 5e-6 ^
    --beta 0.1 ^
    --output_dir "models/dpo_hallucination_resistant" ^
    --device cpu
```

**⚠️ CRITICAL:** Learning rate is **100x lower** than SFT (5e-6 vs 2e-4)!

**Expected time:**
- GPU (RTX 3090): 2-4 hours
- CPU: 12-20 hours

### 6B. Monitor DPO Training

You'll see:
```
Epoch 1/2:
============================================================
Train Loss: 0.68
Val Loss: 0.65
Chosen Preference: 55%
```

**Good signs:**
- Loss decreasing
- Chosen Preference increasing (50% → 60% → 70% → 80%)
- No crashes

**Target:** Chosen Preference should reach 75-85% by the end.

### 6C. Verify DPO Output

```cmd
dir models\dpo_hallucination_resistant\final_model
```

You should see model files like Phase 1.

**If files are present:** ✅ Phase 2 complete! Training finished!

---

## 🎉 Step 7: Test Your Model (5 minutes)

### 7A. Run Inference

```cmd
python sft_inference.py ^
    --model_path "models/dpo_hallucination_resistant/final_model" ^
    --clinical_note "Patient reports fever of 38.5°C and cough for 3 days. Chest X-ray shows mild infiltrates. Started on antibiotics."
```

**Expected output:**
```
======================================================================
Clinical Note:
Patient reports fever of 38.5°C and cough for 3 days...

--------------------------------------------------------------
Generated Summary:
The patient presented with fever and cough. Chest imaging revealed mild infiltrates and antibiotic therapy was initiated.
======================================================================
```

### 7B. Compare Models

Test both Stage A (SFT) and Stage B (DPO) to see the improvement:

**Stage A (SFT only):**
```cmd
python sft_inference.py ^
    --model_path "models/sft_specialist/final_model" ^
    --clinical_note "Patient reports fever of 38.5°C..."
```

**Stage B (DPO - final model):**
```cmd
python sft_inference.py ^
    --model_path "models/dpo_hallucination_resistant/final_model" ^
    --clinical_note "Patient reports fever of 38.5°C..."
```

**Expected:** Stage B should have fewer hallucinations and be more grounded in the input.

---

## 💾 Step 8: Save Your Work

### 8A. Copy Models to USB Drive

```cmd
REM Copy to USB drive (assuming D:)
xcopy models D:\models\ /E /I /H /Y
```

### 8B. Copy Training Logs

```cmd
xcopy models\sft_specialist\training_stats.json D:\logs\ /Y
xcopy models\dpo_hallucination_resistant\dpo_training_stats.json D:\logs\ /Y
```

### 8C. Verify Backups

```cmd
dir D:\models\sft_specialist\final_model
dir D:\models\dpo_hallucination_resistant\final_model
```

---

## 🐛 Troubleshooting Guide

### Problem: "Python not recognized"

**Solution:**
```cmd
REM Add Python to PATH
set PATH=%PATH%;C:\Users\YourFriend\AppData\Local\Programs\Python\Python3X
```

Or reinstall Python with "Add to PATH" checked.

### Problem: "CUDA out of memory"

**Solution:**
```cmd
REM Reduce batch size
python stage_a_sft_training.py --batch_size 4
python stage_b_dpo_training.py --batch_size 2
```

### Problem: "Module not found"

**Solution:**
```cmd
pip install --upgrade pip
pip install pandas scikit-learn torch transformers peft tqdm openpyxl
```

### Problem: Training is very slow

**Check GPU usage:**
```cmd
nvidia-smi
```

- If GPU utilization is low: Increase batch size
- If GPU is not detected: Training is using CPU (will be slow)

### Problem: Model generates gibberish after DPO

**Solution:** Learning rate might be too high. Restart with lower LR:
```cmd
python stage_b_dpo_training.py --learning_rate 1e-6
```

---

## ⏱️ Expected Timeline

| Phase | Task | GPU Time | CPU Time |
|-------|------|----------|----------|
| 0 | Setup | 15 min | 15 min |
| 1A | Data Processing | 15 min | 15 min |
| 1B | SFT Training | 2-3 hrs | 10-15 hrs |
| 2A | DPO Data Generation | 10 min | 10 min |
| 2B | DPO Training | 2-4 hrs | 12-20 hrs |
| 3 | Testing | 5 min | 5 min |
| **TOTAL** | | **~5-7 hours** | **~23-30 hours** |

**Recommendation:** If using CPU, let it run overnight for both training phases.

---

## 📝 Checklist for Completion

### Phase 1 (SFT):
- [ ] Data copied to `Sets/` folder
- [ ] `preprocess_data.py` completed successfully
- [ ] Files exist in `phase1_data/sft/`
- [ ] `stage_a_sft_training.py` completed
- [ ] Model files in `models/sft_specialist/final_model/`
- [ ] Training loss decreased over epochs

### Phase 2 (DPO):
- [ ] `generate_phase2_data.py` completed
- [ ] Files exist in `phase2_data/dpo/`
- [ ] `stage_b_dpo_training.py` completed
- [ ] Chosen Preference reached 75%+
- [ ] Model files in `models/dpo_hallucination_resistant/final_model/`

### Verification:
- [ ] Inference works on test examples
- [ ] DPO model shows reduced hallucinations
- [ ] Models backed up to USB drive
- [ ] Training logs saved

---

## 🎯 Quick Commands Summary

```cmd
REM Setup (once)
python setup_directories.py
pip install pandas scikit-learn torch transformers peft tqdm openpyxl

REM Phase 1
python preprocess_data.py --generate-adversarial
python stage_a_sft_training.py --num_epochs 2 --batch_size 8

REM Phase 2
python generate_phase2_data.py
python stage_b_dpo_training.py --learning_rate 5e-6 --batch_size 4

REM Test
python sft_inference.py --model_path "models/dpo_hallucination_resistant/final_model"

REM Backup
xcopy models D:\models\ /E /I /H /Y
```

---

## 📞 Emergency Contacts

If something goes wrong:
1. Check error message carefully
2. Review troubleshooting section above
3. Check `Guides/STAGE_A_SFT_GUIDE.md` for SFT issues
4. Check `Guides/STAGE_B_DPO_GUIDE.md` for DPO issues
5. Save error logs: `python script.py > error.log 2>&1`

---

## ✅ Success Criteria

You're done when:
1. ✅ Both models trained without errors
2. ✅ DPO Chosen Preference ≥ 75%
3. ✅ Inference generates reasonable summaries
4. ✅ DPO model shows improvement over SFT
5. ✅ All models backed up

**🎉 Congratulations! You've successfully trained a hallucination-resistant medical text model!**

---

**Print this document and bring it with you!**
