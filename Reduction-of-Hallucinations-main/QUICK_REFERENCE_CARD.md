# QUICK REFERENCE CARD - Print This!

**📌 Pin this page near your friend's computer during training**

---

## ⚡ Essential Commands (Copy-Paste Ready)

### Setup (Once)
```cmd
python setup_directories.py
pip install pandas scikit-learn torch transformers peft tqdm openpyxl
```

### Phase 1: SFT Training
```cmd
REM 1. Process data
python preprocess_data.py --generate-adversarial

REM 2. Train SFT (GPU - 2-3 hours)
python stage_a_sft_training.py --num_epochs 2 --batch_size 8

REM 2. Train SFT (CPU - 10-15 hours)
python stage_a_sft_training.py --num_epochs 1 --batch_size 2 --device cpu
```

### Phase 2: DPO Training
```cmd
REM 1. Generate DPO data
python generate_phase2_data.py

REM 2. Train DPO (GPU - 2-4 hours)
python stage_b_dpo_training.py --learning_rate 5e-6 --batch_size 4

REM 2. Train DPO (CPU - 12-20 hours)
python stage_b_dpo_training.py --learning_rate 5e-6 --batch_size 1 --device cpu
```

### Test Model
```cmd
python sft_inference.py --model_path "models/dpo_hallucination_resistant/final_model"
```

### Backup Models
```cmd
xcopy models D:\models\ /E /I /H /Y
```

---

## 🚨 Emergency Fixes

### "CUDA out of memory"
```cmd
REM Reduce batch size
python stage_a_sft_training.py --batch_size 4
python stage_b_dpo_training.py --batch_size 2
```

### "Python not recognized"
```cmd
set PATH=%PATH%;C:\Users\YourFriend\AppData\Local\Programs\Python\Python3X
```

### "Module not found"
```cmd
pip install --upgrade pip
pip install pandas scikit-learn torch transformers peft tqdm openpyxl
```

### Training stuck/frozen
```cmd
REM Press Ctrl+C to stop
REM Check GPU: nvidia-smi
REM Reduce batch size and retry
```

---

## 📊 What to Watch For

### During SFT Training
✅ **Good:** Loss decreasing (2.5 → 2.0 → 1.8 → 1.5)
❌ **Bad:** Loss increasing or stuck
❌ **Bad:** "CUDA out of memory" errors

### During DPO Training
✅ **Good:** Chosen Preference increasing (50% → 60% → 75%)
✅ **Good:** Loss decreasing (0.7 → 0.5 → 0.4)
❌ **Bad:** Chosen Preference stuck at ~50%
❌ **Bad:** Model outputs gibberish

---

## ⏱️ Time Estimates

| Task | GPU | CPU |
|------|-----|-----|
| Setup | 15 min | 15 min |
| Data Processing | 15 min | 15 min |
| SFT Training | 2-3 hrs | 10-15 hrs |
| DPO Data Gen | 10 min | 10 min |
| DPO Training | 2-4 hrs | 12-20 hrs |
| **TOTAL** | **~6 hrs** | **~24 hrs** |

**💡 Tip:** Run overnight if using CPU

---

## 📁 Critical Files to Check

### After Phase 1:
```cmd
dir phase1_data\sft
dir models\sft_specialist\final_model
```
Should see: CSV files and model files

### After Phase 2:
```cmd
dir phase2_data\dpo
dir models\dpo_hallucination_resistant\final_model
```
Should see: JSONL files and model files

---

## 🎯 Success Checklist

Phase 1:
- [ ] phase1_data\sft\ has CSV files
- [ ] models\sft_specialist\final_model\ exists
- [ ] Training loss decreased

Phase 2:
- [ ] phase2_data\dpo\ has JSONL files
- [ ] models\dpo_hallucination_resistant\final_model\ exists
- [ ] Chosen Preference ≥ 75%

Final:
- [ ] Inference works
- [ ] Models backed up to USB
- [ ] Training completed without crashes

---

## 🔧 Key Parameters to Remember

### SFT (Stage A)
- Learning Rate: **2e-4**
- Batch Size: **8** (GPU) or **2** (CPU)
- Epochs: **2**

### DPO (Stage B)
- Learning Rate: **5e-6** ⚠️ (100x lower!)
- Batch Size: **4** (GPU) or **1** (CPU)
- Beta: **0.1**
- Epochs: **2**

---

## 📞 If Something Breaks

1. **Read error message carefully**
2. Check "Emergency Fixes" section above
3. Check GPU: `nvidia-smi`
4. Try reducing batch size
5. Save error log: `python script.py > error.log 2>&1`
6. Look in `EXECUTION_GUIDE_WINDOWS.md` for details

---

## 💡 Pro Tips

✅ **DO:**
- Let training finish completely
- Monitor GPU temperature (< 85°C)
- Save models after each phase
- Test both SFT and DPO models

❌ **DON'T:**
- Interrupt training mid-epoch
- Use same batch size on different GPUs
- Skip Phase 1 before Phase 2
- Forget to backup models

---

## 🎉 You're Done When...

1. Both models trained successfully
2. DPO Chosen Preference ≥ 75%
3. Inference generates good summaries
4. Models saved to USB drive

**Total Training Time (GPU): ~6 hours**
**Total Training Time (CPU): ~24 hours**

---

**Keep this page visible during training!**
**Full guide: EXECUTION_GUIDE_WINDOWS.md**
