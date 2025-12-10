# Windows Quick Start Guide

## ✅ Windows Compatibility

All files in this repository are **fully compatible with Windows**. This guide will help you set up and run the hallucination reduction pipeline on Windows.

---

## 📋 Prerequisites

### 1. Python Installation

Install Python 3.8 or higher:
- Download from [python.org](https://www.python.org/downloads/)
- ✅ Check "Add Python to PATH" during installation
- Verify installation:
```cmd
python --version
```

### 2. Git Installation (Optional but Recommended)

Install Git for Windows:
- Download from [git-scm.com](https://git-scm.com/download/win)
- Or download this repository as ZIP

### 3. GPU Support (Optional but Recommended)

For NVIDIA GPU support:
- Install [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-downloads)
- Install [cuDNN](https://developer.nvidia.com/cudnn)

---

## 🚀 Quick Setup (5 Minutes)

### Step 1: Clone or Download Repository

**Option A: Using Git**
```cmd
git clone https://github.com/YOUR_USERNAME/Reduction-of-Hallucinations.git
cd Reduction-of-Hallucinations
```

**Option B: Download ZIP**
- Download ZIP from GitHub
- Extract to your desired location
- Open Command Prompt in that folder

### Step 2: Install Dependencies

```cmd
pip install pandas scikit-learn torch transformers peft tqdm openpyxl
```

Or install from requirements file (if available):
```cmd
pip install -r requirements.txt
```

### Step 3: Setup Directories

```cmd
python setup_directories.py
```

This creates all necessary directories:
- `Sets/` - For your raw data
- `phase1_data/sft/` - Phase 1 processed data
- `phase2_data/dpo/` - Phase 2 DPO data
- `models/` - Trained models
- And more...

---

## 📊 Complete Workflow

### Phase 1: Data Preparation & SFT Training

#### 1A. Prepare Your Data

Place your raw data CSV in the `Sets/` directory with these columns:
- `id` - Unique identifier
- `clinical_note` - The clinical text
- `model_summary` - The summary
- `label` - "factual" or "hallucinated"

Example: `Sets/train_set.csv`

#### 1B. Process Data

```cmd
python preprocess_data.py ^
    --input-dir "Sets" ^
    --output-dir "phase1_data/sft" ^
    --generate-adversarial
```

#### 1C. Train Stage A (SFT)

```cmd
python stage_a_sft_training.py ^
    --model_name "meta-llama/Llama-2-7b-hf" ^
    --train_data_path "phase1_data/sft/train_set_processed.csv" ^
    --val_data_path "phase1_data/sft/validation_set_processed.csv" ^
    --num_epochs 2 ^
    --output_dir "models/sft_specialist"
```

**Note:** Use `^` for line continuation in Windows CMD, or `\`` in PowerShell.

---

### Phase 2: DPO Training

#### 2A. Generate DPO Data

```cmd
python generate_phase2_data.py ^
    --phase1_dir "phase1_data/sft" ^
    --phase2_dir "phase2_data/dpo" ^
    --adversarial_ratio 1.0
```

This bridges Phase 1 → Phase 2 by creating triplet data.

#### 2B. Train Stage B (DPO)

```cmd
python stage_b_dpo_training.py ^
    --sft_model_path "models/sft_specialist/final_model" ^
    --train_data_path "phase2_data/dpo/train_dpo.jsonl" ^
    --val_data_path "phase2_data/dpo/val_dpo.jsonl" ^
    --num_epochs 2 ^
    --learning_rate 5e-6 ^
    --output_dir "models/dpo_hallucination_resistant"
```

---

## 🔧 Windows-Specific Notes

### Path Separators

✅ **All scripts use cross-platform paths** - they work the same on Windows, Mac, and Linux.

You can use either:
```cmd
python script.py --path "phase1_data/sft"        # Forward slashes (RECOMMENDED)
python script.py --path "phase1_data\sft"        # Backslashes (also works)
```

### Command Line Continuation

**Command Prompt (CMD):**
```cmd
python script.py ^
    --arg1 value1 ^
    --arg2 value2
```

**PowerShell:**
```powershell
python script.py `
    --arg1 value1 `
    --arg2 value2
```

### Virtual Environment (Recommended)

Create isolated Python environment:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

To deactivate:
```cmd
deactivate
```

---

## 💾 Hardware Requirements

### Minimum (CPU Only)
- **RAM:** 16GB
- **Storage:** 50GB free
- **Training Time:** 10-20 hours per stage

### Recommended (GPU)
- **GPU:** NVIDIA RTX 3090/4090 (24GB VRAM)
- **RAM:** 32GB
- **Storage:** 100GB free
- **Training Time:** 2-4 hours per stage

### Cloud Alternative
If you don't have a GPU, consider:
- Google Colab (Free tier: Tesla T4)
- AWS EC2 (p3.2xlarge with V100)
- Azure ML (NC-series VMs)

---

## 🐛 Common Windows Issues

### Issue 1: "Python not recognized"

**Solution:**
Add Python to PATH:
1. Search "Environment Variables" in Windows
2. Edit PATH variable
3. Add: `C:\Users\YourName\AppData\Local\Programs\Python\Python3X`
4. Restart Command Prompt

### Issue 2: "CUDA out of memory"

**Solution:**
Reduce batch size:
```cmd
python stage_a_sft_training.py --batch_size 4
python stage_b_dpo_training.py --batch_size 2
```

### Issue 3: "Module not found"

**Solution:**
Reinstall dependencies:
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue 4: Long paths (>260 characters)

**Solution:**
Enable long paths in Windows:
```cmd
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

Or use shorter directory names.

---

## 📁 Expected Directory Structure

After setup, your directory should look like:

```
Reduction-of-Hallucinations/
├── Sets/                           # Your raw data
│   ├── train_set.csv
│   ├── validation_set.csv
│   └── test_set.csv
│
├── phase1_data/                    # Phase 1 outputs
│   └── sft/
│       ├── train_set_processed.csv
│       └── validation_set_processed.csv
│
├── phase2_data/                    # Phase 2 outputs
│   └── dpo/
│       ├── train_dpo.jsonl
│       └── val_dpo.jsonl
│
├── models/                         # Trained models
│   ├── sft_specialist/
│   │   └── final_model/
│   └── dpo_hallucination_resistant/
│       └── final_model/
│
├── Python scripts (.py files)
├── setup_directories.py            # Run this first!
├── generate_phase2_data.py         # Bridge Phase 1→2
└── README.md
```

---

## ✅ Verification Checklist

Before training:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip list`)
- [ ] Directories created (`python setup_directories.py`)
- [ ] Raw data in `Sets/` directory
- [ ] GPU available (optional): `nvidia-smi`

After Phase 1:
- [ ] Files exist in `phase1_data/sft/`
- [ ] SFT model in `models/sft_specialist/final_model/`

After Phase 2:
- [ ] Files exist in `phase2_data/dpo/`
- [ ] DPO model in `models/dpo_hallucination_resistant/final_model/`

---

## 🎯 Quick Test

Test if everything is set up correctly:

```cmd
REM Test imports
python -c "import torch; import transformers; import peft; print('All imports OK')"

REM Setup directories
python setup_directories.py

REM Check GPU (if available)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 📚 Additional Resources

- **Full Documentation:** See `Guides/` folder
- **Stage A Guide:** `Guides/STAGE_A_SFT_GUIDE.md`
- **Stage B Guide:** `Guides/STAGE_B_DPO_GUIDE.md`
- **Troubleshooting:** `Guides/WINDOWS_SETUP.md`

---

## 🆘 Getting Help

If you encounter issues:

1. Check this guide first
2. Review error messages carefully
3. Ensure all prerequisites are installed
4. Check GPU memory: `nvidia-smi`
5. Try with smaller batch sizes
6. Open an issue on GitHub with:
   - Windows version
   - Python version
   - Full error message
   - Command you ran

---

## 🎉 Success!

Once you've completed both phases, you'll have:
- ✅ Hallucination-resistant medical text model
- ✅ Trained on your domain-specific data
- ✅ Ready for inference and deployment

Test your model:
```cmd
python sft_inference.py ^
    --model_path "models/dpo_hallucination_resistant/final_model" ^
    --clinical_note "Your clinical note here..."
```

---

## License

[Your License Here]

## Acknowledgments

- DPO Paper: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- LoRA: [Low-Rank Adaptation](https://arxiv.org/abs/2106.09714)
