# Windows Setup Guide

Complete guide for running the Hallucination Reduction project on Windows.

---

## ⚠️ Known Windows Issues

### 1. **bitsandbytes** (8-bit quantization)
- **Issue**: `bitsandbytes` doesn't work reliably on Windows
- **Solution**: Skip 8-bit quantization
- **Impact**: Slightly higher memory usage

### 2. **Path Separators**
- **Issue**: Windows uses backslash (`\`) instead of forward slash (`/`)
- **Solution**: Python handles this automatically with `Path()` or `/`
- **Impact**: None (our code is cross-platform)

### 3. **CUDA Toolkit**
- **Issue**: Windows requires manual CUDA installation
- **Solution**: Install from NVIDIA website
- **Impact**: Required for GPU training

---

## 🚀 Installation Steps

### 1. Check Prerequisites

#### Python Version
```cmd
python --version
```
**Required:** Python 3.8 or higher

#### Install Python (if needed)
Download from: https://www.python.org/downloads/

**Important:** Check "Add Python to PATH" during installation

#### Check GPU (Optional)
```cmd
nvidia-smi
```
If this works, you have an NVIDIA GPU. If not, you'll use CPU training.

---

### 2. Install CUDA Toolkit (For GPU)

**Only needed if you have an NVIDIA GPU**

1. Download CUDA Toolkit 11.8 or 12.1:
   - https://developer.nvidia.com/cuda-downloads
   
2. Install with default options

3. Verify:
   ```cmd
   nvcc --version
   ```

---

### 3. Clone Repository

#### Using Git
```cmd
git clone https://github.com/yourusername/Reduction-of-Hallucinations.git
cd Reduction-of-Hallucinations
```

#### Or Download ZIP
1. Download from GitHub
2. Extract to folder
3. Open Command Prompt in that folder

---

### 4. Create Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

---

### 5. Install PyTorch

**For GPU (CUDA 11.8):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For GPU (CUDA 12.1):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU Only:**
```cmd
pip install torch torchvision torchaudio
```

**Verify PyTorch:**
```cmd
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

### 6. Install Other Dependencies

Create `requirements_training_windows.txt`:

```text
# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Transformers and NLP
transformers>=4.35.0
tokenizers>=0.13.0

# Parameter Efficient Fine-Tuning (LoRA)
peft>=0.6.0

# Training utilities
tqdm>=4.65.0
accelerate>=0.24.0

# NOTE: bitsandbytes skipped for Windows compatibility
```

Install:
```cmd
pip install -r requirements_training_windows.txt
```

**Or install from original (will skip bitsandbytes automatically):**
```cmd
pip install -r requirements_training.txt
```
If bitsandbytes fails, that's OK! Just skip it:
```cmd
pip install pandas numpy transformers tokenizers peft tqdm accelerate
```

---

### 7. Verify Installation

```cmd
python -c "import torch; import transformers; import peft; print('✅ All dependencies installed!')"
```

---

## 🎯 Training on Windows

### Stage A: SFT Training

#### GPU Training
```cmd
python stage_a_sft_training.py ^
    --model_name "meta-llama/Llama-2-7b-hf" ^
    --num_epochs 2 ^
    --learning_rate 2e-4 ^
    --batch_size 8
```

**Note:** Windows uses `^` for line continuation (not `\`)

#### CPU Training (No GPU)
```cmd
python stage_a_sft_training.py ^
    --device cpu ^
    --batch_size 2 ^
    --num_epochs 1
```

#### Memory-Constrained GPU
```cmd
python stage_a_sft_training.py ^
    --batch_size 4 ^
    --lora_r 8
```

---

### Stage B: DPO Training

#### GPU Training
```cmd
python stage_b_dpo_training.py ^
    --sft_model_path "./models/sft_specialist/final_model" ^
    --num_epochs 2 ^
    --learning_rate 5e-6 ^
    --beta 0.1 ^
    --batch_size 4
```

#### CPU Training
```cmd
python stage_b_dpo_training.py ^
    --sft_model_path "./models/sft_specialist/final_model" ^
    --device cpu ^
    --batch_size 1 ^
    --num_epochs 1
```

---

## 🔧 Windows-Specific Issues & Solutions

### Issue 1: "bitsandbytes not found"

**Error:**
```
ModuleNotFoundError: No module named 'bitsandbytes'
```

**Solution:**
```cmd
# Option 1: Skip 8-bit quantization in training
python stage_a_sft_training.py --use_8bit False

# Option 2: Try Windows-compatible version (experimental)
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

---

### Issue 2: "CUDA out of memory"

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:**
```cmd
# Reduce batch size
python stage_a_sft_training.py --batch_size 2

# Or reduce LoRA rank
python stage_a_sft_training.py --batch_size 4 --lora_r 8

# Or use CPU
python stage_a_sft_training.py --device cpu --batch_size 2
```

---

### Issue 3: Long Path Names

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution 1:** Enable long paths in Windows

1. Press Win+R
2. Type `gpedit.msc`
3. Navigate to: Computer Configuration → Administrative Templates → System → Filesystem
4. Enable "Enable Win32 long paths"

**Solution 2:** Use shorter folder names

Instead of:
```
C:\Users\YourName\Documents\My Projects\Reduction of Hallucinations
```

Use:
```
C:\Projects\hallucination-reduction
```

---

### Issue 4: Permission Errors

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**

1. Run Command Prompt as Administrator
2. Or move project to a folder you own (not Program Files)

---

### Issue 5: Python Not Found

**Error:**
```
'python' is not recognized as an internal or external command
```

**Solution:**

Try `py` instead of `python`:
```cmd
py -m pip install -r requirements_training.txt
py stage_a_sft_training.py --num_epochs 2
```

Or add Python to PATH:
1. Search "Environment Variables"
2. Add Python installation folder to PATH
3. Restart Command Prompt

---

## 💡 Windows Tips

### Use PowerShell (Alternative to CMD)

PowerShell has better features than Command Prompt:

```powershell
# Activate venv in PowerShell
venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Use Windows Terminal

Download from Microsoft Store for better experience:
- Multiple tabs
- Better colors
- GPU acceleration

### Monitor GPU Usage

```cmd
# In a separate terminal
nvidia-smi -l 1
```

This refreshes GPU stats every second.

---

## 📊 Expected Performance on Windows

### With NVIDIA GPU (RTX 3090, RTX 4090, etc.)

| Stage | Time | Notes |
|-------|------|-------|
| Stage A (SFT) | 1.5-3 hours | Similar to Linux |
| Stage B (DPO) | 3-4 hours | May be 10-20% slower |

### CPU Only

| Stage | Time | Notes |
|-------|------|-------|
| Stage A (SFT) | 8-16 hours | Very slow |
| Stage B (DPO) | 16-24 hours | Consider cloud GPU |

---

## ☁️ Cloud GPU Alternatives

If your Windows PC doesn't have a GPU, consider:

### Google Colab (Free)
- Free tier with GPU
- 12 hours per session
- Upload code and data
- Run training there

### Paperspace Gradient
- Pay-per-use GPU
- Windows-friendly interface
- ~$0.50/hour for RTX 4000

### Lambda Labs
- Powerful GPUs
- ~$1.10/hour for A100
- Best for serious training

---

## 🔍 Debugging Tips

### Check if GPU is Being Used

```python
# test_gpu.py
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

Run:
```cmd
python test_gpu.py
```

### Monitor During Training

**GPU Memory:**
```cmd
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

**CPU Usage:**
- Open Task Manager (Ctrl+Shift+Esc)
- Check Performance tab

---

## 📁 File Paths in Windows

Our code handles Windows paths automatically:

```python
# These all work on Windows:
"phase1_data/sft/train_set_processed.csv"  # Forward slash (works!)
"phase1_data\\sft\\train_set_processed.csv"  # Backslash
Path("phase1_data") / "sft" / "train_set_processed.csv"  # Path object
```

You don't need to change anything in the code!

---

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] CUDA Toolkit installed (if using GPU)
- [ ] Git installed or repository downloaded
- [ ] Virtual environment created
- [ ] PyTorch installed and GPU detected
- [ ] All dependencies installed
- [ ] Verification test passed
- [ ] Training data exists in `phase1_data/`

---

## 🆘 Getting Help

1. **Check error message carefully** - Windows errors are often about paths or permissions
2. **Try CPU training** - If GPU issues persist, CPU works (just slower)
3. **Use Google Colab** - Free GPU alternative
4. **Check paths** - Make sure data files exist

---

## 🚀 Quick Start (Windows)

Copy-paste this entire block:

```cmd
REM Clone repository
git clone https://github.com/yourusername/Reduction-of-Hallucinations.git
cd Reduction-of-Hallucinations

REM Create virtual environment
python -m venv venv
venv\Scripts\activate

REM Install PyTorch (CPU version)
pip install torch torchvision torchaudio

REM Install other dependencies
pip install pandas numpy transformers tokenizers peft tqdm accelerate

REM Verify
python -c "import torch; import transformers; import peft; print('Ready!')"

REM Start training
python stage_a_sft_training.py --device cpu --batch_size 2 --num_epochs 1
```

For GPU version, install PyTorch with CUDA support instead.

---

## 📝 Summary

| Feature | Windows Status | Notes |
|---------|----------------|-------|
| Python code | ✅ Works | Fully compatible |
| File paths | ✅ Works | Auto-handled |
| GPU training | ✅ Works | Requires CUDA |
| CPU training | ✅ Works | Slower but reliable |
| bitsandbytes | ⚠️ Skip | Not needed |
| All features | ✅ 95% | Only 8-bit quantization unsupported |

**Bottom line:** The project works great on Windows with or without GPU!
