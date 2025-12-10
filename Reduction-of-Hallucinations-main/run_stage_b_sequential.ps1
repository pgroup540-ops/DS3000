# Stage B Sequential DPO Training (Option 2)
# Fits in 12GB GPU by caching reference logits
# ============================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Stage B: Sequential DPO Training (Cached Reference Logits)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check GPU
Write-Host "Checking GPU..." -ForegroundColor Yellow
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
Write-Host ""

# Configuration
$SFT_MODEL = "models/sft_specialist_fast_fp16/final_model"
$TRAIN_DATA = "phase2_data/dpo/train_dpo.jsonl"
$VAL_DATA = "phase2_data/dpo/val_dpo.jsonl"
$CACHE_DIR = "cache/reference_logits"
$OUTPUT_DIR = "models/dpo_hallucination_resistant_sequential"

Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Model: $SFT_MODEL"
Write-Host "  Training Data: $TRAIN_DATA"
Write-Host "  Validation Data: $VAL_DATA"
Write-Host "  Cache Directory: $CACHE_DIR"
Write-Host "  Output Directory: $OUTPUT_DIR"
Write-Host ""

Write-Host "How it works:" -ForegroundColor Yellow
Write-Host "  Phase 1: Cache reference model logits - 1-2 hours"
Write-Host "           - Loads reference model only, needs 6GB"
Write-Host "           - Computes and saves logits to disk"
Write-Host "           - Unloads model from GPU"
Write-Host ""
Write-Host "  Phase 2: Train with cached logits - 4-6 hours"
Write-Host "           - Loads active model only, needs 6GB"
Write-Host "           - Reads reference logits from disk"
Write-Host "           - Trains with DPO loss"
Write-Host ""
Write-Host "  Total Time: 6-8 hours"
Write-Host "  Disk Space: 50-100GB for cache"
Write-Host "  GPU Memory: 6GB, ONE model at a time"
Write-Host ""

# Check disk space
$drive = (Get-Location).Drive
$freeSpace = (Get-PSDrive $drive.Name).Free / 1GB
Write-Host "Available disk space: $([math]::Round($freeSpace, 2)) GB" -ForegroundColor $(if ($freeSpace -gt 100) { "Green" } else { "Yellow" })
if ($freeSpace -lt 100) {
    Write-Host "  WARNING: You may need at least 100GB free for cache" -ForegroundColor Yellow
}
Write-Host ""

# Check files
Write-Host "Checking required files..." -ForegroundColor Yellow
if (-not (Test-Path $SFT_MODEL)) {
    Write-Host "ERROR: SFT model not found at $SFT_MODEL" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $TRAIN_DATA)) {
    Write-Host "ERROR: Training data not found at $TRAIN_DATA" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $VAL_DATA)) {
    Write-Host "ERROR: Validation data not found at $VAL_DATA" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ All files found" -ForegroundColor Green
Write-Host ""

# Confirm
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Ready to start sequential training!" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will:" -ForegroundColor Yellow
Write-Host "  1. Cache reference logits - 1-2 hours, needs 50-100GB disk" -ForegroundColor Yellow
Write-Host "  2. Train model with cached logits - 4-6 hours" -ForegroundColor Yellow
Write-Host "  3. Total time - 6-8 hours" -ForegroundColor Yellow
Write-Host ""
Write-Host "Memory: Only 6GB GPU at a time (fits in 12GB!)" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Enter to continue or Ctrl+C to cancel..." -ForegroundColor Yellow
Read-Host

# Run training
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

python stage_b_dpo_sequential.py `
    --sft_model_path $SFT_MODEL `
    --train_data_path $TRAIN_DATA `
    --val_data_path $VAL_DATA `
    --cache_dir $CACHE_DIR `
    --output_dir $OUTPUT_DIR `
    --num_epochs 2 `
    --batch_size 1 `
    --gradient_accumulation_steps 4 `
    --learning_rate 5e-6 `
    --max_length 512

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Model saved to: $OUTPUT_DIR/final_model" -ForegroundColor Green
    Write-Host "Cache saved to: $CACHE_DIR (can be deleted after training)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Evaluate the model"
    Write-Host "2. Compare Stage B vs Stage A hallucination rates"
    Write-Host "3. Delete cache directory to free up disk space"
} else {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host "Training failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check the error messages above for details." -ForegroundColor Yellow
}
