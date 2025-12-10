# Stage B DPO Training with 4-bit QLoRA
# Optimized for 12GB GPU (RTX 5070)
# ===========================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Stage B: 4-bit QLoRA DPO Training" -ForegroundColor Cyan
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
$OUTPUT_DIR = "models/dpo_hallucination_resistant_4bit"

# Training parameters
$NUM_EPOCHS = 2
$BATCH_SIZE = 1
$GRAD_ACCUM = 4  # Effective batch size = 1 * 4 = 4
$LEARNING_RATE = "5e-6"
$MAX_LENGTH = 512

Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  SFT Model: $SFT_MODEL"
Write-Host "  Training Data: $TRAIN_DATA"
Write-Host "  Validation Data: $VAL_DATA"
Write-Host "  Output Directory: $OUTPUT_DIR"
Write-Host "  Epochs: $NUM_EPOCHS"
Write-Host "  Batch Size: $BATCH_SIZE (effective: $($BATCH_SIZE * $GRAD_ACCUM))"
Write-Host "  Learning Rate: $LEARNING_RATE"
Write-Host "  Max Length: $MAX_LENGTH"
Write-Host ""

# Check files exist
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
Write-Host "Ready to start training!" -ForegroundColor Cyan
Write-Host "Expected duration: 4-6 hours" -ForegroundColor Cyan
Write-Host "Expected memory usage: 9-10GB VRAM" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Enter to continue or Ctrl+C to cancel..." -ForegroundColor Yellow
Read-Host

# Run training
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

python stage_b_dpo_training_4bit.py `
    --sft_model_path $SFT_MODEL `
    --train_data_path $TRAIN_DATA `
    --val_data_path $VAL_DATA `
    --output_dir $OUTPUT_DIR `
    --num_epochs $NUM_EPOCHS `
    --batch_size $BATCH_SIZE `
    --gradient_accumulation_steps $GRAD_ACCUM `
    --learning_rate $LEARNING_RATE `
    --max_length $MAX_LENGTH `
    --save_steps 100 `
    --eval_steps 50 `
    --lora_r 16 `
    --lora_alpha 32

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Model saved to: $OUTPUT_DIR/final_model" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Evaluate the model with evaluation script"
    Write-Host "2. Compare Stage B vs Stage A hallucination rates"
    Write-Host "3. Run manual assessment"
} else {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host "Training failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check the error messages above for details." -ForegroundColor Yellow
}
