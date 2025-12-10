# Stage B DPO Training Runner
# This script runs the training with proper logging and can be left running in the background

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stage B DPO Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting training on CPU..." -ForegroundColor Yellow
Write-Host "Expected time: 20-24 hours for 2 epochs" -ForegroundColor Yellow
Write-Host "Output will be logged to: stage_b_training_log.txt" -ForegroundColor Yellow
Write-Host ""

# Set Python to unbuffered mode
$env:PYTHONUNBUFFERED = "1"

# Run training with output to both console and log file
python -u stage_b_dpo_training.py `
    --sft_model_path models/sft_specialist_fast_fp16/final_model `
    --batch_size 1 `
    --device cpu `
    --num_epochs 2 `
    2>&1 | Tee-Object -FilePath "stage_b_training_log.txt"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Model saved to: models/dpo_hallucination_resistant/final_model" -ForegroundColor Green
    Write-Host "Training log: stage_b_training_log.txt" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Training failed with error code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Check stage_b_training_log.txt for details" -ForegroundColor Yellow
}
