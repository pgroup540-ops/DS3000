# Training Progress Monitor
# Run this script to check the current status of Stage B training

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stage B Training Progress Monitor" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if training log exists
if (Test-Path "stage_b_training_log.txt") {
    Write-Host "Last 30 lines of training log:" -ForegroundColor Yellow
    Write-Host "----------------------------------------" -ForegroundColor Gray
    Get-Content "stage_b_training_log.txt" -Tail 30
    Write-Host "----------------------------------------" -ForegroundColor Gray
    Write-Host ""
    
    # Count completed steps
    $completedSteps = (Select-String -Path "stage_b_training_log.txt" -Pattern "Step.*completed!" | Measure-Object).Count
    Write-Host "Completed steps: $completedSteps" -ForegroundColor Green
    
    # Check for errors
    $errors = (Select-String -Path "stage_b_training_log.txt" -Pattern "error|Error|ERROR|Exception" | Measure-Object).Count
    if ($errors -gt 0) {
        Write-Host "Errors detected: $errors" -ForegroundColor Red
    } else {
        Write-Host "No errors detected" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Full log available at: stage_b_training_log.txt" -ForegroundColor Cyan
    
    # Check if model checkpoints exist
    if (Test-Path "models/dpo_hallucination_resistant") {
        Write-Host ""
        Write-Host "Model checkpoints:" -ForegroundColor Yellow
        Get-ChildItem "models/dpo_hallucination_resistant" -Directory | ForEach-Object {
            Write-Host "  - $($_.Name)" -ForegroundColor Green
        }
    }
} else {
    Write-Host "Training log not found. Training may not have started yet." -ForegroundColor Yellow
    Write-Host "Run: .\run_stage_b_training.ps1 to start training" -ForegroundColor Cyan
}

Write-Host ""
