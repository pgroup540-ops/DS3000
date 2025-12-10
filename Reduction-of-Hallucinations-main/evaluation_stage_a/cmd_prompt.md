# 1. EVALUATION
python evaluate_stage_a.py \
    --model_path ./models/sft_specialist/final_model \
    --test_data phase1_data/sft/validation_set_processed.csv \
    --output_dir ./evaluation_results

cat evaluation_results/evaluation_report.txt
open evaluation_results/evaluation_results.csv

# Manual: Review and calculate hallucination rate
# If <15%, proceed. If >15%, consider Stage B.

# 2. PREPARE MODEL
python prepare_for_deployment.py \
    --model_path ./models/sft_specialist/final_model \
    --output_dir ./models/sft_specialist_merged

python sft_inference.py \
    --model_path ./models/sft_specialist_merged \
    --clinical_note "Patient reports fever of 38.5°C and cough. Tested positive for influenza A."

# 3. INSTALL API DEPENDENCIES
pip install fastapi uvicorn

# 4. START API SERVER
python deploy_api.py \
    --model_path ./models/sft_specialist_merged \
    --host 0.0.0.0 \
    --port 8000

# Keep server running, open NEW terminal for testing:

# 5. TEST API (in new terminal)
curl http://localhost:8000/health

curl -X POST http://localhost:8000/summarize \
    -H "Content-Type: application/json" \
    -d '{"clinical_note": "Patient reports fever of 38.5°C and cough. Tested positive for influenza A."}'

open http://localhost:8000/docs