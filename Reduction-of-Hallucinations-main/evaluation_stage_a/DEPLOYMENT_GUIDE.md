# Complete Deployment Guide: Stage A to Production
## From Evaluation to Deployment - Step by Step

This guide takes you from a trained Stage A model to production deployment.

---

## Prerequisites

You have completed Stage A training and have:
- ✓ Trained model at `./models/sft_specialist/final_model`
- ✓ Training completed successfully
- ✓ Validation data available

---

## STEP 1: Evaluate Model Quality

### 1.1 Run Evaluation Script

```bash
python evaluate_stage_a.py \
    --model_path ./models/sft_specialist/final_model \
    --test_data phase1_data/sft/validation_set_processed.csv \
    --output_dir ./evaluation_results
```

**Time**: 10-30 minutes depending on dataset size

**Output**:
```
evaluation_results/
├── evaluation_results.csv       # All generated summaries
├── evaluation_stats.json        # Statistics
└── evaluation_report.txt        # Human-readable report
```

### 1.2 Review Results

```bash
# View the report
cat evaluation_results/evaluation_report.txt

# Or open CSV for detailed review
open evaluation_results/evaluation_results.csv
```

### 1.3 Calculate Hallucination Rate

**Manual review** (recommended for first 20-50 examples):

```python
# Open evaluation_results.csv
# For each row, check if generated_summary contains:
#   - Information NOT in clinical_note (hallucination)
#   - Wrong dosages, measurements, or conditions
#   - Fabricated details

hallucinations = 0
total_reviewed = 50

for each example:
    if contains_fabricated_info:
        hallucinations += 1

hallucination_rate = hallucinations / total_reviewed
print(f"Hallucination Rate: {hallucination_rate:.1%}")
```

### 1.4 Decision Point

**If hallucination rate < 15%**: ✅ Proceed to Step 2 (Deployment)  
**If hallucination rate > 15%**: ⚠️ Consider running Stage B (DPO)

---

## STEP 2: Prepare Model for Deployment

### 2.1 Merge LoRA Weights

Merge LoRA adapters into base model for faster inference:

```bash
python prepare_for_deployment.py \
    --model_path ./models/sft_specialist/final_model \
    --output_dir ./models/sft_specialist_merged
```

**What this does**:
- Merges LoRA adapter weights into base model
- Creates single model file (no adapters needed)
- Tests merged model with sample inference
- Saves deployment-ready model

**Time**: 5-10 minutes

**Output**:
```
models/sft_specialist_merged/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

### 2.2 Test Merged Model

Quick test to verify model works:

```bash
python sft_inference.py \
    --model_path ./models/sft_specialist_merged \
    --clinical_note "Patient reports fever of 38.5°C and cough. Tested positive for influenza A."
```

**Expected output**: Coherent medical summary

---

## STEP 3: Choose Deployment Method

### Option A: Local Deployment (Simple)

For testing or single-machine deployment.

```bash
# Just use the inference script directly
python sft_inference.py \
    --model_path ./models/sft_specialist_merged \
    --clinical_note "Your clinical note here"
```

### Option B: API Deployment (Recommended)

Deploy as REST API for multiple clients.

---

## STEP 4: Deploy as REST API

### 4.1 Install Dependencies

```bash
pip install fastapi uvicorn
```

### 4.2 Start API Server

```bash
python deploy_api.py \
    --model_path ./models/sft_specialist_merged \
    --host 0.0.0.0 \
    --port 8000
```

**Output**:
```
INFO:     Loading model from ./models/sft_specialist_merged
INFO:     Model loaded successfully!
INFO:     Starting API server on 0.0.0.0:8000
INFO:     API Documentation: http://0.0.0.0:8000/docs
```

### 4.3 Test API Endpoints

**Health Check**:
```bash
curl http://localhost:8000/health
```

**Generate Summary**:
```bash
curl -X POST http://localhost:8000/summarize \
    -H "Content-Type: application/json" \
    -d '{
        "clinical_note": "Patient reports fever of 38.5°C and cough. Tested positive for influenza A.",
        "max_tokens": 150,
        "temperature": 0.7
    }'
```

**Expected Response**:
```json
{
  "clinical_note": "Patient reports fever of 38.5°C and cough...",
  "summary": "The patient tested positive for influenza A and experienced fever.",
  "model_version": "stage_a_v1"
}
```

### 4.4 Interactive API Documentation

Open browser: `http://localhost:8000/docs`

This provides:
- Interactive API testing interface
- Full API documentation
- Request/response examples

---

## STEP 5: Production Deployment

### 5.1 Add Production Features

**Create production configuration**:

```python
# config.py
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_PATH = "./models/sft_specialist_merged"
LOG_LEVEL = "INFO"
MAX_WORKERS = 4
```

**Add authentication** (example):

```python
# In deploy_api.py, add:
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/summarize")
async def summarize(request: SummarizeRequest, credentials=Depends(security)):
    # Verify credentials
    # ... existing code
```

**Add rate limiting**:

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/summarize")
@limiter.limit("100/minute")
async def summarize(request: Request, ...):
    # ... existing code
```

### 5.2 Deploy to Server

**Using systemd** (Linux):

```bash
# Create service file: /etc/systemd/system/medical-summary-api.service
[Unit]
Description=Medical Summary API
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/project
ExecStart=/path/to/venv/bin/python deploy_api.py \
    --model_path /path/to/models/sft_specialist_merged \
    --host 0.0.0.0 \
    --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Start service
sudo systemctl start medical-summary-api
sudo systemctl enable medical-summary-api

# Check status
sudo systemctl status medical-summary-api
```

**Using Docker**:

```dockerfile
# Dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements_training.txt .
RUN pip install -r requirements_training.txt
RUN pip install fastapi uvicorn

COPY . .
COPY models/sft_specialist_merged /app/models/sft_specialist_merged

EXPOSE 8000

CMD ["python", "deploy_api.py", "--model_path", "/app/models/sft_specialist_merged"]
```

```bash
# Build and run
docker build -t medical-summary-api .
docker run -p 8000:8000 medical-summary-api
```

### 5.3 Deploy to Cloud

**AWS EC2**:
1. Launch GPU instance (g4dn.xlarge or similar)
2. Install dependencies
3. Copy model files
4. Run API server
5. Configure security group (port 8000)

**Google Cloud**:
1. Create Compute Engine instance with GPU
2. Install dependencies
3. Deploy model
4. Use Cloud Load Balancer for traffic

**Azure**:
1. Create VM with GPU
2. Install dependencies
3. Deploy model
4. Use Application Gateway

---

## STEP 6: Monitor Production

### 6.1 Add Logging

```python
import logging

# In deploy_api.py
@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    logger.info(f"Received request: {request.clinical_note[:50]}...")
    result = model.generate_summary(...)
    logger.info(f"Generated summary: {result['generated_summary'][:50]}...")
    return result
```

### 6.2 Track Metrics

```python
from datetime import datetime

metrics = {
    'requests': 0,
    'errors': 0,
    'avg_latency': 0
}

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    start_time = datetime.now()
    
    try:
        result = model.generate_summary(...)
        metrics['requests'] += 1
        
        latency = (datetime.now() - start_time).total_seconds()
        metrics['avg_latency'] = (metrics['avg_latency'] * (metrics['requests'] - 1) + latency) / metrics['requests']
        
        return result
    except Exception as e:
        metrics['errors'] += 1
        raise
```

### 6.3 Periodic Quality Checks

```bash
# Weekly quality check
python evaluate_stage_a.py \
    --model_path ./models/sft_specialist_merged \
    --test_data new_test_data.csv \
    --output_dir ./weekly_eval_$(date +%Y%m%d)
```

---

## STEP 7: Continuous Improvement

### 7.1 Collect Feedback

- Log all inputs/outputs
- Sample 100 examples per week for manual review
- Track user feedback/corrections

### 7.2 Retrain When Needed

When hallucination rate increases or new data available:

```bash
# 1. Add new training data
# 2. Retrain Stage A
python stage_a_sft_training.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --train_data_path updated_training_data.csv \
    --num_epochs 2

# 3. Re-evaluate
python evaluate_stage_a.py --model_path ./models/sft_specialist/final_model

# 4. If good, prepare and redeploy
python prepare_for_deployment.py --model_path ./models/sft_specialist/final_model
```

---

## Complete Command Reference

### Evaluation
```bash
python evaluate_stage_a.py \
    --model_path ./models/sft_specialist/final_model \
    --test_data phase1_data/sft/validation_set_processed.csv
```

### Preparation
```bash
python prepare_for_deployment.py \
    --model_path ./models/sft_specialist/final_model \
    --output_dir ./models/sft_specialist_merged
```

### Test
```bash
python sft_inference.py \
    --model_path ./models/sft_specialist_merged \
    --clinical_note "Test note"
```

### Deploy
```bash
python deploy_api.py \
    --model_path ./models/sft_specialist_merged \
    --host 0.0.0.0 \
    --port 8000
```

---

## Troubleshooting

### Model won't load
```bash
# Check model files exist
ls ./models/sft_specialist/final_model/

# Try CPU mode
python sft_inference.py --model_path ./models/sft_specialist/final_model --device cpu
```

### Out of memory
```bash
# Use 8-bit quantization
python sft_inference.py --model_path ./models/sft_specialist/final_model --use_8bit
```

### API not responding
```bash
# Check if port is in use
lsof -i :8000

# Try different port
python deploy_api.py --port 8001
```

### Slow inference
- Use merged model (not LoRA)
- Enable 8-bit quantization
- Use GPU instead of CPU
- Reduce max_tokens parameter

---

## Summary Checklist

- [ ] Stage A training completed
- [ ] Evaluation run and results reviewed
- [ ] Hallucination rate calculated (<15%)
- [ ] LoRA weights merged
- [ ] Merged model tested
- [ ] API deployed and tested
- [ ] Health checks passing
- [ ] Production monitoring setup
- [ ] Documentation created
- [ ] Team trained on API usage

**You're ready for production! 🎉**

---

## Next Steps (Optional)

### If hallucination rate > 15%:
Run Stage B (DPO) for hallucination reduction:
```bash
python stage_b_dpo_training.py \
    --sft_model_path ./models/sft_specialist/final_model \
    --learning_rate 5e-6 \
    --batch_size 2
```

### For even better performance:
- Quantize to 8-bit or 4-bit for faster inference
- Use vLLM or TensorRT for optimized serving
- Set up A/B testing between model versions
- Implement caching for common queries
