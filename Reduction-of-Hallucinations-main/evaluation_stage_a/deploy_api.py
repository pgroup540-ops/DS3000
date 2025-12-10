"""
Stage A Model Deployment - FastAPI Server
==========================================

Deploy the fine-tuned Stage A model as a REST API.

Installation:
    pip install fastapi uvicorn

Usage:
    python deploy_api.py --model_path ./models/sft_specialist_merged

    Then test:
    curl -X POST http://localhost:8000/summarize \
        -H "Content-Type: application/json" \
        -d '{"clinical_note": "Patient has fever and cough."}'
"""

import argparse
import logging
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sft_inference import SFTInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response models
class SummarizeRequest(BaseModel):
    clinical_note: str
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7


class SummarizeResponse(BaseModel):
    clinical_note: str
    summary: str
    model_version: str


# Global model instance
model = None
MODEL_VERSION = "stage_a_v1"


def create_app(model_path: str, device: str = "cuda") -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Medical Summary API",
        description="Generate medical summaries from clinical notes",
        version="1.0.0"
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Load model on startup."""
        global model
        logger.info(f"Loading model from {model_path}")
        model = SFTInference(
            model_path=model_path,
            device=device
        )
        logger.info("Model loaded successfully!")
    
    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_version": MODEL_VERSION,
            "message": "Medical Summary API is running"
        }
    
    @app.get("/health")
    async def health():
        """Detailed health check."""
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_version": MODEL_VERSION
        }
    
    @app.post("/summarize", response_model=SummarizeResponse)
    async def summarize(request: SummarizeRequest):
        """
        Generate medical summary from clinical note.
        
        Args:
            request: SummarizeRequest with clinical_note
        
        Returns:
            SummarizeResponse with generated summary
        """
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not request.clinical_note or len(request.clinical_note.strip()) == 0:
            raise HTTPException(status_code=400, detail="clinical_note is required")
        
        try:
            # Generate summary
            result = model.generate_summary(
                clinical_note=request.clinical_note,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            return SummarizeResponse(
                clinical_note=request.clinical_note,
                summary=result['generated_summary'],
                model_version=MODEL_VERSION
            )
        
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Stage A Model API")
    
    parser.add_argument(
        "--model_path",
        default="./models/sft_specialist_merged",
        help="Path to model (preferably merged)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    
    args = parser.parse_args()
    
    # Create app
    app = create_app(
        model_path=args.model_path,
        device=args.device
    )
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"\nAPI Documentation: http://{args.host}:{args.port}/docs\n")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
