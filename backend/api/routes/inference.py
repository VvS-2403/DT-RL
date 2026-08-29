"""
Inference routes.
"""

from fastapi import APIRouter, HTTPException
from backend.services.pipeline_service import PipelineService
from backend.api.models import PredictRequestSchema
import numpy as np

router = APIRouter()
pipeline_service = PipelineService()

@router.post("/run")
async def run_predict(req: PredictRequestSchema):
    """Run model inference over the current dataset."""
    try:
        results = pipeline_service.run_inference(target_rtg=req.target_rtg)
        
        # Format returns for JSON response: converting numpy structures
        json_results = {
            "timestamps": [str(t) for t in results["timestamps"]],
            "asset_ids": results["asset_ids"],
            "regimes": results["regimes"].tolist(),
            "regime_probs": results["regime_probs"].tolist(),
        }
        
        if "portfolio_weights" in results:
            json_results["portfolio_weights"] = results["portfolio_weights"].tolist()
            json_results["confidence"] = results["confidence"].tolist()
        else:
            json_results["forecasts"] = results["forecasts"].tolist()
            json_results["confidence"] = results["confidence"].tolist()
            
        return json_results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
