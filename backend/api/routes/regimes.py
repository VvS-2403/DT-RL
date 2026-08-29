"""
Regime classification routes.
"""

from fastapi import APIRouter, HTTPException
from backend.services.pipeline_service import PipelineService
from backend.api.models import RegimeResponseSchema
from ml.modules.module3_regime import RegimeClassifier
import numpy as np

router = APIRouter()
pipeline_service = PipelineService()

@router.get("/", response_model=RegimeResponseSchema)
async def get_regimes():
    """Retrieve statistical regime assignments and probabilities."""
    try:
        results = pipeline_service.run_inference()
        return {
            "asset_ids": results["asset_ids"],
            "dates": [str(d) for d in results["timestamps"]],
            "regime_names": RegimeClassifier.REGIME_NAMES,
            "regime_probs": results["regime_probs"].tolist(),
            "regimes": results["regimes"].tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
