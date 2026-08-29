"""
Dependencies hypergraph routing.
"""

from fastapi import APIRouter, HTTPException
from backend.services.pipeline_service import PipelineService
import numpy as np

router = APIRouter()
pipeline_service = PipelineService()

@router.get("/")
async def get_dependencies():
    """Retrieve the incidence matrix H mapping assets to latent sectors."""
    try:
        pipeline = pipeline_service.pipeline
        if not pipeline or not pipeline.modules_initialized:
            # Fallback mock sector assignment for demo
            assets = pipeline_service.market_tensor.asset_ids if pipeline_service.market_tensor else [f"ASSET_{i:03d}" for i in range(20)]
            sectors = [f"Sector {j+1}" for j in range(8)]
            np.random.seed(42)
            H_data = np.random.uniform(0.1, 1.0, (len(assets), len(sectors)))
            H_normalized = (np.exp(H_data) / np.sum(np.exp(H_data), axis=1, keepdims=True)).tolist()
            return {
                "asset_ids": assets,
                "sector_names": sectors,
                "incidence_matrix": H_normalized,
            }
            
        # Get actual matrix H
        H = pipeline.module2.layers[0].H.detach().cpu().numpy()  # (N, K)
        assets = pipeline_service.market_tensor.asset_ids if pipeline_service.market_tensor else [f"ASSET_{i:03d}" for i in range(H.shape[0])]
        sectors = [f"Sector {j+1}" for j in range(H.shape[1])]
        
        # Softmax normalize H along sectors dimension for better dashboard membership representation
        H_normalized = (np.exp(H) / np.sum(np.exp(H), axis=1, keepdims=True)).tolist()
        
        return {
            "asset_ids": assets,
            "sector_names": sectors,
            "incidence_matrix": H_normalized,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
