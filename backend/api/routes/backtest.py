"""
Backtest routes.
"""

from fastapi import APIRouter, HTTPException
from backend.services.pipeline_service import PipelineService
from backend.api.models import BacktestRequestSchema, BacktestResponseSchema

router = APIRouter()
pipeline_service = PipelineService()

@router.post("/run", response_model=BacktestResponseSchema)
async def run_backtest(req: BacktestRequestSchema):
    """Run historical backtest simulation and retrieve performance stats."""
    try:
        results = pipeline_service.run_backtest(transaction_cost_bps=req.transaction_cost_bps)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
