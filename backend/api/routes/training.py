"""
Training routes.
"""

from fastapi import APIRouter, HTTPException
from backend.services.pipeline_service import PipelineService
from backend.api.models import TrainingConfigSchema, TrainingProgressSchema
from ml.pipeline import PipelineConfig

router = APIRouter()
pipeline_service = PipelineService()

@router.post("/start")
async def start_training(config: TrainingConfigSchema):
    """Start training the trading pipeline."""
    try:
        # Convert schema to internal config
        train_config = PipelineConfig(
            lookback_window=config.lookback_window,
            forecast_horizon=config.forecast_horizon,
            num_regimes=config.num_regimes,
            output_mode=config.output_mode,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            num_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
        )
        message = pipeline_service.start_training(train_config)
        return {"status": "success", "message": message}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/progress", response_model=TrainingProgressSchema)
async def get_progress():
    """Get active training job progress."""
    progress = pipeline_service.training_progress
    return {
        "status": pipeline_service.training_status,
        "epoch": progress["epoch"],
        "num_epochs": progress["num_epochs"],
        "train_loss": progress["train_loss"],
        "val_loss": progress["val_loss"],
        "history": progress["history"],
        "error": progress.get("error")
    }
