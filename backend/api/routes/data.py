"""
Data management routes.

Endpoints:
- POST /upload - Upload CSV/Parquet
- GET /validate - Validate dataset
- GET /preview - Preview data
- GET /stats - Dataset statistics
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import torch
from pathlib import Path
import tempfile
from typing import Dict, List

from ml.modules.module1_input import StockDataProcessor, DataValidator
from backend.services.pipeline_service import PipelineService

router = APIRouter()
pipeline_service = PipelineService()

# Storage
DATA_DIR = Path("data/uploads")
DATA_DIR.mkdir(parents=True, exist_ok=True)

class CurrentDataWrapper:
    def __getitem__(self, key):
        if key == "market_tensor":
            return pipeline_service.market_tensor
        elif key == "raw_dataframe":
            return pipeline_service.raw_dataframe
        elif key == "filepath":
            return getattr(pipeline_service, "filepath", None)
        raise KeyError(key)
        
    def __setitem__(self, key, value):
        if key == "market_tensor":
            pipeline_service.market_tensor = value
        elif key == "raw_dataframe":
            pipeline_service.raw_dataframe = value
        elif key == "filepath":
            pipeline_service.filepath = value
        else:
            raise KeyError(key)

current_data = CurrentDataWrapper()


@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload CSV or Parquet file.
    
    Supported formats:
    - CSV with columns: Date, Ticker, Open, High, Low, Close, Volume, [indicators]
    - Parquet with same schema
    """
    try:
        # Validate file type
        if file.content_type not in ["text/csv", "application/octet-stream"]:
            raise HTTPException(400, "Only CSV/Parquet files supported")
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Read file
        if file.filename.endswith(".csv"):
            df = pd.read_csv(tmp_path)
        elif file.filename.endswith(".parquet"):
            df = pd.read_parquet(tmp_path)
        else:
            raise HTTPException(400, "Unsupported file format")
        
        # Basic validation
        required_cols = ["Date", "Ticker", "Close", "Volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(400, f"Missing required columns: {missing}")
        
        # Process into MarketTensor
        processor = StockDataProcessor(lookback_window=60)
        market_tensor = processor.process(df)
        
        # Store
        current_data["market_tensor"] = market_tensor
        current_data["raw_dataframe"] = df
        current_data["filepath"] = tmp_path
        
        return {
            "status": "success",
            "message": f"Uploaded {len(df)} rows, {df['Ticker'].nunique()} assets",
            "shape": market_tensor.shape,
            "file_size_mb": len(content) / (1024 * 1024),
        }
    
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


@router.get("/validate")
async def validate_data():
    """
    Validate current dataset.
    
    Returns validation report including:
    - Data quality issues
    - Coverage statistics
    - Missing value patterns
    """
    if current_data["market_tensor"] is None:
        raise HTTPException(400, "No data loaded. Upload data first.")
    
    mt = current_data["market_tensor"]
    
    # Run validation
    validator = DataValidator()
    report = validator.validate_market_tensor(mt)
    
    return {
        "valid": report["valid"],
        "issues": report["issues"],
        "coverage": report["coverage"],
        "shape": report["shape"],
        "date_range": report["date_range"],
        "quality_score": 1.0 - len(report["issues"]) * 0.1,  # Simple scoring
    }


@router.get("/preview")
async def preview_data(num_assets: int = 5, num_days: int = 10):
    """
    Preview dataset.
    
    Args:
        num_assets: Number of assets to show
        num_days: Number of days to show
    """
    if current_data["market_tensor"] is None:
        raise HTTPException(400, "No data loaded")
    
    mt = current_data["market_tensor"]
    X = mt.features.numpy()
    
    # Extract sample
    sample_X = X[:num_assets, -num_days:, :]
    sample_assets = mt.asset_ids[:num_assets]
    sample_dates = mt.timestamps[-num_days:].tolist()
    sample_features = mt.feature_names[:5]  # First 5 features
    
    return {
        "assets": sample_assets,
        "dates": [str(d) for d in sample_dates],
        "features": sample_features,
        "data": sample_X[:, :, :5].tolist(),
    }


@router.get("/stats")
async def get_statistics():
    """Get detailed dataset statistics."""
    if current_data["market_tensor"] is None:
        raise HTTPException(400, "No data loaded")
    
    mt = current_data["market_tensor"]
    X = mt.features.numpy()
    M = mt.validity_mask.numpy()
    
    # Compute stats
    stats = {
        "shape": {
            "num_assets": int(X.shape[0]),
            "num_days": int(X.shape[1]),
            "num_features": int(X.shape[2]),
        },
        "coverage": {
            "overall": float(M.sum() / M.size),
            "min_per_asset": float(M.sum(axis=1).min() / M.shape[1]),
            "max_per_asset": float(M.sum(axis=1).max() / M.shape[1]),
        },
        "features": {
            "mean": [float(X[:, :, i].mean()) for i in range(min(5, X.shape[2]))],
            "std": [float(X[:, :, i].std()) for i in range(min(5, X.shape[2]))],
            "names": mt.feature_names[:5],
        },
        "date_range": {
            "start": str(mt.timestamps[0]),
            "end": str(mt.timestamps[-1]),
            "num_days": len(mt.timestamps),
        },
    }
    
    return stats


@router.get("/assets")
async def list_assets():
    """List all assets in current dataset."""
    if current_data["market_tensor"] is None:
        raise HTTPException(400, "No data loaded")
    
    mt = current_data["market_tensor"]
    return {
        "assets": mt.asset_ids,
        "num_assets": len(mt.asset_ids),
    }


@router.post("/clear")
async def clear_data():
    """Clear loaded data."""
    current_data["market_tensor"] = None
    current_data["raw_dataframe"] = None
    current_data["filepath"] = None
    
    return {"status": "cleared"}
