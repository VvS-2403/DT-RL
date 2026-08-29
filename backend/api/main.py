"""
FastAPI server for Dependency-Aware Trading System.

Endpoints:
- /api/data/*
- /api/train/*
- /api/predict/*
- /api/backtest/*
- /api/regimes/*
- /api/dependencies/*
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import torch
import os
from pathlib import Path

# Import routers
from backend.api.routes import data, training, inference, backtest, regimes, dependencies

from backend.services.pipeline_service import PipelineService

pipeline_service = PipelineService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifecycle management."""
    print("Server starting...")
    yield
    print("Server shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Dependency-Aware Trading System",
    description="Research-grade multi-asset ML trading system",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(training.router, prefix="/api/train", tags=["training"])
app.include_router(inference.router, prefix="/api/predict", tags=["inference"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(regimes.router, prefix="/api/regimes", tags=["regimes"])
app.include_router(dependencies.router, prefix="/api/dependencies", tags=["dependencies"])


# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "pipeline_loaded": pipeline_service.pipeline is not None,
        "pytorch_version": torch.__version__,
    }


# System info
@app.get("/api/system-info")
async def system_info():
    """Get system information."""
    return {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else None,
        "device_count": torch.cuda.device_count(),
    }


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    
    # Development server
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
