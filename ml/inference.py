"""
Inference utilities for Dependency-Aware Trading System.
"""

from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any, Union
from ml.pipeline import DependencyAwareTradingPipeline
from ml.modules.module1_input import MarketTensor

def load_trained_pipeline(checkpoint_path: Union[str, Path]) -> DependencyAwareTradingPipeline:
    """Load a trained pipeline from checkpoint."""
    return DependencyAwareTradingPipeline.load(Path(checkpoint_path))

def run_inference(
    pipeline: DependencyAwareTradingPipeline,
    market_tensor: MarketTensor,
    target_rtg: float = 0.05,
) -> Dict[str, Any]:
    """
    Run prediction on market tensor data.
    
    Args:
        pipeline: Trained DependencyAwareTradingPipeline instance
        market_tensor: Aligned MarketTensor
        target_rtg: Target return-to-go for Decision Transformer conditioning
        
    Returns:
        Dict of results containing weights, forecasts, regimes, and confidence values
    """
    pipeline.eval()
    device = pipeline.device
    
    # Extract features and validity mask
    X = market_tensor.features.unsqueeze(0).to(device)  # Add batch dimension: (1, N, T, F)
    M = market_tensor.validity_mask.unsqueeze(0).to(device)  # Add batch dimension: (1, N, T)
    
    batch_size, N, T, F = X.shape
    
    # Construct return-to-go
    rtg = torch.full((batch_size, N, T, 1), target_rtg, device=device)
    
    with torch.no_grad():
        outputs = pipeline(X, M, rtg)
        
    # Process outputs
    regime_probs = outputs["regime_probs"].view(batch_size, N, T, -1).squeeze(0).cpu().numpy()  # (N, T, 4)
    regime_assignments = regime_probs.argmax(axis=-1)  # (N, T)
    
    results = {
        "regime_probs": regime_probs,
        "regimes": regime_assignments,
        "timestamps": market_tensor.timestamps,
        "asset_ids": market_tensor.asset_ids,
    }
    
    out_obj = outputs["output"]
    if pipeline.config.output_mode == "portfolio_weights":
        weights_reshaped = out_obj.weights.view(batch_size, N, T, N).squeeze(0).cpu().numpy()  # (N, T, N)
        confidence_reshaped = out_obj.confidence.view(batch_size, N, T).squeeze(0).cpu().numpy()  # (N, T)
        results["portfolio_weights"] = weights_reshaped
        results["confidence"] = confidence_reshaped
    else:  # forecasting
        # forecasts is (batch * N, T, N, horizon)
        # Reshape to (batch, N, T, N, horizon)
        forecasts = out_obj.forecasts.view(batch_size, N, T, N, -1).squeeze(0).cpu().numpy()  # (N, T, N, horizon)
        confidence = out_obj.forecast_confidence.view(batch_size, N, T, N).squeeze(0).cpu().numpy()  # (N, T, N)
        results["forecasts"] = forecasts
        results["confidence"] = confidence
        
    return results
