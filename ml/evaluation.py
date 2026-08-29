"""
Evaluation metrics for Dependency-Aware Trading System.

Computes regression metrics for forecasting mode and portfolio performance metrics
for portfolio weight mode.
"""

import numpy as np
import torch
from typing import Dict, Any

def evaluate_predictions(
    outputs: Dict[str, Any],
    targets: torch.Tensor,
    output_mode: str,
) -> Dict[str, Any]:
    """
    Evaluate predicted output against targets.
    
    Args:
        outputs: Dictionary returned by pipeline forward pass
        targets: (batch, N, horizon) future returns
        output_mode: "portfolio_weights" or "forecasting"
        
    Returns:
        Dict of evaluation metrics
    """
    metrics = {}
    
    # Move to CPU for numpy operations
    targets_np = targets.detach().cpu().numpy()
    
    if output_mode == "portfolio_weights":
        weights = outputs["output"].weights.detach().cpu().numpy()  # (batch * N, T, N)
        batch_size = targets_np.shape[0]
        N = targets_np.shape[1]
        T = weights.shape[1]
        
        # Reshape to (batch, N, T, N)
        weights = weights.reshape(batch_size, N, T, N)
        pred_weights = weights[:, :, -1, :]  # (batch, N, N) - weights at the last step
        
        # Targets: (batch, N, horizon)
        # Average returns over the horizon for evaluation
        avg_returns = targets_np.mean(axis=-1)  # (batch, N)
        
        # Calculate returns for each asset's conditioning portfolio
        # portfolio_returns shape: (batch, N)
        portfolio_returns = np.zeros((batch_size, N))
        for b in range(batch_size):
            portfolio_returns[b] = np.dot(pred_weights[b], avg_returns[b])
        
        # We can average across assets to get a single portfolio return per batch step
        step_returns = portfolio_returns.mean(axis=1)  # (batch,)
        
        financial_metrics = compute_financial_metrics(step_returns)
        metrics.update(financial_metrics)
        
    else:  # forecasting
        forecasts = outputs["output"].forecasts.detach().cpu().numpy()  # (batch * N, T, N, horizon)
        batch_size = targets_np.shape[0]
        N = targets_np.shape[1]
        horizon = targets_np.shape[2]
        
        # Reshape to (batch, N, N, horizon)
        forecasts = forecasts.reshape(batch_size, N, N, horizon)
        # Take forecasts at the last timestep
        last_forecasts = forecasts[:, -1, :, :]  # (batch, N, N, horizon)
        
        # Extract self-predictions (diagonal over dimensions 1 and 2)
        diag_forecasts = np.zeros((batch_size, N, horizon))
        for b in range(batch_size):
            for i in range(N):
                diag_forecasts[b, i, :] = last_forecasts[b, i, i, :]
        
        # Compute regression metrics
        mse = np.mean((diag_forecasts - targets_np) ** 2)
        mae = np.mean(np.abs(diag_forecasts - targets_np))
        
        # Directional accuracy
        pred_sign = np.sign(diag_forecasts)
        true_sign = np.sign(targets_np)
        directional_accuracy = np.mean(pred_sign == true_sign)
        
        # R2 score (simple overall)
        ss_res = np.sum((targets_np - diag_forecasts) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        
        metrics.update({
            "mse": float(mse),
            "mae": float(mae),
            "directional_accuracy": float(directional_accuracy),
            "r2_score": float(r2),
        })
        
    return metrics


def compute_financial_metrics(
    returns: np.ndarray,
    weights_diff: np.ndarray = None,
    transaction_cost_bps: float = 10.0,
) -> Dict[str, float]:
    """
    Compute trading financial performance metrics.
    
    Args:
        returns: (num_steps,) portfolio returns
        weights_diff: (num_steps, N) daily changes in portfolio weights (for turnover)
        transaction_cost_bps: Cost per basis point of turnover
        
    Returns:
        Dict of financial metrics
    """
    if len(returns) == 0:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "transaction_costs": 0.0,
        }
        
    # Calculate cumulative return
    cum_returns = np.cumprod(1.0 + returns) - 1.0
    total_return = cum_returns[-1] if len(cum_returns) > 0 else 0.0
    
    # Annualization factor (assuming daily data, 252 trading days)
    ann_factor = 252
    
    mean_return = np.mean(returns)
    vol = np.std(returns) * np.sqrt(ann_factor)
    
    ann_return = mean_return * ann_factor
    
    # Sharpe Ratio (assuming risk-free rate = 0)
    sharpe = (mean_return / (np.std(returns) + 1e-8)) * np.sqrt(ann_factor)
    
    # Sortino Ratio (downside risk only)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(ann_factor) if len(downside_returns) > 0 else 1e-8
    sortino = (mean_return * ann_factor) / (downside_std + 1e-8)
    
    # Max Drawdown
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    
    # Turnover and transaction costs
    turnover = 0.0
    tx_costs = 0.0
    if weights_diff is not None:
        # Turnover is the sum of absolute changes in weights
        turnover = np.mean(np.sum(np.abs(weights_diff), axis=-1))
        # Transaction costs in percent of portfolio value
        tx_costs = turnover * (transaction_cost_bps / 10000.0)
        
    return {
        "cumulative_return": float(total_return),
        "annualized_return": float(ann_return),
        "volatility": float(vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "turnover": float(turnover),
        "transaction_costs": float(tx_costs),
    }
