"""
Module 5: Final Output Module (Execution)

Convert latent decision vectors into executable trades.

Option A: Portfolio Weights (Direct Execution)
- w_t^(i) = Masked Softmax(a_t^(i)) respecting validity masks & constraints
- Pros: End-to-end RL optimizing Sharpe
- Cons: Must implicitly learn execution mechanics

Option B: Forecasting (Separated Execution)
- Ŷ = MLP(H) predicting future returns
- Downstream RL (PPO/DDPG) converts to trades
- Pros: Separates prediction from execution
- Cons: Compounding errors
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PortfolioOutput:
    """Portfolio weights output."""
    weights: torch.Tensor          # (batch, seq_len, N) - portfolio weights
    logits: torch.Tensor           # (batch, seq_len, N) - pre-softmax logits
    confidence: torch.Tensor       # (batch, seq_len) - max softmax prob
    validity_mask_applied: bool    # Whether mask was applied


@dataclass
class ForecastingOutput:
    """Forecasting output."""
    forecasts: torch.Tensor        # (batch, seq_len, N, horizon) - predicted returns
    forecast_confidence: torch.Tensor  # (batch, seq_len, N) - uncertainty estimates


class PortfolioWeightsHead(nn.Module):
    """
    Direct execution: latent decision → portfolio weights via masked softmax.
    
    w_t^(i) = exp(a_t^(i)) · M_t^(i) / Σⱼ exp(a_t^(j)) · M_t^(j) ∈ [-1, 1]
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_assets: int,
        hidden_dim: int = 256,
    ):
        """
        Initialize portfolio weights head.
        
        Args:
            latent_dim: Latent decision vector dimension
            num_assets: Number of assets
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.num_assets = num_assets
        
        self.weight_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_assets),
        )
    
    def forward(
        self,
        latent_decision: torch.Tensor,  # (batch, seq_len, latent_dim)
        validity_mask: Optional[torch.Tensor] = None,  # (batch, seq_len, N)
        constraints: Optional[Dict] = None,
    ) -> PortfolioOutput:
        """
        Compute portfolio weights from latent decisions.
        
        Args:
            latent_decision: (batch, seq_len, latent_dim)
            validity_mask: (batch, seq_len, N) - binary mask for valid assets
            constraints: {
                "max_weight": 0.1,
                "leverage": 1.0,
                "long_only": False,
                "min_weight": -0.1,
            }
            
        Returns:
            PortfolioOutput with weights, logits, confidence
        """
        batch_size, seq_len, _ = latent_decision.shape
        device = latent_decision.device
        
        # Generate logits
        logits = self.weight_generator(latent_decision)  # (batch, seq_len, N)
        
        # Apply validity mask to logits (prevent inactive assets)
        if validity_mask is not None:
            # validity_mask: (batch, seq_len, N) - must be broadcast properly
            if validity_mask.dim() == 2:  # (batch, seq_len)
                # Expand to (batch, seq_len, N)
                validity_mask = validity_mask.unsqueeze(-1).expand(-1, -1, self.num_assets)
            
            # Mask out invalid assets with very negative logits
            logits = torch.where(validity_mask > 0, logits, torch.tensor(float('-inf'), device=device))
        
        # Masked softmax
        # Handle -inf by subtracting max (numerical stability)
        logits_safe = logits.clone()
        min_val = -30000.0 if logits_safe.dtype == torch.float16 else -1e9
        logits_safe[torch.isinf(logits_safe)] = min_val
        logits_max = logits_safe.amax(dim=2, keepdim=True)
        exp_logits = torch.exp(logits_safe - logits_max)
        
        # Mask out invalid assets
        if validity_mask is not None:
            exp_logits = exp_logits * validity_mask
        
        # Softmax: sums to 1.0 over valid assets
        weights_softmax = exp_logits / (exp_logits.sum(dim=2, keepdim=True) + 1e-8)  # (batch, seq_len, N)
        
        # Scale to [-1, 1] range for leverage and shorts
        weights = 2.0 * weights_softmax - 1.0  # Maps [0,1] → [-1, 1]
        
        # Apply constraints
        if constraints is not None:
            weights = self._apply_constraints(weights, constraints)
        
        # Confidence: max softmax probability
        confidence = weights_softmax.amax(dim=2)  # (batch, seq_len)
        
        return PortfolioOutput(
            weights=weights,
            logits=logits,
            confidence=confidence,
            validity_mask_applied=(validity_mask is not None),
        )
    
    def _apply_constraints(
        self,
        weights: torch.Tensor,  # (batch, seq_len, N)
        constraints: Dict,
    ) -> torch.Tensor:
        """Apply portfolio constraints."""
        w = weights.clone()
        
        # Long-only constraint
        if constraints.get("long_only", False):
            w = F.relu(w)  # Clip negative to 0
        
        # Max weight constraint
        max_weight = constraints.get("max_weight", 1.0)
        w = torch.clamp(w, -max_weight, max_weight)
        
        # Leverage constraint: |w|_1 ≤ leverage
        leverage = constraints.get("leverage", 2.0)
        l1_norm = torch.abs(w).sum(dim=2, keepdim=True)
        scale_factor = torch.minimum(
            torch.tensor(1.0, device=w.device),
            leverage / (l1_norm + 1e-8)
        )
        w = w * scale_factor
        
        return w


class ForecastingHead(nn.Module):
    """
    Separated execution: latent decision → future return forecasts.
    
    Ŷ = MLP(H) ∈ R^(N×T_f) predicting T_f-period returns for each asset.
    Downstream RL agent converts forecasts to trades.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_assets: int,
        forecast_horizon: int = 5,
        hidden_dim: int = 256,
    ):
        """
        Initialize forecasting head.
        
        Args:
            latent_dim: Latent decision vector dimension
            num_assets: Number of assets
            forecast_horizon: Number of periods to forecast ahead
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.num_assets = num_assets
        self.forecast_horizon = forecast_horizon
        
        # Main forecast network
        self.forecast_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_assets * forecast_horizon),
        )
        
        # Uncertainty estimation
        self.uncertainty_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_assets),
        )
    
    def forward(
        self,
        latent_decision: torch.Tensor,  # (batch, seq_len, latent_dim)
    ) -> ForecastingOutput:
        """
        Forecast future returns from latent decisions.
        
        Args:
            latent_decision: (batch, seq_len, latent_dim)
            
        Returns:
            ForecastingOutput with predictions and confidence
        """
        batch_size, seq_len, latent_dim = latent_decision.shape
        
        # Flatten for MLP
        latent_flat = latent_decision.view(-1, latent_dim)  # (batch*seq_len, latent_dim)
        
        # Generate forecasts
        forecast_flat = self.forecast_network(latent_flat)  # (batch*seq_len, N*horizon)
        forecasts = forecast_flat.view(
            batch_size, seq_len, self.num_assets, self.forecast_horizon
        )
        
        # Generate uncertainty (log-std)
        log_std = self.uncertainty_network(latent_flat)  # (batch*seq_len, N)
        uncertainty = torch.exp(log_std).view(batch_size, seq_len, self.num_assets)
        
        return ForecastingOutput(
            forecasts=forecasts,
            forecast_confidence=1.0 / (uncertainty + 1e-8),  # Inverse of uncertainty
        )


class OutputModule(nn.Module):
    """
    Unified output module supporting both portfolio weights and forecasting.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_assets: int,
        output_mode: str = "portfolio_weights",  # or "forecasting"
        forecast_horizon: int = 5,
        hidden_dim: int = 256,
    ):
        """
        Initialize output module.
        
        Args:
            latent_dim: Latent decision dimension
            num_assets: Number of assets
            output_mode: "portfolio_weights" or "forecasting"
            forecast_horizon: For forecasting mode
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        assert output_mode in ["portfolio_weights", "forecasting"]
        
        self.output_mode = output_mode
        self.num_assets = num_assets
        
        if output_mode == "portfolio_weights":
            self.head = PortfolioWeightsHead(
                latent_dim=latent_dim,
                num_assets=num_assets,
                hidden_dim=hidden_dim,
            )
        else:  # forecasting
            self.head = ForecastingHead(
                latent_dim=latent_dim,
                num_assets=num_assets,
                forecast_horizon=forecast_horizon,
                hidden_dim=hidden_dim,
            )
    
    def forward(
        self,
        latent_decision: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
        constraints: Optional[Dict] = None,
    ):
        """
        Generate outputs based on mode.
        
        Args:
            latent_decision: (batch, seq_len, latent_dim)
            validity_mask: (batch, seq_len) or (batch, seq_len, N)
            constraints: Portfolio constraints (for portfolio weights mode)
            
        Returns:
            PortfolioOutput or ForecastingOutput
        """
        if self.output_mode == "portfolio_weights":
            return self.head(
                latent_decision=latent_decision,
                validity_mask=validity_mask,
                constraints=constraints,
            )
        else:
            return self.head(latent_decision=latent_decision)


class ExecutionPolicy:
    """Convert outputs to actual trades."""
    
    @staticmethod
    def weights_to_trades(
        current_weights: torch.Tensor,  # (N,)
        target_weights: torch.Tensor,   # (N,)
        prices: torch.Tensor,           # (N,)
        constraints: Optional[Dict] = None,
    ) -> Dict:
        """
        Convert weight changes to trade sizes.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            prices: Current prices
            constraints: {
                "max_turnover": 0.05,
                "transaction_cost_bps": 10,
                "min_trade_size": 100,
            }
            
        Returns:
            {
                "trades": (N,),
                "trade_sizes": (N,),
                "transaction_cost": float,
                "turnover": float,
            }
        """
        device = current_weights.device
        
        # Weight difference
        weight_diff = target_weights - current_weights  # (N,)
        
        # Apply max turnover constraint
        if constraints and "max_turnover" in constraints:
            max_turnover = constraints["max_turnover"]
            actual_turnover = torch.abs(weight_diff).sum().item()
            if actual_turnover > max_turnover:
                weight_diff = weight_diff * (max_turnover / (actual_turnover + 1e-8))
        
        # Compute trade sizes (in units)
        trade_sizes = weight_diff / prices  # Simple division for now
        
        # Filter small trades
        if constraints and "min_trade_size" in constraints:
            min_size = constraints["min_trade_size"]
            trade_sizes[torch.abs(trade_sizes) < min_size] = 0.0
        
        # Compute transaction cost
        transaction_cost = 0.0
        if constraints and "transaction_cost_bps" in constraints:
            cost_bps = constraints["transaction_cost_bps"]
            transaction_cost = (torch.abs(weight_diff).sum() * cost_bps / 10000).item()
        
        return {
            "weight_diff": weight_diff.cpu().numpy().tolist(),
            "trade_sizes": trade_sizes.cpu().numpy().tolist(),
            "transaction_cost": transaction_cost,
            "turnover": torch.abs(weight_diff).sum().item(),
        }
    
    @staticmethod
    def forecasts_to_trades_greedy(
        forecasts: torch.Tensor,  # (N, horizon)
        current_prices: torch.Tensor,  # (N,)
        capital: float = 100000.0,
        constraints: Optional[Dict] = None,
    ) -> Dict:
        """
        Simple greedy execution: rank by expected return, allocate capital.
        
        Args:
            forecasts: (N, horizon) or (N,) if using 1st period
            current_prices: (N,)
            capital: Available capital
            constraints: Portfolio constraints
            
        Returns:
            {
                "weights": (N,),
                "shares": (N,),
                "cash_remaining": float,
            }
        """
        N = forecasts.shape[0]
        
        # Use first forecast period
        if forecasts.dim() > 1:
            expected_returns = forecasts[:, 0]
        else:
            expected_returns = forecasts
        
        # Rank by expected return
        sorted_idx = torch.argsort(expected_returns, descending=True)
        
        # Allocate capital greedily
        weights = torch.zeros(N, device=forecasts.device)
        cash = capital
        
        for idx in sorted_idx:
            if expected_returns[idx] > 0 and cash > 0:
                # Allocate up to max_weight
                max_weight = constraints.get("max_weight", 0.1) if constraints else 0.1
                position_value = min(max_weight * capital, cash)
                weights[idx] = position_value / capital
                cash -= position_value
        
        # Compute shares
        shares = weights * capital / (current_prices + 1e-8)
        
        return {
            "weights": weights.cpu().numpy().tolist(),
            "shares": shares.cpu().numpy().tolist(),
            "cash_remaining": cash,
        }
