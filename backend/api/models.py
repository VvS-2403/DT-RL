"""
Pydantic schemas for FastAPI routes.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TrainingConfigSchema(BaseModel):
    """Configuration schema for triggering training."""
    lookback_window: int = Field(default=60, description="Timeline lookback window in days")
    forecast_horizon: int = Field(default=5, description="Forecast horizon in days")
    num_regimes: int = Field(default=4, description="Number of statistical regimes")
    output_mode: str = Field(default="portfolio_weights", description="'portfolio_weights' or 'forecasting'")
    batch_size: int = Field(default=16, description="Training batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    num_epochs: int = Field(default=5, description="Number of epochs to train")
    weight_decay: float = Field(default=1e-5, description="Weight decay factor")

class TrainingProgressSchema(BaseModel):
    """Schema for training status and metrics."""
    status: str = Field(..., description="Active status: idle, training, completed, failed")
    epoch: int = Field(..., description="Current trained epoch")
    num_epochs: int = Field(..., description="Total training epochs configured")
    train_loss: float = Field(..., description="Current training loss value")
    val_loss: float = Field(..., description="Current validation loss value")
    history: List[Dict[str, Any]] = Field(default=[], description="Detailed metrics timeline per epoch")
    error: Optional[str] = Field(default=None, description="Error message if failed")

class PredictRequestSchema(BaseModel):
    """Request params for inference predict endpoint."""
    target_rtg: float = Field(default=0.05, description="Target return-to-go for conditioning")

class BacktestRequestSchema(BaseModel):
    """Request params for running backtest simulation."""
    transaction_cost_bps: float = Field(default=10.0, description="Cost in basis points per turnover transaction")

class BacktestResponseSchema(BaseModel):
    """Performance metrics schema of historical backtest."""
    cumulative_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    turnover: float
    transaction_costs: float
    equity_curve: List[float]
    benchmark_curve: List[float]
    dates: List[str]

class RegimeResponseSchema(BaseModel):
    """Regime classifications representation schema."""
    asset_ids: List[str]
    dates: List[str]
    regime_names: List[str]
    regime_probs: List[List[List[float]]]  # (N, T, 4)
    regimes: List[List[int]]              # (N, T)
