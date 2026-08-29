"""
Module 1: Stock Data Input

Collect and align multi-asset data into cohesive tensors.
Input: OHLCV, Micro/Macro indicators, Sentiment/News, Fundamentals
Output: X ∈ R^(N×T×F), M ∈ {0,1}^(N×T)

Handles survivorship bias, IPOs, delistings via validity masks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import pandas as pd
from pathlib import Path


@dataclass
class MarketTensor:
    """Clean structure for aligned market data with metadata."""
    
    features: torch.Tensor          # Shape: (N, T, F) - asset, time, feature
    validity_mask: torch.Tensor     # Shape: (N, T) - binary mask for IPO/delist
    asset_ids: List[str]            # Length: N - ticker symbols
    timestamps: np.ndarray          # Length: T - trading dates
    feature_names: List[str]        # Length: F - feature names
    feature_means: torch.Tensor     # Shape: (F,) - for normalization
    feature_stds: torch.Tensor      # Shape: (F,) - for normalization
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return shape as (N, T, F)."""
        return self.features.shape
    
    @property
    def num_assets(self) -> int:
        """Number of assets."""
        return self.features.shape[0]
    
    @property
    def lookback_window(self) -> int:
        """Lookback window length."""
        return self.features.shape[1]
    
    @property
    def num_features(self) -> int:
        """Number of features."""
        return self.features.shape[2]
    
    def to_device(self, device: torch.device) -> "MarketTensor":
        """Move tensors to device."""
        return MarketTensor(
            features=self.features.to(device),
            validity_mask=self.validity_mask.to(device),
            asset_ids=self.asset_ids,
            timestamps=self.timestamps,
            feature_names=self.feature_names,
            feature_means=self.feature_means.to(device),
            feature_stds=self.feature_stds.to(device),
        )


class StockDataProcessor:
    """Process raw stock data into aligned tensors."""
    
    # Standard OHLCV columns
    OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
    
    # Micro indicators (asset-specific)
    MICRO_INDICATORS = [
        "RSI_14", "MACD", "MACD_Signal", "BBand_Upper", "BBand_Middle", "BBand_Lower"
    ]
    
    # Macro indicators (systemic)
    MACRO_INDICATORS = [
        "VIX", "DXY", "TNX", "MOVE", "SKEW"
    ]
    
    # Sentiment & Fundamentals
    SENTIMENT_COLS = ["Sentiment_Score"]
    FUNDAMENTAL_COLS = ["PE_Ratio", "Dividend_Yield", "Market_Cap"]
    
    def __init__(
        self,
        lookback_window: int = 60,
        normalization: str = "zscore",
        fill_method: str = "forward",
        device: torch.device = None,
    ):
        """
        Initialize processor.
        
        Args:
            lookback_window: Number of trading days to look back
            normalization: "zscore" or "minmax"
            fill_method: "forward" for forward-fill, "drop" to skip
            device: torch device for tensors
        """
        self.lookback_window = lookback_window
        self.normalization = normalization
        self.fill_method = fill_method
        self.device = device or torch.device("cpu")
        
    def process(
        self,
        df: pd.DataFrame,
        asset_col: str = "Ticker",
        date_col: str = "Date",
    ) -> MarketTensor:
        """
        Process raw DataFrame into aligned MarketTensor.
        
        Args:
            df: DataFrame with columns [Date, Ticker, OHLCV, indicators, ...]
            asset_col: Column name for asset ticker
            date_col: Column name for trading date
            
        Returns:
            MarketTensor with X, M, metadata
        """
        # Validate input
        if date_col not in df.columns or asset_col not in df.columns:
            raise ValueError(f"Missing required columns: {date_col}, {asset_col}")
        
        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Get unique assets and dates
        assets = sorted(df[asset_col].unique())
        dates = sorted(df[date_col].unique())
        
        # Select feature columns
        feature_cols = self._select_features(df.columns)
        
        # Create aligned tensor (N, T, F)
        N = len(assets)
        T = len(dates)
        F = len(feature_cols)
        
        X = np.full((N, T, F), np.nan, dtype=np.float32)
        M = np.zeros((N, T), dtype=np.int32)  # Validity mask
        
        # Fill tensor
        for i, asset in enumerate(assets):
            asset_df = df[df[asset_col] == asset].set_index(date_col)
            
            for j, date in enumerate(dates):
                if date in asset_df.index:
                    row = asset_df.loc[date]
                    # Get feature values
                    for f, col in enumerate(feature_cols):
                        if col in row.index:
                            val = row[col]
                            if pd.notna(val):
                                X[i, j, f] = float(val)
                                M[i, j] = 1  # Valid
        
        # Handle missing data
        X = self._fill_missing(X, M)
        
        # Normalize
        feature_means, feature_stds = self._compute_stats(X, M)
        X = self._normalize(X, feature_means, feature_stds)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X).to(self.device)
        M_tensor = torch.from_numpy(M).to(self.device)
        feature_means_tensor = torch.from_numpy(feature_means).to(self.device)
        feature_stds_tensor = torch.from_numpy(feature_stds).to(self.device)
        
        return MarketTensor(
            features=X_tensor,
            validity_mask=M_tensor,
            asset_ids=assets,
            timestamps=np.array(dates),
            feature_names=feature_cols,
            feature_means=feature_means_tensor,
            feature_stds=feature_stds_tensor,
        )
    
    def _select_features(self, columns) -> List[str]:
        """Select relevant feature columns from DataFrame."""
        selected = []
        
        for col in self.OHLCV_COLS:
            if col in columns:
                selected.append(col)
        
        for col in self.MICRO_INDICATORS:
            if col in columns:
                selected.append(col)
        
        for col in self.MACRO_INDICATORS:
            if col in columns:
                selected.append(col)
        
        for col in self.SENTIMENT_COLS:
            if col in columns:
                selected.append(col)
        
        for col in self.FUNDAMENTAL_COLS:
            if col in columns:
                selected.append(col)
        
        if not selected:
            raise ValueError("No valid features found in DataFrame")
        
        return selected
    
    def _fill_missing(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Fill missing values."""
        X = X.copy()
        N, T, F = X.shape
        
        if self.fill_method == "forward":
            # Forward-fill per asset per feature
            for i in range(N):
                for f in range(F):
                    for t in range(1, T):
                        if np.isnan(X[i, t, f]) and not np.isnan(X[i, t-1, f]):
                            X[i, t, f] = X[i, t-1, f]
            
            # If still NaN, use feature mean
            for f in range(F):
                mean_val = np.nanmean(X[:, :, f])
                X[:, :, f] = np.where(np.isnan(X[:, :, f]), mean_val, X[:, :, f])
        
        return X
    
    def _compute_stats(
        self,
        X: np.ndarray,
        M: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean/std only on valid data."""
        F = X.shape[2]
        means = np.zeros(F)
        stds = np.ones(F)
        
        for f in range(F):
            valid_data = X[:, :, f][M == 1]
            if len(valid_data) > 0:
                means[f] = np.nanmean(valid_data)
                stds[f] = np.nanstd(valid_data)
                if stds[f] < 1e-6:
                    stds[f] = 1.0  # Prevent division by zero
        
        return means.astype(np.float32), stds.astype(np.float32)
    
    def _normalize(
        self,
        X: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        """Normalize features."""
        if self.normalization == "zscore":
            return (X - means) / stds
        elif self.normalization == "minmax":
            # Simplified minmax
            return (X - np.min(X, axis=(0, 1), keepdims=True)) / \
                   (np.max(X, axis=(0, 1), keepdims=True) - np.min(X, axis=(0, 1), keepdims=True) + 1e-8)
        else:
            return X


class DataValidator:
    """Validate data quality and detect issues."""
    
    @staticmethod
    def validate_market_tensor(mt: MarketTensor) -> Dict[str, any]:
        """
        Validate MarketTensor and return report.
        
        Returns:
            {
                "valid": bool,
                "issues": List[str],
                "coverage": Dict[str, float],
                "stats": Dict
            }
        """
        issues = []
        X = mt.features.numpy() if isinstance(mt.features, torch.Tensor) else mt.features
        M = mt.validity_mask.numpy() if isinstance(mt.validity_mask, torch.Tensor) else mt.validity_mask
        
        N, T, F = mt.shape
        
        # Check NaN
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values")
        
        # Check coverage per asset
        coverage_per_asset = M.sum(axis=1) / T
        if (coverage_per_asset < 0.5).any():
            sparse_assets = [
                mt.asset_ids[i] for i in range(N) if coverage_per_asset[i] < 0.5
            ]
            issues.append(f"Low coverage (<50%): {sparse_assets}")
        
        # Check temporal continuity
        dates_diff = np.diff(mt.timestamps.astype('datetime64[D]'))
        if (dates_diff > np.timedelta64(5, 'D')).any():
            issues.append("Large gaps in time series (>5 days)")
        
        # Check for constant features
        for f, fname in enumerate(mt.feature_names):
            if np.nanstd(X[:, :, f]) < 1e-6:
                issues.append(f"Constant feature: {fname}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "coverage": {
                "overall": float(M.sum() / (N * T)),
                "per_asset": {
                    asset: float(M[i].sum() / T)
                    for i, asset in enumerate(mt.asset_ids)
                }
            },
            "shape": {
                "num_assets": N,
                "lookback_window": T,
                "num_features": F
            },
            "date_range": {
                "start": str(mt.timestamps[0]),
                "end": str(mt.timestamps[-1])
            }
        }
