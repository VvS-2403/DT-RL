"""
Synthetic market data generator for testing and demo purposes.

Generates realistic synthetic OHLCV data with:
- Multiple correlated assets
- Time-varying regimes
- Sector dependencies
- Missing data (IPO/delist simulation)
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

from ml.modules.module1_input import MarketTensor, StockDataProcessor


class SyntheticMarketDataGenerator:
    """Generate synthetic market data for testing."""
    
    def __init__(
        self,
        num_assets: int = 50,
        num_days: int = 250,
        num_sectors: int = 8,
        seed: int = 42,
    ):
        """
        Initialize data generator.
        
        Args:
            num_assets: Number of assets to generate
            num_days: Number of trading days
            num_sectors: Number of sectors
            seed: Random seed
        """
        np.random.seed(seed)
        self.num_assets = num_assets
        self.num_days = num_days
        self.num_sectors = num_sectors
    
    def generate_ohlcv(self) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data.
        
        Returns:
            DataFrame with columns: Date, Ticker, Open, High, Low, Close, Volume
        """
        data = []
        
        # Generate base prices for each asset
        base_prices = np.random.uniform(20, 500, self.num_assets)
        asset_ids = [f"ASSET_{i:03d}" for i in range(self.num_assets)]
        
        # Assign assets to sectors
        sector_assignment = np.random.randint(0, self.num_sectors, self.num_assets)
        
        # Generate time series
        dates = [
            datetime(2023, 1, 1) + timedelta(days=int(d))
            for d in range(self.num_days)
        ]
        
        # Sector-level shocks
        sector_shocks = np.random.normal(1.0, 0.02, (self.num_sectors, self.num_days))
        
        for asset_idx, asset_id in enumerate(asset_ids):
            price = base_prices[asset_idx]
            sector = sector_assignment[asset_idx]
            
            for day_idx, date in enumerate(dates):
                # Random walk with sector effect
                daily_return = (
                    np.random.normal(0.0005, 0.02) +  # Idiosyncratic
                    0.3 * (sector_shocks[sector, day_idx] - 1.0)  # Sector effect
                )
                price = price * (1.0 + daily_return)
                
                # Generate OHLCV
                open_price = price * (1 + np.random.normal(0, 0.005))
                close_price = price
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.02))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.02))
                volume = np.random.uniform(1e6, 1e8)
                
                data.append({
                    "Date": date,
                    "Ticker": asset_id,
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume,
                })
        
        df = pd.DataFrame(data)
        return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        df = df.copy()
        
        # Add macro indicators
        dates = pd.to_datetime(df["Date"]).unique()
        macro_values = {
            "VIX": np.random.uniform(10, 30, len(dates)),
            "DXY": np.random.uniform(95, 105, len(dates)),
            "TNX": np.random.uniform(3, 5, len(dates)),
        }
        
        date_to_idx = {d: i for i, d in enumerate(dates)}
        
        for macro_col, values in macro_values.items():
            df[macro_col] = df["Date"].apply(lambda x: values[date_to_idx[pd.to_datetime(x)]])
        
        # Add per-asset indicators
        for asset_id in df["Ticker"].unique():
            mask = df["Ticker"] == asset_id
            asset_data = df.loc[mask, "Close"].values
            
            # RSI (simplified)
            if len(asset_data) > 14:
                deltas = np.diff(asset_data)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rs = avg_gain / (avg_loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
            
            df.loc[mask, "RSI_14"] = rsi
            
            # MACD (simplified)
            df.loc[mask, "MACD"] = np.random.normal(0, 1)
            df.loc[mask, "MACD_Signal"] = np.random.normal(0, 1)
            
            # Bollinger Bands
            mean = np.mean(asset_data[-20:]) if len(asset_data) >= 20 else np.mean(asset_data)
            std = np.std(asset_data[-20:]) if len(asset_data) >= 20 else np.std(asset_data)
            df.loc[mask, "BBand_Upper"] = mean + 2 * std
            df.loc[mask, "BBand_Middle"] = mean
            df.loc[mask, "BBand_Lower"] = mean - 2 * std
        
        # Sentiment (random)
        df["Sentiment_Score"] = np.random.uniform(-1, 1, len(df))
        
        return df
    
    def add_missing_data(self, df: pd.DataFrame, missing_prob: float = 0.05) -> pd.DataFrame:
        """Simulate IPO/delisting by adding missing data."""
        df = df.copy()
        
        # Randomly remove some data for each asset (simulate IPO/delist)
        for asset_id in df["Ticker"].unique():
            mask = df["Ticker"] == asset_id
            dates = df.loc[mask, "Date"].unique()
            
            # Simulate IPO: remove first N% of dates
            ipo_idx = np.random.randint(0, int(0.3 * len(dates)))
            
            # Simulate delist: remove last N% of dates
            delist_idx = len(dates) - np.random.randint(0, int(0.3 * len(dates)))
            
            # Remove dates outside IPO-delist range
            valid_dates = set(dates[ipo_idx:delist_idx])
            df = df[~(mask & ~df["Date"].isin(valid_dates))]
        
        return df
    
    def generate(self) -> pd.DataFrame:
        """Generate complete synthetic dataset."""
        print(f"Generating synthetic market data: {self.num_assets} assets, {self.num_days} days")
        
        # Generate OHLCV
        df = self.generate_ohlcv()
        
        # Add indicators
        df = self.add_technical_indicators(df)
        
        # Add missing data
        df = self.add_missing_data(df)
        
        # Convert dates to datetime
        df["Date"] = pd.to_datetime(df["Date"])
        
        print(f"Generated {len(df)} rows")
        return df


class SyntheticDataLoader:
    """Load synthetic data into MarketTensor."""
    
    @staticmethod
    def create_demo_market_tensor(
        num_assets: int = 50,
        lookback: int = 60,
        seed: int = 42,
    ) -> Tuple[MarketTensor, pd.DataFrame]:
        """
        Create demo MarketTensor with synthetic data.
        
        Returns:
            (MarketTensor, raw_dataframe)
        """
        # Generate data
        generator = SyntheticMarketDataGenerator(
            num_assets=num_assets,
            num_days=lookback + 100,  # Extra for train/val/test
            seed=seed,
        )
        df = generator.generate()
        
        # Process into MarketTensor
        processor = StockDataProcessor(lookback_window=lookback)
        
        # For demo, take last 'lookback' days
        last_date = df["Date"].max()
        df_windowed = df[df["Date"] >= last_date - pd.Timedelta(days=lookback)]
        
        market_tensor = processor.process(df_windowed)
        
        return market_tensor, df
    
    @staticmethod
    def create_sequences(
        market_tensor: MarketTensor,
        lookback: int = 60,
        horizon: int = 5,
        stride: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create training sequences from MarketTensor.
        
        Args:
            market_tensor: Input data
            lookback: Lookback window
            horizon: Forecast horizon
            stride: Stride between sequences
            
        Returns:
            (X, M, Y) - features, validity masks, future returns
        """
        X = market_tensor.features.numpy()  # (N, T, F)
        M = market_tensor.validity_mask.numpy()  # (N, T)
        N, T, F = X.shape
        
        sequences_X = []
        sequences_M = []
        sequences_Y = []  # Future returns
        
        # Extract price (assume Close is feature 3)
        close_idx = 3 if F > 3 else 0
        prices = X[:, :, close_idx]  # (N, T)
        
        # Generate sequences
        for start in range(0, T - lookback - horizon, stride):
            end = start + lookback
            
            seq_X = X[:, start:end, :]  # (N, lookback, F)
            seq_M = M[:, start:end]  # (N, lookback)
            
            # Future returns
            future_prices = prices[:, end:end + horizon]  # (N, horizon)
            current_prices = prices[:, end:end + 1]  # (N, 1)
            returns = (future_prices / (current_prices + 1e-8)) - 1.0  # (N, horizon)
            
            sequences_X.append(seq_X)
            sequences_M.append(seq_M)
            sequences_Y.append(returns)
        
        X_tensor = torch.from_numpy(np.stack(sequences_X, axis=0)).float()  # (seq, N, lookback, F)
        M_tensor = torch.from_numpy(np.stack(sequences_M, axis=0)).float()  # (seq, N, lookback)
        Y_tensor = torch.from_numpy(np.stack(sequences_Y, axis=0)).float()  # (seq, N, horizon)
        
        return X_tensor, M_tensor, Y_tensor


def create_simple_demo():
    """Create a simple demo dataset."""
    loader = SyntheticDataLoader()
    market_tensor, raw_df = loader.create_demo_market_tensor(num_assets=20, lookback=60)
    
    print("Demo MarketTensor created:")
    print(f"  Shape: {market_tensor.shape}")
    print(f"  Assets: {market_tensor.num_assets}")
    print(f"  Lookback: {market_tensor.lookback_window}")
    print(f"  Features: {market_tensor.num_features}")
    
    # Create sequences
    X, M, Y = loader.create_sequences(market_tensor)
    print(f"\nSequences created:")
    print(f"  X shape: {X.shape}")
    print(f"  M shape: {M.shape}")
    print(f"  Y shape: {Y.shape}")
    
    return market_tensor, X, M, Y
