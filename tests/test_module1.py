"""
Tests for Module 1: Stock Data Input

Verifies:
- Data alignment and tensor creation
- Validity mask handling
- Survivorship bias prevention
"""

import pytest
import torch
import numpy as np
import pandas as pd
from ml.modules.module1_input import StockDataProcessor, DataValidator, MarketTensor


@pytest.fixture
def sample_data():
    """Create sample market data."""
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    assets = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    for asset in assets:
        for date in dates:
            data.append({
                'Date': date,
                'Ticker': asset,
                'Open': 100 + np.random.randn(),
                'High': 102 + np.random.randn(),
                'Low': 99 + np.random.randn(),
                'Close': 101 + np.random.randn(),
                'Volume': 1e6,
                'RSI_14': 50 + np.random.randn() * 10,
            })
    
    return pd.DataFrame(data)


def test_data_processor_shapes(sample_data):
    """Test data processor creates correct tensor shapes."""
    processor = StockDataProcessor(lookback_window=60)
    mt = processor.process(sample_data)
    
    assert mt.shape[0] == 3  # num_assets
    assert mt.shape[1] == 60  # lookback_window
    assert mt.shape[2] >= 6  # num_features (at least OHLCV + RSI)


def test_validity_mask(sample_data):
    """Test validity mask is created."""
    processor = StockDataProcessor()
    mt = processor.process(sample_data)
    
    # Validity mask should be binary
    assert mt.validity_mask.dtype == torch.int32
    assert (mt.validity_mask == 0).sum() + (mt.validity_mask == 1).sum() == mt.validity_mask.numel()


def test_normalization(sample_data):
    """Test feature normalization."""
    processor = StockDataProcessor(normalization="zscore")
    mt = processor.process(sample_data)
    
    X = mt.features.numpy()
    
    # Check that features are approximately normalized (mean ~0, std ~1)
    for f in range(X.shape[2]):
        valid_data = X[:, :, f][mt.validity_mask.numpy() == 1]
        if len(valid_data) > 1:
            mean = np.nanmean(valid_data)
            std = np.nanstd(valid_data)
            assert abs(mean) < 1.0  # Should be close to 0
            assert abs(std - 1.0) < 2.0  # Should be close to 1


def test_data_validator(sample_data):
    """Test data validation."""
    processor = StockDataProcessor()
    mt = processor.process(sample_data)
    
    validator = DataValidator()
    report = validator.validate_market_tensor(mt)
    
    assert "valid" in report
    assert "issues" in report
    assert "coverage" in report


def test_missing_data_handling():
    """Test handling of missing data (IPO/delist)."""
    # Create data with gaps
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    assets = ['AAPL', 'MSFT']
    
    data = []
    for asset in assets:
        for i, date in enumerate(dates):
            # MSFT only has data from day 20
            if asset == 'MSFT' and i < 20:
                continue
            
            data.append({
                'Date': date,
                'Ticker': asset,
                'Close': 100 + np.random.randn(),
                'Volume': 1e6,
            })
    
    df = pd.DataFrame(data)
    processor = StockDataProcessor()
    mt = processor.process(df)
    
    # Check that MSFT has lower validity in early period
    msft_idx = mt.asset_ids.index('MSFT')
    early_validity = mt.validity_mask[msft_idx, :20].sum().item()
    assert early_validity < mt.validity_mask[msft_idx, 20:].sum().item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
