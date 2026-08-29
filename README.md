# Dependency-Aware Trading System

A research-grade deep learning system for multi-asset prediction and portfolio optimization using hypergraph-based cross-asset encoding, continuous regime classification, and decision transformers.

## Architecture Overview

```
Raw Data
  ↓
Module 1: Stock Data Input (X ∈ R^(N×T×F), M ∈ {0,1}^(N×T))
  ↓
Module 2: Cross-Asset Hypergraph Encoder (Z ∈ R^(N×T×D))
  ↓
Module 3: Dependency/Memory Regime Classifier (P_regime, E_regime)
  ↓
Module 4: Decision Transformer (enriched sequences τ)
  ↓
Module 5: Final Output Module (Portfolio Weights or Forecasts)
```

### Key Design Principles

- **Modular & Testable**: Each module has explicit typed contracts
- **Scalable**: Hypergraph avoids O(N²) complexity via degree normalization
- **Continuous Regime Logic**: No hard switching; smooth transitions via embeddings
- **Non-Survivorship-Biased**: Validity masks handle IPO/delisting
- **Production-Quality**: Type hints, error handling, extensive tests

## Quick Start

### Prerequisites
- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (optional, CPU fallback supported)

### Installation

```bash
# Clone and setup
cd dependency-trading-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run End-to-End Demo

```bash
# Start backend server
python -m backend.api.main

# In another terminal, start frontend
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to access the dashboard.

### Docker

```bash
docker-compose up
```

## Project Structure

```
dependency-trading-system/
├── ml/                           # ML pipeline
│   ├── data/
│   │   ├── module1_stock_data.py    # Data loading & alignment
│   │   └── synthetic_data.py         # Demo data generation
│   ├── modules/
│   │   ├── module1_input.py          # Stock tensor creation
│   │   ├── module2_encoder.py        # Hypergraph encoder
│   │   ├── module3_regime.py         # Regime classifier
│   │   ├── module4_transformer.py    # Decision transformer
│   │   └── module5_output.py         # Output heads
│   ├── pipeline.py               # Main pipeline orchestration
│   ├── training.py               # Training loop
│   ├── evaluation.py             # Evaluation metrics
│   └── inference.py              # Inference utilities
│
├── backend/                      # FastAPI server
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── data.py
│   │   │   ├── training.py
│   │   │   ├── inference.py
│   │   │   ├── backtest.py
│   │   │   └── regimes.py
│   │   └── models.py            # Pydantic schemas
│   ├── services/
│   │   ├── pipeline_service.py
│   │   ├── backtest_engine.py
│   │   └── storage.py
│   └── jobs/
│       └── training_job.py       # Async training
│
├── frontend/                     # Next.js + React
│   ├── app/
│   │   ├── page.tsx             # Dashboard
│   │   ├── data/page.tsx         # Data upload
│   │   ├── train/page.tsx        # Training control
│   │   ├── models/page.tsx       # Model management
│   │   ├── predictions/page.tsx  # Prediction display
│   │   ├── backtest/page.tsx     # Backtest results
│   │   ├── regimes/page.tsx      # Regime visualization
│   │   ├── dependencies/page.tsx # Dependency analysis
│   │   └── settings/page.tsx     # Configuration
│   ├── components/
│   │   ├── Sidebar.tsx
│   │   ├── DataUpload.tsx
│   │   ├── EquityCurve.tsx
│   │   ├── RegimeChart.tsx
│   │   └── MetricsCard.tsx
│   └── lib/
│       ├── api.ts               # API client
│       └── types.ts             # Shared types
│
├── configs/
│   ├── default.yaml             # Default hyperparameters
│   └── templates/               # Config templates
│
├── tests/
│   ├── test_module1.py
│   ├── test_module2.py
│   ├── test_module3.py
│   ├── test_module4.py
│   ├── test_module5.py
│   └── test_integration.py
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── pytest.ini
```

## Data Schema

### Input CSV/Parquet Format

```
Date,Ticker,Open,High,Low,Close,Volume,VIX,SectorIndex,...
2024-01-01,AAPL,150.0,151.5,149.8,151.2,50000000,15.2,...
2024-01-01,MSFT,380.0,385.0,379.5,382.1,20000000,15.2,...
```

Required columns:
- `Date`: Trading date (ISO format)
- `Ticker`: Asset symbol
- `Close`: Closing price
- `Volume`: Trading volume
- `Open`, `High`, `Low`: OHLC data

Optional columns:
- Micro indicators: `RSI`, `MACD`, `BBand_*`
- Macro indicators: `VIX`, `DXY`, `TNX`
- Sentiment: `Sentiment_Score`
- Fundamentals: `PE_Ratio`, `Market_Cap`

## API Endpoints

### Data Management
- `POST /api/data/upload` — Upload CSV/Parquet
- `GET /api/data/validate` — Validate dataset
- `GET /api/data/preview` — Dataset preview and stats

### Training
- `POST /api/train/config` — Set training parameters
- `POST /api/train/start` — Start training job
- `GET /api/train/status` — Training progress
- `GET /api/train/metrics` — Training metrics (loss, validation)

### Models
- `GET /api/models/list` — List saved models
- `POST /api/models/load` — Load model for inference
- `GET /api/models/{id}/info` — Model metadata

### Inference
- `POST /api/predict` — Run inference on new data
- `GET /api/predictions/latest` — Latest predictions
- `GET /api/predictions/history` — Prediction history

### Regimes
- `GET /api/regimes/current` — Current regime distribution
- `GET /api/regimes/timeseries` — Regime timeseries
- `GET /api/regimes/transitions` — Regime transitions

### Backtest
- `POST /api/backtest/run` — Run backtest
- `GET /api/backtest/{id}` — Backtest results
- `GET /api/backtest/{id}/trades` — Trade log

## Configuration

Edit `configs/default.yaml`:

```yaml
data:
  lookback_window: 60
  forecast_horizon: 5
  normalization: "zscore"

modules:
  module2:
    num_sectors: 16
    hidden_dim: 64
  module3:
    num_regimes: 4
  module4:
    embedding_dim: 128
    num_layers: 4
    num_heads: 8
    dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10
  mixed_precision: true
  gradient_clip: 1.0

output:
  mode: "portfolio_weights"  # or "forecasting"
  max_weight: 0.1
  leverage: 1.0
  long_only: false
  transaction_cost: 0.001

backtest:
  initial_capital: 100000
  transaction_cost_bps: 10
  slippage_bps: 5
```

## Module Specifications

### Module 1: Stock Data Input
- **Input**: OHLCV, micro/macro indicators, sentiment, fundamentals
- **Output**: 
  - `X`: (N, T, F) feature tensor
  - `M`: (N, T) validity mask
- **Handles**: Survivorship bias, IPOs, delistings

### Module 2: Cross-Asset Encoder
- **Input**: X (N, T, F)
- **Output**: Z (N, T, D) context-aware embeddings
- **Key Feature**: Hypergraph with degree normalization avoids O(N²) complexity
- **Cold-start**: New assets inherit sector characteristics

### Module 3: Regime Classifier
- **Input**: Z (N, T, D)
- **Output**: 
  - `P_regime`: (N, T, 4) softmax probabilities
  - `E_regime`: (N, T, D) continuous embeddings
- **Axes**:
  - Memory: Long (trending) vs Short (mean-reverting)
  - Credit: Long (slow macro) vs Short (instant reaction)
- **Critical**: Continuous embeddings for smooth behavior

### Module 4: Decision Transformer
- **Input**: Enriched sequences τ = (R̂ₜ, Z̃ₜ, aₜ)
  - R̂: Return-to-Go
  - Z̃ = Z + E_regime: Regime-conditioned state
  - a: Actions
- **Default**: Shared transformer with regime conditioning
- **Output**: H (latent decision vector)

### Module 5: Output Module
- **Option A (Portfolio Weights)**:
  - w_t^(i) = Masked Softmax(a_t) ∈ [-1, 1]
  - Respects validity mask, leverage, long-only constraints
- **Option B (Forecasting)**:
  - Ŷ = MLP(H) ∈ R^(N×T_f)
  - Predicts future returns for downstream RL

## Training

```python
from ml.pipeline import DependencyAwareTradingPipeline
from ml.data import load_market_data

# Load data
data = load_market_data("data/market_data.parquet")

# Initialize pipeline
pipeline = DependencyAwareTradingPipeline(
    num_assets=data.num_assets,
    lookback=60,
    horizon=5,
    embedding_dim=128,
    num_regimes=4,
    output_mode="portfolio_weights"
)

# Train
metrics = pipeline.fit(
    data=data,
    batch_size=32,
    learning_rate=0.001,
    num_epochs=100,
    validation_split=0.2,
    test_split=0.1
)

# Save
pipeline.save("models/model_v1.pt")
```

## Inference

```python
# Load trained model
pipeline = DependencyAwareTradingPipeline.load("models/model_v1.pt")

# Make predictions
predictions = pipeline.predict(new_data)
# Returns: {
#   "portfolio_weights": (N, max_weight=0.1, leverage=1.0),
#   "confidences": (N,),
#   "regimes": (N, 4),
#   "dependencies": {...}
# }

# Evaluate
metrics = pipeline.evaluate(test_data)
# Returns: Sharpe, Sortino, max_drawdown, volatility, ...
```

## Backtesting

```python
from backend.services.backtest_engine import BacktestEngine

engine = BacktestEngine(
    initial_capital=100000,
    transaction_cost_bps=10,
    slippage_bps=5
)

results = engine.run(
    predictions=predictions,
    historical_prices=prices,
    constraints={
        "max_weight": 0.1,
        "leverage": 1.0,
        "long_only": False
    }
)

# Results include: equity curve, returns, Sharpe, Sortino, max_drawdown, turnover, ...
```

## Testing

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_module1.py -v

# With coverage
pytest --cov=ml tests/
```

## Monitoring & Logging

- **Training logs**: `logs/training_{timestamp}.log`
- **Inference logs**: `logs/inference_{timestamp}.log`
- **Metrics tracking**: TensorBoard support
  ```bash
  tensorboard --logdir logs/tensorboard
  ```

## Performance

- **Data loading**: ~1M rows/sec
- **Training (per epoch)**: ~5-10 sec (32-batch, 60 assets, 250 days)
- **Inference**: ~100 ms (including data prep)
- **Backtest**: ~500 ms (1 year data, 60 assets)

## Hardware

- **Recommended**: NVIDIA GPU (4GB+ VRAM)
- **Fallback**: CPU (slower training, inference OK)
- **Memory**: ~2-4 GB for training typical datasets

## Troubleshooting

### GPU Memory Issues
```python
pipeline = DependencyAwareTradingPipeline(..., device="cpu")
# Or reduce batch_size, num_layers, embedding_dim
```

### Missing Data
- Module 1 handles NaN via validity masks
- Imputation: forward-fill for prices, zero for indicators
- Delistings: automatic masking

### Convergence Issues
- Lower learning rate (0.0001 → 0.00005)
- Increase gradient clip (1.0 → 5.0)
- Check data normalization
- Increase early stopping patience

## Research Notes

- **Regime Classification**: Prevents treating trending vs mean-reverting assets identically
- **Hypergraph Encoding**: Captures supply-chain co-movements without N² complexity
- **Continuous Embeddings**: Avoids unstable trading at regime boundaries
- **Decision Transformer**: Better long-term credit assignment than traditional RL

## References

- Dependency-Aware Architecture Document (Aug 26, 2026)
- Decision Transformer (Chen et al., 2021)
- Hypergraph Neural Networks (Feng et al., 2019)

## License

Research use only. See LICENSE.

## Contributors

Senior ML Researcher, Quantitative Engineer, Backend Architect, Frontend Engineer
