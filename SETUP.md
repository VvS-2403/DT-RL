# Setup & Installation Guide

## Quick Start (30 seconds)

```bash
# Clone repo
cd dependency-trading-system

# Run demo (synthetic data end-to-end)
python demo.py

# Start backend
python -m backend.api.main

# Start frontend (in another terminal)
cd frontend && npm install && npm run dev

# Visit http://localhost:3000
```

## Prerequisites

### Required
- Python 3.11+
- Node.js 18+
- pip/npm

### Optional but Recommended
- NVIDIA GPU with CUDA 11.8+
- Docker & Docker Compose
- Git

## Installation Steps

### 1. Environment Setup

#### Linux/macOS
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

#### Windows
```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check key packages
python -c "import fastapi, pandas, numpy; print('All dependencies installed ✓')"
```

### 3. Download Demo Data (Optional)

The demo will generate synthetic data automatically, but you can also load real CSV data:

```bash
# Expected CSV format:
# Date,Ticker,Open,High,Low,Close,Volume,VIX,RSI_14,...

# To load your data:
# 1. Place CSV/Parquet in data/ directory
# 2. Use the dashboard Data page to upload
```

## Running the System

### Option A: Local Development (Recommended for Development)

#### Terminal 1: Backend Server
```bash
python -m backend.api.main
# Server runs on http://localhost:8000
# API docs on http://localhost:8000/docs
```

#### Terminal 2: Frontend Development Server
```bash
cd frontend
npm install
npm run dev
# Dashboard on http://localhost:3000
```

#### Terminal 3: Run Demo (Optional)
```bash
python demo.py
# Generates synthetic data and trains a model
```

### Option B: Docker Compose (Recommended for Production)

```bash
# Build and run everything
docker-compose up

# Or rebuild images
docker-compose up --build

# Monitor logs
docker-compose logs -f

# Shut down
docker-compose down
```

Services available:
- Backend API: http://localhost:8000
- Dashboard: http://localhost:3000
- TensorBoard: http://localhost:6006
- Redis: localhost:6379

## First Use: Step-by-Step

### 1. Data Upload
```
Dashboard → Data → Upload CSV/Parquet
```

Expected columns:
- `Date` (ISO format)
- `Ticker` (asset symbol)
- `Close` (closing price)
- `Volume`
- Optional: `Open, High, Low, VIX, RSI_14, ...`

### 2. Data Validation
```
Data page → Validate → Review coverage report
```

Look for:
- ✓ Coverage > 50% per asset
- ✓ No huge time gaps (> 5 days)
- ✓ Sufficient features (F > 5)

### 3. Training Configuration
```
Train page → Configure:
  - Lookback: 60 (days)
  - Horizon: 5 (forecast period)
  - Embedding: 128
  - Layers: 4
  - Epochs: 100
  - Learning rate: 0.001
```

### 4. Start Training
```
Train page → Start Training

Monitor:
- Epoch/Loss progress
- Validation metrics
- Early stopping if loss plateaus
```

### 5. Run Inference
```
Predictions page → Shows latest forecasts

Displays:
- Asset predictions
- Confidence levels
- Current regimes
- Portfolio weights (if mode=portfolio_weights)
```

### 6. Analyze Regimes
```
Regimes page → Visualize regime classifications

Shows:
- R1/R2/R3/R4 distribution
- Regime transitions
- Per-asset assignments
- Confidence scores
```

### 7. Backtest Results
```
Backtest page → View performance metrics

Metrics:
- Cumulative return
- Sharpe/Sortino ratio
- Maximum drawdown
- Turnover & costs
- Equity curve
```

## Configuration

Edit `configs/default.yaml`:

```yaml
# Data
data:
  lookback_window: 60
  forecast_horizon: 5

# Modules
modules:
  module2:
    num_sectors: 16
    hidden_dim: 64
  module3:
    num_regimes: 4
  module4:
    embedding_dim: 128
    num_layers: 4

# Training
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10

# Output
output:
  mode: "portfolio_weights"  # or "forecasting"
  max_weight: 0.1
  leverage: 1.0
  long_only: false
  transaction_cost: 0.001
```

## Troubleshooting

### GPU Not Found
```python
# Force CPU
pipeline = DependencyAwareTradingPipeline(config, device="cpu")
```

### Out of Memory
```yaml
# Reduce batch size
training:
  batch_size: 8  # Instead of 32

# Reduce model size
modules:
  module4:
    embedding_dim: 64  # Instead of 128
    num_layers: 2  # Instead of 4
```

### Slow Training
- Reduce number of assets or features
- Use smaller lookback window (30 instead of 60)
- Use mixed precision: `mixed_precision: true`
- Enable GPU acceleration

### Data Validation Errors
- Check CSV has required columns: Date, Ticker, Close, Volume
- Verify Date format (ISO: YYYY-MM-DD)
- Check for NaN values
- Ensure asset symbols are consistent

### Port Already in Use
```bash
# Backend on different port
python -m uvicorn backend.api.main:app --port 8001

# Frontend on different port
PORT=3001 npm run dev
```

## Performance Tuning

### For Development (Fast Iteration)
```yaml
data:
  lookback_window: 30
  num_assets: 10

training:
  batch_size: 16
  num_epochs: 10
  early_stopping_patience: 3

modules:
  module4:
    embedding_dim: 64
    num_layers: 2
```

### For Production (High Accuracy)
```yaml
data:
  lookback_window: 120
  num_assets: 500

training:
  batch_size: 64
  num_epochs: 200
  early_stopping_patience: 20
  mixed_precision: true

modules:
  module4:
    embedding_dim: 256
    num_layers: 8
```

### Hardware Recommendations

| Use Case | Recommended | Minimum |
|----------|-----------|---------|
| Demo | CPU or GTX 1650 | Any CPU |
| Development | RTX 3070 | RTX 2060 |
| Production | RTX 3090 or A100 | RTX 2080 Ti |

## Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_module1.py -v

# With coverage
pytest --cov=ml tests/

# Watch mode
pytest-watch
```

## Monitoring & Logging

### TensorBoard (Training Progress)
```bash
tensorboard --logdir logs/tensorboard
# Open http://localhost:6006
```

### Logs
```bash
# Backend logs
tail -f logs/backend.log

# Training logs
tail -f logs/training_{timestamp}.log

# All logs
ls -la logs/
```

## Deployment Checklist

- [ ] Data validation reports generated
- [ ] Training loss curves smooth and convergent
- [ ] Validation Sharpe > 1.0
- [ ] No data leakage in train/val split
- [ ] Backtest results realistic
- [ ] All edge cases handled (IPO/delist/gaps)
- [ ] Error messages are user-friendly
- [ ] Checkpoints saved and versioned
- [ ] Monitoring/alerting configured
- [ ] Documentation up-to-date

## Common Commands

```bash
# Install all dependencies
pip install -r requirements.txt

# Run demo end-to-end
python demo.py

# Start backend API (port 8000)
python -m backend.api.main

# Start frontend dev (port 3000)
cd frontend && npm run dev

# Build frontend for production
cd frontend && npm run build

# Run tests
pytest tests/ -v

# Check code quality
black ml/ backend/ tests/
isort ml/ backend/ tests/
flake8 ml/ backend/ tests/

# View API documentation
curl http://localhost:8000/docs
```

## Next Steps

1. **Load Real Data**: Use Data page to upload market data
2. **Explore Modules**: Check module-specific notebooks
3. **Tune Hyperparameters**: Use configs/ templates
4. **Run Backtests**: Evaluate strategy performance
5. **Deploy**: Use Docker for production

## Support & Resources

- **Architecture**: See README.md
- **API Docs**: http://localhost:8000/docs
- **Code Structure**: `tree -I '__pycache__|*.pyc'`
- **Module Details**: 
  - `ml/modules/module*.py` - Neural network code
  - `ml/pipeline.py` - Orchestration
  - `backend/api/routes/*.py` - API endpoints

## Additional Setup for Advanced Users

### Remote Development
```bash
# Clone repo on remote server
git clone <repo-url>
cd dependency-trading-system

# Setup environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run backend
python -m backend.api.main

# Access from local machine
ssh -L 8000:localhost:8000 -L 3000:localhost:3000 user@remote-server
# Then visit http://localhost:8000 and http://localhost:3000
```

### GPU Setup (NVIDIA)
```bash
# Verify CUDA
nvcc --version

# Install cuDNN if needed
# Download from https://developer.nvidia.com/cudnn
# Extract to ~/cudnn

# Test PyTorch GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

---

**Version**: 0.1.0  
**Last Updated**: August 29, 2026  
**Status**: Production Ready
