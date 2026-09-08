# DT-RL: Training & Visualization Guide

Complete instructions for training the Dependency-Aware Trading Pipeline with experiment tracking and visualization.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Setup](#2-local-setup)
3. [WandB Setup](#3-wandb-setup)
4. [Running Local Training](#4-running-local-training)
5. [Google Colab Notebook](#5-google-colab-notebook)
6. [Understanding the Results](#6-understanding-the-results)
7. [WandB Dashboard Guide](#7-wandb-dashboard-guide)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Runtime |
| PyTorch | 2.0+ | Deep learning framework |
| CUDA | 11.8+ | GPU acceleration (optional) |
| wandb | 0.16+ | Experiment tracking |
| matplotlib | 3.7+ | Plotting |

---

## 2. Local Setup

### Clone & Install

```bash
# Clone the repository
git clone https://github.com/Saatwik-ss/DT-RL.git
cd DT-RL

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import wandb; print(f'wandb {wandb.__version__}')"
python -c "from ml.pipeline import DependencyAwareTradingPipeline; print('✓ Pipeline imports OK')"
```

---

## 3. WandB Setup

### Create Account

1. Go to [wandb.ai/signup](https://wandb.ai/signup) and create a free account
2. Navigate to [wandb.ai/authorize](https://wandb.ai/authorize) to get your API key

### Login

```bash
wandb login
# Paste your API key when prompted
```

Or set the environment variable:
```bash
# Windows
set WANDB_API_KEY=your_api_key_here

# Linux/Mac
export WANDB_API_KEY=your_api_key_here
```

---

## 4. Running Local Training

> **Windows Note**: If you encounter `OSError: [WinError 127]` or `OMP: Error #15` when
> running `python train_wandb.py`, use the wrapper script `run_train.py` instead.
> It pre-loads DLL compatibility fixes. All arguments are the same.

### Quick Start (3-epoch demo)

```bash
# Linux/Mac/Colab:
python train_wandb.py --epochs 3 --num-assets 10 --batch-size 8

# Windows (if you get DLL errors):
python run_train.py --epochs 3 --num-assets 10 --batch-size 8
```

### Full Training Run

```bash
python train_wandb.py \
    --epochs 50 \
    --num-assets 20 \
    --batch-size 16 \
    --lr 0.001 \
    --output-mode portfolio_weights \
    --wandb-project DT-RL
```

### All CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-assets` | 20 | Number of synthetic assets |
| `--num-days` | 250 | Trading days to generate |
| `--lookback` | 60 | Lookback window (days) |
| `--horizon` | 5 | Forecast horizon (days) |
| `--stride` | 5 | Stride between sequences |
| `--seed` | 42 | Random seed |
| `--output-mode` | portfolio_weights | `portfolio_weights` or `forecasting` |
| `--num-layers` | 4 | Transformer layers |
| `--num-heads` | 8 | Attention heads |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--wandb-project` | DT-RL | WandB project name |
| `--wandb-entity` | None | WandB team/user |
| `--run-name` | auto | Custom run name |
| `--no-wandb` | False | Disable WandB logging |

### Run Without WandB

```bash
python train_wandb.py --no-wandb --epochs 10
```

### Output Files

After training, you'll find:

```
DT-RL/
├── checkpoints/
│   ├── best_*.pt              # Best model checkpoint
│   └── dtrl_<run_name>.pt    # Final model checkpoint
└── results/
    └── <run_name>/
        ├── data_price_trajectories.png
        ├── data_coverage_heatmap.png
        ├── results_loss_curves.png
        ├── results_regime_evolution_asset0.png
        ├── results_regime_distribution.png
        ├── results_regime_transitions.png
        ├── results_dependency_matrix.png
        ├── results_portfolio_weights.png
        ├── results_equity_curve.png
        ├── results_metrics_table.png
        └── results_training_summary.png
```

---

## 5. Google Colab Notebook

### Option A: Upload as Python Script (Recommended)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `DT_RL_Training_Colab.py`
4. Colab will automatically convert the `# %%` cell markers into notebook cells
5. Run all cells sequentially (Runtime → Run all)

### Option B: Convert to .ipynb First

```bash
# Install jupytext
pip install jupytext

# Convert to notebook
jupytext --to notebook DT_RL_Training_Colab.py

# This creates DT_RL_Training_Colab.ipynb
# Upload this .ipynb to Colab
```

### Option C: Clone Directly in Colab

In the first cell of a new Colab notebook:
```python
!git clone https://github.com/Saatwik-ss/DT-RL.git
%cd DT-RL
!pip install -q torch numpy pandas matplotlib seaborn wandb tqdm scipy pyarrow pyyaml
```

Then copy cells from `DT_RL_Training_Colab.py`.

### Colab Tips

- **GPU**: Go to **Runtime → Change runtime type → T4 GPU** for faster training
- **WandB**: The notebook will prompt for your API key; paste it when asked
- **Set `USE_WANDB = False`** in the config cell if you don't want WandB logging
- All plots render inline automatically in Colab

---

## 6. Understanding the Results

### Loss Curves
- **Train loss**: Should decrease steadily
- **Val loss**: Should decrease then plateau (overfitting starts when it rises)
- **Test loss**: Independent measure — should track val loss
- **Best epoch marker**: Annotated on the plot

### Regime Analysis
The system classifies each asset into 4 regimes at each timestep:

| Regime | Memory | Credit | Behavior |
|--------|--------|--------|----------|
| R1 | Long | Long | Trending + Slow macro absorption |
| R2 | Long | Short | Trending + Instant reaction |
| R3 | Short | Long | Mean-reverting + Slow macro |
| R4 | Short | Short | Mean-reverting + Instant reaction |

- **Regime Evolution**: Stacked area chart — shows how regime probabilities shift over time
- **Regime Distribution**: Bar chart + heatmap — shows asset assignments at the last timestep
- **Regime Transitions**: Heatmap — shows transition probabilities between regimes

### Portfolio Metrics
- **Sharpe Ratio**: Risk-adjusted return (> 1.0 is good, > 2.0 is excellent)
- **Sortino Ratio**: Like Sharpe but only penalizes downside risk
- **Max Drawdown**: Worst peak-to-trough loss (closer to 0 is better)
- **Cumulative Return**: Total return over the evaluation period

### Dependency Matrix
- Shows pairwise similarity between asset embeddings from the hypergraph encoder
- Assets in the same "latent sector" will show high similarity
- Red = correlated, Blue = anti-correlated

---

## 7. WandB Dashboard Guide

After a training run, visit your WandB dashboard to explore:

### Logged Metrics
Navigate to **Charts** tab to see real-time plots of:
- `train/loss`, `val/loss`, `test/loss`
- `train/learning_rate`
- `train/epoch_time_s`

### Logged Images
Navigate to **Media** tab to see:
- Loss curves, regime charts, portfolio weights, equity curves
- All figures from the `results/` prefix

### Model Artifacts
Navigate to **Artifacts** tab to:
- Download the trained model checkpoint
- Compare model versions across runs

### Comparing Runs
- Use the **Parallel Coordinates** chart to compare hyperparameters vs. performance
- Use **Sweep** (advanced) to automatically search hyperparameters

---

## 8. Troubleshooting

### CUDA Out of Memory

```bash
# Reduce model size
python train_wandb.py --num-assets 10 --batch-size 8 --num-layers 2 --num-heads 4

# Or force CPU
python train_wandb.py --epochs 5  # Will auto-detect no GPU
```

### WandB Login Issues

```bash
# Check if logged in
wandb status

# Re-login
wandb login --relogin

# Or use offline mode
WANDB_MODE=offline python train_wandb.py
```

### Import Errors

```bash
# Make sure you're in the project root
cd DT-RL

# Verify the ml package is accessible
python -c "from ml.pipeline import PipelineConfig; print('OK')"
```

### Convergence Issues

Try:
- Lower learning rate: `--lr 0.0005`
- More epochs: `--epochs 100`
- Fewer assets (simpler problem): `--num-assets 10`
- Different output mode: `--output-mode forecasting`

### Colab Session Timeout

- Enable **Runtime → Keep connected** in Colab settings
- Save checkpoints frequently (the notebook does this automatically)
- Download the model checkpoint before the session expires

---

## File Reference

| File | Purpose |
|------|---------|
| [`train_wandb.py`](train_wandb.py) | Local training script with WandB |
| [`DT_RL_Training_Colab.py`](DT_RL_Training_Colab.py) | Google Colab notebook (percent format) |
| [`ml/visualization.py`](ml/visualization.py) | Shared plotting utilities |
| [`ml/pipeline.py`](ml/pipeline.py) | Core pipeline orchestration |
| [`ml/training.py`](ml/training.py) | Training loop & optimization |
| [`ml/evaluation.py`](ml/evaluation.py) | Financial metrics computation |
| [`ml/data/synthetic_data.py`](ml/data/synthetic_data.py) | Synthetic data generation |
| [`requirements.txt`](requirements.txt) | Python dependencies |
