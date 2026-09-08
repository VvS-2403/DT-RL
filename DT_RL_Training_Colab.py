# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # 🚀 DT-RL: Dependency-Aware Trading Pipeline
#
# **Complete Training & Visualization Notebook**
#
# This notebook trains the full Dependency-Aware Reinforcement Learning pipeline
# with hypergraph cross-asset encoding, continuous regime classification, and
# a Decision Transformer. Results are logged to **Weights & Biases** and
# displayed inline with **matplotlib**.
#
# ---
#
# | Module | Purpose |
# |--------|---------|
# | Module 1 | Stock data alignment & validity masks |
# | Module 2 | Hypergraph cross-asset encoder |
# | Module 3 | Continuous regime classifier (4 regimes) |
# | Module 4 | Decision Transformer (offline RL) |
# | Module 5 | Portfolio weights / Forecasting output |

# %% [markdown]
# ## 1. 📦 Setup & Installation

# %%
# Clone repository (uncomment if running on Colab for the first time)
# !git clone https://github.com/Saatwik-ss/DT-RL.git
# %cd DT-RL

# Install dependencies
# !pip install -q torch numpy pandas matplotlib seaborn wandb tqdm scipy pyarrow pyyaml

import sys
import os

# Ensure project root is on path
if os.path.exists("/content/DT-RL"):
    sys.path.insert(0, "/content/DT-RL")
elif os.path.exists("."):
    sys.path.insert(0, ".")

print("✓ Dependencies ready")

# %%
# Core imports
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## 2. 🔑 Weights & Biases Login
#
# Create a free account at [wandb.ai](https://wandb.ai) and paste your API key below.

# %%
USE_WANDB = True  # Set to False to skip WandB logging

if USE_WANDB:
    try:
        import wandb
        wandb.login()  # Will prompt for API key in Colab
        print("✓ WandB authenticated")
    except Exception as e:
        print(f"⚠ WandB login failed: {e}")
        print("  Continuing without WandB logging.")
        USE_WANDB = False

# %% [markdown]
# ## 3. 📊 Configuration

# %%
# ═══════════════════════════════════════════
# CONFIGURATION — Edit these values as needed
# ═══════════════════════════════════════════

CONFIG = {
    # Data
    "num_assets": 20,         # Number of synthetic assets
    "num_days": 250,          # Trading days to generate
    "lookback_window": 60,    # Lookback window (days)
    "forecast_horizon": 5,    # Forecast horizon (days)
    "stride": 5,              # Stride between training sequences
    "seed": 42,

    # Model
    "output_mode": "portfolio_weights",   # "portfolio_weights" or "forecasting"
    "num_regimes": 4,
    "embedding_dim": 128,
    "num_transformer_layers": 4,
    "num_attention_heads": 8,

    # Training
    "batch_size": 16,
    "learning_rate": 0.001,
    "num_epochs": 30,         # Increase for better convergence
    "early_stopping_patience": 10,

    # WandB
    "wandb_project": "DT-RL",
}

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 4. 🏗️ Data Generation & Exploration

# %%
from ml.data.synthetic_data import (
    SyntheticMarketDataGenerator,
    SyntheticDataLoader,
)
from ml.modules.module1_input import (
    MarketTensor,
    StockDataProcessor,
    DataValidator,
)

# Generate synthetic market data
generator = SyntheticMarketDataGenerator(
    num_assets=CONFIG["num_assets"],
    num_days=CONFIG["num_days"],
    seed=CONFIG["seed"],
)
raw_df = generator.generate()
print(f"\nRaw DataFrame: {raw_df.shape}")
print(f"Columns: {list(raw_df.columns)}")
print(f"Date range: {raw_df['Date'].min()} → {raw_df['Date'].max()}")
print(f"Assets: {raw_df['Ticker'].nunique()}")
raw_df.head()

# %%
# Process into MarketTensor
processor = StockDataProcessor(lookback_window=CONFIG["lookback_window"])
market_tensor = processor.process(raw_df)

# Validate
validator = DataValidator()
report = validator.validate_market_tensor(market_tensor)

print(f"\n📋 Data Validation Report:")
print(f"  Valid: {'✓' if report['valid'] else '✗'}")
print(f"  Shape: N={report['shape']['num_assets']}, "
      f"T={report['shape']['lookback_window']}, "
      f"F={report['shape']['num_features']}")
print(f"  Coverage: {report['coverage']['overall']:.1%}")
if not report["valid"]:
    for issue in report["issues"]:
        print(f"  ⚠ {issue}")

# %% [markdown]
# ### 📈 Data Visualizations

# %%
from ml.visualization import (
    plot_price_trajectories,
    plot_coverage_heatmap,
)

# Price trajectories
features = market_tensor.features.numpy()
close_idx = 3 if features.shape[2] > 3 else 0
prices = features[:, :, close_idx]

fig_prices = plot_price_trajectories(
    prices, market_tensor.asset_ids, max_assets=10,
    title="Synthetic Market Price Trajectories",
)
plt.show()

# %%
# Coverage heatmap
fig_coverage = plot_coverage_heatmap(
    market_tensor.validity_mask.numpy(),
    market_tensor.asset_ids,
)
plt.show()

# %%
# Feature correlation matrix
fig, ax = plt.subplots(figsize=(8, 6))
feature_data = features[0, :, :].T  # First asset, all features
corr = np.corrcoef(feature_data)
feature_names = market_tensor.feature_names[:len(corr)]
sns.heatmap(corr, xticklabels=feature_names, yticklabels=feature_names,
            cmap="coolwarm", center=0, annot=True, fmt=".2f",
            ax=ax, square=True, linewidths=0.5)
ax.set_title("Feature Correlation Matrix (Asset 0)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. 🧠 Pipeline Initialization

# %%
from ml.pipeline import DependencyAwareTradingPipeline, PipelineConfig

# Create training sequences
loader = SyntheticDataLoader()
X, M, Y = loader.create_sequences(
    market_tensor,
    lookback=CONFIG["lookback_window"],
    horizon=CONFIG["forecast_horizon"],
    stride=CONFIG["stride"],
)
print(f"Training data shapes:")
print(f"  X (features):  {X.shape}  →  (sequences, assets, lookback, features)")
print(f"  M (mask):      {M.shape}  →  (sequences, assets, lookback)")
print(f"  Y (targets):   {Y.shape}  →  (sequences, assets, horizon)")

# %%
# Initialize pipeline
config = PipelineConfig(
    lookback_window=CONFIG["lookback_window"],
    forecast_horizon=CONFIG["forecast_horizon"],
    output_mode=CONFIG["output_mode"],
    batch_size=CONFIG["batch_size"],
    learning_rate=CONFIG["learning_rate"],
    num_epochs=CONFIG["num_epochs"],
    early_stopping_patience=CONFIG["early_stopping_patience"],
    num_transformer_layers=CONFIG["num_transformer_layers"],
    num_attention_heads=CONFIG["num_attention_heads"],
    device=device,
)

pipeline = DependencyAwareTradingPipeline(config).to(device)

# Initialize modules
dummy_mt = MarketTensor(
    features=torch.zeros(X.shape[1], X.shape[2], X.shape[3]),
    validity_mask=torch.ones(X.shape[1], X.shape[2]),
    asset_ids=[f"ASSET_{i}" for i in range(X.shape[1])],
    timestamps=np.arange(X.shape[2]),
    feature_names=[f"F{i}" for i in range(X.shape[3])],
    feature_means=torch.zeros(X.shape[3]),
    feature_stds=torch.ones(X.shape[3]),
)
pipeline.initialize_modules(dummy_mt)

# Model summary
print(f"\n🧠 Model Architecture:")
total_params = 0
param_counts = {}
for name, module in [("Module2_Encoder", pipeline.module2),
                     ("Module3_Regime", pipeline.module3),
                     ("Module4_Transformer", pipeline.module4),
                     ("Module5_Output", pipeline.module5)]:
    count = sum(p.numel() for p in module.parameters())
    param_counts[name] = count
    total_params += count
    print(f"  {name}: {count:,} parameters")
print(f"  {'─'*40}")
print(f"  Total: {total_params:,} parameters")

# %%
# Visualize parameter distribution
from ml.visualization import plot_model_summary

fig_params = plot_model_summary(param_counts)
plt.show()

# %% [markdown]
# ## 6. 🏋️ Training

# %%
# Initialize WandB run
if USE_WANDB:
    import wandb
    run = wandb.init(
        project=CONFIG["wandb_project"],
        name=f"colab_{CONFIG['output_mode']}_{datetime.now().strftime('%H%M%S')}",
        config=CONFIG,
        reinit=True,
    )
    print(f"📊 WandB Dashboard: {run.url}")

# %%
import time
from ml.training import SequenceTrainer
from torch.utils.data import DataLoader, TensorDataset

# Split data chronologically
n_total = X.shape[0]
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.2)

X_train, M_train, Y_train = X[:n_train], M[:n_train], Y[:n_train]
X_val, M_val, Y_val = X[n_train:n_train+n_val], M[n_train:n_train+n_val], Y[n_train:n_train+n_val]
X_test, M_test, Y_test = X[n_train+n_val:], M[n_train+n_val:], Y[n_train+n_val:]

train_loader = DataLoader(TensorDataset(X_train, M_train, Y_train),
                          batch_size=CONFIG["batch_size"], shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, M_val, Y_val),
                        batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, M_test, Y_test),
                         batch_size=CONFIG["batch_size"], shuffle=False)

print(f"Data split: Train={n_train}, Val={n_val}, Test={n_total-n_train-n_val}")

# %%
# Train!
trainer = SequenceTrainer(pipeline, config, checkpoint_dir=Path("checkpoints"))
history = {"train_loss": [], "val_loss": [], "test_loss": [], "lr": []}

print(f"\n{'═'*60}")
print(f"Training for {CONFIG['num_epochs']} epochs...")
print(f"{'═'*60}\n")

for epoch in range(CONFIG["num_epochs"]):
    t0 = time.time()

    train_loss = trainer.train_epoch(train_loader)
    val_loss = trainer.validate(val_loader)
    test_loss = trainer.validate(test_loader)
    lr = trainer.optimizer.param_groups[0]["lr"]
    elapsed = time.time() - t0

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["test_loss"].append(test_loss)
    history["lr"].append(lr)

    print(f"  Epoch {epoch+1:3d}/{CONFIG['num_epochs']}: "
          f"Train={train_loss:.4f} | Val={val_loss:.4f} | "
          f"Test={test_loss:.4f} | LR={lr:.6f} | {elapsed:.1f}s")

    # WandB logging
    if USE_WANDB:
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "test/loss": test_loss,
            "train/learning_rate": lr,
            "train/epoch_time": elapsed,
        }, step=epoch + 1)

    # LR scheduler
    trainer.scheduler.step(val_loss)

    # Early stopping
    if val_loss < trainer.best_val_loss:
        trainer.best_val_loss = val_loss
        trainer.patience_counter = 0
        trainer._save_checkpoint(epoch, val_loss)
    else:
        trainer.patience_counter += 1
        if trainer.patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break

# Load best checkpoint
best_ckpt = list(Path("checkpoints").glob("best_*.pt"))
if best_ckpt:
    pipeline.load_state_dict(
        torch.load(best_ckpt[0], map_location=device)["state_dict"]
    )
    print(f"\n✓ Loaded best checkpoint: {best_ckpt[0].name}")

print(f"\n{'═'*60}")
print(f"Training complete! Best val loss: {min(history['val_loss']):.4f}")
print(f"{'═'*60}")

# %% [markdown]
# ## 7. 📉 Training Results

# %%
from ml.visualization import plot_loss_curves

fig_loss = plot_loss_curves(history, title="DT-RL Training Loss Curves")
plt.show()

if USE_WANDB:
    wandb.log({"results/loss_curves": wandb.Image(fig_loss)})

# %%
# Learning rate schedule
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, len(history["lr"]) + 1), history["lr"],
        color="#7B1FA2", linewidth=2, marker="o", markersize=3)
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. 🎯 Regime Analysis

# %%
# Run inference for analysis
pipeline.eval()
with torch.no_grad():
    sample_X = X[:1].to(device)
    sample_M = M[:1].to(device)
    sample_Y = Y[:1].to(device)

    rtg = sample_Y.sum(dim=-1, keepdim=True).unsqueeze(-1).expand(
        -1, -1, CONFIG["lookback_window"], -1
    )
    outputs = pipeline(sample_X, sample_M, rtg)

regime_probs = outputs["regime_probs"].squeeze(0).cpu().numpy()  # (N, T, 4)
asset_ids = market_tensor.asset_ids

print(f"Regime probabilities shape: {regime_probs.shape}")
print(f"  → {regime_probs.shape[0]} assets × {regime_probs.shape[1]} timesteps × "
      f"{regime_probs.shape[2]} regimes")

# %%
from ml.visualization import (
    plot_regime_evolution,
    plot_regime_distribution,
    plot_regime_transitions,
)

# Regime evolution for first 3 assets
for idx in range(min(3, len(asset_ids))):
    fig = plot_regime_evolution(regime_probs, asset_ids, asset_idx=idx)
    plt.show()
    if USE_WANDB:
        wandb.log({f"results/regime_evolution_asset{idx}": wandb.Image(fig)})

# %%
# Regime distribution
fig_dist = plot_regime_distribution(regime_probs, asset_ids)
plt.show()
if USE_WANDB:
    wandb.log({"results/regime_distribution": wandb.Image(fig_dist)})

# %%
# Regime transitions
from ml.modules.module3_regime import RegimeVisualizer

transitions = RegimeVisualizer.compute_regime_transitions(
    outputs["regime_probs"].squeeze(0),
)
fig_trans = plot_regime_transitions(np.array(transitions["transition_matrix"]))
plt.show()
if USE_WANDB:
    wandb.log({"results/regime_transitions": wandb.Image(fig_trans)})

# %% [markdown]
# ## 9. 💼 Portfolio & Dependency Analysis

# %%
from ml.visualization import (
    plot_portfolio_weights,
    plot_equity_curve,
    plot_dependency_matrix,
)
from ml.modules.module2_encoder import DependencyAnalyzer

# Cross-asset dependency matrix
Z = outputs["encoder_embeddings"].squeeze(0).cpu()  # (N, T, D)
M_single = sample_M.squeeze(0).cpu()
dep_matrix = DependencyAnalyzer.compute_dependency_matrix(Z, M_single)

fig_deps = plot_dependency_matrix(dep_matrix.numpy(), asset_ids)
plt.show()
if USE_WANDB:
    wandb.log({"results/dependency_matrix": wandb.Image(fig_deps)})

# %%
# Portfolio weights (only for portfolio_weights mode)
if CONFIG["output_mode"] == "portfolio_weights":
    weights = outputs["output"].weights.cpu().numpy()
    N = regime_probs.shape[0]
    T = CONFIG["lookback_window"]
    weights_reshaped = weights.reshape(1, N, T, N)
    last_weights = weights_reshaped[0, :, -1, :]  # (N, N)

    fig_w = plot_portfolio_weights(last_weights, asset_ids)
    plt.show()
    if USE_WANDB:
        wandb.log({"results/portfolio_weights": wandb.Image(fig_w)})

    # Top allocations
    avg_weights = np.abs(last_weights).mean(axis=0)
    sorted_idx = np.argsort(avg_weights)[::-1]
    print("\n📊 Top 10 Average Absolute Allocations:")
    for rank, idx in enumerate(sorted_idx[:10]):
        name = asset_ids[idx] if idx < len(asset_ids) else f"Asset_{idx}"
        print(f"  {rank+1}. {name}: {avg_weights[idx]:.4f}")

# %%
# Equity curve (simulated from target returns)
avg_returns = Y[:, :, 0].mean(dim=1).numpy()
if len(avg_returns) > 5:
    fig_eq = plot_equity_curve(avg_returns,
                               title="Simulated Portfolio Equity Curve")
    plt.show()
    if USE_WANDB:
        wandb.log({"results/equity_curve": wandb.Image(fig_eq)})

# %% [markdown]
# ## 10. 📋 Evaluation Metrics

# %%
from ml.evaluation import evaluate_predictions, compute_financial_metrics
from ml.visualization import plot_metrics_table

# Evaluate
eval_metrics = evaluate_predictions(outputs, sample_Y, CONFIG["output_mode"])
eval_metrics["final_train_loss"] = history["train_loss"][-1]
eval_metrics["final_val_loss"] = history["val_loss"][-1]
eval_metrics["best_val_loss"] = min(history["val_loss"])
eval_metrics["total_epochs"] = len(history["train_loss"])

fig_met = plot_metrics_table(eval_metrics,
                              title="DT-RL Performance Metrics")
plt.show()

if USE_WANDB:
    wandb.log({"results/metrics_table": wandb.Image(fig_met)})
    for k, v in eval_metrics.items():
        wandb.summary[f"eval/{k}"] = v

print("\n📋 Full Metrics:")
for k, v in eval_metrics.items():
    print(f"  {k}: {v:.4f}")

# %% [markdown]
# ## 11. 📊 Training Summary

# %%
from ml.visualization import plot_training_summary

fig_summary = plot_training_summary(
    history,
    eval_metrics,
    regime_probs,
    avg_returns if len(avg_returns) > 5 else None,
)
plt.show()
if USE_WANDB:
    wandb.log({"results/training_summary": wandb.Image(fig_summary)})

# %% [markdown]
# ## 12. 💾 Save & Finish

# %%
# Save model checkpoint
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / "dtrl_colab_model.pt"
pipeline.save(checkpoint_path)
print(f"✓ Model saved to {checkpoint_path}")

# Save to WandB
if USE_WANDB:
    artifact = wandb.Artifact(
        name="dtrl-colab-model",
        type="model",
        description="DT-RL pipeline trained in Colab",
    )
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact)
    wandb.finish()
    print("✓ WandB run finished and artifact uploaded")

# %%
print(f"\n{'═'*60}")
print("🎉 ALL DONE!")
print(f"{'═'*60}")
print(f"""
Summary:
  • Trained for {len(history['train_loss'])} epochs
  • Best validation loss: {min(history['val_loss']):.4f}
  • Final test loss: {history['test_loss'][-1]:.4f}
  • Model saved to: {checkpoint_path}
  • Output mode: {CONFIG['output_mode']}

To load the trained model:
  from ml.pipeline import DependencyAwareTradingPipeline
  pipeline = DependencyAwareTradingPipeline.load("{checkpoint_path}")
""")
