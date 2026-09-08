#!/usr/bin/env python3
"""
WandB-Integrated Training Script for DT-RL Pipeline.

Trains the full Dependency-Aware Trading Pipeline with comprehensive
Weights & Biases logging, including:
  - Per-epoch loss metrics
  - Regime distribution histograms
  - Portfolio weight heatmaps
  - Financial performance metrics (Sharpe, Sortino, Max Drawdown)
  - Training summary figures
  - Model checkpoint artifacts

Usage:
    python train_wandb.py --epochs 50 --batch-size 16 --num-assets 20
    python train_wandb.py --epochs 100 --output-mode forecasting --wandb-project my-project
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Fix Windows DLL loading issues with torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script
import matplotlib.pyplot as plt

# ── Ensure project root is on path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.synthetic_data import SyntheticDataLoader, SyntheticMarketDataGenerator
from ml.pipeline import DependencyAwareTradingPipeline, PipelineConfig
from ml.training import SequenceTrainer
from ml.evaluation import evaluate_predictions, compute_financial_metrics
from ml.modules.module1_input import MarketTensor, StockDataProcessor, DataValidator
from ml.modules.module2_encoder import DependencyAnalyzer
from ml.modules.module3_regime import RegimeVisualizer
from ml.visualization import (
    plot_loss_curves,
    plot_regime_evolution,
    plot_regime_distribution,
    plot_regime_transitions,
    plot_portfolio_weights,
    plot_equity_curve,
    plot_dependency_matrix,
    plot_metrics_table,
    plot_training_summary,
    plot_price_trajectories,
    plot_coverage_heatmap,
    plot_model_summary,
    log_figures_to_wandb,
    close_figures,
)


# ════════════════════════════════════════════════════════════════════════════
# WandB-Instrumented Trainer
# ════════════════════════════════════════════════════════════════════════════

class WandBSequenceTrainer(SequenceTrainer):
    """Extends SequenceTrainer with WandB logging per epoch."""

    def __init__(self, pipeline, config, checkpoint_dir=None, use_wandb=True):
        super().__init__(pipeline, config, checkpoint_dir)
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("[WARN] wandb not installed, falling back to console logging")
                self.use_wandb = False

    def fit(self, X, M, Y, validation_split=0.2, test_split=0.1):
        """Override fit to add per-epoch WandB logging."""
        from torch.utils.data import DataLoader, TensorDataset

        # Chronological split
        n_total = X.shape[0]
        n_train = int(n_total * (1 - validation_split - test_split))
        n_val = int(n_total * validation_split)

        self._check_leakage(X, M, Y, n_train, n_val)

        X_train, M_train, Y_train = X[:n_train], M[:n_train], Y[:n_train]
        X_val, M_val, Y_val = (X[n_train:n_train+n_val],
                                M[n_train:n_train+n_val],
                                Y[n_train:n_train+n_val])
        X_test, M_test, Y_test = (X[n_train+n_val:],
                                   M[n_train+n_val:],
                                   Y[n_train+n_val:])

        train_loader = DataLoader(
            TensorDataset(X_train, M_train, Y_train),
            batch_size=self.config.batch_size, shuffle=False,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, M_val, Y_val),
            batch_size=self.config.batch_size, shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(X_test, M_test, Y_test),
            batch_size=self.config.batch_size, shuffle=False,
        )

        print(f"Training set: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        history = {"train_loss": [], "val_loss": [], "test_loss": [], "lr": []}

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            test_loss = self.validate(test_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["test_loss"].append(test_loss)
            history["lr"].append(current_lr)

            print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                  f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
                  f"Test={test_loss:.4f}, LR={current_lr:.6f}, "
                  f"Time={epoch_time:.1f}s")

            # ── WandB logging ───────────────────────────────────────────
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "test/loss": test_loss,
                    "train/learning_rate": current_lr,
                    "train/epoch_time_s": epoch_time,
                }
                self.wandb.log(log_dict, step=epoch + 1)

            # LR scheduler
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best checkpoint
        best_checkpoint = list(self.checkpoint_dir.glob("best_*.pt"))
        if best_checkpoint:
            self.pipeline.load_state_dict(
                torch.load(best_checkpoint[0], map_location=self.device)["state_dict"]
            )
            print(f"Loaded best checkpoint from {best_checkpoint[0]}")

        return history


# ════════════════════════════════════════════════════════════════════════════
# Main Training Function
# ════════════════════════════════════════════════════════════════════════════

def train_with_wandb(args):
    """Run full training pipeline with WandB logging and visualization."""

    # ── 1. Initialize WandB ─────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity if args.wandb_entity else None,
                name=args.run_name,
                config={
                    "num_assets": args.num_assets,
                    "num_days": args.num_days,
                    "lookback_window": args.lookback,
                    "forecast_horizon": args.horizon,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "num_epochs": args.epochs,
                    "output_mode": args.output_mode,
                    "num_regimes": 4,
                    "embedding_dim": 128,
                    "num_transformer_layers": args.num_layers,
                    "num_attention_heads": args.num_heads,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                },
            )
            print(f"[OK] WandB initialized: {wandb.run.url}")
        except Exception as e:
            print(f"[WARN] WandB init failed: {e}. Continuing without WandB.")
            use_wandb = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"DT-RL Training Pipeline (device: {device})")
    print(f"{'='*70}\n")

    # ── 2. Generate Synthetic Data ──────────────────────────────────────
    print("[1/6] Generating synthetic market data...")
    generator = SyntheticMarketDataGenerator(
        num_assets=args.num_assets,
        num_days=args.num_days,
        seed=args.seed,
    )
    raw_df = generator.generate()

    processor = StockDataProcessor(lookback_window=args.lookback)
    full_market_tensor = processor.process(raw_df)

    # Validate
    validator = DataValidator()
    report = validator.validate_market_tensor(full_market_tensor)
    print(f"  Data valid: {report['valid']}, "
          f"Coverage: {report['coverage']['overall']:.1%}")
    if not report["valid"]:
        for issue in report["issues"]:
            print(f"  [WARN] {issue}")

    # Create sequences
    loader = SyntheticDataLoader()
    X, M, Y = loader.create_sequences(
        full_market_tensor,
        lookback=args.lookback,
        horizon=args.horizon,
        stride=args.stride,
    )
    print(f"  X: {X.shape}, M: {M.shape}, Y: {Y.shape}")

    # ── 3. Visualize Data ───────────────────────────────────────────────
    print("\n[2/6] Generating data visualizations...")
    figures = {}

    # Price trajectories
    prices = full_market_tensor.features[:, :, 3].numpy()  # Close prices
    fig_prices = plot_price_trajectories(
        prices, full_market_tensor.asset_ids, max_assets=10,
    )
    figures["data/price_trajectories"] = fig_prices

    # Coverage heatmap
    fig_coverage = plot_coverage_heatmap(
        full_market_tensor.validity_mask.numpy(),
        full_market_tensor.asset_ids,
    )
    figures["data/coverage_heatmap"] = fig_coverage

    if use_wandb:
        log_figures_to_wandb(figures, step=0)
    print("  [OK] Data plots generated")

    # ── 4. Initialize Pipeline ──────────────────────────────────────────
    print("\n[3/6] Initializing pipeline...")
    config = PipelineConfig(
        lookback_window=args.lookback,
        forecast_horizon=args.horizon,
        output_mode=args.output_mode,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        num_transformer_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        device=device,
    )

    pipeline = DependencyAwareTradingPipeline(config).to(device)

    # Initialize modules with data shape
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

    # Log model summary
    param_counts = {}
    for name, module in [("Module2_Encoder", pipeline.module2),
                         ("Module3_Regime", pipeline.module3),
                         ("Module4_Transformer", pipeline.module4),
                         ("Module5_Output", pipeline.module5)]:
        count = sum(p.numel() for p in module.parameters())
        param_counts[name] = count
        print(f"  {name}: {count:,} parameters")

    total_params = sum(param_counts.values())
    print(f"  Total: {total_params:,} parameters")

    fig_params = plot_model_summary(param_counts)
    if use_wandb:
        log_figures_to_wandb({"model/parameter_distribution": fig_params})

    if use_wandb:
        import wandb
        wandb.log({"model/total_parameters": total_params})

    # ── 5. Train ────────────────────────────────────────────────────────
    print(f"\n[4/6] Training ({config.num_epochs} epochs)...")
    trainer = WandBSequenceTrainer(
        pipeline, config,
        checkpoint_dir=Path("checkpoints"),
        use_wandb=use_wandb,
    )
    history = trainer.fit(X, M, Y)

    # ── 6. Post-Training Analysis ───────────────────────────────────────
    print("\n[5/6] Generating post-training visualizations...")
    post_figures = {}

    # Loss curves
    fig_loss = plot_loss_curves(history)
    post_figures["results/loss_curves"] = fig_loss

    # Run inference for analysis
    pipeline.eval()
    with torch.no_grad():
        test_X = X[:1].to(device)
        test_M = M[:1].to(device)
        test_Y = Y[:1].to(device)

        rtg = test_Y.sum(dim=-1, keepdim=True).unsqueeze(-1).expand(
            -1, -1, args.lookback, -1
        )
        outputs = pipeline(test_X, test_M, rtg)

    # Regime analysis
    regime_probs = outputs["regime_probs"].squeeze(0).cpu().numpy()  # (N, T, 4)
    asset_ids = full_market_tensor.asset_ids

    fig_regime_evo = plot_regime_evolution(regime_probs, asset_ids, asset_idx=0)
    post_figures["results/regime_evolution_asset0"] = fig_regime_evo

    fig_regime_dist = plot_regime_distribution(regime_probs, asset_ids)
    post_figures["results/regime_distribution"] = fig_regime_dist

    # Regime transitions
    regime_vis = RegimeVisualizer()
    transitions = regime_vis.compute_regime_transitions(
        outputs["regime_probs"].squeeze(0),
    )
    fig_transitions = plot_regime_transitions(
        np.array(transitions["transition_matrix"]),
    )
    post_figures["results/regime_transitions"] = fig_transitions

    # Dependency matrix
    Z = outputs["encoder_embeddings"].squeeze(0).cpu()  # (N, T, D)
    M_single = test_M.squeeze(0).cpu()
    dep_matrix = DependencyAnalyzer.compute_dependency_matrix(Z, M_single)
    fig_deps = plot_dependency_matrix(dep_matrix.numpy(), asset_ids)
    post_figures["results/dependency_matrix"] = fig_deps

    # Portfolio / Forecasting analysis
    if config.output_mode == "portfolio_weights":
        weights = outputs["output"].weights.cpu().numpy()  # (batch*N, T, N)
        N = regime_probs.shape[0]
        weights_reshaped = weights.reshape(1, N, args.lookback, N)
        last_weights = weights_reshaped[0, :, -1, :]  # (N, N)

        fig_weights = plot_portfolio_weights(last_weights, asset_ids)
        post_figures["results/portfolio_weights"] = fig_weights

        # Simulate simple returns for equity curve
        avg_returns = Y[:, :, 0].mean(dim=1).numpy()  # (num_seq,)
        if len(avg_returns) > 5:
            fig_equity = plot_equity_curve(avg_returns)
            post_figures["results/equity_curve"] = fig_equity

    # Evaluate metrics
    eval_outputs = outputs.copy()
    eval_metrics = evaluate_predictions(eval_outputs, test_Y, config.output_mode)
    eval_metrics["final_train_loss"] = history["train_loss"][-1]
    eval_metrics["final_val_loss"] = history["val_loss"][-1]
    eval_metrics["best_val_loss"] = min(history["val_loss"])
    eval_metrics["total_epochs"] = len(history["train_loss"])

    fig_metrics = plot_metrics_table(eval_metrics)
    post_figures["results/metrics_table"] = fig_metrics

    print("\n  Final Metrics:")
    for k, v in eval_metrics.items():
        print(f"    {k}: {v:.4f}")

    # Training summary
    avg_returns_for_summary = Y[:, :, 0].mean(dim=1).numpy() if Y.shape[0] > 5 else None
    fig_summary = plot_training_summary(
        history, eval_metrics, regime_probs, avg_returns_for_summary,
    )
    post_figures["results/training_summary"] = fig_summary

    # Log all figures to WandB
    if use_wandb:
        log_figures_to_wandb(post_figures)

        # Log metrics
        import wandb
        for k, v in eval_metrics.items():
            wandb.summary[f"eval/{k}"] = v

    # ── 7. Save & Finish ────────────────────────────────────────────────
    print("\n[6/6] Saving model checkpoint...")
    checkpoint_path = Path("checkpoints") / f"dtrl_{args.run_name}.pt"
    pipeline.save(checkpoint_path)

    if use_wandb:
        import wandb
        artifact = wandb.Artifact(
            name=f"dtrl-model-{args.run_name}",
            type="model",
            description="Trained DT-RL pipeline checkpoint",
        )
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
        wandb.finish()
        print("  [OK] WandB run finished and artifact uploaded")

    # Save figures locally
    fig_dir = Path("results") / args.run_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in {**figures, **post_figures}.items():
        safe_name = name.replace("/", "_") + ".png"
        fig.savefig(fig_dir / safe_name, dpi=150, bbox_inches="tight")
    print(f"  [OK] Figures saved to {fig_dir}/")

    # Clean up
    for fig in list(figures.values()) + list(post_figures.values()):
        plt.close(fig)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")

    return pipeline, history, eval_metrics


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DT-RL Pipeline with WandB logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--num-assets", type=int, default=20,
                        help="Number of synthetic assets")
    parser.add_argument("--num-days", type=int, default=250,
                        help="Number of trading days to generate")
    parser.add_argument("--lookback", type=int, default=60,
                        help="Lookback window size")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forecast horizon")
    parser.add_argument("--stride", type=int, default=5,
                        help="Stride between training sequences")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Model
    parser.add_argument("--output-mode", type=str, default="portfolio_weights",
                        choices=["portfolio_weights", "forecasting"],
                        help="Output mode")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")

    # WandB
    parser.add_argument("--wandb-project", type=str, default="DT-RL",
                        help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="WandB entity (team/user)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="WandB run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging")

    args = parser.parse_args()

    # Auto-generate run name
    if args.run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.run_name = f"dtrl_{args.output_mode}_{args.num_assets}a_{timestamp}"

    return args


if __name__ == "__main__":
    args = parse_args()
    train_with_wandb(args)
