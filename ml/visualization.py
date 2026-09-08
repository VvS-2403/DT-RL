"""
Visualization utilities for Dependency-Aware Trading System.

Provides matplotlib-based plotting functions for:
- Training loss curves
- Regime evolution and transitions
- Portfolio weight heatmaps
- Equity curves with drawdown shading
- Cross-asset dependency matrices
- Multi-panel training summaries

All functions return matplotlib Figure objects for flexible use
in notebooks, scripts, and WandB logging.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Use non-interactive backend when not in notebook
try:
    get_ipython()  # noqa: F821
except NameError:
    matplotlib.use("Agg")


# ── Style defaults ──────────────────────────────────────────────────────────

COLORS = {
    "train": "#2196F3",
    "val": "#FF9800",
    "test": "#4CAF50",
    "regime_0": "#E53935",
    "regime_1": "#1E88E5",
    "regime_2": "#43A047",
    "regime_3": "#FDD835",
    "equity": "#1565C0",
    "drawdown": "#EF5350",
    "benchmark": "#9E9E9E",
}

REGIME_NAMES = [
    "R1: Long Mem + Long Credit",
    "R2: Long Mem + Short Credit",
    "R3: Short Mem + Long Credit",
    "R4: Short Mem + Short Credit",
]

REGIME_COLORS = [COLORS["regime_0"], COLORS["regime_1"],
                 COLORS["regime_2"], COLORS["regime_3"]]


def _apply_style(ax: plt.Axes, title: str = "", xlabel: str = "",
                 ylabel: str = "", grid: bool = True):
    """Apply consistent styling to an axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if grid:
        ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Loss Curves ─────────────────────────────────────────────────────────────

def plot_loss_curves(
    history: Dict[str, List[float]],
    title: str = "Training Loss Curves",
) -> plt.Figure:
    """
    Plot train/val/test loss curves.

    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'test_loss'
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], color=COLORS["train"],
            linewidth=2, label="Train Loss", marker="o", markersize=4)
    ax.plot(epochs, history["val_loss"], color=COLORS["val"],
            linewidth=2, label="Val Loss", marker="s", markersize=4)
    if "test_loss" in history and history["test_loss"]:
        ax.plot(epochs, history["test_loss"], color=COLORS["test"],
                linewidth=2, label="Test Loss", marker="^", markersize=4)

    # Mark best validation epoch
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val = min(history["val_loss"])
    ax.axvline(x=best_epoch, color=COLORS["val"], linestyle="--", alpha=0.5)
    ax.annotate(f"Best val: {best_val:.4f}\n(epoch {best_epoch})",
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 0.5, best_val * 1.1),
                fontsize=9, color=COLORS["val"],
                arrowprops=dict(arrowstyle="->", color=COLORS["val"]))

    ax.legend(fontsize=11, framealpha=0.9)
    _apply_style(ax, title=title, xlabel="Epoch", ylabel="Loss")

    fig.tight_layout()
    return fig


# ── Regime Evolution ────────────────────────────────────────────────────────

def plot_regime_evolution(
    regime_probs: np.ndarray,
    asset_ids: Optional[List[str]] = None,
    asset_idx: int = 0,
    title: str = "Regime Probability Evolution",
) -> plt.Figure:
    """
    Plot regime probabilities over time as a stacked area chart for one asset.

    Args:
        regime_probs: (N, T, 4) regime probabilities
        asset_ids: List of asset names
        asset_idx: Which asset to plot
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    probs = regime_probs[asset_idx]  # (T, 4)
    T, K = probs.shape
    timesteps = np.arange(T)

    ax.stackplot(timesteps, probs.T,
                 labels=REGIME_NAMES[:K],
                 colors=REGIME_COLORS[:K],
                 alpha=0.8)

    asset_name = asset_ids[asset_idx] if asset_ids else f"Asset {asset_idx}"
    _apply_style(ax, title=f"{title} — {asset_name}",
                 xlabel="Timestep", ylabel="Probability")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, T - 1)

    fig.tight_layout()
    return fig


def plot_regime_distribution(
    regime_probs: np.ndarray,
    asset_ids: Optional[List[str]] = None,
    title: str = "Regime Distribution (Last Timestep)",
) -> plt.Figure:
    """
    Plot regime assignments across all assets at the last timestep.

    Args:
        regime_probs: (N, T, 4) regime probabilities
        asset_ids: List of asset names
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Last-timestep probabilities
    P_last = regime_probs[:, -1, :]  # (N, 4)
    N, K = P_last.shape

    # Bar chart of regime assignments
    assignments = P_last.argmax(axis=1)
    regime_counts = [np.sum(assignments == k) for k in range(K)]

    axes[0].bar(range(K), regime_counts, color=REGIME_COLORS[:K], alpha=0.85,
                edgecolor="white", linewidth=1.5)
    axes[0].set_xticks(range(K))
    axes[0].set_xticklabels([f"R{k+1}" for k in range(K)], fontsize=9)
    _apply_style(axes[0], title="Regime Assignment Counts",
                 xlabel="Regime", ylabel="Count")

    # Heatmap of probabilities
    labels = asset_ids[:min(N, 20)] if asset_ids else [f"A{i}" for i in range(min(N, 20))]
    im = axes[1].imshow(P_last[:min(N, 20)], aspect="auto", cmap="YlOrRd",
                        interpolation="nearest")
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_xticks(range(K))
    axes[1].set_xticklabels([f"R{k+1}" for k in range(K)])
    _apply_style(axes[1], title="Regime Probabilities per Asset", grid=False)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ── Regime Transitions ──────────────────────────────────────────────────────

def plot_regime_transitions(
    transition_matrix: np.ndarray,
    title: str = "Regime Transition Probabilities",
) -> plt.Figure:
    """
    Plot regime transition matrix as a heatmap.

    Args:
        transition_matrix: (4, 4) transition probability matrix
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    K = transition_matrix.shape[0]
    im = ax.imshow(transition_matrix, cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest")

    # Annotate cells
    for i in range(K):
        for j in range(K):
            val = transition_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    labels = [f"R{k+1}" for k in range(K)]
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(K))
    ax.set_yticklabels(labels)
    _apply_style(ax, title=title, xlabel="To Regime", ylabel="From Regime",
                 grid=False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability")

    fig.tight_layout()
    return fig


# ── Portfolio Weights ───────────────────────────────────────────────────────

def plot_portfolio_weights(
    weights: np.ndarray,
    asset_ids: Optional[List[str]] = None,
    title: str = "Portfolio Weight Allocation",
) -> plt.Figure:
    """
    Plot portfolio weights as a heatmap over time.

    Args:
        weights: (T, N) or (N,) portfolio weights
        asset_ids: List of asset names
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    N = weights.shape[-1]
    labels = asset_ids[:N] if asset_ids else [f"Asset_{i}" for i in range(N)]

    im = ax.imshow(weights.T, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-0.15, vmax=0.15)
    ax.set_yticks(range(min(N, 30)))
    ax.set_yticklabels(labels[:30], fontsize=7)
    _apply_style(ax, title=title, xlabel="Timestep", ylabel="Asset",
                 grid=False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Weight")

    fig.tight_layout()
    return fig


# ── Equity Curve ────────────────────────────────────────────────────────────

def plot_equity_curve(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    title: str = "Portfolio Equity Curve",
    initial_capital: float = 100000.0,
) -> plt.Figure:
    """
    Plot equity curve with drawdown shading.

    Args:
        returns: (T,) portfolio returns
        benchmark_returns: (T,) optional benchmark returns
        title: Plot title
        initial_capital: Starting capital

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    height_ratios=[3, 1],
                                    sharex=True)

    # Equity curve
    equity = initial_capital * np.cumprod(1.0 + returns)
    ax1.plot(equity, color=COLORS["equity"], linewidth=2, label="Portfolio")

    if benchmark_returns is not None:
        bench_equity = initial_capital * np.cumprod(1.0 + benchmark_returns)
        ax1.plot(bench_equity, color=COLORS["benchmark"],
                 linewidth=1.5, linestyle="--", label="Benchmark")

    ax1.fill_between(range(len(equity)), initial_capital, equity,
                     where=equity >= initial_capital,
                     alpha=0.1, color=COLORS["equity"])
    ax1.axhline(y=initial_capital, color="gray", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=10, framealpha=0.9)
    _apply_style(ax1, title=title, ylabel="Portfolio Value ($)")

    # Drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    ax2.fill_between(range(len(drawdown)), 0, drawdown,
                     color=COLORS["drawdown"], alpha=0.4)
    ax2.plot(drawdown, color=COLORS["drawdown"], linewidth=1)
    _apply_style(ax2, xlabel="Trading Day", ylabel="Drawdown (%)")
    ax2.set_ylim(min(drawdown.min() * 1.2, -1), 1)

    fig.tight_layout()
    return fig


# ── Dependency Matrix ───────────────────────────────────────────────────────

def plot_dependency_matrix(
    dep_matrix: np.ndarray,
    asset_ids: Optional[List[str]] = None,
    title: str = "Cross-Asset Dependency Matrix",
) -> plt.Figure:
    """
    Plot cross-asset dependency/correlation matrix as a heatmap.

    Args:
        dep_matrix: (N, N) dependency matrix
        asset_ids: List of asset names
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    N = dep_matrix.shape[0]
    labels = asset_ids[:N] if asset_ids else [f"A{i}" for i in range(N)]

    im = ax.imshow(dep_matrix, cmap="coolwarm", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(min(N, 20)))
    ax.set_xticklabels(labels[:20], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(min(N, 20)))
    ax.set_yticklabels(labels[:20], fontsize=7)
    _apply_style(ax, title=title, grid=False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Similarity")

    fig.tight_layout()
    return fig


# ── Financial Metrics Table ─────────────────────────────────────────────────

def plot_metrics_table(
    metrics: Dict[str, float],
    title: str = "Portfolio Performance Metrics",
) -> plt.Figure:
    """
    Render financial metrics as a styled table figure.

    Args:
        metrics: Dictionary of metric_name -> value
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, max(3, len(metrics) * 0.45)))
    ax.axis("off")

    # Format values
    rows = []
    for key, val in metrics.items():
        display_name = key.replace("_", " ").title()
        if "return" in key or "drawdown" in key or "cost" in key or "turnover" in key:
            display_val = f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
        elif "ratio" in key:
            display_val = f"{val:.3f}"
        else:
            display_val = f"{val:.4f}"
        rows.append([display_name, display_val])

    table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     cellLoc="center", loc="center",
                     colColours=["#E3F2FD", "#E3F2FD"])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for j in range(2):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(2):
            table[i, j].set_facecolor(color)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# ── Training Summary ────────────────────────────────────────────────────────

def plot_training_summary(
    history: Dict[str, List[float]],
    metrics: Optional[Dict[str, float]] = None,
    regime_probs: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Multi-panel training summary figure.

    Args:
        history: Training history dictionary
        metrics: Optional financial metrics
        regime_probs: Optional (N, T, 4) regime probabilities
        returns: Optional (T,) portfolio returns

    Returns:
        matplotlib Figure
    """
    n_panels = 2
    if regime_probs is not None:
        n_panels += 1
    if returns is not None:
        n_panels += 1

    fig = plt.figure(figsize=(16, 4 * n_panels))
    gs = gridspec.GridSpec(n_panels, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Loss curves (spans full width)
    ax_loss = fig.add_subplot(gs[0, :])
    epochs = range(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"], color=COLORS["train"],
                 linewidth=2, label="Train", marker="o", markersize=3)
    ax_loss.plot(epochs, history["val_loss"], color=COLORS["val"],
                 linewidth=2, label="Val", marker="s", markersize=3)
    if "test_loss" in history and history["test_loss"]:
        ax_loss.plot(epochs, history["test_loss"], color=COLORS["test"],
                     linewidth=2, label="Test", marker="^", markersize=3)
    ax_loss.legend(fontsize=10)
    _apply_style(ax_loss, title="Training Loss Curves",
                 xlabel="Epoch", ylabel="Loss")

    panel_idx = 1

    # Panel 2: Regime evolution
    if regime_probs is not None:
        ax_regime = fig.add_subplot(gs[panel_idx, :])
        probs = regime_probs[0]  # First asset
        T, K = probs.shape
        ax_regime.stackplot(range(T), probs.T,
                           labels=REGIME_NAMES[:K],
                           colors=REGIME_COLORS[:K], alpha=0.8)
        ax_regime.legend(loc="upper left", fontsize=8)
        ax_regime.set_ylim(0, 1)
        _apply_style(ax_regime, title="Regime Evolution (Asset 0)",
                     xlabel="Timestep", ylabel="Probability")
        panel_idx += 1

    # Panel 3: Equity curve
    if returns is not None:
        ax_eq = fig.add_subplot(gs[panel_idx, 0])
        equity = 100000 * np.cumprod(1.0 + returns)
        ax_eq.plot(equity, color=COLORS["equity"], linewidth=2)
        ax_eq.axhline(y=100000, color="gray", linestyle=":", alpha=0.5)
        _apply_style(ax_eq, title="Equity Curve",
                     xlabel="Day", ylabel="Value ($)")

        ax_dd = fig.add_subplot(gs[panel_idx, 1])
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        ax_dd.fill_between(range(len(drawdown)), 0, drawdown,
                          color=COLORS["drawdown"], alpha=0.4)
        _apply_style(ax_dd, title="Drawdown",
                     xlabel="Day", ylabel="Drawdown (%)")
        panel_idx += 1

    # Panel 4: Metrics table (if available)
    if metrics is not None:
        ax_met = fig.add_subplot(gs[panel_idx, :])
        ax_met.axis("off")
        rows = [[k.replace("_", " ").title(), f"{v:.4f}"]
                for k, v in metrics.items()]
        table = ax_met.table(cellText=rows,
                             colLabels=["Metric", "Value"],
                             cellLoc="center", loc="center",
                             colColours=["#E3F2FD", "#E3F2FD"])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        for j in range(2):
            table[0, j].set_facecolor("#1565C0")
            table[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle("DT-RL Training Summary", fontsize=16, fontweight="bold",
                 y=1.01)
    return fig


# ── Price Trajectories ──────────────────────────────────────────────────────

def plot_price_trajectories(
    prices: np.ndarray,
    asset_ids: Optional[List[str]] = None,
    max_assets: int = 10,
    title: str = "Synthetic Price Trajectories",
) -> plt.Figure:
    """
    Plot price trajectories for multiple assets.

    Args:
        prices: (N, T) price matrix
        asset_ids: List of asset names
        max_assets: Maximum number of assets to plot
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    N = min(prices.shape[0], max_assets)
    for i in range(N):
        label = asset_ids[i] if asset_ids else f"Asset {i}"
        ax.plot(prices[i], linewidth=1.2, alpha=0.8, label=label)

    ax.legend(fontsize=8, ncol=2, framealpha=0.9, loc="upper left")
    _apply_style(ax, title=title, xlabel="Trading Day", ylabel="Price ($)")

    fig.tight_layout()
    return fig


# ── Coverage Heatmap ────────────────────────────────────────────────────────

def plot_coverage_heatmap(
    validity_mask: np.ndarray,
    asset_ids: Optional[List[str]] = None,
    title: str = "Data Coverage (Validity Mask)",
) -> plt.Figure:
    """
    Plot data coverage heatmap showing valid/invalid timesteps per asset.

    Args:
        validity_mask: (N, T) binary mask
        asset_ids: List of asset names
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    N = validity_mask.shape[0]
    labels = asset_ids[:N] if asset_ids else [f"Asset_{i}" for i in range(N)]

    im = ax.imshow(validity_mask, aspect="auto", cmap="Greens",
                   interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(min(N, 30)))
    ax.set_yticklabels(labels[:30], fontsize=7)
    _apply_style(ax, title=title, xlabel="Timestep", ylabel="Asset",
                 grid=False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Valid")

    fig.tight_layout()
    return fig


# ── Model Architecture Summary ─────────────────────────────────────────────

def plot_model_summary(
    param_counts: Dict[str, int],
    title: str = "Model Parameter Distribution",
) -> plt.Figure:
    """
    Plot model parameter distribution as a horizontal bar chart.

    Args:
        param_counts: Dictionary of module_name -> param_count
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    modules = list(param_counts.keys())
    counts = list(param_counts.values())
    total = sum(counts)

    colors = plt.cm.Set2(np.linspace(0, 1, len(modules)))
    bars = ax.barh(modules, counts, color=colors, edgecolor="white",
                   linewidth=1.5, height=0.6)

    # Add count labels
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)",
                ha="left", va="center", fontsize=9)

    _apply_style(ax, title=f"{title}\nTotal: {total:,} parameters",
                 xlabel="Parameter Count")
    ax.set_xlim(0, max(counts) * 1.3)

    fig.tight_layout()
    return fig


# ── WandB Integration ───────────────────────────────────────────────────────

def log_figures_to_wandb(
    figures: Dict[str, plt.Figure],
    step: Optional[int] = None,
) -> None:
    """
    Log multiple matplotlib figures to Weights & Biases.

    Args:
        figures: Dictionary of name -> matplotlib Figure
        step: Optional global step for logging
    """
    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping figure logging")
        return

    log_dict = {}
    for name, fig in figures.items():
        log_dict[name] = wandb.Image(fig)

    if step is not None:
        log_dict["global_step"] = step

    wandb.log(log_dict)


def close_figures(*figures: plt.Figure) -> None:
    """Close multiple figures to free memory."""
    for fig in figures:
        plt.close(fig)
