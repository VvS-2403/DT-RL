"""
Module 3: Dependency/Memory Regime Classifier

Classify assets across 4 statistical regimes using continuous embeddings.
Input: Z ∈ R^(N×T×D)
Output: P_regime ∈ R^(N×T×4), E_regime ∈ R^(N×T×D)

4 Regimes (2×2):
- Memory Axis: Long (Trending) vs Short (Mean-reverting)
- Credit Axis: Long (Slow macro) vs Short (Instant reaction)

CRITICAL: Use continuous embeddings, NOT hard routing.
Prevents unstable trading at regime boundaries.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RegimeClassifier(nn.Module):
    """Classify assets into 4 regimes with continuous embeddings."""
    
    # Regime names
    REGIME_NAMES = [
        "R1: Long Memory + Long Credit",      # Trending + Slow macro absorption
        "R2: Long Memory + Short Credit",     # Trending + Instant reaction
        "R3: Short Memory + Long Credit",     # Mean-reverting + Slow macro
        "R4: Short Memory + Short Credit",    # Mean-reverting + Instant reaction
    ]
    
    def __init__(
        self,
        input_dim: int,
        num_regimes: int = 4,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize regime classifier.
        
        Args:
            input_dim: Input embedding dimension (D)
            num_regimes: Number of regimes (fixed at 4)
            embedding_dim: Regime embedding dimension
            hidden_dim: Hidden layer dimension for MLP
            dropout: Dropout rate
        """
        assert num_regimes == 4, "Must have exactly 4 regimes"
        
        super().__init__()
        self.input_dim = input_dim
        self.num_regimes = num_regimes
        self.embedding_dim = embedding_dim
        
        # MLP to compute regime probabilities
        # Input: (N, T, D) → Output: (N, T, 4)
        self.regime_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_regimes),
        )
        
        # Learnable regime embeddings W_regime ∈ R^(4, D)
        self.regime_embeddings = nn.Parameter(
            torch.randn(num_regimes, embedding_dim) / np.sqrt(embedding_dim)
        )
        
        # Optional: per-regime scaling
        self.regime_scales = nn.Parameter(torch.ones(num_regimes))
        
    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify regimes and compute continuous embeddings.
        
        Args:
            Z: (N, T, D) - context-aware latent embeddings
            
        Returns:
            P_regime: (N, T, 4) - softmax probabilities over regimes
            E_regime: (N, T, D) - continuous regime embeddings
        """
        N, T, D = Z.shape
        
        # Compute regime probabilities: (N, T, D) → (N, T, 4)
        Z_flat = Z.view(N * T, D)
        logits = self.regime_classifier(Z_flat)  # (N*T, 4)
        P_regime_flat = F.softmax(logits, dim=1)  # (N*T, 4)
        P_regime = P_regime_flat.view(N, T, self.num_regimes)  # (N, T, 4)
        
        # Scale each regime contribution: (N, T, 4) * (1, 1, 4)
        P_regime_scaled = P_regime * self.regime_scales.unsqueeze(0).unsqueeze(0)
        
        # Compute continuous embedding: E_regime = P_regime_scaled @ W_regime
        # (N, T, 4) @ (4, D) = (N, T, D)
        E_regime = torch.matmul(P_regime_scaled, self.regime_embeddings)  # (N, T, D)
        
        return P_regime, E_regime


class RegimeMemoryAnalyzer(nn.Module):
    """Analyze memory properties (autocorrelation) of each asset."""
    
    def __init__(self, window_size: int = 20):
        """
        Initialize memory analyzer.
        
        Args:
            window_size: Window for autocorrelation computation
        """
        super().__init__()
        self.window_size = window_size
    
    def forward(self, Z: torch.Tensor, validity_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute autocorrelation (memory indicator) per asset per time.
        
        Args:
            Z: (N, T, D) - embeddings
            validity_mask: (N, T) - validity mask
            
        Returns:
            autocorr: (N, T) - autocorrelation values
        """
        N, T, D = Z.shape
        device = Z.device
        
        autocorr = torch.zeros(N, T, device=device)
        
        # Compute autocorrelation per asset
        for n in range(N):
            # Take first feature dimension as proxy for trend
            z_series = Z[n, :, 0]  # (T,)
            mask = validity_mask[n, :]  # (T,)
            
            # Compute autocorrelation at lag=1
            for t in range(self.window_size, T):
                window_start = max(0, t - self.window_size)
                window = z_series[window_start:t]
                mask_window = mask[window_start:t]
                
                if mask_window.sum() < 2:
                    autocorr[n, t] = 0.0
                    continue
                
                # Pearson correlation lag-1
                valid_idx = (mask_window > 0).nonzero(as_tuple=True)[0]
                if len(valid_idx) < 2:
                    continue
                
                w = window[valid_idx]
                corr = torch.corrcoef(torch.stack([w[:-1], w[1:]]))
                if not torch.isnan(corr[0, 1]):
                    autocorr[n, t] = corr[0, 1]
        
        return autocorr


class CreditAxisAnalyzer(nn.Module):
    """Analyze information delay (credit axis) via recent vs historical changes."""
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        """
        Initialize credit analyzer.
        
        Args:
            short_window: Short-term window (instant reaction)
            long_window: Long-term window (slow absorption)
        """
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
    
    def forward(self, Z: torch.Tensor, validity_mask: torch.Tensor) -> torch.Tensor:
        """
        Estimate credit axis: short credit = high recent volatility / low historical.
        
        Args:
            Z: (N, T, D) - embeddings
            validity_mask: (N, T) - validity mask
            
        Returns:
            credit_scores: (N, T) - credit axis indicator (-1=long, +1=short)
        """
        N, T, D = Z.shape
        device = Z.device
        
        credit_scores = torch.zeros(N, T, device=device)
        
        for n in range(N):
            z_series = Z[n, :, 0]  # (T,)
            mask = validity_mask[n, :]
            
            for t in range(self.long_window, T):
                # Recent volatility (last short_window bars)
                short_idx = (mask[t-self.short_window:t] > 0).nonzero(as_tuple=True)[0]
                if len(short_idx) > 1:
                    short_vol = z_series[t-self.short_window:t][short_idx].std()
                else:
                    short_vol = 0.0
                
                # Historical volatility (last long_window bars)
                long_idx = (mask[t-self.long_window:t] > 0).nonzero(as_tuple=True)[0]
                if len(long_idx) > 1:
                    long_vol = z_series[t-self.long_window:t][long_idx].std()
                else:
                    long_vol = 1.0
                
                # Ratio: recent/historical
                # Short credit = high ratio (immediate reaction)
                # Long credit = low ratio (slow absorption)
                if long_vol > 1e-6:
                    ratio = (short_vol / long_vol).clamp(0, 10)
                    # Normalize to [-1, 1]
                    credit_scores[n, t] = 2.0 * (ratio / 10.0) - 1.0
        
        return credit_scores


class RegimeVisualizer:
    """Utilities for visualizing regime classification."""
    
    @staticmethod
    def get_regime_names() -> list:
        """Return regime names."""
        return RegimeClassifier.REGIME_NAMES
    
    @staticmethod
    def extract_regime_assignments(
        P_regime: torch.Tensor,
        asset_ids: list,
    ) -> dict:
        """
        Extract per-asset regime assignments from probabilities.
        
        Args:
            P_regime: (N, T, 4) - regime probabilities
            asset_ids: List[str] - asset identifiers
            
        Returns:
            {
                "current": {"AAPL": 0, "MSFT": 2, ...},
                "confidence": {"AAPL": 0.85, ...},
                "all_probabilities": {
                    "AAPL": [0.05, 0.1, 0.8, 0.05],
                    ...
                }
            }
        """
        N, T, K = P_regime.shape
        
        # Get latest probabilities (last timestep)
        P_latest = P_regime[:, -1, :]  # (N, 4)
        
        # Get regime assignments (argmax)
        assignments = P_latest.argmax(dim=1)  # (N,)
        confidences = P_latest.max(dim=1)[0]  # (N,)
        
        current_regimes = {
            asset: int(assignments[i].item())
            for i, asset in enumerate(asset_ids)
        }
        
        confidence_dict = {
            asset: float(confidences[i].item())
            for i, asset in enumerate(asset_ids)
        }
        
        all_probs = {
            asset: P_latest[i].detach().cpu().numpy().tolist()
            for i, asset in enumerate(asset_ids)
        }
        
        return {
            "current": current_regimes,
            "confidence": confidence_dict,
            "all_probabilities": all_probs,
        }
    
    @staticmethod
    def compute_regime_transitions(
        P_regime: torch.Tensor,
        window: int = 10,
    ) -> dict:
        """
        Compute regime transition frequencies.
        
        Args:
            P_regime: (N, T, 4) - regime probabilities
            window: Number of recent timesteps to analyze
            
        Returns:
            Transition matrix (4, 4) and statistics
        """
        N, T, K = P_regime.shape
        
        # Extract recent regime assignments
        regimes = P_regime[:, max(0, T-window):, :].argmax(dim=2)  # (N, window)
        
        # Compute transitions
        transitions = torch.zeros(K, K, device=P_regime.device)
        
        for i in range(1, regimes.shape[1]):
            prev_regime = regimes[:, i-1]
            curr_regime = regimes[:, i]
            
            for r_from in range(K):
                for r_to in range(K):
                    count = ((prev_regime == r_from) & (curr_regime == r_to)).sum().item()
                    transitions[r_from, r_to] += count
        
        # Normalize by row
        row_sums = transitions.sum(dim=1, keepdim=True).clamp(min=1)
        transition_probs = transitions / row_sums
        
        return {
            "transition_matrix": transition_probs.detach().cpu().numpy().tolist(),
            "raw_counts": transitions.detach().cpu().numpy().tolist(),
        }
