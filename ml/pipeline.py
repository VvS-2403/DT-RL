"""
Main Pipeline: DependencyAwareTradingPipeline

Orchestrates all 5 modules into a complete end-to-end system.
Tensor flow:
  Raw Data → M1 (X, M) → M2 (Z) → M3 (P_regime, E_regime)
  → M4 (H) → M5 (weights or forecasts)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from ml.modules.module1_input import MarketTensor, StockDataProcessor
from ml.modules.module2_encoder import HypergraphEncoder, MultiLayerHypergraphEncoder
from ml.modules.module3_regime import RegimeClassifier
from ml.modules.module4_transformer import RegimeConditionedTransformer, TrajectoryEncoder, ActionPolicyHead
from ml.modules.module5_output import OutputModule, PortfolioOutput, ForecastingOutput


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    # Data
    lookback_window: int = 60
    forecast_horizon: int = 5
    
    # Module 2: Encoder
    num_sectors: int = 16
    encoder_hidden_dim: int = 64
    encoder_dropout: float = 0.1
    num_encoder_layers: int = 2
    
    # Module 3: Regime
    num_regimes: int = 4
    regime_embedding_dim: int = 128
    regime_hidden_dim: int = 256
    regime_dropout: float = 0.1
    
    # Module 4: Transformer
    transformer_embedding_dim: int = 128
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    transformer_ffn_dim: int = 512
    transformer_dropout: float = 0.1
    
    # Module 5: Output
    output_mode: str = "portfolio_weights"  # or "forecasting"
    output_hidden_dim: int = 256
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    
    # Portfolio constraints
    max_weight: float = 0.1
    leverage: float = 1.0
    long_only: bool = False
    transaction_cost_bps: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DependencyAwareTradingPipeline(nn.Module):
    """
    Complete 5-module trading pipeline.
    
    Architecture:
    1. Module 1: Stock Data Alignment
    2. Module 2: Hypergraph Encoder
    3. Module 3: Regime Classifier
    4. Module 4: Decision Transformer
    5. Module 5: Output Heads
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline."""
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Will be initialized after seeing data
        self.num_assets = None
        self.feature_dim = None
        self.modules_initialized = False
        
        # Data processor
        self.data_processor = StockDataProcessor(
            lookback_window=config.lookback_window,
            device=self.device,
        )
    
    def initialize_modules(self, market_tensor: MarketTensor):
        """Initialize neural network modules based on data."""
        if self.modules_initialized:
            return
        
        self.num_assets = market_tensor.num_assets
        self.feature_dim = market_tensor.num_features
        
        print(f"Initializing pipeline for {self.num_assets} assets, {self.feature_dim} features")
        
        # Module 2: Hypergraph Encoder (X → Z)
        self.module2 = MultiLayerHypergraphEncoder(
            input_dim=self.feature_dim,
            output_dim=self.config.regime_embedding_dim,
            num_sectors=self.config.num_sectors,
            num_assets=self.num_assets,
            num_layers=self.config.num_encoder_layers,
            dropout=self.config.encoder_dropout,
        ).to(self.device)
        
        # Module 3: Regime Classifier (Z → P_regime, E_regime)
        self.module3 = RegimeClassifier(
            input_dim=self.config.regime_embedding_dim,
            num_regimes=self.config.num_regimes,
            embedding_dim=self.config.regime_embedding_dim,
            hidden_dim=self.config.regime_hidden_dim,
            dropout=self.config.regime_dropout,
        ).to(self.device)
        
        # Module 4: Decision Transformer (τ → H)
        # Action dim = num_assets (for portfolio weights)
        self.module4 = RegimeConditionedTransformer(
            state_dim=self.config.regime_embedding_dim,
            regime_embedding_dim=self.config.regime_embedding_dim,
            action_dim=self.num_assets,
            transformer_embedding_dim=self.config.transformer_embedding_dim,
            num_layers=self.config.num_transformer_layers,
            num_heads=self.config.num_attention_heads,
            ffn_dim=self.config.transformer_ffn_dim,
            dropout=self.config.transformer_dropout,
        ).to(self.device)
        
        # Module 5: Output Module (H → weights or forecasts)
        self.module5 = OutputModule(
            latent_dim=self.config.transformer_embedding_dim,
            num_assets=self.num_assets,
            output_mode=self.config.output_mode,
            forecast_horizon=self.config.forecast_horizon,
            hidden_dim=self.config.output_hidden_dim,
        ).to(self.device)
        
        self.modules_initialized = True
        print("Pipeline modules initialized successfully")
    
    def forward(
        self,
        X: torch.Tensor,  # (batch, N, T, F)
        M: torch.Tensor,  # (batch, N, T)
        return_to_go: torch.Tensor,  # (batch, N, T, 1)
        actions: Optional[torch.Tensor] = None,  # (batch, N, T, N)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through entire pipeline.
        
        Args:
            X: (batch, N, T, F) - feature tensors
            M: (batch, N, T) - validity masks
            return_to_go: (batch, N, T, 1) - return-to-go
            actions: (batch, N, T, N) - optional previous actions
            
        Returns:
            {
                "portfolio_weights" or "forecasts": outputs from Module 5,
                "regime_probs": Module 3 regime probabilities,
                "regime_embeddings": Module 3 regime embeddings,
                "encoder_embeddings": Module 2 embeddings,
            }
        """
        batch_size, N, T, F = X.shape
        
        # Reshape for processing: (batch*N, T, F)
        X_flat = X.view(batch_size * N, T, F)
        M_flat = M.view(batch_size * N, T)
        
        # Module 1: Data already aligned
        # (implicitly done in preprocessing)
        
        # Module 2: Hypergraph Encoder
        Z = self.module2(X, M)  # (batch, N, T, D)
        Z_flat = Z.view(batch_size * N, T, -1)
        
        # Module 3: Regime Classifier
        Z_for_regime = Z_flat.view(batch_size * N, T, -1)
        P_regime_flat, E_regime_flat = self.module3(Z_for_regime)  # (batch*N, T, 4) & (batch*N, T, D)
        P_regime = P_regime_flat.view(batch_size, N, T, self.config.num_regimes)
        E_regime = E_regime_flat.view(batch_size, N, T, -1)
        
        # Module 4: Decision Transformer
        # Prepare trajectory tensors
        return_to_go_flat = return_to_go.view(batch_size * N, T, 1)
        Z_cond = Z_flat + E_regime_flat  # Enriched state
        
        if actions is not None:
            actions_flat = actions.view(batch_size * N, T, N)
        else:
            actions_flat = None
        
        H_flat = self.module4(
            return_to_go=return_to_go_flat,
            states=Z_cond,
            regime_embeddings=E_regime_flat,
            actions=actions_flat,
        )  # (batch*N, T, embedding_dim)
        
        # Module 5: Output Module
        constraints = {
            "max_weight": self.config.max_weight,
            "leverage": self.config.leverage,
            "long_only": self.config.long_only,
            "transaction_cost_bps": self.config.transaction_cost_bps,
        }
        
        # M is (batch, N, T). We need a mask of shape (batch*N, T, N) where
        # mask[b*N + i, t, j] = M[b, j, t] (is asset j valid at time t?)
        M_transposed = M.transpose(1, 2)  # (batch, T, N)
        # Expand so every predicting asset i sees the same portfolio validity mask
        M_for_weights = M_transposed.unsqueeze(1).expand(batch_size, N, T, N).reshape(batch_size * N, T, N)
        
        output = self.module5(
            latent_decision=H_flat,
            validity_mask=M_for_weights,
            constraints=constraints,
        )
        
        return {
            "output": output,
            "regime_probs": P_regime,
            "regime_embeddings": E_regime,
            "encoder_embeddings": Z,
            "latent_decision": H_flat.view(batch_size, N, T, -1),
        }
    
    def get_parameters_by_module(self) -> Dict[str, list]:
        """Get parameters grouped by module for debugging."""
        params = {}
        if self.modules_initialized:
            params["module2"] = list(self.module2.parameters())
            params["module3"] = list(self.module3.parameters())
            params["module4"] = list(self.module4.parameters())
            params["module5"] = list(self.module5.parameters())
        return params
    
    def save(self, path: Path):
        """Save pipeline state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
            "num_assets": self.num_assets,
            "feature_dim": self.feature_dim,
            "modules_initialized": self.modules_initialized,
        }, path)
        print(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "DependencyAwareTradingPipeline":
        """Load pipeline from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        
        pipeline = cls(config)
        # Dummy initialization to create modules
        if checkpoint["modules_initialized"]:
            # Create dummy market tensor for initialization
            from ml.modules.module1_input import MarketTensor
            dummy_mt = MarketTensor(
                features=torch.zeros(checkpoint["num_assets"], 60, checkpoint["feature_dim"]),
                validity_mask=torch.ones(checkpoint["num_assets"], 60),
                asset_ids=[f"ASSET_{i}" for i in range(checkpoint["num_assets"])],
                timestamps=np.array([f"2024-01-{i:02d}" for i in range(1, 61)]),
                feature_names=[f"F{i}" for i in range(checkpoint["feature_dim"])],
                feature_means=torch.zeros(checkpoint["feature_dim"]),
                feature_stds=torch.ones(checkpoint["feature_dim"]),
            )
            pipeline.initialize_modules(dummy_mt)
            pipeline.load_state_dict(checkpoint["state_dict"])
        
        print(f"Pipeline loaded from {path}")
        return pipeline
    
    def to_device(self, device: str):
        """Move to device."""
        self.device = torch.device(device)
        self.to(self.device)
        return self

    def fit(self, X: torch.Tensor, M: torch.Tensor, Y: torch.Tensor) -> Dict[str, list]:
        """
        Train the pipeline end-to-end.
        
        Args:
            X: (batch, N, T, F) features
            M: (batch, N, T) validity mask
            Y: (batch, N, horizon) targets / returns
            
        Returns:
            Dict containing training history (loss per epoch, etc.)
        """
        from ml.training import SequenceTrainer
        # Make sure modules are initialized
        if not self.modules_initialized:
            from ml.modules.module1_input import MarketTensor
            dummy_mt = MarketTensor(
                features=torch.zeros(X.shape[1], X.shape[2], X.shape[3]),
                validity_mask=torch.ones(X.shape[1], X.shape[2]),
                asset_ids=[f"ASSET_{i}" for i in range(X.shape[1])],
                timestamps=np.arange(X.shape[2]),
                feature_names=[f"F{i}" for i in range(X.shape[3])],
                feature_means=torch.zeros(X.shape[3]),
                feature_stds=torch.ones(X.shape[3]),
            )
            self.initialize_modules(dummy_mt)
        
        # Fit using SequenceTrainer
        trainer = SequenceTrainer(self, self.config)
        history = trainer.fit(X, M, Y)
        return history

    def predict(self, X: torch.Tensor, M: torch.Tensor, target_rtg: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Run inference on the pipeline.
        
        Args:
            X: (batch, N, T, F) features
            M: (batch, N, T) validity mask
            target_rtg: (batch, N, T, 1) target return-to-go. If None, default is constructed.
            
        Returns:
            Dictionary with outputs ("output", "regime_probs", "regime_embeddings", "encoder_embeddings", "latent_decision")
        """
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            M = M.to(self.device)
            
            if target_rtg is None:
                # Default RTG: target of 5% (0.05) return per step
                batch_size, N, T, _ = X.shape
                target_rtg = torch.full((batch_size, N, T, 1), 0.05, device=self.device)
            else:
                target_rtg = target_rtg.to(self.device)
                
            return self.forward(X, M, target_rtg)

    def evaluate(self, X: torch.Tensor, M: torch.Tensor, Y: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluate model performance on a test set.
        
        Args:
            X: (batch, N, T, F) features
            M: (batch, N, T) validity mask
            Y: (batch, N, horizon) targets
            
        Returns:
            Dictionary with losses and performance metrics
        """
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            M = M.to(self.device)
            Y = Y.to(self.device)
            
            # Construct RTG from target Y (sum of future returns as benchmark target)
            rtg = Y.sum(dim=-1, keepdim=True).unsqueeze(-1)  # (batch, N, 1, 1)
            # Expand to T
            rtg = rtg.expand(-1, -1, X.shape[2], -1)          # (batch, N, T, 1)
            
            outputs = self.forward(X, M, rtg)
            
            # Compute loss
            from ml.training import SequenceTrainer
            trainer = SequenceTrainer(self, self.config)
            loss = trainer._compute_loss(outputs["output"], Y)
            
            # Additional evaluation metrics
            from ml.evaluation import evaluate_predictions
            metrics = evaluate_predictions(outputs, Y, self.config.output_mode)
            
            metrics["loss"] = loss.item()
            return metrics
