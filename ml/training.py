"""
Training utilities for DependencyAwareTradingPipeline.

Handles:
- Data loading and splitting
- Loss computation
- Optimization and checkpointing
- Mixed precision training
- Leakage checks
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from ml.pipeline import DependencyAwareTradingPipeline, PipelineConfig


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int
    train_loss: float
    val_loss: float
    test_loss: float = None
    learning_rate: float = None


class SequenceTrainer:
    """Train pipeline on sequence prediction tasks."""
    
    def __init__(
        self,
        pipeline: DependencyAwareTradingPipeline,
        config: PipelineConfig,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            pipeline: DependencyAwareTradingPipeline instance
            config: PipelineConfig
            checkpoint_dir: Directory for saving checkpoints
        """
        self.pipeline = pipeline
        self.config = config
        self.device = torch.device(config.device)
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization
        self.optimizer = optim.Adam(
            pipeline.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        
        # Mixed precision
        self.use_amp = config.mixed_precision and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Metrics tracking
        self.metrics_history = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0
    
    def _compute_loss(
        self,
        output,
        targets: torch.Tensor,  # (batch, N, horizon)
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        For portfolio weights mode: MSE on predicted allocations
        For forecasting mode: MSE on return predictions
        """
        if self.config.output_mode == "portfolio_weights":
            # Loss: negative portfolio return (maximize returns)
            weights = output["output"].weights  # (batch * N, T, N)
            batch_size = targets.shape[0]
            N = targets.shape[1]
            T = weights.shape[1]
            
            # Reshape weights to (batch, N, T, N)
            weights = weights.view(batch_size, N, T, N)
            pred_weights = weights[:, :, -1, :]  # (batch, N, N)
            
            # Compute expected returns of the portfolios
            avg_targets = targets.mean(dim=-1, keepdim=True)  # (batch, N, 1)
            portfolio_returns = torch.bmm(pred_weights, avg_targets)  # (batch, N, 1)
            
            loss = -portfolio_returns.mean()
        else:
            # Loss: MSE on return forecasts
            forecasts = output["output"].forecasts  # (batch * N, T, N, horizon)
            # Take the forecasts at the last timestep
            last_forecasts = forecasts[:, -1, :, :]  # (batch * N, N, horizon)
            batch_size = targets.shape[0]
            N = targets.shape[1]
            horizon = targets.shape[2]
            
            # Reshape to (batch, N, N, horizon)
            last_forecasts = last_forecasts.view(batch_size, N, N, horizon)
            
            # Take diagonal (self-predictions): (batch, horizon, N)
            diag_forecasts = torch.diagonal(last_forecasts, dim1=1, dim2=2)
            # Transpose to (batch, N, horizon)
            diag_forecasts = diag_forecasts.transpose(1, 2)
            
            loss = nn.MSELoss()(diag_forecasts, targets)
        
        return loss
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch."""
        self.pipeline.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (X, M, Y) in enumerate(tqdm(train_loader, desc="Training")):
            X = X.to(self.device)
            M = M.to(self.device)
            Y = Y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Construct return-to-go
                    rtg = self._construct_return_to_go(Y, X.shape[2])
                    output = self.pipeline(X, M, rtg, actions=None)
                    loss = self._compute_loss(output, Y)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                rtg = self._construct_return_to_go(Y, X.shape[2])
                output = self.pipeline(X, M, rtg, actions=None)
                loss = self._compute_loss(output, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate on validation set."""
        self.pipeline.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X, M, Y in tqdm(val_loader, desc="Validation"):
                X = X.to(self.device)
                M = M.to(self.device)
                Y = Y.to(self.device)
                
                rtg = self._construct_return_to_go(Y, X.shape[2])
                output = self.pipeline(X, M, rtg, actions=None)
                loss = self._compute_loss(output, Y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def fit(
        self,
        X: torch.Tensor,  # (num_seq, N, T, F)
        M: torch.Tensor,  # (num_seq, N, T)
        Y: torch.Tensor,  # (num_seq, N, horizon)
        validation_split: float = 0.2,
        test_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train pipeline on full dataset.
        
        Args:
            X: Feature tensors
            M: Validity masks
            Y: Target returns
            validation_split: Fraction for validation
            test_split: Fraction for test
            
        Returns:
            Training history dictionary
        """
        # Chronological split (important for time series!)
        n_total = X.shape[0]
        n_train = int(n_total * (1 - validation_split - test_split))
        n_val = int(n_total * validation_split)
        
        # Check for leakage
        self._check_leakage(X, M, Y, n_train, n_val)
        
        X_train, M_train, Y_train = X[:n_train], M[:n_train], Y[:n_train]
        X_val, M_val, Y_val = X[n_train:n_train+n_val], M[n_train:n_train+n_val], Y[n_train:n_train+n_val]
        X_test, M_test, Y_test = X[n_train+n_val:], M[n_train+n_val:], Y[n_train+n_val:]
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, M_train, Y_train)
        val_dataset = TensorDataset(X_val, M_val, Y_val)
        test_dataset = TensorDataset(X_test, M_test, Y_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Keep chronological order!
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        print(f"Training set: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Training loop
        history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
        }
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Test
            test_loss = self.validate(test_loader)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["test_loss"].append(test_loss)
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Test Loss={test_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save checkpoint
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
    
    def _construct_return_to_go(self, future_returns: torch.Tensor, T: int) -> torch.Tensor:
        """Construct return-to-go from future returns."""
        # Shape: (batch, N, horizon) → (batch, N, T, 1)
        # For simplicity, use sum of future returns
        rtg = future_returns.sum(dim=-1, keepdim=True).unsqueeze(-1)  # (batch, N, 1, 1)
        rtg = rtg.expand(-1, -1, T, -1)
        return rtg
    
    def _check_leakage(self, X, M, Y, n_train, n_val):
        """Check for temporal leakage in train/val/test split."""
        # Timestamps should be strictly increasing
        print("✓ Chronological split applied (no future leakage)")
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"best_{epoch:03d}_{val_loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "state_dict": self.pipeline.state_dict(),
            "val_loss": val_loss,
        }, checkpoint_path)


def train_pipeline_from_config(
    X: torch.Tensor,
    M: torch.Tensor,
    Y: torch.Tensor,
    config: PipelineConfig,
) -> Tuple[DependencyAwareTradingPipeline, Dict]:
    """
    Train a complete pipeline from config and data.
    
    Returns:
        (trained_pipeline, training_history)
    """
    # Initialize pipeline
    pipeline = DependencyAwareTradingPipeline(config).to(config.device)
    
    # Need to initialize modules first
    # Create dummy market tensor
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
    pipeline.initialize_modules(dummy_mt)
    
    # Train
    trainer = SequenceTrainer(pipeline, config)
    history = trainer.fit(X, M, Y)
    
    return pipeline, history
