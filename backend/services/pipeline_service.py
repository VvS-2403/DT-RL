"""
Pipeline state management service.

Coordinates dataset loading, model state, asynchronous training execution,
inference runs, and portfolio backtesting.
"""

import threading
import time
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from ml.pipeline import DependencyAwareTradingPipeline, PipelineConfig
from ml.modules.module1_input import StockDataProcessor, DataValidator, MarketTensor
from ml.data.synthetic_data import SyntheticDataLoader
from ml.evaluation import compute_financial_metrics

class PipelineService:
    """Singleton service to manage model training and execution state."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PipelineService, cls).__new__(cls, *args, **kwargs)
            cls._instance._init()
        return cls._instance
        
    def _init(self):
        self.market_tensor: Optional[MarketTensor] = None
        self.raw_dataframe: Optional[pd.DataFrame] = None
        self.pipeline: Optional[DependencyAwareTradingPipeline] = None
        self.config: Optional[PipelineConfig] = None
        
        # Training state
        self.training_status = "idle"  # idle, training, completed, failed
        self.training_progress = {
            "epoch": 0,
            "num_epochs": 0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "history": []
        }
        self.training_thread: Optional[threading.Thread] = None
        
        # Checkpoint directories
        self.checkpoints_dir = Path("checkpoints")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Load synthetic data for initial setup so the app is immediately ready
        self.load_demo_data()
        
    def load_demo_data(self):
        """Load synthetic demo data immediately to make the system testable."""
        try:
            loader = SyntheticDataLoader()
            self.market_tensor, self.raw_dataframe = loader.create_demo_market_tensor(
                num_assets=20,
                lookback=60,
                seed=42
            )
            # Default config
            self.config = PipelineConfig(
                lookback_window=60,
                forecast_horizon=5,
                num_regimes=4,
                output_mode="portfolio_weights",
                batch_size=16,
                num_epochs=5,
                learning_rate=0.001,
            )
            # Pre-initialize pipeline
            self.pipeline = DependencyAwareTradingPipeline(self.config).to(self.config.device)
            self.pipeline.initialize_modules(self.market_tensor)
        except Exception as e:
            print(f"Error loading demo data: {e}")
            
    def set_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process and set the current dataset."""
        try:
            processor = StockDataProcessor(lookback_window=60)
            self.market_tensor = processor.process(df)
            self.raw_dataframe = df
            
            # Re-initialize pipeline modules for new asset dimensions if needed
            if self.pipeline and self.config:
                new_pipeline = DependencyAwareTradingPipeline(self.config).to(self.config.device)
                new_pipeline.initialize_modules(self.market_tensor)
                self.pipeline = new_pipeline
                
            return {
                "num_assets": self.market_tensor.num_assets,
                "num_days": len(self.market_tensor.timestamps),
                "num_features": self.market_tensor.num_features,
            }
        except Exception as e:
            raise ValueError(f"Failed to process dataset: {e}")
            
    def start_training(self, train_config: PipelineConfig) -> str:
        """Start model training in a background thread."""
        if self.training_status == "training":
            return "Training is already in progress"
            
        if self.market_tensor is None:
            raise ValueError("No dataset loaded. Upload data first.")
            
        self.config = train_config
        # Re-create pipeline with target config
        self.pipeline = DependencyAwareTradingPipeline(self.config).to(self.config.device)
        self.pipeline.initialize_modules(self.market_tensor)
        
        self.training_status = "training"
        self.training_progress = {
            "epoch": 0,
            "num_epochs": self.config.num_epochs,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "history": []
        }
        
        self.training_thread = threading.Thread(target=self._run_training_job)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        return "Training started"
        
    def _run_training_job(self):
        """Training job target run in background thread."""
        try:
            # Generate sequences from current market tensor
            loader = SyntheticDataLoader()
            # In a real system, we'd use the current uploaded market_tensor to create sequences
            # Since market_tensor is already processed and aligned, create sequences:
            X, M, Y = loader.create_sequences(
                self.market_tensor,
                lookback=self.config.lookback_window,
                horizon=self.config.forecast_horizon,
                stride=5
            )
            
            # Simple hook into the trainer to update progress per epoch
            # We override fit parameters or write a simple training loop here to log epoch updates
            from torch.utils.data import DataLoader, TensorDataset
            import torch.optim as optim
            import torch.nn as nn
            
            # Chronological splits
            n_total = X.shape[0]
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.2)
            
            train_dataset = TensorDataset(X[:n_train], M[:n_train], Y[:n_train])
            val_dataset = TensorDataset(X[n_train:n_train+n_val], M[n_train:n_train+n_val], Y[n_train:n_train+n_val])
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            optimizer = optim.Adam(self.pipeline.parameters(), lr=self.config.learning_rate)
            
            # Mock trainer fit, but updating self.training_progress
            for epoch in range(self.config.num_epochs):
                self.pipeline.train()
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_X, batch_M, batch_Y in train_loader:
                    batch_X = batch_X.to(self.pipeline.device)
                    batch_M = batch_M.to(self.pipeline.device)
                    batch_Y = batch_Y.to(self.pipeline.device)
                    
                    optimizer.zero_grad()
                    
                    # RTG
                    rtg = batch_Y.sum(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, batch_X.shape[2], -1)
                    outputs = self.pipeline(batch_X, batch_M, rtg)
                    
                    # Compute loss
                    # Mimic trainer _compute_loss but inline
                    if self.config.output_mode == "portfolio_weights":
                        weights = outputs["output"].weights.view(batch_Y.shape[0], batch_Y.shape[1], -1, batch_Y.shape[1])
                        pred_weights = weights[:, :, -1, :]
                        avg_targets = batch_Y.mean(dim=-1, keepdim=True)
                        portfolio_returns = torch.bmm(pred_weights, avg_targets)
                        loss = -portfolio_returns.mean()
                    else:
                        forecasts = outputs["output"].forecasts
                        last_forecasts = forecasts[:, -1, :, :].view(batch_Y.shape[0], batch_Y.shape[1], batch_Y.shape[1], -1)
                        diag_forecasts = torch.diagonal(last_forecasts, dim1=1, dim2=2).transpose(1, 2)
                        loss = nn.MSELoss()(diag_forecasts, batch_Y)
                        
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                train_loss = epoch_loss / num_batches
                
                # Validation
                self.pipeline.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch_X, batch_M, batch_Y in val_loader:
                        batch_X = batch_X.to(self.pipeline.device)
                        batch_M = batch_M.to(self.pipeline.device)
                        batch_Y = batch_Y.to(self.pipeline.device)
                        rtg = batch_Y.sum(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, batch_X.shape[2], -1)
                        outputs = self.pipeline(batch_X, batch_M, rtg)
                        
                        if self.config.output_mode == "portfolio_weights":
                            weights = outputs["output"].weights.view(batch_Y.shape[0], batch_Y.shape[1], -1, batch_Y.shape[1])
                            pred_weights = weights[:, :, -1, :]
                            avg_targets = batch_Y.mean(dim=-1, keepdim=True)
                            portfolio_returns = torch.bmm(pred_weights, avg_targets)
                            loss = -portfolio_returns.mean()
                        else:
                            forecasts = outputs["output"].forecasts
                            last_forecasts = forecasts[:, -1, :, :].view(batch_Y.shape[0], batch_Y.shape[1], batch_Y.shape[1], -1)
                            diag_forecasts = torch.diagonal(last_forecasts, dim1=1, dim2=2).transpose(1, 2)
                            loss = nn.MSELoss()(diag_forecasts, batch_Y)
                        val_loss += loss.item()
                        val_batches += 1
                val_loss = val_loss / val_batches
                
                # Log epoch stats
                epoch_data = {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "time": time.time(),
                }
                self.training_progress["epoch"] = epoch + 1
                self.training_progress["train_loss"] = float(train_loss)
                self.training_progress["val_loss"] = float(val_loss)
                self.training_progress["history"].append(epoch_data)
                
                # Checkpoint saving
                if epoch == self.config.num_epochs - 1 or val_loss < min([h["val_loss"] for h in self.training_progress["history"][:-1]] or [float("inf")]):
                    self.pipeline.save(self.checkpoints_dir / "latest_model.pt")
                    
                time.sleep(0.5)  # Simulate small delay for UI updates
                
            self.training_status = "completed"
        except Exception as e:
            print(f"Training failed: {e}")
            self.training_status = "failed"
            self.training_progress["error"] = str(e)
            
    def run_inference(self, target_rtg: float = 0.05) -> Dict[str, Any]:
        """Run predictions using current model on loaded dataset."""
        if not self.pipeline:
            raise ValueError("No model loaded/trained. Start training first.")
        if not self.market_tensor:
            raise ValueError("No data loaded. Upload data first.")
            
        from ml.inference import run_inference
        return run_inference(self.pipeline, self.market_tensor, target_rtg)
        
    def run_backtest(self, transaction_cost_bps: float = 10.0) -> Dict[str, Any]:
        """Simulate historical trading using model predictions and evaluate."""
        if not self.pipeline:
            raise ValueError("No model loaded.")
        if not self.market_tensor:
            raise ValueError("No data loaded.")
            
        # Run inference across historical timeline
        inference_results = self.run_inference()
        
        # Calculate daily portfolio returns
        # For simplicity, extract portfolio returns from predicted weights
        if "portfolio_weights" in inference_results:
            weights = inference_results["portfolio_weights"]  # (N, T, N)
            
            # Compute daily asset returns from raw_dataframe
            # Assume we have Close prices. Let's compute daily return
            raw_prices = self.market_tensor.features[:, :, 0].cpu().numpy()  # (N, T) Close is index 0 or similar
            daily_returns = np.zeros_like(raw_prices)
            daily_returns[:, 1:] = (raw_prices[:, 1:] / (raw_prices[:, :-1] + 1e-8)) - 1.0
            
            # weights at last step: (N, N) where asset i chooses weights.
            # Take the diagonal weights (own asset allocation model)
            pred_weights = weights[:, -1, :]  # (N, N)
            diagonal_weights = np.diagonal(pred_weights)  # (N,)
            diagonal_weights = np.clip(diagonal_weights, -1.0, 1.0)
            
            # Normalize to leverage
            leverage = self.config.leverage if self.config else 1.0
            l1_sum = np.sum(np.abs(diagonal_weights)) + 1e-8
            diagonal_weights = diagonal_weights * (leverage / l1_sum)
            
            # Compute daily portfolio return timeline
            T = daily_returns.shape[1]
            portfolio_returns = np.zeros(T)
            for t in range(T):
                portfolio_returns[t] = np.sum(diagonal_weights * daily_returns[:, t])
                
            # Compute transaction costs (simplified turnover)
            # Assume we hold constant weights, so turnover is low
            weights_diff = np.zeros((T, len(diagonal_weights)))
            weights_diff[0] = diagonal_weights
            
            metrics = compute_financial_metrics(
                portfolio_returns[60:],  # Exclude lookback window
                weights_diff[60:],
                transaction_cost_bps
            )
            
            # Generate equity curve
            equity_curve = np.cumprod(1.0 + portfolio_returns[60:])
            benchmark = np.cumprod(1.0 + daily_returns.mean(axis=0)[60:])
            
            metrics["equity_curve"] = equity_curve.tolist()
            metrics["benchmark_curve"] = benchmark.tolist()
            metrics["dates"] = [str(d) for d in self.market_tensor.timestamps[60:]]
            
            return metrics
        else:
            # If forecasting, simulate a basic long-top-5-short-bottom-5 policy
            forecasts = inference_results["forecasts"]  # (N, T, N, horizon)
            last_forecast = forecasts[:, -1, :, 0]  # (N, N)
            diagonal_forecast = np.diagonal(last_forecast)  # (N,)
            
            # Sort and allocate weights
            N = len(diagonal_forecast)
            weights = np.zeros(N)
            top_k = min(5, N // 2)
            indices = np.argsort(diagonal_forecast)
            
            weights[indices[-top_k:]] = 1.0 / top_k  # Long top forecasts
            weights[indices[:top_k]] = -1.0 / top_k  # Short bottom forecasts
            
            raw_prices = self.market_tensor.features[:, :, 0].cpu().numpy()
            daily_returns = np.zeros_like(raw_prices)
            daily_returns[:, 1:] = (raw_prices[:, 1:] / (raw_prices[:, :-1] + 1e-8)) - 1.0
            
            T = daily_returns.shape[1]
            portfolio_returns = np.zeros(T)
            for t in range(T):
                portfolio_returns[t] = np.sum(weights * daily_returns[:, t])
                
            metrics = compute_financial_metrics(
                portfolio_returns[60:],
                transaction_cost_bps=transaction_cost_bps
            )
            
            equity_curve = np.cumprod(1.0 + portfolio_returns[60:])
            benchmark = np.cumprod(1.0 + daily_returns.mean(axis=0)[60:])
            
            metrics["equity_curve"] = equity_curve.tolist()
            metrics["benchmark_curve"] = benchmark.tolist()
            metrics["dates"] = [str(d) for d in self.market_tensor.timestamps[60:]]
            
            return metrics
