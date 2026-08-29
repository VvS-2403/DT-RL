#!/usr/bin/env python3
"""
Quick-start demo: Generate synthetic data → Initialize pipeline → Run training

Run this to verify the entire system works end-to-end.
"""

import torch
import numpy as np
from pathlib import Path

from ml.data.synthetic_data import SyntheticDataLoader
from ml.pipeline import DependencyAwareTradingPipeline, PipelineConfig
from ml.training import train_pipeline_from_config
from ml.modules.module1_input import DataValidator


def main():
    print("=" * 80)
    print("DEPENDENCY-AWARE TRADING SYSTEM: END-TO-END DEMO")
    print("=" * 80)
    
    # Step 1: Create synthetic data
    print("\n[1/4] Generating synthetic market data...")
    loader = SyntheticDataLoader()
    market_tensor, raw_df = loader.create_demo_market_tensor(
        num_assets=20,
        lookback=60,
        seed=42
    )
    print(f"  ✓ Created MarketTensor: {market_tensor.shape}")
    print(f"    - Assets: {market_tensor.num_assets}")
    print(f"    - Lookback: {market_tensor.lookback_window} days")
    print(f"    - Features: {market_tensor.num_features}")
    
    # Step 2: Validate data
    print("\n[2/4] Validating data...")
    validator = DataValidator()
    report = validator.validate_market_tensor(market_tensor)
    print(f"  ✓ Data is {'VALID' if report['valid'] else 'INVALID'}")
    if not report['valid']:
        for issue in report['issues']:
            print(f"    ⚠ {issue}")
    print(f"  ✓ Overall coverage: {report['coverage']['overall']:.1%}")
    
    # Step 3: Create sequences for training
    print("\n[3/4] Creating training sequences...")
    from ml.modules.module1_input import StockDataProcessor
    processor = StockDataProcessor(lookback_window=60)
    full_market_tensor = processor.process(raw_df)
    X, M, Y = loader.create_sequences(full_market_tensor, lookback=60, horizon=5, stride=5)
    print(f"  ✓ X shape: {X.shape}")
    print(f"  ✓ M shape: {M.shape}")
    print(f"  ✓ Y shape: {Y.shape}")
    
    # Step 4: Initialize and train pipeline
    print("\n[4/4] Training pipeline...")
    config = PipelineConfig(
        lookback_window=60,
        forecast_horizon=5,
        num_regimes=4,
        output_mode="portfolio_weights",
        batch_size=16,
        num_epochs=3,  # Short demo
        learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    
    # Train
    pipeline, history = train_pipeline_from_config(X, M, Y, config)
    
    print(f"  ✓ Training complete!")
    print(f"    Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"    Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Step 5: Run inference
    print("\n[5/5] Running inference on test data...")
    pipeline.eval()
    
    with torch.no_grad():
        # Get a single batch for inference
        test_X = X[:1].to(config.device)
        test_M = M[:1].to(config.device)
        test_Y = Y[:1].to(config.device)
        
        # Construct return-to-go
        rtg = test_Y.sum(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, 60, -1)
        
        # Forward pass
        output = pipeline(test_X, test_M, rtg)
        
        # Extract results
        portfolio_weights = output["output"].weights
        regimes = output["regime_probs"].argmax(dim=-1)
        
        print(f"  ✓ Inference successful!")
        print(f"    Portfolio weights shape: {portfolio_weights.shape}")
        print(f"    Regime assignments shape: {regimes.shape}")
        print(f"    Sample regime: {regimes[0, 0, -1].item()}")  # Last timestep of first asset
        print(f"    Sample weight: {portfolio_weights[0, -1, 0].item():.4f}")
    
    # Save checkpoint
    print("\n[6/6] Saving checkpoint...")
    checkpoint_path = Path("checkpoints/demo_model.pt")
    pipeline.save(checkpoint_path)
    print(f"  ✓ Model saved to {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE ✓")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Start the backend server: python -m backend.api.main")
    print("  2. Start the frontend: cd frontend && npm run dev")
    print("  3. Open http://localhost:3000 in your browser")
    print("\nOr use Docker:")
    print("  docker-compose up")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
