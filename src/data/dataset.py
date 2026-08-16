"""
PyTorch Dataset and DataLoader utilities for time-series forecasting.

Creates sliding-window (X, y) pairs:
    X: (lookback, n_features)  — past window of all features
    y: (horizon,)              — future values of target only
    
    ┌───────── lookback ─────────┐┌─── horizon ───┐
    │  features at t-168 ... t-1 ││ PM2.5 at t...t+23 │
    └────────────────────────────┘└─────────────────┘

Usage:
    from src.data.dataset import TimeSeriesDataset, create_dataloaders
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class TimeSeriesDataset(Dataset):
    """
    Sliding window dataset for multi-step time-series forecasting.
    
    Args:
        data: numpy array (timesteps, features) — all columns including target
        target_idx: column index of the target variable
        lookback: number of past timesteps as input
        horizon: number of future timesteps to predict
    """
    
    def __init__(self, data, target_idx=0, lookback=168, horizon=24):
        self.data = torch.FloatTensor(data)
        self.target_idx = target_idx
        self.lookback = lookback
        self.horizon = horizon
    
    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1
    
    def __getitem__(self, idx):
        # Input: all features for the lookback window
        x = self.data[idx : idx + self.lookback]  # (lookback, n_features)
        
        # Target: only the target column for the horizon window
        y = self.data[
            idx + self.lookback : idx + self.lookback + self.horizon,
            self.target_idx
        ]  # (horizon,)
        
        return x, y


def create_dataloaders(train_df, val_df, test_df, target="pm25",
                       lookback=168, horizon=24, batch_size=64,
                       num_workers=0):
    """
    Create PyTorch DataLoaders from train/val/test DataFrames.
    
    Args:
        train_df, val_df, test_df: DataFrames with target column
        target: target column name
        lookback: input window size (hours)
        horizon: prediction window size (hours)
        batch_size: batch size for training
        num_workers: DataLoader workers (set 0 for Windows compatibility)
    
    Returns:
        train_loader, val_loader, test_loader, feature_info dict
    """
    # Ensure target is the first column (index 0)
    cols = [target] + [c for c in train_df.columns if c != target]
    
    train_arr = train_df[cols].values.astype(np.float32)
    val_arr = val_df[cols].values.astype(np.float32)
    test_arr = test_df[cols].values.astype(np.float32)
    
    target_idx = 0  # target is first column now
    
    # Create datasets
    train_ds = TimeSeriesDataset(train_arr, target_idx, lookback, horizon)
    val_ds = TimeSeriesDataset(val_arr, target_idx, lookback, horizon)
    test_ds = TimeSeriesDataset(test_arr, target_idx, lookback, horizon)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    feature_info = {
        "n_features": train_arr.shape[1],
        "target_idx": target_idx,
        "feature_cols": cols,
        "lookback": lookback,
        "horizon": horizon,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }
    
    print(f"DataLoaders created:")
    print(f"  Features: {feature_info['n_features']}")
    print(f"  Lookback: {lookback}h, Horizon: {horizon}h")
    print(f"  Train: {len(train_ds):,} samples")
    print(f"  Val:   {len(val_ds):,} samples")
    print(f"  Test:  {len(test_ds):,} samples")
    print(f"  Batch: {batch_size}")
    
    return train_loader, val_loader, test_loader, feature_info
