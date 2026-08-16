
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from pathlib import Path
import joblib


def temporal_split(df, val_frac=0.15, test_frac=0.15, verbose=True):

    assert df.index.is_monotonic_increasing, \
        "DataFrame index must be sorted chronologically"
    
    n = len(df)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    if verbose:
        print("Temporal Split:")
        print(f"  Train: {train.index.min()} → {train.index.max()} "
              f"({len(train):,} rows, {len(train)/n*100:.0f}%)")
        print(f"  Val:   {val.index.min()} → {val.index.max()} "
              f"({len(val):,} rows, {len(val)/n*100:.0f}%)")
        print(f"  Test:  {test.index.min()} → {test.index.max()} "
              f"({len(test):,} rows, {len(test)/n*100:.0f}%)")
    
    return train, val, test


def scale_data(train, val, test, target="pm25",
               method="robust", save_path=None):

    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get feature columns (all except target)
    feature_cols = [c for c in train.columns if c != target]
    
    # Fit scaler on training data only
    scaler.fit(train[feature_cols])
    
    # Transform all splits
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
    train_scaled[feature_cols] = scaler.transform(train[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])
    
    # Also scale target separately (for inverse transform during evaluation)
    target_scaler = RobustScaler() if method == "robust" else StandardScaler()
    target_scaler.fit(train[[target]])
    
    train_scaled[target] = target_scaler.transform(train[[target]])
    val_scaled[target] = target_scaler.transform(val[[target]])
    test_scaled[target] = target_scaler.transform(test[[target]])
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path / "feature_scaler.pkl")
        joblib.dump(target_scaler, save_path / "target_scaler.pkl")
        print(f"  Scalers saved to: {save_path}")
    
    print(f"  Scaling method: {method}")
    print(f"  Features scaled: {len(feature_cols)}")
    
    return train_scaled, val_scaled, test_scaled, scaler, target_scaler


def walk_forward_cv(df, n_splits=5, test_size_days=30, gap_hours=24):

    n = len(df)
    test_size = test_size_days * 24  # hours
    gap = gap_hours
    splits = []
    
    for i in range(n_splits):
        test_end = n - i * test_size
        test_start = test_end - test_size
        train_end = test_start - gap
        
        if train_end < test_size:
            print(f"  Warning: not enough training data for fold {n_splits - i}")
            break
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    
    splits.reverse()
    
    print(f"Walk-forward CV: {len(splits)} folds")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {i+1}: train={len(train_idx):,} rows, "
              f"test={len(test_idx):,} rows")
    
    return splits
