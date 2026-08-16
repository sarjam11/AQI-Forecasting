"""
Evaluation metrics for time-series forecasting.

Includes:
  - MAE, RMSE, MAPE, MASE (standard forecasting metrics)
  - Diebold-Mariano test (statistical significance of model differences)
  - Plotting utilities for forecast comparison

Usage:
    from src.evaluation.metrics import calculate_metrics, plot_forecast
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred, naive_mae=None):
    """
    Calculate standard forecasting metrics.
    
    Args:
        y_true: actual values (array-like)
        y_pred: predicted values (array-like)
        naive_mae: MAE of naive persistence forecast (for MASE).
                   If None, computed from y_true.
    
    Returns:
        dict with MAE, RMSE, MAPE, MASE
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Remove NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "MASE": np.nan}
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE — exclude near-zero actuals to avoid division explosion
    nonzero = y_true > 1.0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    else:
        mape = np.nan
    
    # MASE — scale-independent metric
    if naive_mae is None:
        naive_mae = mean_absolute_error(y_true[1:], y_true[:-1])
    mase = mae / (naive_mae + 1e-8)
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "MASE": round(mase, 4),
    }


def calculate_metrics_by_horizon(y_true_multi, y_pred_multi, horizons=None):
    """
    Calculate metrics at different forecast horizons.
    
    Args:
        y_true_multi: 2D array (n_samples, max_horizon)
        y_pred_multi: 2D array (n_samples, max_horizon)
        horizons: list of horizons to evaluate (default: [1, 6, 12, 24])
    
    Returns:
        DataFrame with metrics per horizon
    """
    if horizons is None:
        max_h = y_pred_multi.shape[1]
        horizons = [h for h in [1, 3, 6, 12, 24, 48, 72] if h <= max_h]
    
    results = []
    for h in horizons:
        metrics = calculate_metrics(
            y_true_multi[:, h - 1],
            y_pred_multi[:, h - 1]
        )
        metrics["horizon"] = h
        results.append(metrics)
    
    return pd.DataFrame(results).set_index("horizon")


def diebold_mariano_test(errors_1, errors_2, horizon=1):
    """
    Diebold-Mariano test for equal predictive accuracy.
    
    Tests H0: Model 1 and Model 2 have equal forecast accuracy.
    
    Args:
        errors_1: forecast errors from model 1
        errors_2: forecast errors from model 2
        horizon: forecast horizon (for autocorrelation adjustment)
    
    Returns:
        (dm_statistic, p_value, interpretation)
    """
    from scipy import stats
    
    e1 = np.asarray(errors_1, dtype=float)
    e2 = np.asarray(errors_2, dtype=float)
    
    # Loss differential (squared errors)
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    mean_d = np.mean(d)
    
    # Newey-West variance estimator (accounts for autocorrelation)
    var_d = np.var(d, ddof=1)
    for k in range(1, horizon):
        if k < T:
            gamma_k = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
            var_d += 2 * (1 - k / horizon) * gamma_k
    
    if var_d <= 0:
        return 0.0, 1.0, "Inconclusive (zero variance)"
    
    dm_stat = mean_d / np.sqrt(var_d / T)
    p_value = 2 * stats.t.sf(abs(dm_stat), df=T - 1)
    
    if p_value < 0.01:
        interp = "Highly significant difference"
    elif p_value < 0.05:
        interp = "Significant difference"
    elif p_value < 0.10:
        interp = "Marginally significant"
    else:
        interp = "No significant difference"
    
    # Which model is better?
    if p_value < 0.05:
        if mean_d > 0:
            interp += " — Model 2 is better"
        else:
            interp += " — Model 1 is better"
    
    return dm_stat, p_value, interp


def print_comparison_table(results_dict):
    """
    Print a formatted comparison table of model results.
    
    Args:
        results_dict: {"Model Name": {"MAE": x, "RMSE": y, ...}, ...}
    """
    df = pd.DataFrame(results_dict).T
    df.index.name = "Model"
    
    # Highlight best values
    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)
    print(df.to_string())
    print("-" * 65)
    
    best_model = df["MAE"].idxmin()
    print(f"\n  Best model (by MAE): {best_model}")
    print(f"  MAE improvement over naive: "
          f"{(1 - df.loc[best_model, 'MAE'] / df.iloc[0]['MAE']) * 100:.1f}%")
    
    return df
