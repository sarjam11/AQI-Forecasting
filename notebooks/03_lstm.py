# %% [markdown]
# # 🧠 Phase 5 — LSTM from Scratch in PyTorch
#
# Building a multi-step PM2.5 forecaster using a custom LSTM architecture.
#
# **Target to beat:** Linear Ridge baseline — MAE 7.01, MAPE 26.4%
#
# Architecture:
# ```
# Input (168h, 80+ features)
#   → 2-layer LSTM (128 hidden)
#   → LayerNorm
#   → FC decoder (128 → 64 → 24)
#   → Output (24h forecast)
# ```

# %% — Setup
import sys
from pathlib import Path

# ── IMPORTANT: Set project root ──
# If running in Colab, adjust this path after uploading files.
# If running locally, set to your aqi-forecasting directory.
project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name()}")

FIGURES_DIR = Path("../figures")
FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)

# %% — Load Feature-Engineered Data
data_path = project_root / "data" / "processed" / "delhi_aqi_features.csv"
if not data_path.exists():
    # Colab fallback: look in current directory
    data_path = Path("delhi_aqi_features.csv")

print(f"Loading: {data_path}")
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Auto-detect target
target = "pm25"
if target not in df.columns:
    for col in df.columns:
        if "pm2" in col.lower():
            target = col
            break

# Convert all to numeric, drop non-numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle NaN/inf
df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill().fillna(0)

print(f"Shape: {df.shape}")
print(f"Target: {target}")
print(f"Date range: {df.index.min()} → {df.index.max()}")

# %% — Train/Val/Test Split
from src.training.split import temporal_split, scale_data

train_raw, val_raw, test_raw = temporal_split(df, val_frac=0.15, test_frac=0.15)

# Scale using RobustScaler (fit on train only)
train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler = scale_data(
    train_raw, val_raw, test_raw, target=target, method="robust"
)

print(f"\nScaled train sample — {target}: "
      f"mean={train_scaled[target].mean():.3f}, "
      f"std={train_scaled[target].std():.3f}")

# %% — Create DataLoaders
from src.data.dataset import create_dataloaders

LOOKBACK = 168   # 7 days of hourly input
HORIZON = 24     # 24 hours ahead forecast
BATCH_SIZE = 64

train_loader, val_loader, test_loader, feat_info = create_dataloaders(
    train_scaled, val_scaled, test_scaled,
    target=target,
    lookback=LOOKBACK,
    horizon=HORIZON,
    batch_size=BATCH_SIZE,
    num_workers=0,  # Set 0 for Windows/Colab compatibility
)

# Quick sanity check
x_sample, y_sample = next(iter(train_loader))
print(f"\nSanity check:")
print(f"  X batch: {x_sample.shape}  (batch, lookback, features)")
print(f"  Y batch: {y_sample.shape}  (batch, horizon)")

# %% [markdown]
# ## Model Definition

# %% — Create LSTM Model
from src.models.lstm_model import LSTMForecaster, LSTMForecasterWithAttention

model = LSTMForecaster(
    input_size=feat_info["n_features"],
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    horizon=HORIZON,
    bidirectional=False,
)

print(f"Model: {model.__class__.__name__}")
print(f"Parameters: {model.count_parameters():,}")
print(f"\nArchitecture:\n{model}")

# %% [markdown]
# ## Training

# %% — Train the Model
from src.training.train_lstm import train_model, plot_training_history

training_config = {
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "grad_clip": 1.0,
    "loss": "huber",       # "huber", "mse", or "mae"
    "lookback": LOOKBACK,
    "horizon": HORIZON,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": BATCH_SIZE,
}

model, history = train_model(
    model, train_loader, val_loader,
    config=training_config,
    model_name="lstm_v1",
    use_wandb=False,       # Set True if you have W&B configured
    save_dir=str(MODELS_DIR),
)

# %% — Plot Training Curves
plot_training_history(history, save_path=FIGURES_DIR / "19_lstm_training.png")

# %% [markdown]
# ## Evaluation

# %% — Generate Test Predictions
from src.training.train_lstm import evaluate_model

preds_scaled, actuals_scaled = evaluate_model(model, test_loader, device)

# Inverse transform to original scale
# target_scaler was fit on the target column only
preds_original = target_scaler.inverse_transform(preds_scaled)
actuals_original = target_scaler.inverse_transform(actuals_scaled)

print(f"\nOriginal scale:")
print(f"  Predictions range: [{preds_original.min():.1f}, {preds_original.max():.1f}]")
print(f"  Actuals range: [{actuals_original.min():.1f}, {actuals_original.max():.1f}]")

# %% — Calculate Metrics
from src.evaluation.metrics import calculate_metrics, calculate_metrics_by_horizon

# 1-step ahead metrics (first hour of each window)
metrics_1step = calculate_metrics(actuals_original[:, 0], preds_original[:, 0])
print(f"\n1-Step Ahead (next hour):")
print(f"  MAE:  {metrics_1step['MAE']}")
print(f"  RMSE: {metrics_1step['RMSE']}")
print(f"  MAPE: {metrics_1step['MAPE']}%")

# Full horizon average
all_mae = np.mean([
    calculate_metrics(actuals_original[:, h], preds_original[:, h])["MAE"]
    for h in range(HORIZON)
])
all_mape = np.mean([
    calculate_metrics(actuals_original[:, h], preds_original[:, h])["MAPE"]
    for h in range(HORIZON)
])
print(f"\nAverage across all {HORIZON} horizons:")
print(f"  MAE:  {all_mae:.2f}")
print(f"  MAPE: {all_mape:.2f}%")

# Metrics by horizon
print(f"\nMetrics by forecast horizon:")
horizon_metrics = calculate_metrics_by_horizon(actuals_original, preds_original)
print(horizon_metrics.to_string())

# %% — Horizon Degradation Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

horizons = range(1, HORIZON + 1)
maes = [calculate_metrics(actuals_original[:, h-1], preds_original[:, h-1])["MAE"]
        for h in horizons]
mapes = [calculate_metrics(actuals_original[:, h-1], preds_original[:, h-1])["MAPE"]
         for h in horizons]

axes[0].plot(horizons, maes, "o-", color="#2563EB", linewidth=2, markersize=4)
axes[0].axhline(y=7.01, color="#DC2626", linestyle="--",
                linewidth=1.5, label="Ridge Baseline (7.01)")
axes[0].set_xlabel("Forecast Horizon (hours ahead)")
axes[0].set_ylabel("MAE (µg/m³)")
axes[0].set_title("MAE vs Forecast Horizon — LSTM")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(horizons, mapes, "o-", color="#059669", linewidth=2, markersize=4)
axes[1].axhline(y=26.43, color="#DC2626", linestyle="--",
                linewidth=1.5, label="Ridge Baseline (26.4%)")
axes[1].set_xlabel("Forecast Horizon (hours ahead)")
axes[1].set_ylabel("MAPE (%)")
axes[1].set_title("MAPE vs Forecast Horizon — LSTM")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% — Sample Forecast Plots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Pick 3 random test windows
np.random.seed(42)
sample_indices = np.random.choice(len(actuals_original), 3, replace=False)

for i, idx in enumerate(sample_indices):
    ax = axes[i]
    hours = range(1, HORIZON + 1)

    ax.plot(hours, actuals_original[idx], "o-", color="#000000",
            linewidth=2, markersize=4, label="Actual")
    ax.plot(hours, preds_original[idx], "s--", color="#2563EB",
            linewidth=2, markersize=4, label="LSTM Prediction")

    # Error band
    errors = np.abs(actuals_original[idx] - preds_original[idx])
    ax.fill_between(hours,
                     preds_original[idx] - errors,
                     preds_original[idx] + errors,
                     alpha=0.1, color="#2563EB")

    mae_sample = np.mean(errors)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"Sample {i+1} — 24h Forecast (MAE: {mae_sample:.1f})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Hours Ahead")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Comparison with Baselines

# %% — Summary Table
print(f"""
╔══════════════════════════════════════════════════════════╗
║                 LSTM vs BASELINES                        ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Baselines (1-step ahead):                               ║
║    Naive Persistence:   MAE = 9.98,  MAPE = 35.2%       ║
║    Seasonal Naive:      MAE = 9.98,  MAPE = 35.7%       ║
║    Moving Avg (48h):    MAE = 9.53,  MAPE = 38.0%       ║
║    Linear Ridge:        MAE = 7.01,  MAPE = 26.4%       ║
║                                                          ║
║  LSTM (1-step ahead):                                    ║
║    MAE:  {metrics_1step['MAE']:<8}                               ║
║    MAPE: {metrics_1step['MAPE']:<8}%                              ║
║                                                          ║
║  LSTM (avg across 24h horizon):                          ║
║    MAE:  {all_mae:<8.2f}                                       ║
║    MAPE: {all_mape:<8.2f}%                                      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

improvement_mae = (1 - metrics_1step["MAE"] / 7.01) * 100
improvement_naive = (1 - metrics_1step["MAE"] / 9.98) * 100
print(f"MAE improvement over Ridge baseline: {improvement_mae:.1f}%")
print(f"MAE improvement over Naive: {improvement_naive:.1f}%")

# %% [markdown]
# ## Optional: LSTM with Attention

# %% — Train LSTM + Attention (uncomment to run)
"""
model_attn = LSTMForecasterWithAttention(
    input_size=feat_info["n_features"],
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    horizon=HORIZON,
)

print(f"LSTM + Attention parameters: {model_attn.count_parameters():,}")

model_attn, history_attn = train_model(
    model_attn, train_loader, val_loader,
    config=training_config,
    model_name="lstm_attention_v1",
    save_dir=str(MODELS_DIR),
)

# Evaluate
preds_attn_scaled, _ = evaluate_model(model_attn, test_loader, device)
preds_attn = target_scaler.inverse_transform(preds_attn_scaled)

metrics_attn = calculate_metrics(actuals_original[:, 0], preds_attn[:, 0])
print(f"LSTM+Attention 1-step: MAE={metrics_attn['MAE']}, MAPE={metrics_attn['MAPE']}%")

# Attention weight visualization
# Shows which past timesteps the model focuses on
model_attn.eval()
with torch.no_grad():
    sample_x, _ = next(iter(test_loader))
    _ = model_attn(sample_x.to(device))
    attn_weights = model_attn.get_attention_weights()

fig, ax = plt.subplots(figsize=(14, 3))
avg_attn = attn_weights.mean(axis=0)  # average across batch
ax.bar(range(LOOKBACK), avg_attn, color="#7C3AED", alpha=0.7)
ax.set_xlabel("Lookback Hour (0 = 168h ago, 167 = most recent)")
ax.set_ylabel("Attention Weight")
ax.set_title("LSTM Temporal Attention — Which Past Hours Matter Most?")

# Mark key lags
for lag, label in [(LOOKBACK-1, "t-1"), (LOOKBACK-24, "t-24"),
                    (LOOKBACK-48, "t-48"), (0, "t-168")]:
    if 0 <= lag < LOOKBACK:
        ax.axvline(lag, color="#DC2626", linestyle=":", alpha=0.5)
        ax.text(lag, avg_attn.max() * 0.95, label, ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "22_attention_weights.png", dpi=150, bbox_inches="tight")
plt.show()
"""

print("Uncomment the cell above to train LSTM + Attention variant.")
print("\nPhase 5 complete. Next: Phase 6 — Temporal Fusion Transformer (TFT)")
