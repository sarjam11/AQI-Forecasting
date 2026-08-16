# %% [markdown]
# # ⚡ Phase 6 — Temporal Fusion Transformer & N-BEATS
#
# **This is the showstopper model.**
#
# TFT gives us three things the LSTM can't:
# 1. **Variable selection** — learns which features matter at each timestep
# 2. **Temporal attention** — interpretable attention over past timesteps
# 3. **Quantile forecasts** — prediction intervals, not just point estimates
#
# **Targets to beat:**
# - Ridge baseline: MAE 7.01, MAPE 26.4%
# - LSTM: MAE 7.13, MAPE 27.3%
#
# **⚠️ Enable GPU:** Runtime → Change runtime type → T4 GPU

# %% — Install Dependencies (Colab)
# !pip install pytorch-forecasting pytorch-lightning --quiet

# %% — Setup
import sys
import os
from pathlib import Path

# Colab: upload src.zip and data CSV first, or mount Google Drive
# Uncomment the approach you're using:

# APPROACH 1: Upload files
# from google.colab import files
# uploaded = files.upload()  # upload src.zip, then delhi_aqi_features.csv

# APPROACH 2: If src is already extracted
if os.path.exists("/content/src"):
    sys.path.insert(0, "/content")
elif os.path.exists("../src"):
    sys.path.insert(0, str(Path("..").resolve()))

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
else:
    print("  ⚠️ No GPU detected! TFT will be very slow on CPU.")
    print("  Go to Runtime → Change runtime type → T4 GPU")

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# %% — Load Data
# Find the features CSV
candidates = [
    Path("delhi_aqi_features.csv"),
    Path("data/processed/delhi_aqi_features.csv"),
    Path("../data/processed/delhi_aqi_features.csv"),
]
data_path = None
for p in candidates:
    if p.exists():
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError(
        "delhi_aqi_features.csv not found! Upload it or check the path."
    )

print(f"Loading: {data_path}")
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Auto-detect target
target = "pm25"
if target not in df.columns:
    for col in df.columns:
        if "pm2" in col.lower():
            target = col
            break

# Clean up
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill().fillna(0)

print(f"Shape: {df.shape}, Target: {target}")
print(f"Date range: {df.index.min()} → {df.index.max()}")

# %% [markdown]
# ## 1. Prepare Data for PyTorch Forecasting
#
# PyTorch Forecasting's `TimeSeriesDataSet` requires a specific format:
# - Integer `time_idx` column (sequential timestep counter)
# - `group_ids` column (even for single series)
# - Explicit classification of features into known/unknown, real/categorical

# %% — Format Data for TFT
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data import NaNLabelEncoder
print(f"PyTorch Forecasting: {pytorch_forecasting.__version__}")

# Reset index to get datetime as a column
df_tft = df.copy()
df_tft = df_tft.reset_index()
df_tft = df_tft.rename(columns={df_tft.columns[0]: "datetime"})

# Create required columns
df_tft["time_idx"] = np.arange(len(df_tft))
df_tft["group"] = "delhi"  # single series

# ── Feature classification ──
# TFT distinguishes between:
#   - time_varying_known_reals: features known in advance (calendar, flags)
#   - time_varying_unknown_reals: features only known up to present (target, weather, pollutants)

# Known future features (we know the calendar in advance)
known_reals = []
known_candidates = [
    "hour_sin", "hour_cos", "hour_sin2", "hour_cos2",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "doy_sin", "doy_cos",
    "is_weekend", "rush_hour", "night_flag",
    "diwali_window", "stubble_season", "winter_flag",
    "solar_proxy",
]
for col in known_candidates:
    if col in df_tft.columns:
        known_reals.append(col)

print(f"Known future reals ({len(known_reals)}): {known_reals}")

# Unknown features (only known up to current time)
# Start with target, then add key covariates
unknown_reals = [target]

# Add raw pollutants and weather (if available)
raw_candidates = [
    "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "wind_speed", "pressure",
]
for col in raw_candidates:
    if col in df_tft.columns:
        unknown_reals.append(col)

# Add domain features
domain_candidates = [
    "stagnation_index", "inversion_proxy", "ventilation_coeff",
    "dewpoint_depression", "fog_risk",
    "pm25_pm10_ratio",
]
for col in domain_candidates:
    if col in df_tft.columns:
        unknown_reals.append(col)

# Add key lag features (not all — TFT has internal memory)
lag_candidates = [
    "pm25_lag_1", "pm25_lag_24", "pm25_lag_168",
    "pm25_rmean_24", "pm25_rstd_24",
    "pm25_rmean_168",
    "pm25_diff_1", "pm25_diff_24",
]
for col in lag_candidates:
    if col in df_tft.columns:
        unknown_reals.append(col)

# Remove duplicates
unknown_reals = list(dict.fromkeys(unknown_reals))

print(f"Unknown reals ({len(unknown_reals)}): {unknown_reals}")

# %% — Train/Val/Test Split Indices
# PyTorch Forecasting handles splitting via time_idx cutoffs

n = len(df_tft)
train_cutoff = int(n * 0.70)
val_cutoff = int(n * 0.85)

print(f"\nSplit points:")
print(f"  Train: time_idx 0 → {train_cutoff} ({train_cutoff} rows)")
print(f"  Val:   time_idx {train_cutoff} → {val_cutoff} ({val_cutoff - train_cutoff} rows)")
print(f"  Test:  time_idx {val_cutoff} → {n} ({n - val_cutoff} rows)")

# %% — Create TimeSeriesDataSet
MAX_ENCODER_LENGTH = 168   # 7 days lookback
MAX_PREDICTION_LENGTH = 24  # 24 hours forecast

# Check for and remove any columns with all-zero variance (TFT doesn't like these)
low_var_cols = []
for col in known_reals + unknown_reals:
    if col in df_tft.columns and col != target:
        if df_tft.loc[:train_cutoff, col].std() < 1e-8:
            low_var_cols.append(col)

if low_var_cols:
    print(f"\nRemoving {len(low_var_cols)} zero-variance columns: {low_var_cols}")
    known_reals = [c for c in known_reals if c not in low_var_cols]
    unknown_reals = [c for c in unknown_reals if c not in low_var_cols]

training = TimeSeriesDataSet(
    df_tft[df_tft.time_idx <= train_cutoff],
    time_idx="time_idx",
    target=target,
    group_ids=["group"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    static_categoricals=["group"],
    time_varying_known_reals=["time_idx"] + known_reals,
    time_varying_unknown_reals=unknown_reals,
    target_normalizer=GroupNormalizer(
        groups=["group"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

# Validation set (from training dataset definition)
validation = TimeSeriesDataSet.from_dataset(
    training,
    df_tft[(df_tft.time_idx > train_cutoff) & (df_tft.time_idx <= val_cutoff)],
    predict=True,
    stop_randomization=True,
)

# Test set
testing = TimeSeriesDataSet.from_dataset(
    training,
    df_tft[df_tft.time_idx > val_cutoff],
    predict=True,
    stop_randomization=True,
)

print(f"\nDatasets created:")
print(f"  Train: {len(training)} samples")
print(f"  Val:   {len(validation)} samples")
print(f"  Test:  {len(testing)} samples")

# %% — DataLoaders
BATCH_SIZE = 64

train_dataloader = training.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0
)
test_dataloader = testing.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0
)

# Quick check
x, y = next(iter(train_dataloader))
print(f"\nBatch shapes:")
for key in ["encoder_cont", "decoder_cont", "encoder_target", "decoder_target"]:
    if key in x:
        print(f"  {key}: {x[key].shape}")
    elif key == "decoder_target":
        print(f"  target: {y[0].shape}")

# %% [markdown]


# %% — Configure TFT
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    output_size=7,            # 7 quantiles: 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=5,
    optimizer="adam",
)

print(f"TFT parameters: {tft.size()/1e3:.1f}k")

# %% — Train TFT
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

trainer = pl.Trainer(
    max_epochs=80,
    accelerator="auto",           # uses GPU if available
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="best-tft-{epoch}-{val_loss:.4f}",
            save_top_k=1,
        ),
    ],
    enable_model_summary=True,
    log_every_n_steps=20,
)

print("Starting TFT training...")
print("(This takes 10-30 minutes on T4 GPU, much longer on CPU)\n")

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# %% — Load Best Model
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"Best model: {best_model_path}")

best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# %% [markdown]
# ## 3. TFT Evaluation

# %% — Generate Predictions
# Raw predictions include all quantiles
raw_predictions = best_tft.predict(
    test_dataloader,
    mode="raw",
    return_x=True,
    trainer_kwargs=dict(accelerator="auto"),
)

# Point predictions (median = 0.5 quantile)
point_predictions = best_tft.predict(
    test_dataloader,
    mode="prediction",
    trainer_kwargs=dict(accelerator="auto"),
)

# Get actuals
actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])

print(f"Predictions shape: {point_predictions.shape}")
print(f"Actuals shape: {actuals.shape}")

# %% — Calculate Metrics
from src.evaluation.metrics import calculate_metrics

tft_preds = point_predictions.cpu().numpy()
tft_actuals = actuals.cpu().numpy()

# 1-step ahead
tft_metrics_1step = calculate_metrics(tft_actuals[:, 0], tft_preds[:, 0])

# Per-horizon metrics
horizon_results = []
for h in range(MAX_PREDICTION_LENGTH):
    m = calculate_metrics(tft_actuals[:, h], tft_preds[:, h])
    m["horizon"] = h + 1
    horizon_results.append(m)

horizon_df = pd.DataFrame(horizon_results).set_index("horizon")

# Average across all horizons
avg_mae = horizon_df["MAE"].mean()
avg_mape = horizon_df["MAPE"].mean()

print(f"\nTFT Results:")
print(f"  1-Step:  MAE = {tft_metrics_1step['MAE']}, MAPE = {tft_metrics_1step['MAPE']}%")
print(f"  24h Avg: MAE = {avg_mae:.2f}, MAPE = {avg_mape:.2f}%")
print(f"\nPer-horizon:")
print(horizon_df.to_string())

# %% — Horizon Degradation Comparison (TFT vs LSTM vs Ridge)
fig, ax = plt.subplots(figsize=(12, 5))

# TFT
tft_maes = [horizon_df.loc[h+1, "MAE"] for h in range(MAX_PREDICTION_LENGTH)]
ax.plot(range(1, 25), tft_maes, "o-", color="#2563EB",
        linewidth=2, markersize=5, label="TFT")

# LSTM (approximate — flat at ~7.15 from your results)
ax.axhline(y=7.15, color="#7C3AED", linestyle="-.",
           linewidth=1.5, label="LSTM (~7.15 flat)")

# Ridge baseline
ax.axhline(y=7.01, color="#DC2626", linestyle="--",
           linewidth=1.5, label="Ridge Baseline (7.01)")

# Naive
ax.axhline(y=9.98, color="#94A3B8", linestyle=":",
           linewidth=1, label="Naive (9.98)")

ax.set_xlabel("Forecast Horizon (hours ahead)")
ax.set_ylabel("MAE (µg/m³)")
ax.set_title("MAE vs Forecast Horizon — All Models")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, 25))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "23_all_models_horizon.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. TFT Interpretability — The Key Differentiator
#
# This section is what makes TFT worth implementing. These plots go straight
# into your README and are powerful interview talking points.

# %% — Variable Importance
interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")

fig = best_tft.plot_interpretation(interpretation)
plt.suptitle("TFT Feature Importance — What Drives Delhi's Air Quality?", y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "24_tft_variable_importance.png",
            dpi=150, bbox_inches="tight")
plt.show()

# Print top features
print("\nEncoder Variable Importance (past inputs):")
enc_imp = interpretation["encoder_variables"]
for name, imp in sorted(zip(training.encoder_variables, enc_imp),
                         key=lambda x: -x[1])[:15]:
    bar = "█" * int(imp / enc_imp.max() * 30)
    print(f"  {name:35s} {imp:.4f} {bar}")

print("\nDecoder Variable Importance (known future inputs):")
dec_imp = interpretation["decoder_variables"]
for name, imp in sorted(zip(training.decoder_variables, dec_imp),
                         key=lambda x: -x[1])[:10]:
    bar = "█" * int(imp / dec_imp.max() * 30)
    print(f"  {name:35s} {imp:.4f} {bar}")

# %% — Attention Weights Visualization
# Shows which past timesteps the model attends to most
fig = best_tft.plot_interpretation(interpretation)
plt.tight_layout()
plt.show()

# Custom attention heatmap for specific predictions
fig, ax = plt.subplots(figsize=(14, 3))

# Average attention across all test samples
attn = interpretation["attention"]  # (encoder_length,)
hours_back = range(-MAX_ENCODER_LENGTH, 0)

ax.bar(hours_back, attn.cpu().numpy(), color="#2563EB", alpha=0.7, width=1.0)
ax.set_xlabel("Hours Before Forecast (0 = forecast start)")
ax.set_ylabel("Attention Weight")
ax.set_title("TFT Temporal Attention — Which Past Hours Matter Most?")

# Mark key lags
for lag, label in [(-1, "1h ago"), (-24, "24h ago"),
                    (-48, "2 days"), (-168, "1 week")]:
    if lag >= -MAX_ENCODER_LENGTH:
        ax.axvline(lag, color="#DC2626", linestyle=":", alpha=0.5)
        ax.text(lag, attn.max().item() * 0.9, label,
                ha="center", fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "25_tft_attention_weights.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Probabilistic Forecasts (Quantile Predictions)
#
# TFT outputs prediction intervals, not just point estimates.
# This is critical for real-world air quality warnings.

# %% — Forecast with Confidence Intervals
# Pick a few test samples to visualize
n_samples = 4
fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3.5 * n_samples))

for i in range(n_samples):
    ax = axes[i]
    idx = i * (len(tft_actuals) // n_samples)  # spread across test set

    hours = range(1, MAX_PREDICTION_LENGTH + 1)
    actual = tft_actuals[idx]

    # Get quantile predictions for this sample
    # raw_predictions.output shape: (n_samples, horizon, n_quantiles)
    quantiles = raw_predictions.output[idx].cpu().numpy()
    # quantile indices: 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98

    median = quantiles[:, 3]   # 0.5 quantile (median)
    q10 = quantiles[:, 1]      # 10th percentile
    q90 = quantiles[:, 5]      # 90th percentile
    q25 = quantiles[:, 2]      # 25th percentile
    q75 = quantiles[:, 4]      # 75th percentile

    # Plot
    ax.plot(hours, actual, "o-", color="#000000", linewidth=2,
            markersize=4, label="Actual", zorder=5)
    ax.plot(hours, median, "s-", color="#2563EB", linewidth=2,
            markersize=4, label="TFT Median")

    # 80% interval (10-90)
    ax.fill_between(hours, q10, q90, alpha=0.15, color="#2563EB",
                     label="80% interval")
    # 50% interval (25-75)
    ax.fill_between(hours, q25, q75, alpha=0.25, color="#2563EB",
                     label="50% interval")

    mae_sample = np.mean(np.abs(actual - median))
    coverage = np.mean((actual >= q10) & (actual <= q90)) * 100
    ax.set_title(f"Sample {i+1} — MAE: {mae_sample:.1f} µg/m³, "
                 f"80% interval coverage: {coverage:.0f}%")
    ax.set_ylabel("PM2.5 (µg/m³)")
    if i == 0:
        ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Hours Ahead")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "26_tft_quantile_forecasts.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %% — Overall Calibration Check
# For well-calibrated intervals, ~80% of actuals should fall within 10-90% interval
print("\nQuantile Calibration:")
quantile_preds = raw_predictions.output.cpu().numpy()  # (samples, horizon, 7)
# Quantile order: 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98

for qi, qname, expected in [(0, "2%", 2), (1, "10%", 10), (2, "25%", 25),
                              (4, "75%", 75), (5, "90%", 90), (6, "98%", 98)]:
    below = (tft_actuals < quantile_preds[:, :, qi]).mean() * 100
    print(f"  Below {qname} quantile: {below:.1f}% (expected ~{expected}%)")

coverage_80 = ((tft_actuals >= quantile_preds[:, :, 1]) &
               (tft_actuals <= quantile_preds[:, :, 5])).mean() * 100
print(f"\n  80% interval coverage: {coverage_80:.1f}% (expected ~80%)")

# %% [markdown]
# ## 6. N-BEATS (Bonus Model)
#
# Pure deep learning — no hand-crafted features, decomposes into
# trend + seasonality. Quick to train and a good comparison point.

# %% — Train N-BEATS
from pytorch_forecasting import NBeats

# N-BEATS uses a simpler dataset (target only, no covariates)
nbeats_training = TimeSeriesDataSet(
    df_tft[df_tft.time_idx <= train_cutoff],
    time_idx="time_idx",
    target=target,
    group_ids=["group"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    time_varying_unknown_reals=[target],
    target_normalizer=GroupNormalizer(
        groups=["group"], transformation="softplus"
    ),
    allow_missing_timesteps=True,
)

nbeats_validation = TimeSeriesDataSet.from_dataset(
    nbeats_training,
    df_tft[(df_tft.time_idx > train_cutoff) & (df_tft.time_idx <= val_cutoff)],
    predict=True,
    stop_randomization=True,
)

nbeats_testing = TimeSeriesDataSet.from_dataset(
    nbeats_training,
    df_tft[df_tft.time_idx > val_cutoff],
    predict=True,
    stop_randomization=True,
)

nbeats_train_dl = nbeats_training.to_dataloader(train=True, batch_size=64, num_workers=0)
nbeats_val_dl = nbeats_validation.to_dataloader(train=False, batch_size=64, num_workers=0)
nbeats_test_dl = nbeats_testing.to_dataloader(train=False, batch_size=64, num_workers=0)

nbeats = NBeats.from_dataset(
    nbeats_training,
    learning_rate=0.001,
    widths=[256, 2048],
    backcast_loss_ratio=1.0,
)

print(f"N-BEATS parameters: {nbeats.size()/1e3:.1f}k")

trainer_nbeats = pl.Trainer(
    max_epochs=50,
    accelerator="auto",
    gradient_clip_val=0.5,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=8, mode="min", verbose=True),
        ModelCheckpoint(monitor="val_loss", mode="min",
                        filename="best-nbeats-{epoch}-{val_loss:.4f}"),
    ],
    log_every_n_steps=20,
)

print("Training N-BEATS...")
trainer_nbeats.fit(nbeats, train_dataloaders=nbeats_train_dl,
                    val_dataloaders=nbeats_val_dl)

# %% — N-BEATS Evaluation
best_nbeats = NBeats.load_from_checkpoint(
    trainer_nbeats.checkpoint_callback.best_model_path
)

nbeats_predictions = best_nbeats.predict(
    nbeats_test_dl, mode="prediction",
    trainer_kwargs=dict(accelerator="auto"),
)
nbeats_actuals = torch.cat([y[0] for x, y in iter(nbeats_test_dl)])

nb_preds = nbeats_predictions.cpu().numpy()
nb_actuals = nbeats_actuals.cpu().numpy()

nb_metrics_1step = calculate_metrics(nb_actuals[:, 0], nb_preds[:, 0])
nb_avg_mae = np.mean([
    calculate_metrics(nb_actuals[:, h], nb_preds[:, h])["MAE"]
    for h in range(MAX_PREDICTION_LENGTH)
])

print(f"\nN-BEATS Results:")
print(f"  1-Step:  MAE = {nb_metrics_1step['MAE']}, MAPE = {nb_metrics_1step['MAPE']}%")
print(f"  24h Avg: MAE = {nb_avg_mae:.2f}")

# %% [markdown]
# ## 7. Final Comparison — All Models

# %% — Summary Table
print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    FINAL MODEL COMPARISON                         ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Model                  │ MAE (1-step) │ MAPE (1-step) │ 24h Avg ║
║  ───────────────────────┼──────────────┼───────────────┼─────────║
║  Naive Persistence      │    9.98      │    35.2%      │   —     ║
║  Seasonal Naive (24h)   │    9.98      │    35.7%      │   —     ║
║  Moving Avg (48h)       │    9.53      │    38.0%      │   —     ║
║  Linear Ridge           │    7.01      │    26.4%      │   —     ║
║  LSTM (from scratch)    │    7.13      │    27.3%      │  7.16   ║
║  N-BEATS                │    {nb_metrics_1step['MAE']:<9} │    {nb_metrics_1step['MAPE']:<9}%│  {nb_avg_mae:<6.2f}  ║
║  TFT                    │    {tft_metrics_1step['MAE']:<9} │    {tft_metrics_1step['MAPE']:<9}%│  {avg_mae:<6.2f}  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")

# Improvements over Ridge baseline
for name, mae in [("LSTM", 7.13), ("N-BEATS", nb_metrics_1step["MAE"]),
                   ("TFT", tft_metrics_1step["MAE"])]:
    imp = (1 - mae / 7.01) * 100
    print(f"  {name} vs Ridge: {imp:+.1f}% MAE")

# %% — Combined Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = ["Naive", "Seasonal\nNaive", "MA (48h)", "Ridge", "LSTM",
          "N-BEATS", "TFT"]
maes = [9.98, 9.98, 9.53, 7.01, 7.13, nb_metrics_1step["MAE"],
        tft_metrics_1step["MAE"]]
mapes = [35.2, 35.7, 38.0, 26.4, 27.3, nb_metrics_1step["MAPE"],
         tft_metrics_1step["MAPE"]]

colors = ["#94A3B8", "#94A3B8", "#94A3B8", "#F59E0B",
          "#7C3AED", "#059669", "#2563EB"]

# MAE
axes[0].bar(models, maes, color=colors)
axes[0].set_ylabel("MAE (µg/m³)")
axes[0].set_title("1-Step MAE — All Models")
axes[0].axhline(y=7.01, color="#DC2626", linestyle="--",
                linewidth=1, alpha=0.5)

for i, v in enumerate(maes):
    axes[0].text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=9)

# MAPE
axes[1].bar(models, mapes, color=colors)
axes[1].set_ylabel("MAPE (%)")
axes[1].set_title("1-Step MAPE — All Models")
axes[1].axhline(y=26.4, color="#DC2626", linestyle="--",
                linewidth=1, alpha=0.5)

for i, v in enumerate(mapes):
    axes[1].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "27_final_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Save All Results
results = {
    "Naive Persistence": {"MAE": 9.98, "MAPE": 35.2, "type": "baseline"},
    "Seasonal Naive": {"MAE": 9.98, "MAPE": 35.7, "type": "baseline"},
    "Moving Avg (48h)": {"MAE": 9.53, "MAPE": 38.0, "type": "baseline"},
    "Linear Ridge": {"MAE": 7.01, "MAPE": 26.4, "type": "baseline"},
    "LSTM": {"MAE": 7.13, "MAPE": 27.3, "type": "deep_learning"},
    "N-BEATS": {"MAE": nb_metrics_1step["MAE"],
                "MAPE": nb_metrics_1step["MAPE"], "type": "deep_learning"},
    "TFT": {"MAE": tft_metrics_1step["MAE"],
            "MAPE": tft_metrics_1step["MAPE"], "type": "deep_learning"},
}

results_df = pd.DataFrame(results).T
results_df.to_csv("final_model_comparison.csv")
print("Results saved to: final_model_comparison.csv")

# %% [markdown]
# ## 8. Key Takeaways for README / Interviews
#
# Document these findings:
#
# 1. **Feature engineering matters more than architecture** — Ridge with good
#    features matched or beat a vanilla LSTM. This shows domain knowledge
#    (stagnation index, inversion proxy) adds real predictive value.
#
# 2. **TFT's variable importance reveals the physics** — if wind_speed,
#    humidity, and recent PM2.5 lags dominate, that confirms the atmospheric
#    science: stagnant humid conditions trap particulates.
#
# 3. **Attention weights show learned seasonality** — peaks at 24h and 168h
#    lags mean the model independently discovered the daily and weekly cycles.
#
# 4. **Probabilistic forecasts enable decision-making** — "there's a 90%
#    chance PM2.5 stays below 150" is more useful than "PM2.5 will be 120."
#
# 5. **Winter remains hardest** — all models have higher errors Nov–Feb.
#    This is honest analysis, not a failure.

print("\n✓ Phase 6 complete!")
print("  Next: Phase 7 — Deployment (FastAPI + Streamlit dashboard)")
