# %% [markdown]
# # ⚡ Phase 6 — Temporal Fusion Transformer & N-BEATS
#
# **Targets to beat:**
# - Ridge baseline: MAE 7.01, MAPE 26.4%
# - LSTM: MAE 7.13, MAPE 27.3%
#
# Versions: pytorch_forecasting 1.8.0, lightning 2.6.5, numpy 2.0.2

# %% — Setup
import sys
import os
from pathlib import Path

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
print(f"PyTorch {torch.__version__}, Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name()}")

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# %% — Load Data
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
    raise FileNotFoundError("delhi_aqi_features.csv not found!")

print(f"Loading: {data_path}")
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

target = "pm25"
if target not in df.columns:
    for col in df.columns:
        if "pm2" in col.lower():
            target = col
            break

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill().fillna(0)

print(f"Shape: {df.shape}, Target: {target}")

# %% — Prepare for PyTorch Forecasting
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

print(f"pytorch_forecasting: {pytorch_forecasting.__version__}")

df_tft = df.reset_index()
df_tft = df_tft.rename(columns={df_tft.columns[0]: "datetime"})
df_tft["time_idx"] = np.arange(len(df_tft))
df_tft["group"] = "delhi"

# ── Feature classification ──
known_reals = []
known_candidates = [
    "hour_sin", "hour_cos", "hour_sin2", "hour_cos2",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "doy_sin", "doy_cos",
    "is_weekend", "rush_hour", "night_flag",
    "diwali_window", "stubble_season", "winter_flag",
    "solar_proxy",
]
for col in known_candidates:
    if col in df_tft.columns:
        known_reals.append(col)

unknown_reals = [target]
other_candidates = [
    "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "wind_speed", "pressure",
    "stagnation_index", "inversion_proxy", "ventilation_coeff",
    "dewpoint_depression", "fog_risk", "pm25_pm10_ratio",
    "pm25_lag_1", "pm25_lag_24", "pm25_lag_168",
    "pm25_rmean_24", "pm25_rstd_24", "pm25_rmean_168",
    "pm25_diff_1", "pm25_diff_24",
]
for col in other_candidates:
    if col in df_tft.columns:
        unknown_reals.append(col)

unknown_reals = list(dict.fromkeys(unknown_reals))

print(f"Known reals: {len(known_reals)}")
print(f"Unknown reals: {len(unknown_reals)}")

# %% — Split and remove zero-variance columns
MAX_ENCODER_LENGTH = 168
MAX_PREDICTION_LENGTH = 24

n = len(df_tft)
train_cutoff = int(n * 0.70)
val_cutoff = int(n * 0.85)

# Remove zero-variance columns
low_var = []
for col in known_reals + unknown_reals:
    if col != target and col in df_tft.columns:
        if df_tft.loc[:train_cutoff, col].std() < 1e-8:
            low_var.append(col)
if low_var:
    print(f"Removing zero-variance: {low_var}")
    known_reals = [c for c in known_reals if c not in low_var]
    unknown_reals = [c for c in unknown_reals if c not in low_var]

print(f"Train: 0-{train_cutoff}, Val: {train_cutoff}-{val_cutoff}, Test: {val_cutoff}-{n}")

# %% — Create Datasets
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

# Validation — include lookback buffer, NO predict=True
val_start = train_cutoff - MAX_ENCODER_LENGTH
validation = TimeSeriesDataSet.from_dataset(
    training,
    df_tft[(df_tft.time_idx >= val_start) & (df_tft.time_idx <= val_cutoff)],
    stop_randomization=True,
)

# Test — include lookback buffer, NO predict=True
test_start = val_cutoff - MAX_ENCODER_LENGTH
testing = TimeSeriesDataSet.from_dataset(
    training,
    df_tft[df_tft.time_idx >= test_start],
    stop_randomization=True,
)

print(f"\nDatasets: Train={len(training)}, Val={len(validation)}, Test={len(testing)}")

# %% — DataLoaders
BATCH_SIZE = 64

train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
test_dataloader = testing.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

x, y = next(iter(train_dataloader))
print(f"Batch check — encoder_cont: {x['encoder_cont'].shape}")

# %% [markdown]
# ## Train TFT

# %% — Configure TFT
from pytorch_forecasting import TemporalFusionTransformer

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=5,
    optimizer="adam",
)

print(f"TFT parameters: {tft.size()/1e3:.1f}k")

# %% — Train
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

trainer = pl.Trainer(
    max_epochs=80,
    accelerator="auto",
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=8, mode="min", verbose=True),
        ModelCheckpoint(
            monitor="val_loss", mode="min",
            filename="best-tft-{epoch}-{val_loss:.4f}", save_top_k=1,
        ),
    ],
    enable_model_summary=True,
    log_every_n_steps=20,
)

print("Starting TFT training...\n")
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# %% — Load Best Model
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"Best model: {best_model_path}")
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# %% [markdown]
# ## TFT Evaluation

# %% — Generate Predictions
raw_predictions = best_tft.predict(
    test_dataloader, mode="raw", return_x=True,
    trainer_kwargs=dict(accelerator="auto"),
)

point_predictions = best_tft.predict(
    test_dataloader, mode="prediction",
    trainer_kwargs=dict(accelerator="auto"),
)

actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])

tft_preds = point_predictions.cpu().numpy()
tft_actuals = actuals.cpu().numpy()

print(f"Predictions: {tft_preds.shape}, Actuals: {tft_actuals.shape}")

# %% — Metrics
from src.evaluation.metrics import calculate_metrics

tft_1step = calculate_metrics(tft_actuals[:, 0], tft_preds[:, 0])

horizon_results = []
for h in range(MAX_PREDICTION_LENGTH):
    m = calculate_metrics(tft_actuals[:, h], tft_preds[:, h])
    m["horizon"] = h + 1
    horizon_results.append(m)

horizon_df = pd.DataFrame(horizon_results).set_index("horizon")
avg_mae = horizon_df["MAE"].mean()
avg_mape = horizon_df["MAPE"].mean()

print(f"\nTFT Results:")
print(f"  1-Step:  MAE = {tft_1step['MAE']}, MAPE = {tft_1step['MAPE']}%")
print(f"  24h Avg: MAE = {avg_mae:.2f}, MAPE = {avg_mape:.2f}%")
print(f"\nPer-horizon:")
print(horizon_df.to_string())

# %% — Horizon Comparison Plot
fig, ax = plt.subplots(figsize=(12, 5))

tft_maes = horizon_df["MAE"].values
ax.plot(range(1, 25), tft_maes, "o-", color="#2563EB", linewidth=2,
        markersize=5, label="TFT")
ax.axhline(y=7.15, color="#7C3AED", linestyle="-.", linewidth=1.5,
           label="LSTM (~7.15)")
ax.axhline(y=7.01, color="#DC2626", linestyle="--", linewidth=1.5,
           label="Ridge (7.01)")
ax.axhline(y=9.98, color="#94A3B8", linestyle=":", linewidth=1,
           label="Naive (9.98)")

ax.set_xlabel("Forecast Horizon (hours ahead)")
ax.set_ylabel("MAE (µg/m³)")
ax.set_title("MAE vs Forecast Horizon — All Models")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, 25))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "23_all_models_horizon.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## TFT Interpretability

# %% — Variable Importance
interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")

# Encoder variables
print("Encoder Variable Importance (past inputs):")
enc_vars = training.encoder_variables
enc_imp = interpretation["encoder_variables"]
for name, imp in sorted(zip(enc_vars, enc_imp), key=lambda x: -x[1])[:15]:
    bar = "█" * int(float(imp) / float(enc_imp.max()) * 30)
    print(f"  {name:35s} {float(imp):.4f} {bar}")

print("\nDecoder Variable Importance (future known inputs):")
dec_vars = training.decoder_variables
dec_imp = interpretation["decoder_variables"]
for name, imp in sorted(zip(dec_vars, dec_imp), key=lambda x: -x[1])[:10]:
    bar = "█" * int(float(imp) / float(dec_imp.max()) * 30)
    print(f"  {name:35s} {float(imp):.4f} {bar}")

# %% — Variable Importance Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Encoder
top_n = 15
enc_sorted = sorted(zip(enc_vars, enc_imp.cpu().numpy()), key=lambda x: -x[1])[:top_n]
names, vals = zip(*enc_sorted)
axes[0].barh(range(len(names)), vals, color="#2563EB", alpha=0.8)
axes[0].set_yticks(range(len(names)))
axes[0].set_yticklabels(names, fontsize=9)
axes[0].set_xlabel("Importance")
axes[0].set_title("Encoder Variables (Past Inputs)")
axes[0].invert_yaxis()

# Decoder
dec_sorted = sorted(zip(dec_vars, dec_imp.cpu().numpy()), key=lambda x: -x[1])[:10]
names_d, vals_d = zip(*dec_sorted)
axes[1].barh(range(len(names_d)), vals_d, color="#059669", alpha=0.8)
axes[1].set_yticks(range(len(names_d)))
axes[1].set_yticklabels(names_d, fontsize=9)
axes[1].set_xlabel("Importance")
axes[1].set_title("Decoder Variables (Known Future)")
axes[1].invert_yaxis()

plt.suptitle("TFT Feature Importance — What Drives Delhi's Air Quality?", fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "24_tft_variable_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Attention Weights
fig, ax = plt.subplots(figsize=(14, 3))

attn = interpretation["attention"].cpu().numpy()
hours_back = range(-len(attn), 0)

ax.bar(hours_back, attn, color="#2563EB", alpha=0.7, width=1.0)
ax.set_xlabel("Hours Before Forecast (0 = forecast start)")
ax.set_ylabel("Attention Weight")
ax.set_title("TFT Temporal Attention — Which Past Hours Matter Most?")

for lag, label in [(-1, "1h ago"), (-24, "24h ago"), (-48, "2 days"), (-168, "1 week")]:
    if lag >= -len(attn):
        ax.axvline(lag, color="#DC2626", linestyle=":", alpha=0.5)
        ax.text(lag, max(attn) * 0.9, label, ha="center", fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "25_tft_attention_weights.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Probabilistic Forecasts

# %% — Quantile Forecast Plots
n_samples = 4
fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3.5 * n_samples))

for i in range(n_samples):
    ax = axes[i]
    idx = i * (len(tft_actuals) // n_samples)
    hours = range(1, MAX_PREDICTION_LENGTH + 1)
    actual = tft_actuals[idx]

    quantiles = raw_predictions.output[idx].cpu().numpy()
    median = quantiles[:, 3]
    q10 = quantiles[:, 1]
    q90 = quantiles[:, 5]
    q25 = quantiles[:, 2]
    q75 = quantiles[:, 4]

    ax.plot(hours, actual, "o-", color="#000000", linewidth=2, markersize=4,
            label="Actual", zorder=5)
    ax.plot(hours, median, "s-", color="#2563EB", linewidth=2, markersize=4,
            label="TFT Median")
    ax.fill_between(hours, q10, q90, alpha=0.15, color="#2563EB", label="80% CI")
    ax.fill_between(hours, q25, q75, alpha=0.25, color="#2563EB", label="50% CI")

    mae_i = np.mean(np.abs(actual - median))
    coverage = np.mean((actual >= q10) & (actual <= q90)) * 100
    ax.set_title(f"Sample {i+1} — MAE: {mae_i:.1f}, 80% coverage: {coverage:.0f}%")
    ax.set_ylabel("PM2.5 (µg/m³)")
    if i == 0:
        ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Hours Ahead")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "26_tft_quantile_forecasts.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Calibration Check
print("Quantile Calibration:")
quantile_preds = raw_predictions.output.cpu().numpy()

for qi, qname, expected in [(0,"2%",2), (1,"10%",10), (2,"25%",25),
                              (4,"75%",75), (5,"90%",90), (6,"98%",98)]:
    below = (tft_actuals < quantile_preds[:, :, qi]).mean() * 100
    print(f"  Below {qname}: {below:.1f}% (expected ~{expected}%)")

coverage_80 = ((tft_actuals >= quantile_preds[:, :, 1]) &
               (tft_actuals <= quantile_preds[:, :, 5])).mean() * 100
print(f"\n  80% interval coverage: {coverage_80:.1f}% (target ~80%)")

# %% [markdown]
# ## N-BEATS (Bonus)

# %% — Train N-BEATS
from pytorch_forecasting import NBeats

nbeats_training = TimeSeriesDataSet(
    df_tft[df_tft.time_idx <= train_cutoff],
    time_idx="time_idx",
    target=target,
    group_ids=["group"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    time_varying_unknown_reals=[target],
    target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
    allow_missing_timesteps=True,
)

nb_val_start = train_cutoff - MAX_ENCODER_LENGTH
nbeats_validation = TimeSeriesDataSet.from_dataset(
    nbeats_training,
    df_tft[(df_tft.time_idx >= nb_val_start) & (df_tft.time_idx <= val_cutoff)],
    stop_randomization=True,
)

nb_test_start = val_cutoff - MAX_ENCODER_LENGTH
nbeats_testing = TimeSeriesDataSet.from_dataset(
    nbeats_training,
    df_tft[df_tft.time_idx >= nb_test_start],
    stop_randomization=True,
)

nb_train_dl = nbeats_training.to_dataloader(train=True, batch_size=64, num_workers=0)
nb_val_dl = nbeats_validation.to_dataloader(train=False, batch_size=64, num_workers=0)
nb_test_dl = nbeats_testing.to_dataloader(train=False, batch_size=64, num_workers=0)

nbeats = NBeats.from_dataset(
    nbeats_training,
    learning_rate=0.001,
    widths=[256, 2048],
    backcast_loss_ratio=1.0,
)

print(f"N-BEATS: {nbeats.size()/1e3:.1f}k params")
print(f"Datasets: Train={len(nbeats_training)}, Val={len(nbeats_validation)}, Test={len(nbeats_testing)}")

trainer_nb = pl.Trainer(
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
trainer_nb.fit(nbeats, train_dataloaders=nb_train_dl, val_dataloaders=nb_val_dl)

# %% — N-BEATS Evaluation
best_nb = NBeats.load_from_checkpoint(trainer_nb.checkpoint_callback.best_model_path)

nb_preds_raw = best_nb.predict(nb_test_dl, mode="prediction",
                                trainer_kwargs=dict(accelerator="auto"))
nb_actuals_raw = torch.cat([y[0] for x, y in iter(nb_test_dl)])

nb_preds = nb_preds_raw.cpu().numpy()
nb_acts = nb_actuals_raw.cpu().numpy()

nb_1step = calculate_metrics(nb_acts[:, 0], nb_preds[:, 0])
nb_avg = np.mean([calculate_metrics(nb_acts[:, h], nb_preds[:, h])["MAE"]
                   for h in range(MAX_PREDICTION_LENGTH)])

print(f"\nN-BEATS: 1-Step MAE={nb_1step['MAE']}, MAPE={nb_1step['MAPE']}%")
print(f"         24h Avg MAE={nb_avg:.2f}")

# %% [markdown]
# ## Final Comparison

# %% — Summary
print(f"""
{'='*65}
                    FINAL MODEL COMPARISON
{'='*65}

  Model                  MAE (1-step)  MAPE (1-step)  24h Avg MAE
  ─────────────────────  ────────────  ─────────────  ───────────
  Naive Persistence         9.98         35.2%           —
  Seasonal Naive            9.98         35.7%           —
  Moving Avg (48h)          9.53         38.0%           —
  Linear Ridge              7.01         26.4%           —
  LSTM (from scratch)       7.13         27.3%          7.16
  N-BEATS                   {nb_1step['MAE']:<12} {nb_1step['MAPE']}%{'':<9}{nb_avg:.2f}
  TFT                       {tft_1step['MAE']:<12} {tft_1step['MAPE']}%{'':<9}{avg_mae:.2f}

{'='*65}
""")

for name, mae in [("LSTM", 7.13), ("N-BEATS", nb_1step["MAE"]),
                   ("TFT", tft_1step["MAE"])]:
    imp = (1 - mae / 7.01) * 100
    imp_naive = (1 - mae / 9.98) * 100
    print(f"  {name:10s} vs Ridge: {imp:+.1f}% | vs Naive: {imp_naive:+.1f}%")

# %% — Final Bar Chart
fig, ax = plt.subplots(figsize=(12, 5))

models = ["Naive", "Seasonal", "MA(48h)", "Ridge", "LSTM", "N-BEATS", "TFT"]
maes = [9.98, 9.98, 9.53, 7.01, 7.13, nb_1step["MAE"], tft_1step["MAE"]]
colors = ["#94A3B8", "#94A3B8", "#94A3B8", "#F59E0B", "#7C3AED", "#059669", "#2563EB"]

bars = ax.bar(models, maes, color=colors)
ax.axhline(y=7.01, color="#DC2626", linestyle="--", linewidth=1, alpha=0.5)
ax.set_ylabel("MAE (µg/m³)")
ax.set_title("1-Step Ahead MAE — All Models")

for bar, v in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.15, f"{v:.2f}",
            ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "27_final_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ Phase 6 complete! Next: Deployment (FastAPI + Streamlit)")
