# %% [markdown]
# # 📊 Phase 4 — Baseline Models
#
# Establishing benchmark performance that deep learning models must beat.
#
# Models tested:
# 1. Naive Persistence (forecast = last value)
# 2. Seasonal Naive (forecast = same hour yesterday)
# 3. Moving Average (24h and 48h windows)
# 4. Linear Regression (Ridge on engineered features)
# 5. SARIMA (classical statistical model)

# %% — Setup
import sys
from pathlib import Path

# Add project root to path
project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["figure.dpi"] = 120

FIGURES_DIR = Path("../figures")
FIGURES_DIR.mkdir(exist_ok=True)

print("Setup complete ✓")

# %% — Load Feature-Engineered Data
data_path = project_root / "data" / "processed" / "delhi_aqi_features.csv"

if not data_path.exists():
    # Fallback to combined data
    data_path = project_root / "data" / "processed" / "delhi_aqi_combined.csv"
    if not data_path.exists():
        candidates = list((project_root / "data" / "processed").glob("*.csv"))
        if candidates:
            data_path = candidates[0]
        else:
            raise FileNotFoundError("No data found! Run build_features.py first.")

print(f"Loading: {data_path.name}")
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Auto-detect target column
target = "pm25"
if target not in df.columns:
    for col in df.columns:
        if "pm2" in col.lower() or "pm25" in col.lower():
            target = col
            break
    else:
        target = df.columns[0]

print(f"Shape: {df.shape}")
print(f"Target: {target}")
print(f"Date range: {df.index.min()} → {df.index.max()}")

# %% — Temporal Train/Val/Test Split
from src.training.split import temporal_split

train, val, test = temporal_split(df, val_frac=0.15, test_frac=0.15)

# %% [markdown]
# ## 1. Naive Baselines

# %% — Run Naive Models
from src.evaluation.metrics import calculate_metrics, print_comparison_table

all_results = {}

# --- Naive Persistence ---
# 1-step ahead: forecast = previous hour's value
naive_pred = test[target].shift(1).dropna()
naive_actual = test[target].iloc[1:]

metrics = calculate_metrics(naive_actual.values, naive_pred.values)
all_results["Naive Persistence"] = metrics
print(f"Naive Persistence: {metrics}")

# Store naive MAE for MASE computation
naive_mae = metrics["MAE"]

# --- Seasonal Naive (24h) ---
# 1-step ahead: forecast = same hour yesterday
seasonal_pred = test[target].shift(24).dropna()
seasonal_actual = test[target].iloc[24:]

metrics = calculate_metrics(seasonal_actual.values, seasonal_pred.values)
all_results["Seasonal Naive (24h)"] = metrics
print(f"Seasonal Naive (24h): {metrics}")

# --- Moving Average (24h) ---
ma24_pred = test[target].rolling(window=24, min_periods=1).mean().shift(1).dropna()
ma24_actual = test[target].iloc[1:]

metrics = calculate_metrics(ma24_actual.values, ma24_pred.values)
all_results["Moving Avg (24h)"] = metrics
print(f"Moving Avg (24h): {metrics}")

# --- Moving Average (48h) ---
ma48_pred = test[target].rolling(window=48, min_periods=1).mean().shift(1).dropna()
ma48_actual = test[target].iloc[1:]

# Align lengths
min_len = min(len(ma48_actual), len(ma48_pred))
metrics = calculate_metrics(ma48_actual.values[:min_len], ma48_pred.values[:min_len])
all_results["Moving Avg (48h)"] = metrics
print(f"Moving Avg (48h): {metrics}")

# %% [markdown]
# ## 2. Linear Regression Baseline

# %% — Ridge Regression
from src.models.baselines import LinearBaseline

# Get feature columns (exclude target)
feature_cols = [c for c in train.columns if c != target]

# Check for numeric features only
numeric_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_cols.append(c)

feature_cols = numeric_cols
print(f"Using {len(feature_cols)} numeric features")

# Handle NaN/inf
train_clean = train[feature_cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
test_clean = test[feature_cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()

linear = LinearBaseline(alpha=1.0)
linear.fit(train_clean, target=target)

# Predict on test
linear_pred = linear.predict_all(test_clean)
linear_actual = test_clean[target].values

metrics = calculate_metrics(linear_actual, linear_pred)
all_results["Linear (Ridge)"] = metrics
print(f"Linear (Ridge): {metrics}")

# Top features
print("\nTop 15 features by coefficient magnitude:")
top_feats = linear.get_top_features(15)
for feat, importance in top_feats.items():
    bar = "█" * int(importance / top_feats.max() * 30)
    print(f"  {feat:30s} {importance:.4f}  {bar}")

# %% [markdown]
# ## 3. SARIMA Baseline

# %% — Fit SARIMA on Daily Data
from src.models.baselines import SARIMABaseline

sarima = SARIMABaseline(order=(2, 1, 2), seasonal_order=(1, 1, 1, 7))

try:
    sarima.fit(train, target=target)
    
    # Predict on test period (daily)
    test_daily = test[target].resample("1D").mean().dropna()
    pred_mean, pred_ci = sarima.predict_all_daily(test_daily.index)
    
    sarima_metrics = calculate_metrics(test_daily.values, pred_mean.values)
    all_results["SARIMA (daily)"] = sarima_metrics
    print(f"SARIMA (daily): {sarima_metrics}")
    
    # Plot SARIMA forecast
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Last 30 days of training + full test
    train_daily = train[target].resample("1D").mean().dropna()
    ax.plot(train_daily.index[-30:], train_daily.values[-30:],
            color="#2563EB", linewidth=1.5, label="Train (last 30d)")
    ax.plot(test_daily.index, test_daily.values,
            color="#000000", linewidth=1.5, label="Actual")
    ax.plot(pred_mean.index, pred_mean.values,
            color="#DC2626", linewidth=1.5, linestyle="--", label="SARIMA Forecast")
    
    # Confidence intervals
    ax.fill_between(pred_ci.index,
                     pred_ci.iloc[:, 0],
                     pred_ci.iloc[:, 1],
                     alpha=0.15, color="#DC2626", label="95% CI")
    
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("SARIMA Daily Forecast vs Actual")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "15_sarima_forecast.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    sarima_fitted = True

except Exception as e:
    print(f"SARIMA failed: {e}")
    print("This is OK — SARIMA on long series can be slow/unstable.")
    print("The other baselines are sufficient.")
    sarima_fitted = False

# %% [markdown]
# ## 4. Comparison Table

# %% — Print Results
results_df = print_comparison_table(all_results)

# %% — Visual Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of MAE
models = list(all_results.keys())
maes = [all_results[m]["MAE"] for m in models]
colors = ["#94A3B8"] * len(models)
best_idx = np.argmin(maes)
colors[best_idx] = "#2563EB"

axes[0].barh(range(len(models)), maes, color=colors)
axes[0].set_yticks(range(len(models)))
axes[0].set_yticklabels(models)
axes[0].set_xlabel("MAE (µg/m³)")
axes[0].set_title("Model Comparison — MAE (lower is better)")
axes[0].invert_yaxis()

for i, v in enumerate(maes):
    axes[0].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=10)

# Bar chart of MAPE
mapes = [all_results[m]["MAPE"] for m in models]
colors = ["#94A3B8"] * len(models)
best_idx = np.argmin(mapes)
colors[best_idx] = "#059669"

axes[1].barh(range(len(models)), mapes, color=colors)
axes[1].set_yticks(range(len(models)))
axes[1].set_yticklabels(models)
axes[1].set_xlabel("MAPE (%)")
axes[1].set_title("Model Comparison — MAPE (lower is better)")
axes[1].invert_yaxis()

for i, v in enumerate(mapes):
    axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "16_baseline_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Forecast Plots (Test Period Sample)
# Show a 1-week window from the test set

sample_start = test.index[0]
sample_end = sample_start + pd.Timedelta(days=7)

sample = test.loc[sample_start:sample_end]

fig, ax = plt.subplots(figsize=(14, 5))

# Actual
ax.plot(sample.index, sample[target], color="#000000",
        linewidth=2, label="Actual", zorder=5)

# Naive
ax.plot(sample.index, sample[target].shift(1),
        color="#94A3B8", linewidth=1, linestyle=":", label="Naive Persistence")

# Seasonal Naive
ax.plot(sample.index, sample[target].shift(24),
        color="#F59E0B", linewidth=1, linestyle="--", label="Seasonal Naive (24h)")

# Moving Average
ax.plot(sample.index, sample[target].rolling(24, min_periods=1).mean().shift(1),
        color="#8B5CF6", linewidth=1, linestyle="-.", label="Moving Avg (24h)")

# Linear Ridge (if features available)
if len(feature_cols) > 0:
    sample_clean = sample[feature_cols + [target]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(sample_clean) > 0:
        linear_sample_pred = linear.predict_all(sample_clean)
        ax.plot(sample_clean.index, linear_sample_pred,
                color="#2563EB", linewidth=1.5, label="Linear (Ridge)")

ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title(f"1-Week Forecast Comparison — {sample_start.strftime('%b %d')} to "
             f"{sample_end.strftime('%b %d, %Y')}")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "17_forecast_sample.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Error Analysis by Season

# %% — Seasonal Error Breakdown
test_with_month = test.copy()
test_with_month["month"] = test_with_month.index.month

# Linear model errors
if len(feature_cols) > 0:
    test_preds = test_clean.copy()
    test_preds["linear_pred"] = linear.predict_all(test_clean)
    test_preds["linear_error"] = (test_preds[target] - test_preds["linear_pred"]).abs()
    test_preds["month"] = test_preds.index.month
    
    season_map = {
        1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
        5: "Summer", 6: "Summer", 7: "Monsoon", 8: "Monsoon",
        9: "Monsoon", 10: "Autumn", 11: "Winter", 12: "Winter"
    }
    test_preds["season"] = test_preds["month"].map(season_map)
    
    season_errors = test_preds.groupby("season")["linear_error"].agg(
        ["mean", "median", "std", "count"]
    ).round(2)
    
    print("\nLinear Model — Error by Season:")
    print(season_errors.to_string())
    
    fig, ax = plt.subplots(figsize=(8, 4))
    season_order = ["Winter", "Spring", "Summer", "Monsoon", "Autumn"]
    season_errors = season_errors.reindex(
        [s for s in season_order if s in season_errors.index]
    )
    
    ax.bar(season_errors.index, season_errors["mean"],
           color=["#1E40AF", "#059669", "#D97706", "#7C3AED", "#DC2626"],
           alpha=0.7, yerr=season_errors["std"] / 3, capsize=5)
    ax.set_ylabel("Mean Absolute Error (µg/m³)")
    ax.set_title("Linear Model Error by Season — Winter Is Hardest")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "18_error_by_season.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 6. Save Results

# %% — Save Baseline Results
results_df.to_csv(project_root / "data" / "processed" / "baseline_results.csv")
print("Baseline results saved to: data/processed/baseline_results.csv")

print(f"""
╔══════════════════════════════════════════════════════════╗
║              PHASE 4 COMPLETE — BASELINES                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Baseline numbers are now established.                   ║
║  Deep learning models must beat these to justify         ║
║  their added complexity.                                 ║
║                                                          ║
║  Key benchmarks to beat:                                 ║
║    MAE:  {best_mae:<8.1f} ({best_model})
║    MAPE: {best_mape:<8.1f}% ({best_mape_model})
║                                                          ║
║  Next: Phase 5 — LSTM from Scratch in PyTorch            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""".format(
    best_mae=results_df["MAE"].min(),
    best_model=results_df["MAE"].idxmin(),
    best_mape=results_df["MAPE"].min(),
    best_mape_model=results_df["MAPE"].idxmin(),
))
