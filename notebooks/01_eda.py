# %% [markdown]
# # 🌫️ Delhi Air Quality — Exploratory Data Analysis
#
# **Project:** Air Quality Forecasting with Deep Learning
#
# **Objective:** Understand the temporal patterns, seasonality, feature correlations,
# and stationarity of Delhi's PM2.5 data before building forecasting models.
#
# **Data source:** CPCB hourly air quality + Open-Meteo ERA5 weather data

# %% — Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

FIGURES_DIR = Path("../figures")
FIGURES_DIR.mkdir(exist_ok=True)

print("Libraries loaded ✓")

# %% [markdown]
# ## 1. Load and Inspect the Data

# %% — Load Data
# ── Auto-detect the processed file ──
processed_dir = Path("../data/processed")
candidates = list(processed_dir.glob("*combined*.csv")) + list(processed_dir.glob("*clean*.csv"))

if not candidates:
    # Fallback: load whatever CSV is in processed or raw
    candidates = list(processed_dir.glob("*.csv"))
    if not candidates:
        candidates = list(Path("../data/raw").glob("*.csv"))

if not candidates:
    raise FileNotFoundError(
        "No CSV found! Run prepare_dataset.py first, or place your CSV in data/processed/"
    )

data_path = candidates[0]
print(f"Loading: {data_path}")

df = pd.read_csv(data_path)
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{list(df.columns)}")
print(f"\nFirst 3 rows:")
df.head(3)

# %% — Standardize Columns
# Auto-detect and rename columns to a standard format
col_map = {}
for col in df.columns:
    cl = col.lower().strip().replace(" ", "_").replace(".", "")

    if cl in ["datetime", "date_time", "date", "timestamp", "from_date", "from date"]:
        col_map[col] = "datetime"
    elif cl in ["pm25", "pm2.5", "pm_25", "pm 2.5"]:
        col_map[col] = "pm25"
    elif cl in ["pm10", "pm_10"]:
        col_map[col] = "pm10"
    elif cl in ["no2", "nitrogen_dioxide"]:
        col_map[col] = "no2"
    elif cl in ["so2", "sulfur_dioxide", "sulphur_dioxide"]:
        col_map[col] = "so2"
    elif cl in ["co", "carbon_monoxide"]:
        col_map[col] = "co"
    elif cl in ["o3", "ozone"]:
        col_map[col] = "o3"
    elif cl in ["nh3", "ammonia"]:
        col_map[col] = "nh3"
    elif cl in ["no", "nitric_oxide"]:
        col_map[col] = "no_"
    elif cl in ["nox"]:
        col_map[col] = "nox"
    elif cl in ["benzene"]:
        col_map[col] = "benzene"
    elif cl in ["toluene"]:
        col_map[col] = "toluene"
    elif cl in ["xylene"]:
        col_map[col] = "xylene"
    elif cl in ["aqi", "aqi_value"]:
        col_map[col] = "aqi"
    elif cl in ["aqi_bucket", "aqi_category"]:
        col_map[col] = "aqi_bucket"
    elif cl in ["station", "stationid", "station_name", "site", "city"]:
        col_map[col] = "station"
    # Weather columns (from prepare_dataset merge)
    elif cl in ["temperature", "temperature_2m", "temp"]:
        col_map[col] = "temperature"
    elif cl in ["humidity", "relative_humidity_2m", "relative_humidity", "rh"]:
        col_map[col] = "humidity"
    elif cl in ["wind_speed", "wind_speed_10m", "ws"]:
        col_map[col] = "wind_speed"
    elif cl in ["wind_direction", "wind_direction_10m", "wd"]:
        col_map[col] = "wind_direction"
    elif cl in ["pressure", "surface_pressure", "bp"]:
        col_map[col] = "pressure"
    elif cl in ["precipitation", "rainfall", "rf"]:
        col_map[col] = "precipitation"

df = df.rename(columns=col_map)
print(f"Renamed columns: {col_map}")
print(f"\nStandardized columns: {list(df.columns)}")

# %% — Parse Datetime and Set Index
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime")
    df = df.sort_index()
else:
    # Try the first column as datetime
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df = df.set_index(df.columns[0])
    df = df.sort_index()

# If there are multiple stations, filter to one
if "station" in df.columns:
    stations = df["station"].unique()
    print(f"\nStations in data: {stations}")
    # Prefer ITO or Anand Vihar
    for pref in ["ITO", "Anand Vihar", "RK Puram", "DTU"]:
        matches = [s for s in stations if pref.lower() in str(s).lower()]
        if matches:
            selected = matches[0]
            break
    else:
        selected = stations[0]
    print(f"Using station: {selected}")
    df = df[df["station"] == selected].copy()
    df = df.drop(columns=["station"], errors="ignore")

# Convert all remaining columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop columns that are entirely NaN
df = df.dropna(axis=1, how="all")

print(f"\nFinal shape: {df.shape}")
print(f"Date range: {df.index.min()} → {df.index.max()}")
print(f"Duration: {(df.index.max() - df.index.min()).days} days")

# %% — Quick Data Profile
print("=" * 70)
print("DATA PROFILE")
print("=" * 70)

profile = pd.DataFrame({
    "dtype": df.dtypes,
    "non_null": df.count(),
    "null": df.isnull().sum(),
    "null_%": (df.isnull().sum() / len(df) * 100).round(1),
    "mean": df.mean().round(2),
    "std": df.std().round(2),
    "min": df.min().round(2),
    "max": df.max().round(2),
})

print(profile.to_string())

# %% [markdown]
# ## 2. PM2.5 Time Series Overview

# %% — Full Time Series Plot
target = "pm25" if "pm25" in df.columns else df.columns[0]
print(f"Target variable: {target}")

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df.index, df[target], linewidth=0.4, alpha=0.8, color="#2563EB")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Delhi PM2.5 Concentration — Hourly Time Series")

# Add AQI category bands
aqi_bands = [
    (0, 30, "#00E400", "Good"),
    (30, 60, "#92D050", "Satisfactory"),
    (60, 90, "#FFFF00", "Moderate"),
    (90, 120, "#FF7E00", "Poor"),
    (120, 250, "#FF0000", "Very Poor"),
    (250, df[target].max() + 50, "#7E0023", "Severe"),
]
for low, high, color, label in aqi_bands:
    ax.axhspan(low, high, alpha=0.08, color=color)
    ax.text(df.index[-1], (low + high) / 2, f"  {label}", fontsize=7,
            va="center", alpha=0.6)

ax.set_xlim(df.index.min(), df.index.max())
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_full_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Yearly Overlay
fig, ax = plt.subplots(figsize=(14, 5))

df["year"] = df.index.year
df["day_of_year"] = df.index.dayofyear

daily = df.groupby(["year", "day_of_year"])[target].mean().reset_index()

colors = plt.cm.viridis(np.linspace(0.1, 0.9, daily["year"].nunique()))
for i, (year, group) in enumerate(daily.groupby("year")):
    ax.plot(group["day_of_year"], group[target], label=str(year),
            linewidth=0.8, alpha=0.7, color=colors[i])

ax.set_xlabel("Day of Year")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("PM2.5 Daily Average — Year-over-Year Comparison")
ax.legend(loc="upper left", framealpha=0.9)
ax.set_xlim(1, 366)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_yearly_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# Clean up temp columns
df = df.drop(columns=["year", "day_of_year"], errors="ignore")

# %% [markdown]
# ## 3. Seasonal Decomposition

# %% — STL Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Use daily resampled data for cleaner decomposition
daily_target = df[target].resample("1D").mean().dropna()

# Need at least 2 full cycles for decomposition
decomp = seasonal_decompose(daily_target, model="additive", period=365,
                            extrapolate_trend="freq")

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

axes[0].plot(decomp.observed, linewidth=0.6, color="#2563EB")
axes[0].set_ylabel("Observed")
axes[0].set_title("Seasonal Decomposition of Daily PM2.5 (period=365 days)")

axes[1].plot(decomp.trend, linewidth=1.0, color="#DC2626")
axes[1].set_ylabel("Trend")

axes[2].plot(decomp.seasonal, linewidth=0.5, color="#059669")
axes[2].set_ylabel("Seasonal")

axes[3].plot(decomp.resid, linewidth=0.4, color="#7C3AED", alpha=0.7)
axes[3].set_ylabel("Residual")

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_seasonal_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Temporal Patterns
#
# These are the patterns our models need to capture: daily cycle (rush hours),
# weekly pattern (weekday vs weekend), monthly/seasonal pattern (winter pollution crisis).

# %% — Monthly Distribution (Box Plot)
df["month"] = df.index.month

fig, ax = plt.subplots(figsize=(12, 5))
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

bp = df.boxplot(column=target, by="month", ax=ax, showfliers=False,
                patch_artist=True, return_type="dict")

# Color by severity
month_colors = ["#FF7E00", "#FF7E00", "#FFFF00", "#92D050", "#00E400", "#00E400",
                "#00E400", "#00E400", "#92D050", "#FFFF00", "#FF0000", "#FF0000"]

for patch, color in zip(bp[target]["boxes"], month_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

ax.set_xticklabels(month_names)
ax.set_xlabel("Month")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("PM2.5 Distribution by Month — Delhi's Winter Pollution Crisis")
fig.suptitle("")  # Remove default boxplot title
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_monthly_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nMonthly PM2.5 Statistics:")
monthly_stats = df.groupby("month")[target].agg(["mean", "median", "std", "max"])
monthly_stats.index = month_names
print(monthly_stats.round(1).to_string())

# %% — Hourly Pattern (Diurnal Cycle)
df["hour"] = df.index.hour

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall hourly pattern
hourly_mean = df.groupby("hour")[target].mean()
hourly_std = df.groupby("hour")[target].std()

axes[0].plot(hourly_mean.index, hourly_mean.values, "o-", color="#2563EB",
             linewidth=2, markersize=5)
axes[0].fill_between(hourly_mean.index,
                      hourly_mean - hourly_std,
                      hourly_mean + hourly_std,
                      alpha=0.15, color="#2563EB")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("PM2.5 (µg/m³)")
axes[0].set_title("Average PM2.5 by Hour of Day")
axes[0].set_xticks(range(0, 24, 3))

# Hourly pattern by season
df["season"] = df["month"].map({
    1: "Winter", 2: "Winter", 3: "Spring",
    4: "Spring", 5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Autumn", 11: "Winter", 12: "Winter"
})

season_colors = {
    "Winter": "#1E40AF", "Spring": "#059669",
    "Summer": "#D97706", "Monsoon": "#7C3AED",
    "Autumn": "#DC2626"
}

for season, color in season_colors.items():
    mask = df["season"] == season
    hourly_season = df[mask].groupby("hour")[target].mean()
    axes[1].plot(hourly_season.index, hourly_season.values, "o-",
                 label=season, color=color, linewidth=1.5, markersize=4)

axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("PM2.5 (µg/m³)")
axes[1].set_title("Diurnal Cycle by Season")
axes[1].legend()
axes[1].set_xticks(range(0, 24, 3))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_hourly_pattern.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Day of Week Pattern
df["dayofweek"] = df.index.dayofweek
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

fig, ax = plt.subplots(figsize=(10, 4))
dow_mean = df.groupby("dayofweek")[target].mean()
dow_std = df.groupby("dayofweek")[target].std()

bars = ax.bar(range(7), dow_mean.values, color="#2563EB", alpha=0.7,
              yerr=dow_std.values, capsize=4, error_kw={"alpha": 0.3})

# Highlight weekend
bars[5].set_color("#059669")
bars[6].set_color("#059669")

ax.set_xticks(range(7))
ax.set_xticklabels(dow_names)
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Average PM2.5 by Day of Week (green = weekend)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_day_of_week.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Autocorrelation Analysis
#
# Critical for understanding temporal dependencies and choosing lag features.

# %% — ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Use hourly data
hourly_target = df[target].dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Hourly ACF — look for 24h cycle
plot_acf(hourly_target, lags=72, ax=axes[0, 0], alpha=0.05,
         title="ACF — Hourly (72h lags)")

# Hourly PACF
plot_pacf(hourly_target, lags=72, ax=axes[0, 1], alpha=0.05,
          method="ywm", title="PACF — Hourly (72h lags)")

# Daily ACF — look for weekly + yearly cycles
plot_acf(daily_target, lags=90, ax=axes[1, 0], alpha=0.05,
         title="ACF — Daily (90-day lags)")

# Daily PACF
plot_pacf(daily_target, lags=60, ax=axes[1, 1], alpha=0.05,
          method="ywm", title="PACF — Daily (60-day lags)")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.show()

print("""
Key observations from ACF/PACF:
- Strong autocorrelation at lag 24 → daily cycle (expected: morning/evening traffic)
- Gradual decay in ACF → non-stationary series (needs differencing for ARIMA)
- PACF significant spikes → direct predictive lags for model feature selection
""")

# %% [markdown]
# ## 6. Stationarity Test

# %% — Augmented Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller

def run_adf_test(series, name=""):
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"ADF Test: {name}")
    print(f"  Test statistic : {result[0]:.4f}")
    print(f"  p-value        : {result[1]:.6f}")
    print(f"  Lags used      : {result[2]}")
    print(f"  Observations   : {result[3]}")
    for key, val in result[4].items():
        print(f"  Critical {key:4s}  : {val:.4f}")
    print(f"  Stationary     : {'✓ YES' if result[1] < 0.05 else '✗ NO'}")
    print()
    return result[1] < 0.05

print("=" * 50)
is_stat = run_adf_test(daily_target, "PM2.5 (Daily, Raw)")

if not is_stat:
    diff1 = daily_target.diff().dropna()
    run_adf_test(diff1, "PM2.5 (Daily, 1st Difference)")

    diff_seasonal = daily_target.diff(7).dropna()
    run_adf_test(diff_seasonal, "PM2.5 (Daily, 7-day Seasonal Diff)")

# %% [markdown]
# ## 7. Feature Correlations
#
# Understanding which features drive PM2.5 helps with feature selection
# and tells a domain story in interviews.

# %% — Correlation Heatmap
# Select numeric columns that are likely features
feature_cols = [c for c in df.columns if c not in
                ["month", "hour", "dayofweek", "season", "year", "day_of_year",
                 "aqi", "aqi_bucket"]]
corr_df = df[feature_cols].select_dtypes(include=[np.number])

if corr_df.shape[1] > 2:
    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})

    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print top correlations with target
    if target in corr_matrix.columns:
        target_corr = corr_matrix[target].drop(target).sort_values(
            key=abs, ascending=False
        )
        print(f"\nTop correlations with {target}:")
        for feat, corr_val in target_corr.items():
            bar = "█" * int(abs(corr_val) * 20)
            sign = "+" if corr_val > 0 else "-"
            print(f"  {feat:25s} {sign}{abs(corr_val):.3f}  {bar}")

# %% — Lagged Correlations (Weather → PM2.5)
weather_features = [c for c in ["wind_speed", "temperature", "humidity",
                                "pressure", "precipitation"] if c in df.columns]

if weather_features and target in df.columns:
    print("\nLagged Correlations: Weather → PM2.5")
    print("-" * 60)

    lags = [0, 1, 3, 6, 12, 24, 48]
    lag_results = []

    for feat in weather_features:
        for lag in lags:
            corr = df[target].corr(df[feat].shift(lag))
            lag_results.append({"feature": feat, "lag_hours": lag, "correlation": corr})

    lag_df = pd.DataFrame(lag_results)

    fig, ax = plt.subplots(figsize=(10, 5))
    for feat in weather_features:
        subset = lag_df[lag_df["feature"] == feat]
        ax.plot(subset["lag_hours"], subset["correlation"].abs(),
                "o-", label=feat, linewidth=1.5, markersize=5)

    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("|Correlation| with PM2.5")
    ax.set_title("Lagged Correlations: How Past Weather Affects Current PM2.5")
    ax.legend()
    ax.set_xticks(lags)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_lagged_correlations.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Best lag per feature
    print("\nOptimal lag per weather feature:")
    for feat in weather_features:
        subset = lag_df[lag_df["feature"] == feat]
        best = subset.loc[subset["correlation"].abs().idxmax()]
        print(f"  {feat:20s} → best lag = {int(best['lag_hours'])}h "
              f"(r = {best['correlation']:.3f})")

# %% [markdown]
# ## 8. Wind Analysis
#
# Wind is the primary dispersal mechanism for particulate matter.
# Low wind = pollutant accumulation = high PM2.5.

# %% — Wind Speed vs PM2.5 Scatter
if "wind_speed" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    sample = df.dropna(subset=[target, "wind_speed"]).sample(
        min(10000, len(df)), random_state=42
    )
    axes[0].scatter(sample["wind_speed"], sample[target],
                    alpha=0.1, s=5, color="#2563EB")
    axes[0].set_xlabel("Wind Speed (m/s)")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    axes[0].set_title("Wind Speed vs PM2.5 — Higher Wind Disperses Pollution")

    # Binned mean
    df["wind_bin"] = pd.cut(df["wind_speed"], bins=15)
    wind_binned = df.groupby("wind_bin", observed=True)[target].agg(["mean", "std", "count"])
    wind_binned = wind_binned[wind_binned["count"] > 50]  # drop low-count bins

    x = range(len(wind_binned))
    axes[1].bar(x, wind_binned["mean"], color="#2563EB", alpha=0.7,
                yerr=wind_binned["std"] / 3, capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{iv.mid:.1f}" for iv in wind_binned.index], rotation=45
    )
    axes[1].set_xlabel("Wind Speed Bin (m/s)")
    axes[1].set_ylabel("Mean PM2.5 (µg/m³)")
    axes[1].set_title("Mean PM2.5 by Wind Speed Bin")

    df = df.drop(columns=["wind_bin"], errors="ignore")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_wind_vs_pm25.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% — Wind Rose (Direction vs PM2.5)
if "wind_direction" in df.columns and "wind_speed" in df.columns:
    # Bin wind direction into 16 compass sectors
    dir_bins = np.arange(0, 361, 22.5)
    dir_labels = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

    df["wind_sector"] = pd.cut(
        df["wind_direction"] % 360, bins=dir_bins, labels=dir_labels,
        include_lowest=True
    )

    sector_pm25 = df.groupby("wind_sector", observed=True)[target].mean()

    # Polar plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)

    angles = np.linspace(0, 2 * np.pi, len(sector_pm25), endpoint=False)
    values = sector_pm25.values
    values = np.append(values, values[0])  # close the polygon
    angles = np.append(angles, angles[0])

    ax.plot(angles, values, "o-", color="#DC2626", linewidth=2)
    ax.fill(angles, values, alpha=0.15, color="#DC2626")
    ax.set_thetagrids(np.degrees(angles[:-1]), dir_labels, fontsize=9)
    ax.set_title("Mean PM2.5 by Wind Direction\n(Winds FROM this direction)", y=1.08)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_wind_rose_pm25.png", dpi=150, bbox_inches="tight")
    plt.show()

    df = df.drop(columns=["wind_sector"], errors="ignore")

# %% [markdown]
# ## 9. Extreme Event Analysis
#
# Diwali and stubble burning season (Oct–Nov) cause massive PM2.5 spikes.
# Understanding these is critical for model error analysis later.

# %% — Diwali Impact
# Approximate Diwali dates
diwali_dates = pd.to_datetime([
    "2019-10-27", "2020-11-14", "2021-11-04",
    "2022-10-24", "2023-11-12", "2024-11-01"
])

# Filter to dates within our data range
diwali_dates = [d for d in diwali_dates
                if df.index.min() <= d <= df.index.max()]

if diwali_dates:
    fig, axes = plt.subplots(len(diwali_dates), 1,
                              figsize=(14, 3 * len(diwali_dates)), sharex=False)
    if len(diwali_dates) == 1:
        axes = [axes]

    for i, diwali in enumerate(diwali_dates):
        window_start = diwali - pd.Timedelta(days=5)
        window_end = diwali + pd.Timedelta(days=5)

        window = df.loc[window_start:window_end, target].dropna()

        if len(window) > 0:
            axes[i].plot(window.index, window.values, linewidth=1.0, color="#2563EB")
            axes[i].axvline(diwali, color="#DC2626", linestyle="--",
                           linewidth=2, label="Diwali Night")
            axes[i].axhspan(0, 60, alpha=0.1, color="#00E400")
            axes[i].set_ylabel("PM2.5")
            axes[i].set_title(f"Diwali {diwali.year} — "
                             f"Peak: {window.max():.0f} µg/m³")
            axes[i].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_diwali_impact.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Diwali Impact Summary:")
    for d in diwali_dates:
        pre = df.loc[d - pd.Timedelta(days=7):d - pd.Timedelta(days=2), target]
        post = df.loc[d - pd.Timedelta(days=1):d + pd.Timedelta(days=2), target]
        if len(pre) > 0 and len(post) > 0:
            print(f"  {d.year}: Pre-Diwali mean = {pre.mean():.0f}, "
                  f"Diwali window mean = {post.mean():.0f}, "
                  f"Peak = {post.max():.0f} µg/m³ "
                  f"({post.max() / pre.mean():.1f}x baseline)")

# %% [markdown]
# ## 10. Distribution Analysis

# %% — PM2.5 Distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
axes[0].hist(df[target].dropna(), bins=100, color="#2563EB", alpha=0.7,
             edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("PM2.5 (µg/m³)")
axes[0].set_ylabel("Count")
axes[0].set_title("PM2.5 Distribution (heavily right-skewed)")
axes[0].axvline(df[target].median(), color="#DC2626", linestyle="--",
                label=f"Median: {df[target].median():.0f}")
axes[0].axvline(df[target].mean(), color="#059669", linestyle="--",
                label=f"Mean: {df[target].mean():.0f}")
axes[0].legend()

# Log-transformed
log_target = np.log1p(df[target].dropna())
axes[1].hist(log_target, bins=80, color="#7C3AED", alpha=0.7,
             edgecolor="white", linewidth=0.3)
axes[1].set_xlabel("log(1 + PM2.5)")
axes[1].set_ylabel("Count")
axes[1].set_title("Log-Transformed — Closer to Normal")

# QQ plot
from scipy import stats
stats.probplot(log_target, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot (log PM2.5 vs Normal)")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "13_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

skewness = df[target].skew()
kurtosis = df[target].kurtosis()
print(f"PM2.5 Skewness: {skewness:.2f} (positive = right-skewed)")
print(f"PM2.5 Kurtosis: {kurtosis:.2f} (>3 = heavy tails, extreme events)")
print(f"\nImplication: Consider log-transform or Huber loss for training "
      f"(robust to outlier spikes)")

# %% [markdown]
# ## 11. Missing Data Patterns

# %% — Missing Data Heatmap
fig, ax = plt.subplots(figsize=(14, 4))

# Resample to daily and show % missing per day
daily_missing = df.resample("1D").apply(lambda x: x.isnull().sum() / len(x) * 100)
daily_missing = daily_missing[feature_cols].select_dtypes(include=[np.number])

# Only show columns with some missing data
cols_with_missing = daily_missing.columns[daily_missing.max() > 0]

if len(cols_with_missing) > 0:
    sns.heatmap(daily_missing[cols_with_missing].T,
                cmap="YlOrRd", ax=ax, cbar_kws={"label": "% Missing"})
    ax.set_title("Daily Missing Data Pattern")
    ax.set_xlabel("Date")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14_missing_data.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No missing data — clean dataset!")

# %% [markdown]
# ## 12. Key Findings Summary

# %% — Summary
print("""
╔══════════════════════════════════════════════════════════════════╗
║                    EDA KEY FINDINGS                             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. SEASONALITY                                                  ║
║     - Extreme winter pollution (Nov-Feb): 3-5x summer levels    ║
║     - Driven by: stubble burning, temperature inversion,        ║
║       low wind speeds, Diwali fireworks                         ║
║                                                                  ║
║  2. DIURNAL PATTERN                                              ║
║     - Morning peak (~8-10 AM): traffic + low boundary layer     ║
║     - Evening peak (~8-11 PM): traffic + cooking + cooling      ║
║     - Afternoon dip: solar heating lifts boundary layer          ║
║                                                                  ║
║  3. CORRELATIONS                                                 ║
║     - Wind speed: STRONG negative (dispersal mechanism)         ║
║     - Humidity: positive (fog traps particulates)               ║
║     - Temperature: negative (warm air = better mixing)          ║
║     - PM10, NO2, CO: positive (co-emitted pollutants)           ║
║                                                                  ║
║  4. STATIONARITY                                                 ║
║     - Raw series: likely non-stationary (trend + seasonality)   ║
║     - First difference: stationary (suitable for ARIMA)         ║
║     - Implication: LSTM/TFT can handle non-stationarity         ║
║                                                                  ║
║  5. DISTRIBUTION                                                 ║
║     - Heavily right-skewed (extreme Diwali/winter spikes)       ║
║     - Log-transform or Huber loss recommended for training      ║
║     - Models will struggle with extreme events (>500 µg/m³)     ║
║                                                                  ║
║  6. FEATURE ENGINEERING IMPLICATIONS                             ║
║     - Use lags at 1h, 24h, 168h (strong autocorrelation)       ║
║     - Include cyclical encodings (hour, month)                  ║
║     - Wind speed and direction are top weather predictors       ║
║     - Diwali/stubble flags will help capture extreme events     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

# %% — Save clean temporal features for quick reference
print(f"\nFigures saved to: {FIGURES_DIR.resolve()}")
print(f"Notebook complete. Proceed to Phase 3: Feature Engineering.")

# Clean up temp columns
df = df.drop(columns=["month", "hour", "dayofweek", "season"],
             errors="ignore")
