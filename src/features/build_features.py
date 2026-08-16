
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings

warnings.filterwarnings("ignore")

# Resolve paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_config():
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



# 1. TEMPORAL FEATURES

def add_temporal_features(df):
    
    # Raw calendar features (useful for tree-based models)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # Cyclical encodings 
    # Hour: 24h cycle
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # Day of week: 7-day cycle
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # Month: 12-month cycle
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Day of year: 365-day cycle 
    day_of_year = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Higher-order Fourier terms for sub-daily patterns
    # These capture the double-peak pattern (morning + evening rush hours)
    df["hour_sin2"] = np.sin(2 * np.pi * 2 * df["hour"] / 24)
    df["hour_cos2"] = np.cos(2 * np.pi * 2 * df["hour"] / 24)
    
    n_temporal = 16
    print(f"    → {n_temporal} temporal features added")
    return df


# 2. LAG FEATURES

def add_lag_features(df, config):
    
    target = "pm25"
    lags = config["features"]["lags"]  # [1, 2, 3, 6, 12, 24, 48, 72, 168]
    
    n_features = 0
    
    # Target lags (all specified lags)
    for lag in lags:
        df[f"pm25_lag_{lag}"] = df[target].shift(lag)
        n_features += 1
    
    # Covariate lags 
    covariate_lags = config["features"].get("weather_lags", [1, 6, 24])
    
    # Pollutant covariates
    pollutant_cols = [c for c in ["pm10", "no2", "co", "o3", "so2"] if c in df.columns]
    for col in pollutant_cols:
        for lag in covariate_lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            n_features += 1
    
    # Weather covariates
    weather_cols = [c for c in ["wind_speed", "temperature", "humidity", 
                                "pressure", "wind_direction"] if c in df.columns]
    for col in weather_cols:
        for lag in covariate_lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            n_features += 1
    
    return df


# 3. ROLLING STATISTICS

def add_rolling_features(df, config):
    
    target = "pm25"
    windows = config["features"]["rolling_windows"]  # [6, 12, 24, 48, 168]
    
    n_features = 0
    
    for window in windows:
        roll = df[target].rolling(window=window, min_periods=1)
        
        df[f"pm25_rmean_{window}"] = roll.mean()
        df[f"pm25_rstd_{window}"] = roll.std()
        df[f"pm25_rmin_{window}"] = roll.min()
        df[f"pm25_rmax_{window}"] = roll.max()
        n_features += 4
        
        # Range (max - min) as a volatility measure
        df[f"pm25_rrange_{window}"] = df[f"pm25_rmax_{window}"] - df[f"pm25_rmin_{window}"]
        n_features += 1
    
    # Rolling stats for wind speed (key dispersal variable)
    if "wind_speed" in df.columns:
        for window in [6, 24]:
            df[f"wind_rmean_{window}"] = df["wind_speed"].rolling(
                window=window, min_periods=1
            ).mean()
            df[f"wind_rstd_{window}"] = df["wind_speed"].rolling(
                window=window, min_periods=1
            ).std()
            n_features += 2
    
    return df


# 4. RATE OF CHANGE FEATURES

def add_rate_of_change_features(df):
    
    target = "pm25"
    n_features = 0
    
    # First differences (velocity: how fast is PM2.5 changing?)
    df["pm25_diff_1"] = df[target].diff(1)
    df["pm25_diff_3"] = df[target].diff(3)
    df["pm25_diff_6"] = df[target].diff(6)
    df["pm25_diff_24"] = df[target].diff(24)
    n_features += 4
    
    # Percentage change
    df["pm25_pct_1"] = df[target].pct_change(1)
    df["pm25_pct_24"] = df[target].pct_change(24)
    n_features += 2
    
    # Second difference (acceleration: is the rate of change speeding up?)
    df["pm25_accel"] = df["pm25_diff_1"].diff(1)
    n_features += 1
    
    # Ratio to recent average (is current value above or below trend?)
    for window in [24, 168]:
        rmean = df[target].rolling(window=window, min_periods=1).mean()
        df[f"pm25_ratio_{window}"] = df[target] / (rmean + 1e-8)
        n_features += 1
    
    # Cap extreme percentage changes (avoid inf values)
    for col in ["pm25_pct_1", "pm25_pct_24"]:
        df[col] = df[col].clip(-10, 10)
    
    print(f"    → {n_features} rate-of-change features added")
    return df


# 5. DOMAIN-SPECIFIC ATMOSPHERIC FEATURES

def add_atmospheric_features(df):
    """
    Domain-specific features grounded in atmospheric science.
    
    These are the features that differentiate this project from generic
    time-series work. Each one has a physical justification:
    
    1. STAGNATION INDEX
       Low wind + high humidity = air mass stagnation = pollutant accumulation.
       This is the primary meteorological driver of Delhi's pollution episodes.
       Formula: humidity / (wind_speed + epsilon)
    
    2. TEMPERATURE INVERSION PROXY  
       When surface temperature drops below the air above, a "lid" forms that
       traps pollutants near ground level. Common in Delhi winters (Nov-Feb)
       during night/early morning. We approximate this using:
       (30 - temperature) clipped to ≥0, normalized by wind speed.
       Low temp + low wind → strong inversion → trapped pollution.
    
    3. VENTILATION COEFFICIENT
       Product of wind speed and mixing height (approximated by temperature).
       Higher ventilation = better pollutant dispersal.
       Used in actual air quality management by CPCB.
    
    4. DEW POINT DEPRESSION
       Difference between temperature and dewpoint. Small depression means
       near-saturation → fog formation → particulates act as condensation
       nuclei → PM2.5 measurement spikes.
    
    5. SOLAR RADIATION INDEX
       Daytime solar heating lifts the planetary boundary layer, improving
       vertical mixing. This explains the afternoon PM2.5 dip.
    """
    
    n_features = 0
    
    # 1. Stagnation Index
    if "wind_speed" in df.columns and "humidity" in df.columns:
        df["stagnation_index"] = df["humidity"] / (df["wind_speed"] + 0.1)
        n_features += 1
        
        # Sustained stagnation (rolling mean over 6h)
        df["stagnation_6h"] = df["stagnation_index"].rolling(
            window=6, min_periods=1
        ).mean()
        n_features += 1
    
    # 2. Temperature Inversion Proxy
    if "temperature" in df.columns and "wind_speed" in df.columns:
        # Below ~15°C with low wind → inversion likely
        df["inversion_proxy"] = (
            (30 - df["temperature"]).clip(lower=0) / (df["wind_speed"] + 0.1)
        )
        n_features += 1
    
    # 3. Ventilation Coefficient (simplified)
    if "wind_speed" in df.columns and "temperature" in df.columns:
        # Mixing height correlates with temperature (warmer → higher BL)
        mixing_height_proxy = (df["temperature"].clip(lower=5) - 5) * 50 + 200
        df["ventilation_coeff"] = df["wind_speed"] * mixing_height_proxy
        n_features += 1
    
    # 4. Dew Point Depression
    if "temperature" in df.columns and "dewpoint" in df.columns:
        df["dewpoint_depression"] = df["temperature"] - df["dewpoint"]
        # Near-zero depression → fog likely → PM2.5 spike
        df["fog_risk"] = (df["dewpoint_depression"] < 3).astype(int)
        n_features += 2
    elif "temperature" in df.columns and "humidity" in df.columns:
        # Approximate dewpoint from temp and humidity (Magnus formula)
        a, b = 17.27, 237.7
        alpha = (a * df["temperature"]) / (b + df["temperature"]) + np.log(
            df["humidity"] / 100 + 1e-8
        )
        dewpoint_approx = (b * alpha) / (a - alpha)
        df["dewpoint_depression"] = df["temperature"] - dewpoint_approx
        df["fog_risk"] = (df["dewpoint_depression"] < 3).astype(int)
        n_features += 2
    
    # 5. Solar Radiation Index
    if "solar_radiation" in df.columns:
        df["solar_index"] = df["solar_radiation"] / (df["solar_radiation"].max() + 1e-8)
        n_features += 1
    else:
        # Approximate using hour of day (crude but useful)
        hour = df.index.hour
        # Bell curve centered at noon
        df["solar_proxy"] = np.exp(-0.5 * ((hour - 12) / 3.5) ** 2)
        df["solar_proxy"] *= (df.index.month.map(
            {1: 0.6, 2: 0.7, 3: 0.85, 4: 0.95, 5: 1.0, 6: 0.95,
             7: 0.7, 8: 0.7, 9: 0.8, 10: 0.85, 11: 0.7, 12: 0.6}
        ))
        n_features += 1
    
    # 6. Wind Direction Categories (cardinal directions affect source exposure)
    if "wind_direction" in df.columns:
        # NW winds bring stubble burning smoke from Punjab/Haryana
        wd = df["wind_direction"]
        df["wind_from_nw"] = (((wd >= 270) & (wd <= 360)) | (wd < 45)).astype(int)
        # SE winds are generally cleaner (from less industrialized areas)
        df["wind_from_se"] = ((wd >= 90) & (wd < 225)).astype(int)
        n_features += 2
    
    print(f"    → {n_features} atmospheric features added")
    return df


# 6. EVENT FLAGS

def add_event_flags(df):
    
    n_features = 0
    
    # Diwali dates (main celebration night)
    diwali_dates = pd.to_datetime([
        "2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07",
        "2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24",
        "2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08",
    ])
    
    # Mark Diwali window (2 days before to 2 days after)
    df["diwali_window"] = 0
    for d in diwali_dates:
        mask = (df.index >= d - pd.Timedelta(days=2)) & \
               (df.index <= d + pd.Timedelta(days=2))
        df.loc[mask, "diwali_window"] = 1
    n_features += 1
    
    # Stubble burning season (Oct 15 – Nov 30)
    month = df.index.month
    day = df.index.day
    df["stubble_season"] = (
        ((month == 10) & (day >= 15)) | (month == 11)
    ).astype(int)
    n_features += 1
    
    # Winter season flag (Nov – Feb: worst pollution months)
    df["winter_flag"] = df.index.month.isin([11, 12, 1, 2]).astype(int)
    n_features += 1
    
    # Rush hour flag (7-10 AM and 5-9 PM)
    hour = df.index.hour
    df["rush_hour"] = (
        ((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 21))
    ).astype(int)
    n_features += 1
    
    # Night flag (low boundary layer, poor dispersion)
    df["night_flag"] = ((hour >= 22) | (hour <= 5)).astype(int)
    n_features += 1
    
    print(f"    → {n_features} event flags added")
    return df


# 7. INTERACTION FEATURES

def add_interaction_features(df):
    
    n_features = 0
    
    # Winter night stagnation: worst-case scenario combo
    if all(c in df.columns for c in ["winter_flag", "night_flag", "stagnation_index"]):
        df["winter_night_stagnation"] = (
            df["winter_flag"] * df["night_flag"] * df["stagnation_index"]
        )
        n_features += 1
    
    # Stubble + NW wind: smoke blowing from Punjab into Delhi
    if all(c in df.columns for c in ["stubble_season", "wind_from_nw"]):
        df["stubble_nw_wind"] = df["stubble_season"] * df["wind_from_nw"]
        n_features += 1
    
    # PM2.5/PM10 ratio (indicates fine particle dominance, source signature)
    if all(c in df.columns for c in ["pm25", "pm10"]):
        df["pm25_pm10_ratio"] = df["pm25"] / (df["pm10"] + 1e-8)
        df["pm25_pm10_ratio"] = df["pm25_pm10_ratio"].clip(0, 2)
        n_features += 1
    
    return df


# MAIN PIPELINE


def build_features(input_path=None, output_path=None):
 
    config = load_config()
    target = "pm25"
    
    # Load data
    if input_path is None:
        input_path = PROCESSED_DIR / "delhi_aqi_combined.csv"
        # Fallback to clean-only file
        if not input_path.exists():
            input_path = PROCESSED_DIR / "delhi_aqi_clean.csv"
    
    input_path = Path(input_path)
    print("FEATURE ENGINEERING PIPELINE")
    print(f"Input: {input_path}")
    
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    # Standardize column names 
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip().replace(" ", "_").replace(".", "")
        if cl in ["pm25", "pm2.5", "pm_25"]: col_map[col] = "pm25"
        elif cl in ["pm10"]: col_map[col] = "pm10"
        elif cl in ["no2"]: col_map[col] = "no2"
        elif cl in ["so2"]: col_map[col] = "so2"
        elif cl in ["co"]: col_map[col] = "co"
        elif cl in ["o3"]: col_map[col] = "o3"
        elif cl in ["nh3"]: col_map[col] = "nh3"
        elif cl in ["temperature", "temperature_2m"]: col_map[col] = "temperature"
        elif cl in ["humidity", "relative_humidity_2m", "relative_humidity"]: col_map[col] = "humidity"
        elif cl in ["wind_speed", "wind_speed_10m"]: col_map[col] = "wind_speed"
        elif cl in ["wind_direction", "wind_direction_10m"]: col_map[col] = "wind_direction"
        elif cl in ["pressure", "surface_pressure"]: col_map[col] = "pressure"
        elif cl in ["precipitation"]: col_map[col] = "precipitation"
        elif cl in ["dewpoint", "dewpoint_2m"]: col_map[col] = "dewpoint"
        elif cl in ["cloud_cover"]: col_map[col] = "cloud_cover"
        elif cl in ["solar_radiation", "shortwave_radiation"]: col_map[col] = "solar_radiation"
    
    df = df.rename(columns=col_map)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    print(f"Input shape: {df.shape}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    print(f"Target ({target}) — mean: {df[target].mean():.1f}, "
          f"null: {df[target].isnull().sum()}")
    print()
    
    # Run pipeline steps 
    df = add_temporal_features(df)
    df = add_lag_features(df, config)
    df = add_rolling_features(df, config)
    df = add_rate_of_change_features(df)
    df = add_atmospheric_features(df)
    df = add_event_flags(df)
    df = add_interaction_features(df)
    
    # Drop initial rows where lags create NaNs 
    max_lag = max(config["features"]["lags"])
    rows_before = len(df)
    df = df.iloc[max_lag:]
    
    # Drop rows where target is NaN 
    df = df.dropna(subset=[target])
    
    # Handle remaining NaNs in features 
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        print(f"\n  Filling NaNs in {len(cols_with_nulls)} columns:")
        for col in cols_with_nulls.index:
            pct = cols_with_nulls[col] / len(df) * 100
            if pct < 5:
                # Small gaps: forward fill then backward fill
                df[col] = df[col].ffill().bfill()
            else:
                # Larger gaps: fill with median
                df[col] = df[col].fillna(df[col].median())
            print(f"    {col}: {cols_with_nulls[col]} nulls ({pct:.1f}%) → filled")
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Identify feature columns vs target 
    # Exclude raw pollutants and weather (we keep their lagged versions)
    raw_cols = ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3",
                "temperature", "humidity", "wind_speed", "wind_direction",
                "pressure", "precipitation", "dewpoint", "cloud_cover",
                "solar_radiation", "rain", "wind_gusts",
                "direct_radiation", "diffuse_radiation"]
    
    feature_cols = [c for c in df.columns if c != target]
    
    # Save 
    if output_path is None:
        output_path = PROCESSED_DIR / "delhi_aqi_features.csv"
    
    output_path = Path(output_path)
    df.to_csv(output_path)
    
    #  Summary
    print(f"\n{'=' * 60}")
    print(f"FEATURE ENGINEERING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")
    print(f"Shape:  {df.shape}")
    print(f"Target: {target}")
    print(f"Features: {len(feature_cols)}")
    print(f"Rows:   {len(df):,}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    print(f"Remaining NaNs: {df.isnull().sum().sum()}")
    
    # Feature categories
    categories = {
        "Temporal":    [c for c in feature_cols if any(
            c.startswith(p) for p in ["hour", "dow", "month", "doy", "day_", "week_", "is_"])],
        "Lag":         [c for c in feature_cols if "_lag_" in c],
        "Rolling":     [c for c in feature_cols if any(
            c.startswith(p) for p in ["pm25_r", "wind_r"])],
        "Rate":        [c for c in feature_cols if any(
            c.startswith(p) for p in ["pm25_diff", "pm25_pct", "pm25_accel", "pm25_ratio"])],
        "Atmospheric": [c for c in feature_cols if c in [
            "stagnation_index", "stagnation_6h", "inversion_proxy",
            "ventilation_coeff", "dewpoint_depression", "fog_risk",
            "solar_proxy", "solar_index", "wind_from_nw", "wind_from_se"]],
        "Events":      [c for c in feature_cols if c in [
            "diwali_window", "stubble_season", "winter_flag",
            "rush_hour", "night_flag"]],
        "Interaction": [c for c in feature_cols if c in [
            "winter_night_stagnation", "stubble_nw_wind", "pm25_pm10_ratio"]],
        "Raw":         [c for c in feature_cols if c in raw_cols],
    }
    
    print(f"\nFeature breakdown:")
    for cat, cols in categories.items():
        if cols:
            print(f"  {cat:15s}: {len(cols):3d} features")
    
    # Save feature list for reference
    feature_list_path = PROCESSED_DIR / "feature_list.txt"
    with open(feature_list_path, "w") as f:
        f.write(f"Target: {target}\n")
        f.write(f"Total features: {len(feature_cols)}\n\n")
        for cat, cols in categories.items():
            if cols:
                f.write(f"\n{cat} ({len(cols)}):\n")
                for c in sorted(cols):
                    f.write(f"  {c}\n")
    
    print(f"\nFeature list saved to: {feature_list_path}")
    
    return df, feature_cols, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features for AQI forecasting")
    parser.add_argument(
        "--input",
        default=None,
        help="Input CSV path (default: data/processed/delhi_aqi_combined.csv)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/processed/delhi_aqi_features.csv)"
    )
    
    args = parser.parse_args()
    build_features(args.input, args.output)
