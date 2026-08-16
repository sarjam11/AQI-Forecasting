"""
Prepare the final merged dataset from raw AQI + weather data.

Handles multiple input formats (Kaggle, OpenCity, OpenAQ) and merges
with weather data into a single clean hourly DataFrame.

Usage:
    python src/data/prepare_dataset.py
    python src/data/prepare_dataset.py --input-format kaggle
    python src/data/prepare_dataset.py --input-format opencity
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    """Load project config."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    config_path = project_root / "configs" / "default.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Format-specific loaders
# ──────────────────────────────────────────────

def load_kaggle_format():
    """
    Load Kaggle dataset format.
    Expected: single CSV with datetime + pollutant columns.
    Adapt column names based on actual file structure.
    """
    print("Loading Kaggle format...")
    
    # Find the CSV file(s)
    csv_files = list(RAW_DIR.glob("*.csv"))
    aqi_files = [f for f in csv_files if "weather" not in f.name.lower()]
    
    if not aqi_files:
        raise FileNotFoundError("No AQI CSV files found in data/raw/")
    
    print(f"  Found {len(aqi_files)} file(s)")
    
    dfs = []
    for f in aqi_files:
        print(f"  Reading: {f.name}")
        df = pd.read_csv(f)
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {list(df.columns)}")
        dfs.append(df)
    
    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = pd.concat(dfs, ignore_index=True)
    
    # Standardize column names (handle common variations)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        
        if col_lower in ["datetime", "date_time", "date", "timestamp", "from date"]:
            col_map[col] = "datetime"
        elif col_lower in ["pm2.5", "pm25", "pm 2.5"]:
            col_map[col] = "pm25"
        elif col_lower in ["pm10", "pm 10"]:
            col_map[col] = "pm10"
        elif col_lower in ["no2", "nitrogen dioxide"]:
            col_map[col] = "no2"
        elif col_lower in ["so2", "sulfur dioxide", "sulphur dioxide"]:
            col_map[col] = "so2"
        elif col_lower in ["co", "carbon monoxide"]:
            col_map[col] = "co"
        elif col_lower in ["o3", "ozone"]:
            col_map[col] = "o3"
        elif col_lower in ["nh3", "ammonia"]:
            col_map[col] = "nh3"
        elif col_lower in ["no", "nitric oxide"]:
            col_map[col] = "no"
        elif col_lower in ["nox"]:
            col_map[col] = "nox"
        elif col_lower in ["benzene"]:
            col_map[col] = "benzene"
        elif col_lower in ["toluene"]:
            col_map[col] = "toluene"
        elif col_lower in ["xylene"]:
            col_map[col] = "xylene"
        elif col_lower in ["aqi"]:
            col_map[col] = "aqi"
        elif col_lower in ["station", "stationid", "station_name", "site"]:
            col_map[col] = "station"
    
    df = df.rename(columns=col_map)
    print(f"\n  Mapped columns: {col_map}")
    
    return df


def load_opencity_format():
    """
    Load OpenCity.in CKAN format.
    Expected: station-specific CSVs with hourly readings.
    """
    print("Loading OpenCity format...")
    
    csv_files = list(RAW_DIR.glob("*CPCB*.csv")) + list(RAW_DIR.glob("*DPCC*.csv"))
    if not csv_files:
        csv_files = [f for f in RAW_DIR.glob("*.csv") if "weather" not in f.name.lower()]
    
    if not csv_files:
        raise FileNotFoundError("No station CSV files found in data/raw/")
    
    dfs = []
    for f in csv_files:
        print(f"  Reading: {f.name}")
        df = pd.read_csv(f)
        
        # Add station name from filename if not in data
        if "station" not in [c.lower() for c in df.columns]:
            station = f.stem.split("_")[0]
            df["station"] = station
        
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Apply same column mapping
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ["datetime", "date_time", "date", "timestamp", "from date"]:
            col_map[col] = "datetime"
        elif col_lower in ["pm2.5", "pm25"]:
            col_map[col] = "pm25"
        elif col_lower in ["pm10"]:
            col_map[col] = "pm10"
        elif col_lower in ["no2"]:
            col_map[col] = "no2"
        elif col_lower in ["so2"]:
            col_map[col] = "so2"
        elif col_lower in ["co"]:
            col_map[col] = "co"
        elif col_lower in ["o3"]:
            col_map[col] = "o3"
    
    df = df.rename(columns=col_map)
    return df


def load_openaq_format():
    """Load OpenAQ API v3 download format."""
    print("Loading OpenAQ format...")
    
    csv_files = list(RAW_DIR.glob("openaq_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No openaq_*.csv files found in data/raw/")
    
    # Group by location
    locations = {}
    for f in csv_files:
        parts = f.stem.split("_")  # openaq_ITO_pm25
        loc = parts[1]
        param = parts[2]
        
        df = pd.read_csv(f)
        
        if loc not in locations:
            locations[loc] = {}
        locations[loc][param] = df
    
    # Merge parameters per location
    merged_dfs = []
    for loc, params in locations.items():
        loc_df = None
        for param_name, param_df in params.items():
            # Extract datetime and value
            if "datetime" in param_df.columns:
                dt_col = "datetime"
            elif "period" in param_df.columns:
                # OpenAQ v3 hours endpoint returns 'period'
                param_df["datetime"] = pd.to_datetime(
                    param_df["period"].apply(
                        lambda x: eval(x)["datetimeFrom"]["utc"] if isinstance(x, str) else x
                    )
                )
                dt_col = "datetime"
            
            temp = param_df[[dt_col, "value"]].copy()
            temp = temp.rename(columns={"value": param_name, dt_col: "datetime"})
            temp["datetime"] = pd.to_datetime(temp["datetime"])
            
            if loc_df is None:
                loc_df = temp
            else:
                loc_df = loc_df.merge(temp, on="datetime", how="outer")
        
        loc_df["station"] = loc
        merged_dfs.append(loc_df)
    
    df = pd.concat(merged_dfs, ignore_index=True)
    return df


def auto_detect_format():
    """Auto-detect the input data format based on files present."""
    files = list(RAW_DIR.glob("*.csv"))
    names = [f.name.lower() for f in files]
    
    if any("openaq" in n for n in names):
        return "openaq"
    elif any("cpcb" in n or "dpcc" in n for n in names):
        return "opencity"
    else:
        return "kaggle"  # default assumption


# ──────────────────────────────────────────────
# Cleaning Pipeline
# ──────────────────────────────────────────────

def clean_aqi_data(df, config):
    """
    Clean and standardize AQI DataFrame.
    """
    print(f"\nCleaning AQI data...")
    print(f"  Input shape: {df.shape}")
    
    # 1. Parse datetime
    if "datetime" not in df.columns:
        raise ValueError(f"No datetime column found. Columns: {list(df.columns)}")
    
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    
    # Remove timezone info if present (normalize to local time)
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    
    # 2. Filter to target station if multiple exist
    if "station" in df.columns:
        stations = df["station"].unique()
        print(f"  Stations found: {stations}")
        
        # Prefer ITO or use first station
        target = config["data"]["primary_station"]
        target_matches = [s for s in stations if "ito" in str(s).lower()]
        
        if target_matches:
            selected = target_matches[0]
        else:
            selected = stations[0]
        
        print(f"  Using station: {selected}")
        df = df[df["station"] == selected].copy()
    
    # 3. Convert pollutant columns to numeric
    pollutant_cols = ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3", "no", "nox",
                      "benzene", "toluene", "xylene"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 4. Set datetime index and resample to hourly
    df = df.set_index("datetime")
    df = df.sort_index()
    
    # Keep only numeric columns for resampling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    # Resample to hourly (takes median to handle duplicates and sub-hourly data)
    df = df.resample("1h").median()
    
    # 5. Remove physically impossible / erroneous values
    bounds = {
        "pm25": (0, 999),
        "pm10": (0, 999),
        "no2": (0, 500),
        "so2": (0, 500),
        "co": (0, 50),     # in mg/m³ for CPCB
        "o3": (0, 500),
    }
    
    for col, (low, high) in bounds.items():
        if col in df.columns:
            invalid = ((df[col] < low) | (df[col] > high)).sum()
            if invalid > 0:
                print(f"  Removed {invalid} invalid values from {col}")
            df.loc[(df[col] < low) | (df[col] > high), col] = np.nan
    
    # 6. Handle missing values
    print(f"\n  Missing values before imputation:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        pct = missing / len(df) * 100
        print(f"    {col}: {missing:,} ({pct:.1f}%)")
    
    # Forward fill small gaps (up to 3 hours)
    df = df.ffill(limit=3)
    
    # Linear interpolation for medium gaps (up to 6 hours)
    df = df.interpolate(method="linear", limit=6)
    
    print(f"\n  Missing values after imputation:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        pct = missing / len(df) * 100
        if missing > 0:
            print(f"    {col}: {missing:,} ({pct:.1f}%)")
    
    # 7. Drop rows where target PM2.5 is still missing
    if "pm25" in df.columns:
        before = len(df)
        df = df.dropna(subset=["pm25"])
        after = len(df)
        print(f"\n  Dropped {before - after} rows with missing PM2.5")
    
    print(f"  Output shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def load_weather(config):
    """Load and clean weather data."""
    weather_path = RAW_DIR / "weather_delhi.csv"
    
    if not weather_path.exists():
        print(f"\nWARNING: Weather file not found at {weather_path}")
        print("  Run: python src/data/download_weather.py")
        return None
    
    print(f"\nLoading weather data from {weather_path}...")
    weather = pd.read_csv(weather_path, parse_dates=["datetime"])
    weather = weather.set_index("datetime")
    weather = weather.sort_index()
    
    # Resample to hourly (should already be hourly from Open-Meteo)
    weather = weather.resample("1h").mean()
    
    print(f"  Weather shape: {weather.shape}")
    print(f"  Weather range: {weather.index.min()} to {weather.index.max()}")
    
    return weather


def merge_datasets(aqi_df, weather_df):
    """Merge AQI and weather DataFrames on datetime index."""
    print(f"\nMerging AQI and weather data...")
    
    if weather_df is None:
        print("  No weather data — returning AQI only")
        return aqi_df
    
    # Find overlapping date range
    overlap_start = max(aqi_df.index.min(), weather_df.index.min())
    overlap_end = min(aqi_df.index.max(), weather_df.index.max())
    print(f"  Overlapping range: {overlap_start} to {overlap_end}")
    
    # Inner join on datetime to keep only overlapping period
    combined = aqi_df.join(weather_df, how="inner")
    
    print(f"  Merged shape: {combined.shape}")
    print(f"  Columns: {list(combined.columns)}")
    
    return combined


def generate_data_report(df, output_path):
    """Generate a summary report of the final dataset."""
    report = []
    report.append("=" * 60)
    report.append("DATASET REPORT")
    report.append("=" * 60)
    report.append(f"\nShape: {df.shape}")
    report.append(f"Date range: {df.index.min()} to {df.index.max()}")
    report.append(f"Total hours: {len(df):,}")
    report.append(f"Total days: {(df.index.max() - df.index.min()).days:,}")
    
    report.append(f"\nColumns ({df.shape[1]}):")
    for col in df.columns:
        missing = df[col].isnull().sum()
        pct = missing / len(df) * 100
        report.append(
            f"  {col:25s} | "
            f"mean={df[col].mean():8.2f} | "
            f"std={df[col].std():8.2f} | "
            f"min={df[col].min():8.2f} | "
            f"max={df[col].max():8.2f} | "
            f"missing={missing} ({pct:.1f}%)"
        )
    
    if "pm25" in df.columns:
        report.append(f"\nPM2.5 Statistics:")
        report.append(f"  Mean: {df['pm25'].mean():.1f} µg/m³")
        report.append(f"  Median: {df['pm25'].median():.1f} µg/m³")
        report.append(f"  95th percentile: {df['pm25'].quantile(0.95):.1f} µg/m³")
        report.append(f"  Max: {df['pm25'].max():.1f} µg/m³")
        report.append(f"  Days > 100 µg/m³: {(df['pm25'] > 100).sum():,} hours")
        report.append(f"  Days > 250 µg/m³: {(df['pm25'] > 250).sum():,} hours")
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(output_path, "w") as f:
        f.write(report_text)
    
    return report_text


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(input_format=None):
    """Run the full data preparation pipeline."""
    config = load_config()
    
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    # 1. Detect format and load AQI data
    if input_format is None:
        input_format = auto_detect_format()
    
    print(f"\nDetected format: {input_format}")
    
    loaders = {
        "kaggle": load_kaggle_format,
        "opencity": load_opencity_format,
        "openaq": load_openaq_format,
    }
    
    aqi_raw = loaders[input_format]()
    
    # 2. Clean AQI data
    aqi_clean = clean_aqi_data(aqi_raw, config)
    
    # Save intermediate AQI-only file
    aqi_path = PROCESSED_DIR / "delhi_aqi_clean.csv"
    aqi_clean.to_csv(aqi_path)
    print(f"\n  Saved clean AQI data to: {aqi_path}")
    
    # 3. Load weather data
    weather = load_weather(config)
    
    # 4. Merge
    combined = merge_datasets(aqi_clean, weather)
    
    # 5. Final cleanup — fill remaining small gaps in features
    combined = combined.ffill(limit=2).fillna(method="bfill", limit=2)
    
    # 6. Save final dataset
    output_path = PROCESSED_DIR / "delhi_aqi_combined.csv"
    combined.to_csv(output_path)
    print(f"\n✓ Final dataset saved to: {output_path}")
    
    # 7. Generate report
    report_path = PROCESSED_DIR / "data_report.txt"
    generate_data_report(combined, report_path)
    print(f"\n✓ Report saved to: {report_path}")
    
    print(f"\nNext step: Open notebooks/01_eda.ipynb to explore the data")
    
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare merged AQI dataset")
    parser.add_argument(
        "--input-format",
        choices=["kaggle", "opencity", "openaq"],
        default=None,
        help="Input format (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    main(args.input_format)
