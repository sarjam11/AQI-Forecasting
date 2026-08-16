"""
Download historical hourly weather data for Delhi from Open-Meteo API.

Open-Meteo is free, requires no API key, and provides ERA5 reanalysis data
from 1940 onwards at hourly resolution.

Usage:
    python src/data/download_weather.py
    python src/data/download_weather.py --start 2019-01-01 --end 2024-12-31
"""

import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Delhi coordinates
DELHI_LAT = 28.6139
DELHI_LON = 77.2090

# Open-Meteo Historical API endpoint
HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"

# Weather variables to fetch
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "surface_pressure",
    "precipitation",
    "rain",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
]


def fetch_weather_chunk(start_date, end_date):
    """
    Fetch one chunk of weather data from Open-Meteo.
    
    Open-Meteo can handle large date ranges but we chunk by year
    to be safe with response sizes and avoid timeouts.
    """
    params = {
        "latitude": DELHI_LAT,
        "longitude": DELHI_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Kolkata",
    }
    
    response = requests.get(HISTORICAL_API, params=params, timeout=120)
    response.raise_for_status()
    
    data = response.json()
    
    if "hourly" not in data:
        raise ValueError(f"No hourly data in response: {list(data.keys())}")
    
    hourly = data["hourly"]
    
    # Build DataFrame
    df = pd.DataFrame({"datetime": pd.to_datetime(hourly["time"])})
    
    for var in HOURLY_VARIABLES:
        if var in hourly:
            df[var] = hourly[var]
        else:
            print(f"  Warning: {var} not in response")
            df[var] = np.nan
    
    return df


def download_weather(start_date="2019-01-01", end_date="2024-12-31"):
    """
    Download full date range of weather data, chunked by year.
    """
    print("=" * 60)
    print("Downloading Delhi Weather Data from Open-Meteo")
    print(f"  Location: Delhi ({DELHI_LAT}°N, {DELHI_LON}°E)")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Variables: {len(HOURLY_VARIABLES)}")
    print("=" * 60)
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    all_chunks = []
    current_start = start
    
    while current_start <= end:
        # Chunk by year
        current_end = min(
            pd.Timestamp(f"{current_start.year}-12-31"),
            end
        )
        
        chunk_start_str = current_start.strftime("%Y-%m-%d")
        chunk_end_str = current_end.strftime("%Y-%m-%d")
        
        print(f"\n  Fetching {chunk_start_str} to {chunk_end_str}...", end=" ")
        
        try:
            df = fetch_weather_chunk(chunk_start_str, chunk_end_str)
            all_chunks.append(df)
            print(f"✓ ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error: {e}")
            # Try smaller chunks (monthly) on failure
            print(f"    Retrying month by month...")
            monthly_start = current_start
            while monthly_start <= current_end:
                monthly_end = min(
                    monthly_start + pd.DateOffset(months=1) - pd.Timedelta(days=1),
                    current_end
                )
                ms = monthly_start.strftime("%Y-%m-%d")
                me = monthly_end.strftime("%Y-%m-%d")
                
                try:
                    df = fetch_weather_chunk(ms, me)
                    all_chunks.append(df)
                    print(f"      {ms} to {me}: ✓ ({len(df)} rows)")
                except Exception as e2:
                    print(f"      {ms} to {me}: ✗ ({e2})")
                
                monthly_start = monthly_end + pd.Timedelta(days=1)
                time.sleep(1)
        
        current_start = current_end + pd.Timedelta(days=1)
        time.sleep(0.5)  # Rate limit
    
    if not all_chunks:
        print("\nERROR: No weather data downloaded!")
        return None
    
    # Combine all chunks
    weather = pd.concat(all_chunks, ignore_index=True)
    weather = weather.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    weather = weather.reset_index(drop=True)
    
    # Rename columns for clarity
    column_map = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "dewpoint_2m": "dewpoint",
        "surface_pressure": "pressure",
        "precipitation": "precipitation",
        "rain": "rain",
        "cloud_cover": "cloud_cover",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "wind_gusts_10m": "wind_gusts",
        "shortwave_radiation": "solar_radiation",
        "direct_radiation": "direct_radiation",
        "diffuse_radiation": "diffuse_radiation",
    }
    weather = weather.rename(columns=column_map)
    
    # Save
    output_path = RAW_DIR / "weather_delhi.csv"
    weather.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"Weather data saved to: {output_path}")
    print(f"  Rows: {len(weather):,}")
    print(f"  Columns: {weather.shape[1]}")
    print(f"  Date range: {weather['datetime'].min()} to {weather['datetime'].max()}")
    print(f"  Missing values:")
    for col in weather.columns:
        if col == "datetime":
            continue
        missing = weather[col].isnull().sum()
        pct = missing / len(weather) * 100
        if missing > 0:
            print(f"    {col}: {missing} ({pct:.1f}%)")
    
    print(f"\n  Sample (first 3 rows):")
    print(weather.head(3).to_string(index=False))
    print(f"\n✓ Weather download complete!")
    print(f"  Next step: python src/data/prepare_dataset.py")
    
    return weather


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Delhi weather data")
    parser.add_argument("--start", default="2019-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    
    args = parser.parse_args()
    download_weather(args.start, args.end)
