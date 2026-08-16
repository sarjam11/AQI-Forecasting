"""
Download Delhi air quality data from CPCB.

Three strategies (in order of preference):
  1. Kaggle dataset (pre-cleaned, 2019-2024) — requires kaggle CLI
  2. OpenCity.in CKAN (station-wise CSVs, 2017-2023) — direct HTTP
  3. OpenAQ API v3 (mirror of CPCB data) — requires free API key

Usage:
    python src/data/download_aqi.py --source kaggle
    python src/data/download_aqi.py --source opencity
    python src/data/download_aqi.py --source openaq --api-key YOUR_KEY
"""

import argparse
import os
import requests
import time
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Strategy 1: Kaggle Dataset
# ──────────────────────────────────────────────
def download_from_kaggle():
    """
    Download pre-cleaned Delhi NCR hourly AQI dataset from Kaggle.
    
    Prerequisites:
      pip install kaggle
      Place kaggle.json in ~/.kaggle/ (get from kaggle.com > Account > API)
    
    Dataset: aniket0712/delhi-ncr-hourly-air-quality-dataset-20192024
    """
    print("=" * 60)
    print("STRATEGY 1: Kaggle Dataset Download")
    print("=" * 60)

    try:
        os.system(
            f"kaggle datasets download "
            f"-d aniket0712/delhi-ncr-hourly-air-quality-dataset-20192024 "
            f"-p {RAW_DIR} --unzip"
        )
        
        # Check what was downloaded
        downloaded = list(RAW_DIR.glob("*.csv"))
        if downloaded:
            print(f"\nDownloaded {len(downloaded)} file(s):")
            for f in downloaded:
                df = pd.read_csv(f, nrows=5)
                print(f"  {f.name}: {df.shape[1]} columns")
                print(f"    Columns: {list(df.columns[:10])}...")
            return True
        else:
            print("No CSV files found after download.")
            return False
            
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Make sure kaggle CLI is installed and kaggle.json is configured.")
        print("Falling back to alternative sources...\n")
        return False


# ──────────────────────────────────────────────
# Strategy 2: OpenCity.in CKAN Portal
# ──────────────────────────────────────────────

# Known resource URLs from data.opencity.in CKAN portal
# These are direct download links for Delhi CPCB station data (2017-2023)
OPENCITY_RESOURCES = {
    "ITO_CPCB_2017_2023": {
        "url": "https://data.opencity.in/dataset/delhi-hourly-air-quality-reports",
        "station": "ITO",
        "agency": "CPCB",
        "description": "ITO CPCB AQI Data 2017-2023"
    },
    "Anand_Vihar_DPCC_2017_2023": {
        "url": "https://data.opencity.in/dataset/delhi-hourly-air-quality-reports",
        "station": "Anand Vihar",
        "agency": "DPCC",
        "description": "Anand Vihar DPCC AQI Data 2017-2023"
    },
    "RK_Puram_DPCC_2017_2023": {
        "url": "https://data.opencity.in/dataset/delhi-hourly-air-quality-reports",
        "station": "RK Puram",
        "agency": "DPCC",
        "description": "RK Puram DPCC AQI Data 2017-2023"
    },
}


def download_from_opencity():
    """
    Download from OpenCity.in CKAN portal.
    
    NOTE: The portal organizes data as individual CSV resources.
    You'll need to manually download station CSVs from:
    https://data.opencity.in/dataset/delhi-hourly-air-quality-reports
    
    Steps:
    1. Visit the URL above
    2. Find your station (e.g., "ITO CPCB AQI Data 2017-2023")
    3. Click "Download" to get the CSV
    4. Save to data/raw/
    
    This function provides a CKAN API-based approach as well.
    """
    print("=" * 60)
    print("STRATEGY 2: OpenCity.in CKAN Portal")
    print("=" * 60)
    
    ckan_api = "https://data.opencity.in/api/3/action/package_show"
    params = {"id": "delhi-hourly-air-quality-reports"}
    
    try:
        print("Querying CKAN API for resource list...")
        resp = requests.get(ckan_api, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get("success"):
            raise ValueError("CKAN API returned unsuccessful response")
        
        resources = data["result"]["resources"]
        print(f"Found {len(resources)} resources\n")
        
        # Filter for relevant CSVs
        target_keywords = ["ITO", "Anand Vihar", "RK Puram", "DTU"]
        downloaded_count = 0
        
        for resource in resources:
            name = resource.get("name", "")
            url = resource.get("url", "")
            fmt = resource.get("format", "").upper()
            
            if fmt != "CSV":
                continue
                
            # Check if this is a station we want
            is_target = any(kw.lower() in name.lower() for kw in target_keywords)
            if not is_target:
                continue
            
            # Download
            filename = f"{name.replace(' ', '_').replace(',', '')}.csv"
            filepath = RAW_DIR / filename
            
            if filepath.exists():
                print(f"  [SKIP] {filename} already exists")
                continue
                
            print(f"  Downloading: {name}")
            print(f"    URL: {url}")
            
            try:
                file_resp = requests.get(url, timeout=60, stream=True)
                file_resp.raise_for_status()
                
                with open(filepath, "wb") as f:
                    for chunk in file_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Quick validation
                df = pd.read_csv(filepath, nrows=5)
                print(f"    Saved: {filepath} ({df.shape[1]} columns)")
                downloaded_count += 1
                
            except Exception as e:
                print(f"    FAILED: {e}")
            
            time.sleep(1)  # rate limit
        
        if downloaded_count > 0:
            print(f"\nDownloaded {downloaded_count} station file(s)")
            return True
        else:
            print("\nNo files downloaded. Try manual download from:")
            print("  https://data.opencity.in/dataset/delhi-hourly-air-quality-reports")
            return False
            
    except Exception as e:
        print(f"CKAN API failed: {e}")
        print("\nManual download instructions:")
        print("  1. Go to: https://data.opencity.in/dataset/delhi-hourly-air-quality-reports")
        print("  2. Download CSV files for: ITO, Anand Vihar, RK Puram")
        print("  3. Save them to: data/raw/")
        return False


# ──────────────────────────────────────────────
# Strategy 3: OpenAQ API v3
# ──────────────────────────────────────────────

OPENAQ_BASE = "https://api.openaq.org/v3"

# Delhi CPCB station IDs on OpenAQ (found via explore.openaq.org)
OPENAQ_DELHI_LOCATIONS = {
    "DTU_CPCB": 5626,       # DTU, New Delhi - CPCB
    "US_Embassy": 8118,     # US Embassy, New Delhi (AirNow)
    "Anand_Vihar": 8557,    # Anand Vihar, Delhi - CPCB
    "ITO": 8562,            # ITO, Delhi - CPCB
    "RK_Puram": 8564,       # R.K. Puram, Delhi - DPCC
}


def get_location_sensors(location_id, api_key):
    """Get all sensor IDs for a location."""
    url = f"{OPENAQ_BASE}/locations/{location_id}"
    headers = {"X-API-Key": api_key}
    
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    sensors = {}
    for sensor in data["results"][0].get("sensors", []):
        param_name = sensor["parameter"]["name"]
        sensor_id = sensor["id"]
        units = sensor["parameter"]["units"]
        sensors[param_name] = {"id": sensor_id, "units": units}
    
    return sensors


def download_sensor_data(sensor_id, api_key, date_from, date_to):
    """Download hourly measurements for a single sensor."""
    url = f"{OPENAQ_BASE}/sensors/{sensor_id}/hours"
    headers = {"X-API-Key": api_key}
    
    all_data = []
    page = 1
    
    while True:
        params = {
            "date_from": date_from,
            "date_to": date_to,
            "limit": 1000,
            "page": page,
        }
        
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        
        if resp.status_code == 429:
            print("    Rate limited, waiting 60s...")
            time.sleep(60)
            continue
            
        resp.raise_for_status()
        data = resp.json()
        
        results = data.get("results", [])
        if not results:
            break
            
        all_data.extend(results)
        
        if len(results) < 1000:
            break
            
        page += 1
        time.sleep(0.3)  # rate limit
    
    return all_data


def download_from_openaq(api_key, date_from="2019-01-01", date_to="2024-12-31"):
    """
    Download Delhi air quality data from OpenAQ API v3.
    
    Prerequisites:
      Sign up at https://explore.openaq.org (free)
      Get your API key from account settings
    """
    print("=" * 60)
    print("STRATEGY 3: OpenAQ API v3")
    print("=" * 60)
    
    if not api_key:
        print("ERROR: OpenAQ API key required.")
        print("  1. Sign up at https://explore.openaq.org")
        print("  2. Get API key from account settings")
        print("  3. Run with: --api-key YOUR_KEY")
        return False
    
    for loc_name, loc_id in OPENAQ_DELHI_LOCATIONS.items():
        print(f"\n--- {loc_name} (ID: {loc_id}) ---")
        
        try:
            # Get sensors
            sensors = get_location_sensors(loc_id, api_key)
            print(f"  Found sensors: {list(sensors.keys())}")
            
            location_data = {}
            
            for param_name, sensor_info in sensors.items():
                # Only download pollutants we care about
                if param_name not in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
                    continue
                    
                print(f"  Downloading {param_name} (sensor {sensor_info['id']})...")
                
                raw = download_sensor_data(
                    sensor_info["id"], api_key, date_from, date_to
                )
                
                if raw:
                    df = pd.DataFrame(raw)
                    location_data[param_name] = df
                    print(f"    Got {len(df)} hourly records")
                else:
                    print(f"    No data returned")
                
                time.sleep(1)
            
            # Save per location
            if location_data:
                for param_name, df in location_data.items():
                    filepath = RAW_DIR / f"openaq_{loc_name}_{param_name}.csv"
                    df.to_csv(filepath, index=False)
                    print(f"  Saved: {filepath}")
                    
        except Exception as e:
            print(f"  ERROR for {loc_name}: {e}")
            continue
    
    return True


# ──────────────────────────────────────────────
# Manual Download Helper
# ──────────────────────────────────────────────

def print_manual_instructions():
    """Print step-by-step manual download guide."""
    print("""
╔══════════════════════════════════════════════════════════╗
║            MANUAL DOWNLOAD INSTRUCTIONS                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  If automated download fails, use these manual options:  ║
║                                                          ║
║  OPTION A — Kaggle (Recommended, pre-cleaned)            ║
║  1. Go to: kaggle.com/datasets/aniket0712/               ║
║     delhi-ncr-hourly-air-quality-dataset-20192024        ║
║  2. Click "Download" → unzip into data/raw/              ║
║                                                          ║
║  OPTION B — OpenCity.in (Station-wise, official CPCB)    ║
║  1. Go to: data.opencity.in/dataset/                     ║
║     delhi-hourly-air-quality-reports                     ║
║  2. Download CSVs for: ITO, Anand Vihar, RK Puram        ║
║  3. Save to data/raw/                                    ║
║                                                          ║
║  OPTION C — CPCB Portal (Official, month-by-month)       ║
║  1. Go to: app.cpcbccr.com/ccr/#/caaqm-dashboard-all/   ║
║     caaqm-landing                                        ║
║  2. Select: Delhi → ITO → PM2.5,PM10,NO2,SO2,CO,O3      ║
║  3. Set date range (1 month at a time)                   ║
║  4. Export CSV → repeat for all months                   ║
║  5. Save everything to data/raw/                         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Delhi AQI data")
    parser.add_argument(
        "--source", 
        choices=["kaggle", "opencity", "openaq", "all"],
        default="all",
        help="Data source to use (default: tries all in order)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAQ API key (required for openaq source)"
    )
    parser.add_argument(
        "--date-from",
        default="2019-01-01",
        help="Start date (default: 2019-01-01)"
    )
    parser.add_argument(
        "--date-to",
        default="2024-12-31",
        help="End date (default: 2024-12-31)"
    )
    
    args = parser.parse_args()
    
    success = False
    
    if args.source == "kaggle":
        success = download_from_kaggle()
    elif args.source == "opencity":
        success = download_from_opencity()
    elif args.source == "openaq":
        success = download_from_openaq(args.api_key, args.date_from, args.date_to)
    elif args.source == "all":
        # Try each source in order
        print("Attempting all sources in order of preference...\n")
        
        success = download_from_kaggle()
        if not success:
            print("\n" + "-" * 60 + "\n")
            success = download_from_opencity()
        if not success:
            print("\n" + "-" * 60 + "\n")
            if args.api_key:
                success = download_from_openaq(
                    args.api_key, args.date_from, args.date_to
                )
            else:
                print("Skipping OpenAQ (no API key provided)")
    
    if not success:
        print_manual_instructions()
    else:
        print("\n✓ Data download complete!")
        print(f"  Files saved to: {RAW_DIR.resolve()}")
        print(f"  Next step: python src/data/download_weather.py")
