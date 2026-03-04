"""
Step 2: Fetch historical weather data from Open-Meteo for all airports.

Uses the free Open-Meteo Archive API (no API key needed).
Fetches hourly weather for October 2025 at each airport.
"""

import os
import time
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
COORDS_FILE = os.path.join(DATA_DIR, "airport_coordinates.csv")
WEATHER_DIR = os.path.join(DATA_DIR, "weather")
OUTPUT = os.path.join(WEATHER_DIR, "airport_weather_oct2025.csv")

# Open-Meteo Archive API
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Weather variables to fetch (hourly)
HOURLY_VARS = [
    "temperature_2m",        # °C
    "wind_speed_10m",        # km/h
    "wind_gusts_10m",        # km/h
    "precipitation",         # mm
    "rain",                  # mm
    "snowfall",              # cm
    "cloud_cover",           # %
    "visibility",            # meters
    "weather_code",          # WMO code (fog, rain, snow, etc.)
]


def fetch_weather_for_airport(iata, lat, lon, start_date, end_date):
    """Fetch hourly weather for one airport over a date range."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }

    try:
        resp = requests.get(API_URL, params=params, timeout=30)
        if resp.status_code != 200:
            return None

        data = resp.json()
        hourly = data.get("hourly", {})

        if "time" not in hourly:
            return None

        # Build DataFrame
        df = pd.DataFrame({"datetime": hourly["time"]})
        for var in HOURLY_VARS:
            df[var] = hourly.get(var, [None] * len(df))

        df["airport"] = iata
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour

        return df

    except Exception as e:
        print(f"    Error for {iata}: {e}")
        return None


def main():
    print("=" * 55)
    print("  FETCH HISTORICAL WEATHER (Open-Meteo)")
    print("=" * 55)

    os.makedirs(WEATHER_DIR, exist_ok=True)

    # Load airport coordinates
    airports = pd.read_csv(COORDS_FILE)
    total = len(airports)
    print(f"\n  Airports to fetch: {total}")
    print(f"  Date range: 2025-10-01 to 2025-10-31")
    print(f"  Variables: {len(HOURLY_VARS)} ({', '.join(HOURLY_VARS[:4])}...)")
    print()

    all_weather = []
    success = 0
    failed = 0

    for i, row in airports.iterrows():
        iata = row["iata"]
        lat = row["latitude"]
        lon = row["longitude"]

        pct = (i + 1) / total * 100
        print(f"  [{i+1:3d}/{total}] {iata:4s} ({lat:7.2f}, {lon:8.2f}) ...", end=" ", flush=True)

        df = fetch_weather_for_airport(iata, lat, lon, "2025-10-01", "2025-10-31")

        if df is not None and len(df) > 0:
            all_weather.append(df)
            success += 1
            print(f"OK ({len(df)} hours)")
        else:
            failed += 1
            print("FAILED")

        # Rate limit: Open-Meteo allows ~600 req/min for free
        # Be polite with a small delay
        if (i + 1) % 10 == 0:
            time.sleep(1)

    # Combine and save
    if all_weather:
        combined = pd.concat(all_weather, ignore_index=True)
        combined.to_csv(OUTPUT, index=False)
        print(f"\n  {'=' * 45}")
        print(f"  Saved: {OUTPUT}")
        print(f"  Total rows: {len(combined):,}")
        print(f"  Airports: {success} success, {failed} failed")
        print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"  Columns: {list(combined.columns)}")
    else:
        print("\n  ERROR: No weather data fetched!")


if __name__ == "__main__":
    main()
