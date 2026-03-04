"""
Step 1: Get airport coordinates for weather data fetching.

Uses ourairports.com data (public domain) to map IATA codes to lat/lon.
Only fetches coords for airports that appear in our BTS flight data.
"""

import os
import csv
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT = os.path.join(DATA_DIR, "airport_coordinates.csv")

# URL for OurAirports data (public domain)
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def get_unique_airports():
    """Get all unique airport codes from our BTS data."""
    print("[1/3] Reading BTS data for unique airports...")
    df = pd.read_csv(os.path.join(RAW_DIR, "ontime_2025_10.csv"), usecols=["Origin", "Dest"])
    origins = set(df["Origin"].dropna().unique())
    dests = set(df["Dest"].dropna().unique())
    all_airports = origins | dests
    print(f"       Found {len(all_airports)} unique airports")
    return all_airports


def download_airport_coords(target_airports):
    """Download airport coordinates from OurAirports."""
    print("[2/3] Downloading airport coordinates from OurAirports...")

    resp = requests.get(AIRPORTS_URL, timeout=30)
    resp.raise_for_status()

    # Parse CSV
    lines = resp.text.splitlines()
    reader = csv.DictReader(lines)

    coords = {}
    for row in reader:
        iata = row.get("iata_code", "").strip()
        if iata and iata in target_airports:
            try:
                lat = float(row["latitude_deg"])
                lon = float(row["longitude_deg"])
                name = row.get("name", "")
                coords[iata] = {"iata": iata, "name": name, "latitude": lat, "longitude": lon}
            except (ValueError, KeyError):
                continue

    print(f"       Matched {len(coords)}/{len(target_airports)} airports")
    return coords


def save_coordinates(coords):
    """Save to CSV."""
    print("[3/3] Saving airport coordinates...")
    df = pd.DataFrame(coords.values())
    df = df.sort_values("iata").reset_index(drop=True)
    df.to_csv(OUTPUT, index=False)
    print(f"       Saved to {OUTPUT}")
    print(f"       {len(df)} airports with coordinates")
    return df


def main():
    print("=" * 55)
    print("  AIRPORT COORDINATES LOOKUP")
    print("=" * 55)

    airports = get_unique_airports()
    coords = download_airport_coords(airports)
    df = save_coordinates(coords)

    # Show missing
    missing = airports - set(coords.keys())
    if missing:
        print(f"\n  Missing coordinates for {len(missing)} airports:")
        print(f"  {sorted(missing)}")

    print(f"\n  Done! {len(df)} airports ready for weather fetching.")


if __name__ == "__main__":
    main()
