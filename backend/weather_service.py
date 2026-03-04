"""
Weather Service — Real-time weather data from Open-Meteo Forecast API.

Used at prediction time to fetch current/forecast weather for airports,
enabling the primary model's weather features.
Free API, no key needed.
"""

import requests
import pandas as pd
from typing import Optional, Dict
from datetime import datetime, timedelta

# Open-Meteo Forecast API (free, no key)
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Airport coordinates cache loaded from CSV
_airport_coords: Dict[str, tuple] = {}


def load_airport_coords(coords_file: str):
    """Load airport lat/lon from CSV into memory."""
    global _airport_coords
    try:
        df = pd.read_csv(coords_file)
        _airport_coords = {
            row["iata"]: (row["latitude"], row["longitude"])
            for _, row in df.iterrows()
        }
        return len(_airport_coords)
    except Exception as e:
        print(f"[WARN] Could not load airport coordinates: {e}")
        return 0


def get_weather_for_airport(airport: str, target_date: str, target_hour: int) -> Optional[dict]:
    """
    Fetch weather for an airport at a specific date/hour.

    Args:
        airport: IATA code (e.g., JFK, LAX)
        target_date: Date string YYYY-MM-DD
        target_hour: Hour 0-23

    Returns:
        Dict with weather features or None if unavailable.
    """
    if airport not in _airport_coords:
        return None

    lat, lon = _airport_coords[airport]

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m,precipitation,visibility,cloud_cover,weather_code",
            "start_date": target_date,
            "end_date": target_date,
            "timezone": "UTC",
        }

        resp = requests.get(FORECAST_URL, params=params, timeout=5)
        if resp.status_code != 200:
            return None

        data = resp.json()
        hourly = data.get("hourly", {})

        if "time" not in hourly or target_hour >= len(hourly["time"]):
            return None

        return {
            "temperature_2m": hourly.get("temperature_2m", [None])[target_hour],
            "wind_speed_10m": hourly.get("wind_speed_10m", [None])[target_hour],
            "precipitation": hourly.get("precipitation", [None])[target_hour],
            "visibility": hourly.get("visibility", [None])[target_hour],
            "cloud_cover": hourly.get("cloud_cover", [None])[target_hour],
            "weather_code": hourly.get("weather_code", [None])[target_hour],
        }

    except requests.exceptions.Timeout:
        print(f"[WARN] Weather API timeout for {airport} — falling back to base model")
        return None
    except requests.exceptions.ConnectionError:
        print(f"[WARN] Weather API unreachable for {airport} — falling back to base model")
        return None
    except Exception:
        return None


def get_flight_weather(origin: str, dest: str, date_str: str,
                       dep_hour: int, flight_duration_min: float = 120) -> Optional[dict]:
    """
    Fetch weather for both origin and destination airports.

    Returns dict with all 14 weather features needed by the primary model,
    or None if weather data is unavailable for either airport.
    """
    # Estimate arrival hour
    arr_hour = int((dep_hour + flight_duration_min / 60) % 24)

    origin_wx = get_weather_for_airport(origin, date_str, dep_hour)
    dest_wx = get_weather_for_airport(dest, date_str, arr_hour)

    if origin_wx is None or dest_wx is None:
        return None

    # Check for None values in critical fields
    if origin_wx.get("temperature_2m") is None or dest_wx.get("temperature_2m") is None:
        return None

    # Build the 14 weather features matching the primary model
    result = {
        "origin_temp": origin_wx["temperature_2m"] or 15.0,
        "origin_wind": origin_wx["wind_speed_10m"] or 10.0,
        "origin_precip": origin_wx["precipitation"] or 0.0,
        "origin_visibility": origin_wx["visibility"] or 24000.0,
        "origin_clouds": origin_wx["cloud_cover"] or 50.0,
        "origin_wx_code": origin_wx["weather_code"] or 0,
        "dest_temp": dest_wx["temperature_2m"] or 15.0,
        "dest_wind": dest_wx["wind_speed_10m"] or 10.0,
        "dest_precip": dest_wx["precipitation"] or 0.0,
        "dest_visibility": dest_wx["visibility"] or 24000.0,
        "dest_clouds": dest_wx["cloud_cover"] or 50.0,
        "dest_wx_code": dest_wx["weather_code"] or 0,
    }

    # Derived features
    result["bad_wx_origin"] = int(
        (result["origin_precip"] > 0.5) or
        (result["origin_visibility"] < 5000) or
        (result["origin_wind"] > 40)
    )
    result["bad_wx_dest"] = int(
        (result["dest_precip"] > 0.5) or
        (result["dest_visibility"] < 5000) or
        (result["dest_wind"] > 40)
    )

    return result
