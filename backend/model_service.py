"""
Model Service — Dual-Model Flight Delay Prediction.

Loads both the primary (weather-enhanced) and fallback (no weather) XGBoost models.
Logic Gate: Automatically uses the primary model when weather data is available,
            and falls back to the base model when weather is unavailable.

v2: Updated feature construction to match enhanced 45-feature fallback model
    and weather-extended primary model.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
from datetime import date as _date, datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from backend import weather_service

# ============================================================
# US HOLIDAYS  (for IS_HOLIDAY / NEAR_HOLIDAY features)
# ============================================================
_US_HOLIDAYS_RAW = {
    (2024, 12, 24), (2024, 12, 25), (2024, 12, 31),
    (2025,  1,  1), (2025,  1, 20), (2025,  2, 17),
    (2025,  3, 14), (2025,  3, 18), (2025,  3, 21),
    (2025,  5, 26), (2025,  7,  3), (2025,  7,  4), (2025,  7,  5),
    (2025,  9,  1), (2025, 10, 13),
    (2025, 11, 11), (2025, 11, 26), (2025, 11, 27), (2025, 11, 28),
    (2025, 11, 29), (2025, 11, 30),
    (2025, 12, 24), (2025, 12, 25), (2025, 12, 31),
    (2026,  1,  1), (2026,  1, 19), (2026,  2, 16),
    (2026,  5, 25), (2026,  7,  3), (2026,  7,  4),
    (2026,  9,  7), (2026, 11, 11), (2026, 11, 26), (2026, 12, 24), (2026, 12, 25),
}
_NEAR_HOLIDAY = set()
for _yy, _mm, _dd in _US_HOLIDAYS_RAW:
    for _delta in range(-3, 4):
        try:
            _dt = _date(_yy, _mm, _dd) + timedelta(days=_delta)
            _NEAR_HOLIDAY.add((_dt.year, _dt.month, _dt.day))
        except Exception:
            pass

_SEASON_MAP = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
               6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


class PredictionResult:
    """Structured prediction result."""
    def __init__(self, carrier, origin, dest, date, dep_time,
                 is_delayed, delay_probability, on_time_probability,
                 risk_level, distance, elapsed_time, model_used="fallback",
                 weather_available=False, predicted_delay_minutes=0.0):
        self.carrier = carrier
        self.origin = origin
        self.dest = dest
        self.date = date
        self.dep_time = dep_time
        self.is_delayed = is_delayed
        self.delay_probability = delay_probability
        self.on_time_probability = on_time_probability
        self.risk_level = risk_level
        self.distance = distance
        self.elapsed_time = elapsed_time
        self.model_used = model_used
        self.weather_available = weather_available
        self.predicted_delay_minutes = predicted_delay_minutes

    def to_dict(self):
        return {
            "carrier": self.carrier,
            "origin": self.origin,
            "dest": self.dest,
            "date": self.date,
            "dep_time": self.dep_time,
            "is_delayed": self.is_delayed,
            "delay_probability": round(self.delay_probability, 4),
            "on_time_probability": round(self.on_time_probability, 4),
            "risk_level": self.risk_level,
            "distance_miles": round(self.distance, 1),
            "flight_duration_minutes": round(self.elapsed_time, 1),
            "model_used": self.model_used,
            "weather_available": self.weather_available,
            "predicted_delay_minutes": round(self.predicted_delay_minutes, 1),
        }


class ModelService:
    """Dual-model service with automatic weather-based switching."""

    def __init__(self, models_dir: str, data_dir: str):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.fallback_model = None
        self.primary_model = None
        self.primary_features = None
        self.encoders = None
        self.agg_stats = None
        self.config = None
        self.route_lookup = None
        self.has_primary = False
        # Regression models
        self.fallback_reg_model = None
        self.fallback_reg_features = []
        self.primary_reg_model = None
        self.primary_reg_features = []
        self._load()

    def _load(self):
        """Load both models, encoders, aggregate stats, and route lookup."""
        # Always load fallback model
        with open(os.path.join(self.models_dir, "fallback_model.pkl"), 'rb') as f:
            self.fallback_model = pickle.load(f)
        with open(os.path.join(self.models_dir, "encoders.pkl"), 'rb') as f:
            self.encoders = pickle.load(f)
        with open(os.path.join(self.models_dir, "aggregate_stats.pkl"), 'rb') as f:
            self.agg_stats = pickle.load(f)
        with open(os.path.join(self.models_dir, "model_config.pkl"), 'rb') as f:
            self.config = pickle.load(f)

        # Try loading primary model
        primary_path = os.path.join(self.models_dir, "primary_model.pkl")
        primary_config_path = os.path.join(self.models_dir, "primary_model_config.pkl")
        if os.path.exists(primary_path) and os.path.exists(primary_config_path):
            try:
                with open(primary_path, 'rb') as f:
                    self.primary_model = pickle.load(f)
                with open(primary_config_path, 'rb') as f:
                    primary_config = pickle.load(f)
                self.primary_features = primary_config.get("features", [])
                self.has_primary = True
                print(f"[INFO] Primary model loaded ({len(self.primary_features)} features)")
            except Exception as e:
                print(f"[WARN] Could not load primary model: {e}")
                self.has_primary = False
        else:
            print("[INFO] No primary model found, using fallback only")

        # Load fallback regressor
        fb_reg_path = os.path.join(self.models_dir, "fallback_reg_model.pkl")
        fb_reg_cfg  = os.path.join(self.models_dir, "fallback_reg_config.pkl")
        if os.path.exists(fb_reg_path) and os.path.exists(fb_reg_cfg):
            try:
                with open(fb_reg_path, 'rb') as f:
                    self.fallback_reg_model = pickle.load(f)
                with open(fb_reg_cfg, 'rb') as f:
                    cfg = pickle.load(f)
                self.fallback_reg_features = cfg.get("features", [])
                print(f"[INFO] Fallback regressor loaded ({len(self.fallback_reg_features)} features)")
            except Exception as e:
                print(f"[WARN] Could not load fallback regressor: {e}")

        # Load primary regressor
        pr_reg_path = os.path.join(self.models_dir, "primary_reg_model.pkl")
        pr_reg_cfg  = os.path.join(self.models_dir, "primary_reg_config.pkl")
        if os.path.exists(pr_reg_path) and os.path.exists(pr_reg_cfg):
            try:
                with open(pr_reg_path, 'rb') as f:
                    self.primary_reg_model = pickle.load(f)
                with open(pr_reg_cfg, 'rb') as f:
                    cfg = pickle.load(f)
                self.primary_reg_features = cfg.get("features", [])
                print(f"[INFO] Primary regressor loaded ({len(self.primary_reg_features)} features)")
            except Exception as e:
                print(f"[WARN] Could not load primary regressor: {e}")

        # Load airport coordinates for weather service
        coords_file = os.path.join(self.data_dir, "airport_coordinates.csv")
        if os.path.exists(coords_file):
            n = weather_service.load_airport_coords(coords_file)
            print(f"[INFO] Weather service: {n} airport coordinates loaded")

        # Build route distance/duration lookup
        processed_path = os.path.join(self.data_dir, "processed", "fallback_dataset.csv")
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path, usecols=['ORIGIN', 'DEST', 'DISTANCE', 'CRS_ELAPSED_TIME'])
            self.route_lookup = df.groupby(['ORIGIN', 'DEST']).agg({
                'DISTANCE': 'mean', 'CRS_ELAPSED_TIME': 'mean'
            }).to_dict('index')
        else:
            self.route_lookup = {}

    # ─── Reference Data ───────────────────────────────────────

    def get_airlines(self):
        return sorted(list(self.encoders['CARRIER'].classes_))

    def get_airports(self):
        return sorted(list(self.encoders['ORIGIN'].classes_))

    def get_airport_names(self):
        """Return mapping of IATA codes to 'City (CODE)' display names."""
        mapping = {
            'ABQ': 'Albuquerque', 'ANC': 'Anchorage', 'ATL': 'Atlanta',
            'AUS': 'Austin', 'BDL': 'Hartford', 'BHM': 'Birmingham',
            'BNA': 'Nashville', 'BOI': 'Boise', 'BOS': 'Boston',
            'BTV': 'Burlington VT', 'BUF': 'Buffalo', 'BUR': 'Burbank',
            'BWI': 'Baltimore', 'CAK': 'Akron', 'CHS': 'Charleston',
            'CLE': 'Cleveland', 'CLT': 'Charlotte', 'CMH': 'Columbus OH',
            'COS': 'Colorado Springs', 'CVG': 'Cincinnati',
            'DAL': 'Dallas Love', 'DAY': 'Dayton', 'DCA': 'Washington Reagan',
            'DEN': 'Denver', 'DFW': 'Dallas/Fort Worth', 'DSM': 'Des Moines',
            'DTW': 'Detroit', 'ELP': 'El Paso', 'EWR': 'Newark',
            'FAT': 'Fresno', 'FLL': 'Fort Lauderdale', 'GEG': 'Spokane',
            'GRR': 'Grand Rapids', 'GSO': 'Greensboro', 'GSP': 'Greenville SC',
            'HNL': 'Honolulu', 'HOU': 'Houston Hobby', 'IAD': 'Washington Dulles',
            'IAH': 'Houston', 'ICT': 'Wichita', 'IND': 'Indianapolis',
            'ISP': 'Long Island', 'JAX': 'Jacksonville', 'JFK': 'New York JFK',
            'LAS': 'Las Vegas', 'LAX': 'Los Angeles', 'LGA': 'New York LaGuardia',
            'LIT': 'Little Rock', 'MCI': 'Kansas City', 'MCO': 'Orlando',
            'MDW': 'Chicago Midway', 'MEM': 'Memphis', 'MHT': 'Manchester NH',
            'MIA': 'Miami', 'MKE': 'Milwaukee', 'MSN': 'Madison',
            'MSP': 'Minneapolis', 'MSY': 'New Orleans', 'OAK': 'Oakland',
            'OGG': 'Maui', 'OKC': 'Oklahoma City', 'OMA': 'Omaha',
            'ONT': 'Ontario CA', 'ORD': 'Chicago O\'Hare', 'ORF': 'Norfolk',
            'PBI': 'West Palm Beach', 'PDX': 'Portland OR', 'PHL': 'Philadelphia',
            'PHX': 'Phoenix', 'PIT': 'Pittsburgh', 'PSP': 'Palm Springs',
            'PVD': 'Providence', 'PWM': 'Portland ME', 'RDU': 'Raleigh-Durham',
            'RIC': 'Richmond', 'RNO': 'Reno', 'ROC': 'Rochester NY',
            'RSW': 'Fort Myers', 'SAN': 'San Diego', 'SAT': 'San Antonio',
            'SAV': 'Savannah', 'SDF': 'Louisville', 'SEA': 'Seattle',
            'SFO': 'San Francisco', 'SJC': 'San Jose', 'SJU': 'San Juan PR',
            'SLC': 'Salt Lake City', 'SMF': 'Sacramento', 'SNA': 'Orange County',
            'SRQ': 'Sarasota', 'STL': 'St. Louis', 'SYR': 'Syracuse',
            'TPA': 'Tampa', 'TUL': 'Tulsa', 'TUS': 'Tucson',
        }
        return {code: f"{mapping.get(code, code)} ({code})" for code in self.get_airports()}

    def get_airline_names(self):
        mapping = {
            'AA': 'American Airlines', 'DL': 'Delta Air Lines',
            'UA': 'United Airlines', 'WN': 'Southwest Airlines',
            'B6': 'JetBlue Airways', 'AS': 'Alaska Airlines',
            'NK': 'Spirit Airlines', 'F9': 'Frontier Airlines',
            'G4': 'Allegiant Air', 'HA': 'Hawaiian Airlines',
            'OO': 'SkyWest Airlines', 'YX': 'Republic Airways',
            'MQ': 'Envoy Air', 'OH': 'PSA Airlines',
            'QX': 'Horizon Air', '9E': 'Endeavor Air',
        }
        return {code: mapping.get(code, code) for code in self.get_airlines()}

    # ─── Analytics ────────────────────────────────────────────

    def get_stats(self):
        carrier_rates = self.agg_stats['carrier_delay_rate']
        origin_rates = self.agg_stats['origin_delay_rate']
        route_rates = self.agg_stats['route_delay_rate']
        hour_rates = self.agg_stats['hour_delay_rate']

        overall_rate = sum(carrier_rates.values()) / len(carrier_rates) if carrier_rates else 0.203
        worst_carrier = max(carrier_rates, key=carrier_rates.get) if carrier_rates else 'N/A'
        worst_route = max(route_rates, key=route_rates.get) if route_rates else 'N/A'
        best_hour = min(hour_rates, key=hour_rates.get) if hour_rates else 6

        return {
            "overall_delay_rate": round(overall_rate, 4),
            "total_airlines": len(carrier_rates),
            "total_airports": len(origin_rates),
            "total_routes": len(route_rates),
            "worst_carrier": worst_carrier,
            "worst_carrier_rate": round(carrier_rates.get(worst_carrier, 0), 4),
            "worst_route": worst_route,
            "worst_route_rate": round(route_rates.get(worst_route, 0), 4),
            "best_departure_hour": best_hour,
            "best_hour_rate": round(hour_rates.get(best_hour, 0), 4),
            "primary_model_available": self.has_primary,
        }

    def get_analytics_trends(self):
        dow_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
                     4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
        dow_rates = self.agg_stats['dow_delay_rate']
        return [
            {"day": dow_names.get(d, str(d)), "day_num": d,
             "delay_rate": round(r * 100, 2)}
            for d, r in sorted(dow_rates.items())
        ]

    def get_analytics_routes(self, top_n=10):
        route_rates = self.agg_stats['route_delay_rate']
        sorted_routes = sorted(route_rates.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {"route": r.replace('_', ' → '), "delay_rate": round(rate * 100, 2)}
            for r, rate in sorted_routes
        ]

    def get_analytics_heatmap(self):
        hour_rates = self.agg_stats['hour_delay_rate']
        dow_rates = self.agg_stats['dow_delay_rate']
        heatmap = []
        for hour in range(24):
            row = {"hour": hour}
            h_rate = hour_rates.get(hour, 0.203)
            for dow in range(1, 8):
                d_rate = dow_rates.get(dow, 0.203)
                combined = (h_rate + d_rate) / 2
                row[f"day_{dow}"] = round(combined * 100, 1)
            heatmap.append(row)
        return heatmap

    def get_analytics_carriers(self):
        carrier_rates = self.agg_stats['carrier_delay_rate']
        names = self.get_airline_names()
        return sorted([
            {"code": c, "name": names.get(c, c), "delay_rate": round(r * 100, 2)}
            for c, r in carrier_rates.items()
        ], key=lambda x: x['delay_rate'], reverse=True)

    def get_analytics_hours(self):
        hour_rates = self.agg_stats['hour_delay_rate']
        return [
            {"hour": h, "label": f"{h:02d}:00", "delay_rate": round(r * 100, 2)}
            for h, r in sorted(hour_rates.items())
        ]

    # ─── Prediction ───────────────────────────────────────────

    def _build_base_features(self, carrier, origin, dest, date_str, dep_time):
        """
        Build the enhanced 45-feature vector for both fallback and primary models (v2).
        Falls back gracefully to 0 for any feature not computable from inputs.
        """
        carrier = carrier.upper().strip()
        origin  = origin.upper().strip()
        dest    = dest.upper().strip()

        dt           = datetime.strptime(date_str, '%Y-%m-%d')
        month        = dt.month
        day_of_month = dt.day
        day_of_week  = dt.isoweekday()   # 1=Mon … 7=Sun
        year         = dt.year
        is_weekend   = 1 if day_of_week in [6, 7] else 0

        dep_hour = int(dep_time[:2]) if len(dep_time) >= 2 else 0
        dep_hour = min(max(dep_hour, 0), 23)

        # Route info (distance / elapsed time)
        route_info   = self.route_lookup.get((origin, dest), {})
        distance     = float(route_info.get('DISTANCE', 0))
        elapsed_time = float(route_info.get('CRS_ELAPSED_TIME', 0))
        arr_hour     = int((dep_hour + elapsed_time / 60)) % 24 if elapsed_time > 0 else (dep_hour + 2) % 24

        # --- TIME_BLOCK (bins: -1,5,9,13,17,21,24 → labels 0-5) ---
        if   dep_hour <= 5:  time_block = 0
        elif dep_hour <= 9:  time_block = 1
        elif dep_hour <= 13: time_block = 2
        elif dep_hour <= 17: time_block = 3
        elif dep_hour <= 21: time_block = 4
        else:                time_block = 5

        # --- DISTANCE GROUP ---
        if   distance <= 250:  distance_group = 0
        elif distance <= 500:  distance_group = 1
        elif distance <= 1000: distance_group = 2
        elif distance <= 2000: distance_group = 3
        else:                  distance_group = 4

        # --- DURATION BUCKET ---
        if   elapsed_time <= 60:  dur_bucket = 0
        elif elapsed_time <= 120: dur_bucket = 1
        elif elapsed_time <= 180: dur_bucket = 2
        elif elapsed_time <= 300: dur_bucket = 3
        else:                     dur_bucket = 4

        # --- SPEED PROXY ---
        speed_proxy = distance / elapsed_time if elapsed_time > 0 else 0.0

        # --- CYCLICAL ENCODINGS ---
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
        hour_sin  = math.sin(2 * math.pi * dep_hour / 24)
        hour_cos  = math.cos(2 * math.pi * dep_hour / 24)
        dow_sin   = math.sin(2 * math.pi * day_of_week / 7)
        dow_cos   = math.cos(2 * math.pi * day_of_week / 7)
        dom_sin   = math.sin(2 * math.pi * day_of_month / 31)
        dom_cos   = math.cos(2 * math.pi * day_of_month / 31)

        # --- SEASON ---
        season = _SEASON_MAP.get(month, 0)

        # --- HOLIDAY FLAGS ---
        is_holiday   = 1 if (year, month, day_of_month) in _US_HOLIDAYS_RAW else 0
        near_holiday = 1 if (year, month, day_of_month) in _NEAR_HOLIDAY else 0

        # --- PEAK TRAVEL INDICATORS ---
        is_friday_evening  = 1 if (day_of_week == 5 and dep_hour >= 15) else 0
        is_sunday_evening  = 1 if (day_of_week == 7 and dep_hour >= 15) else 0
        is_monday_morning  = 1 if (day_of_week == 1 and dep_hour <= 9)  else 0
        is_peak_hour       = 1 if dep_hour in [7, 8, 16, 17, 18] else 0
        is_early_morning   = 1 if dep_hour <= 6 else 0
        is_red_eye         = 1 if dep_hour >= 22 else 0

        # --- ENCODE CATEGORICALS ---
        carrier_enc = int(self.encoders['CARRIER'].transform([carrier])[0]) if carrier in self.encoders['CARRIER'].classes_ else 0
        origin_enc  = int(self.encoders['ORIGIN'].transform([origin])[0])   if origin  in self.encoders['ORIGIN'].classes_  else 0
        dest_enc    = int(self.encoders['DEST'].transform([dest])[0])        if dest    in self.encoders['DEST'].classes_    else 0

        # --- AGGREGATE DELAY RATES ---
        dr = 0.203  # global default
        agg = self.agg_stats
        carrier_dr  = agg.get('carrier_delay_rate', {}).get(carrier, dr)
        origin_dr   = agg.get('origin_delay_rate',  {}).get(origin,  dr)
        dest_dr     = agg.get('dest_delay_rate',    {}).get(dest,    dr)
        hour_dr     = agg.get('hour_delay_rate',    {}).get(dep_hour, dr)
        dow_dr      = agg.get('dow_delay_rate',     {}).get(day_of_week, dr)
        season_dr   = agg.get('season_delay_rate',  {}).get(season,  dr)
        tb_dr       = agg.get('time_block_delay_rate', {}).get(time_block, dr)
        route_key   = f"{origin}_{dest}"
        route_dr    = agg.get('route_delay_rate',   {}).get(route_key, dr)

        # Level-2 interaction rates
        co_key   = f"{carrier}_{origin}"
        ch_key   = f"{carrier}_{dep_hour}"
        cd_key   = f"{carrier}_{day_of_week}"
        od_key   = f"{origin}_{day_of_week}"
        oh_key   = f"{origin}_{dep_hour}"
        rh_key   = f"{route_key}_{dep_hour}"
        dh_key   = f"{dest}_{arr_hour}"

        co_dr    = agg.get('carrier_origin_delay_rate', {}).get(co_key, dr)
        ch_dr    = agg.get('carrier_hour_delay_rate',   {}).get(ch_key, dr)
        cd_dr    = agg.get('carrier_dow_delay_rate',    {}).get(cd_key, dr)
        od_dr    = agg.get('origin_dow_delay_rate',     {}).get(od_key, dr)
        oh_dr    = agg.get('origin_hour_delay_rate',    {}).get(oh_key, dr)
        rh_dr    = agg.get('route_hour_delay_rate',     {}).get(rh_key, dr)
        dh_dr    = agg.get('dest_hour_delay_rate',      {}).get(dh_key, dr)

        # --- CONGESTION PROXIES ---
        oh_counts  = agg.get('origin_hour_congestion', {})
        dh_counts  = agg.get('dest_hour_congestion',   {})
        max_oh     = float(agg.get('congestion_max_origin', 1) or 1)
        max_dh     = float(agg.get('congestion_max_dest',   1) or 1)
        origin_cong = oh_counts.get(f"{origin}_{dep_hour}", 0) / max_oh
        dest_cong   = dh_counts.get(f"{dest}_{arr_hour}",   0) / max_dh

        # Default tail utilization (unknown at inference)
        tail_today = 3  # median proxy

        # ─── Full 45-feature fallback vector ───
        fallback_feats = {
            # Base temporal
            'MONTH': month, 'DAY_OF_MONTH': day_of_month, 'DAY_OF_WEEK': day_of_week,
            'DEP_HOUR': dep_hour, 'ARR_HOUR': arr_hour, 'IS_WEEKEND': is_weekend,
            'TIME_BLOCK': time_block,
            # Distance / duration
            'DISTANCE': distance, 'CRS_ELAPSED_TIME': elapsed_time,
            'DISTANCE_GROUP': distance_group, 'DURATION_BUCKET': dur_bucket,
            'SPEED_PROXY': speed_proxy,
            # Encoded categoricals
            'CARRIER_ENCODED': carrier_enc, 'ORIGIN_ENCODED': origin_enc, 'DEST_ENCODED': dest_enc,
            # Cyclical
            'MONTH_SIN': month_sin, 'MONTH_COS': month_cos,
            'HOUR_SIN': hour_sin,   'HOUR_COS': hour_cos,
            'DOW_SIN': dow_sin,     'DOW_COS': dow_cos,
            'DOM_SIN': dom_sin,     'DOM_COS': dom_cos,
            # Season / holiday
            'SEASON': season, 'IS_HOLIDAY': is_holiday, 'NEAR_HOLIDAY': near_holiday,
            # Peak travel
            'IS_FRIDAY_EVENING': is_friday_evening, 'IS_SUNDAY_EVENING': is_sunday_evening,
            'IS_MONDAY_MORNING': is_monday_morning, 'IS_PEAK_HOUR': is_peak_hour,
            'IS_EARLY_MORNING': is_early_morning, 'IS_RED_EYE': is_red_eye,
            # Congestion
            'ORIGIN_CONGESTION': origin_cong, 'DEST_CONGESTION': dest_cong,
            # Tail utilization
            'TAIL_FLIGHTS_TODAY': tail_today,
            # Level-1 aggregate rates
            'CARRIER_DELAY_RATE': carrier_dr, 'ORIGIN_DELAY_RATE': origin_dr,
            'DEST_DELAY_RATE': dest_dr, 'HOUR_DELAY_RATE': hour_dr,
            'DOW_DELAY_RATE': dow_dr, 'ROUTE_DELAY_RATE': route_dr,
            'SEASON_DELAY_RATE': season_dr, 'TIME_BLOCK_DELAY_RATE': tb_dr,
            # Level-2 interaction rates
            'CARRIER_ORIGIN_DELAY_RATE': co_dr, 'CARRIER_HOUR_DELAY_RATE': ch_dr,
            'CARRIER_DOW_DELAY_RATE': cd_dr, 'ORIGIN_DOW_DELAY_RATE': od_dr,
            'ORIGIN_HOUR_DELAY_RATE': oh_dr, 'ROUTE_HOUR_DELAY_RATE': rh_dr,
            'DEST_HOUR_DELAY_RATE': dh_dr,
        }

        # Legacy base_features dict for primary model (weather path)
        base_feats = {
            'MONTH': month, 'DAY_OF_WEEK': day_of_week,
            'DAY_OF_MONTH': day_of_month, 'DEP_HOUR': dep_hour,
            'IS_WEEKEND': is_weekend, 'TIME_BLOCK': time_block,
            'DISTANCE': distance, 'DISTANCE_GROUP': distance_group,
            'FLIGHT_DURATION': elapsed_time,
            'CARRIER_ENC': carrier_enc, 'ORIGIN_ENC': origin_enc,
            'DEST_ENC': dest_enc, 'CARRIER_DELAY_RATE': carrier_dr,
            'ORIGIN_DELAY_RATE': origin_dr, 'DEST_DELAY_RATE': dest_dr,
            'ROUTE_DELAY_RATE': route_dr, 'HOUR_DELAY_RATE': hour_dr,
            'DOW_DELAY_RATE': dow_dr,
            # Also pass new features so primary model can use them if needed
            **{k: v for k, v in fallback_feats.items()}
        }

        return {
            "base_features":     base_feats,
            "fallback_features": fallback_feats,
            "carrier": carrier, "origin": origin, "dest": dest,
            "dep_hour": dep_hour, "distance": distance,
            "elapsed_time": elapsed_time,
        }

    def predict(self, carrier: str, origin: str, dest: str,
                date_str: str, dep_time: str) -> PredictionResult:
        """
        Predict delay using the Logic Gate:
        1. Try to get weather data for origin + destination
        2. If weather available AND primary model loaded → use primary model
        3. Otherwise → use fallback model
        Also runs the regression model to predict exact delay minutes.
        """
        info = self._build_base_features(carrier, origin, dest, date_str, dep_time)
        c, o, d = info["carrier"], info["origin"], info["dest"]
        dep_hour = info["dep_hour"]
        distance = info["distance"]
        elapsed_time = info["elapsed_time"]

        model_used = "fallback"
        weather_available = False
        wx = None  # captured at outer scope for regression path

        # ─── Logic Gate: Try primary model first ───
        if self.has_primary:
            wx = weather_service.get_flight_weather(
                origin=o, dest=d, date_str=date_str,
                dep_hour=dep_hour,
                flight_duration_min=elapsed_time if elapsed_time > 0 else 120,
            )

            if wx is not None:
                weather_available = True
                # Build primary feature vector
                primary_feats = {**info["base_features"], **wx}
                features = pd.DataFrame([primary_feats])

                # Ensure correct column order
                features = features.reindex(columns=self.primary_features, fill_value=0)

                prediction = self.primary_model.predict(features)[0]
                probability = self.primary_model.predict_proba(features)[0]
                model_used = "primary"

        # ─── Fallback: No weather → use fallback model ───
        if model_used == "fallback":
            fallback_feats = info["fallback_features"]
            # Use the trained model's feature column list if available
            fallback_cols = self.config.get('feature_columns', list(fallback_feats.keys()))
            features = pd.DataFrame([fallback_feats]).reindex(columns=fallback_cols, fill_value=0)
            prediction = self.fallback_model.predict(features)[0]
            probability = self.fallback_model.predict_proba(features)[0]

        delay_prob = float(probability[1])

        if delay_prob < 0.15: risk = "LOW"
        elif delay_prob < 0.30: risk = "MODERATE"
        elif delay_prob < 0.50: risk = "ELEVATED"
        else: risk = "HIGH"

        # ─── Regression: Predict exact delay minutes ───
        predicted_delay_min = 0.0
        try:
            if model_used == "primary" and self.primary_reg_model is not None and wx is not None:
                reg_feats = pd.DataFrame([{**info["base_features"], **wx}])
                reg_feats = reg_feats.reindex(columns=self.primary_reg_features, fill_value=0)
                predicted_delay_min = float(self.primary_reg_model.predict(reg_feats)[0])
            elif self.fallback_reg_model is not None:
                fb_reg_feats = pd.DataFrame([info["fallback_features"]])
                fb_reg_feats = fb_reg_feats.reindex(columns=self.fallback_reg_features, fill_value=0)
                predicted_delay_min = float(self.fallback_reg_model.predict(fb_reg_feats)[0])
            predicted_delay_min = max(0.0, predicted_delay_min)  # clip negatives
        except Exception as e:
            print(f"[WARN] Regression prediction failed: {e}")
            predicted_delay_min = 0.0

        return PredictionResult(
            carrier=c, origin=o, dest=d, date=date_str, dep_time=dep_time,
            is_delayed=bool(prediction), delay_probability=delay_prob,
            on_time_probability=float(probability[0]),
            risk_level=risk, distance=distance, elapsed_time=elapsed_time,
            model_used=model_used, weather_available=weather_available,
            predicted_delay_minutes=predicted_delay_min,
        )

    def get_airport_map_data(self, top_routes=10):
        """Return airport coordinates + delay rates and top delayed routes for map visualization."""
        coords_file = os.path.join(self.data_dir, "airport_coordinates.csv")
        if not os.path.exists(coords_file):
            return {"airports": [], "routes": []}

        coords_df = pd.read_csv(coords_file)
        origin_rates = self.agg_stats.get('origin_delay_rate', {})
        dest_rates = self.agg_stats.get('dest_delay_rate', {})
        known_airports = set(self.get_airports())

        airports = []
        coord_lookup = {}
        for _, row in coords_df.iterrows():
            iata = row['iata']
            if iata not in known_airports:
                continue
            lat, lon = float(row['latitude']), float(row['longitude'])
            # Filter to AlbersUsa-supported range (CONUS + Alaska + Hawaii)
            # Excludes territories like Guam, American Samoa that crash the projection
            if not (17.0 <= lat <= 72.0 and -180.0 <= lon <= -65.0):
                continue
            coord_lookup[iata] = (lat, lon)
            avg_rate = (origin_rates.get(iata, 0.203) + dest_rates.get(iata, 0.203)) / 2
            airports.append({
                "iata": iata,
                "name": row.get('name', iata),
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "delay_rate": round(avg_rate * 100, 2),
            })

        # Top delayed routes with coordinates
        route_rates = self.agg_stats.get('route_delay_rate', {})
        sorted_routes = sorted(route_rates.items(), key=lambda x: x[1], reverse=True)
        routes = []
        for route_key, rate in sorted_routes:
            if len(routes) >= top_routes:
                break
            parts = route_key.split('_')
            if len(parts) != 2:
                continue
            orig, dest = parts
            if orig in coord_lookup and dest in coord_lookup:
                routes.append({
                    "origin": orig,
                    "dest": dest,
                    "origin_lat": coord_lookup[orig][0],
                    "origin_lon": coord_lookup[orig][1],
                    "dest_lat": coord_lookup[dest][0],
                    "dest_lon": coord_lookup[dest][1],
                    "delay_rate": round(rate * 100, 2),
                })

        return {"airports": airports, "routes": routes}

    def batch_predict(self, flights: list) -> list:
        return [
            self.predict(f['carrier'], f['origin'], f['dest'], f['date'], f['dep_time']).to_dict()
            for f in flights
        ]
