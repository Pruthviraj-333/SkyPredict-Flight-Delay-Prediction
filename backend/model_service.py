"""
Model Service — Wraps the trained XGBoost fallback model for API use.

Loads model, encoders, and aggregate stats once at initialization.
Provides predict() and batch_predict() methods.
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class PredictionResult:
    """Structured prediction result."""
    def __init__(self, carrier, origin, dest, date, dep_time,
                 is_delayed, delay_probability, on_time_probability,
                 risk_level, distance, elapsed_time):
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
        }


class ModelService:
    """Service that loads and serves the fallback ML model."""

    def __init__(self, models_dir: str, data_dir: str):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.model = None
        self.encoders = None
        self.agg_stats = None
        self.config = None
        self.route_lookup = None
        self._load()

    def _load(self):
        """Load model, encoders, aggregate stats, and route lookup."""
        with open(os.path.join(self.models_dir, "fallback_model.pkl"), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(self.models_dir, "encoders.pkl"), 'rb') as f:
            self.encoders = pickle.load(f)
        with open(os.path.join(self.models_dir, "aggregate_stats.pkl"), 'rb') as f:
            self.agg_stats = pickle.load(f)
        with open(os.path.join(self.models_dir, "model_config.pkl"), 'rb') as f:
            self.config = pickle.load(f)

        # Build route distance/duration lookup
        processed_path = os.path.join(self.data_dir, "processed", "fallback_dataset.csv")
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path, usecols=['ORIGIN', 'DEST', 'DISTANCE', 'CRS_ELAPSED_TIME'])
            self.route_lookup = df.groupby(['ORIGIN', 'DEST']).agg({
                'DISTANCE': 'mean',
                'CRS_ELAPSED_TIME': 'mean'
            }).to_dict('index')
        else:
            self.route_lookup = {}

    def get_airlines(self):
        """Return list of known airline codes."""
        return sorted(list(self.encoders['CARRIER'].classes_))

    def get_airports(self):
        """Return list of known airport codes."""
        return sorted(list(self.encoders['ORIGIN'].classes_))

    def get_airline_names(self):
        """Return mapping of airline codes to names."""
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

    def get_stats(self):
        """Return aggregate statistics from training data."""
        carrier_rates = self.agg_stats['carrier_delay_rate']
        origin_rates = self.agg_stats['origin_delay_rate']
        route_rates = self.agg_stats['route_delay_rate']
        hour_rates = self.agg_stats['hour_delay_rate']

        # Overall delay rate
        overall_rate = sum(carrier_rates.values()) / len(carrier_rates) if carrier_rates else 0.203

        # Worst carrier
        worst_carrier = max(carrier_rates, key=carrier_rates.get) if carrier_rates else 'N/A'

        # Worst route
        worst_route = max(route_rates, key=route_rates.get) if route_rates else 'N/A'

        # Best time to fly
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
        }

    def get_analytics_trends(self):
        """Delay rate by day of week."""
        dow_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
                     4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
        dow_rates = self.agg_stats['dow_delay_rate']
        return [
            {"day": dow_names.get(d, str(d)), "day_num": d,
             "delay_rate": round(r * 100, 2)}
            for d, r in sorted(dow_rates.items())
        ]

    def get_analytics_routes(self, top_n=10):
        """Top N most delayed routes."""
        route_rates = self.agg_stats['route_delay_rate']
        sorted_routes = sorted(route_rates.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {"route": r.replace('_', ' → '), "delay_rate": round(rate * 100, 2)}
            for r, rate in sorted_routes
        ]

    def get_analytics_heatmap(self):
        """Delay rates by hour for each day of week."""
        hour_rates = self.agg_stats['hour_delay_rate']
        dow_rates = self.agg_stats['dow_delay_rate']

        heatmap = []
        for hour in range(24):
            row = {"hour": hour}
            h_rate = hour_rates.get(hour, 0.203)
            for dow in range(1, 8):
                d_rate = dow_rates.get(dow, 0.203)
                # Combined estimate
                combined = (h_rate + d_rate) / 2
                row[f"day_{dow}"] = round(combined * 100, 1)
            heatmap.append(row)
        return heatmap

    def get_analytics_carriers(self):
        """Delay rates by carrier."""
        carrier_rates = self.agg_stats['carrier_delay_rate']
        names = self.get_airline_names()
        return sorted([
            {"code": c, "name": names.get(c, c), "delay_rate": round(r * 100, 2)}
            for c, r in carrier_rates.items()
        ], key=lambda x: x['delay_rate'], reverse=True)

    def get_analytics_hours(self):
        """Delay rates by hour of day."""
        hour_rates = self.agg_stats['hour_delay_rate']
        return [
            {"hour": h, "label": f"{h:02d}:00", "delay_rate": round(r * 100, 2)}
            for h, r in sorted(hour_rates.items())
        ]

    def predict(self, carrier: str, origin: str, dest: str,
                date_str: str, dep_time: str) -> PredictionResult:
        """Predict delay for a single flight."""
        carrier = carrier.upper().strip()
        origin = origin.upper().strip()
        dest = dest.upper().strip()

        # Parse date
        date = datetime.strptime(date_str, '%Y-%m-%d')
        month = date.month
        day_of_month = date.day
        day_of_week = date.isoweekday()
        is_weekend = 1 if day_of_week in [6, 7] else 0

        # Parse departure time
        dep_hour = int(dep_time[:2]) if len(dep_time) >= 2 else 0
        dep_hour = min(max(dep_hour, 0), 23)

        # Time block
        if dep_hour <= 5: time_block = 0
        elif dep_hour <= 9: time_block = 1
        elif dep_hour <= 13: time_block = 2
        elif dep_hour <= 17: time_block = 3
        elif dep_hour <= 21: time_block = 4
        else: time_block = 5

        # Encode categoricals
        carrier_enc = self.encoders['CARRIER'].transform([carrier])[0] if carrier in self.encoders['CARRIER'].classes_ else 0
        origin_enc = self.encoders['ORIGIN'].transform([origin])[0] if origin in self.encoders['ORIGIN'].classes_ else 0
        dest_enc = self.encoders['DEST'].transform([dest])[0] if dest in self.encoders['DEST'].classes_ else 0

        # Aggregate delay rates
        default_rate = 0.203
        carrier_delay_rate = self.agg_stats['carrier_delay_rate'].get(carrier, default_rate)
        origin_delay_rate = self.agg_stats['origin_delay_rate'].get(origin, default_rate)
        dest_delay_rate = self.agg_stats['dest_delay_rate'].get(dest, default_rate)
        hour_delay_rate = self.agg_stats['hour_delay_rate'].get(dep_hour, default_rate)
        dow_delay_rate = self.agg_stats['dow_delay_rate'].get(day_of_week, default_rate)
        route_key = f"{origin}_{dest}"
        route_delay_rate = self.agg_stats['route_delay_rate'].get(route_key, default_rate)

        # Route distance and duration
        route_info = self.route_lookup.get((origin, dest), {})
        distance = route_info.get('DISTANCE', 0)
        elapsed_time = route_info.get('CRS_ELAPSED_TIME', 0)

        # Distance group
        if distance <= 250: distance_group = 0
        elif distance <= 500: distance_group = 1
        elif distance <= 1000: distance_group = 2
        elif distance <= 2000: distance_group = 3
        else: distance_group = 4

        # Arrival hour estimate
        arr_hour = (dep_hour + int(elapsed_time / 60)) % 24 if elapsed_time > 0 else (dep_hour + 2) % 24

        # Feature vector
        features = pd.DataFrame([{
            'MONTH': month, 'DAY_OF_MONTH': day_of_month,
            'DAY_OF_WEEK': day_of_week, 'DEP_HOUR': dep_hour,
            'ARR_HOUR': arr_hour, 'IS_WEEKEND': is_weekend,
            'TIME_BLOCK': time_block, 'DISTANCE': distance,
            'CRS_ELAPSED_TIME': elapsed_time, 'DISTANCE_GROUP': distance_group,
            'CARRIER_ENCODED': carrier_enc, 'ORIGIN_ENCODED': origin_enc,
            'DEST_ENCODED': dest_enc, 'CARRIER_DELAY_RATE': carrier_delay_rate,
            'ORIGIN_DELAY_RATE': origin_delay_rate, 'DEST_DELAY_RATE': dest_delay_rate,
            'HOUR_DELAY_RATE': hour_delay_rate, 'DOW_DELAY_RATE': dow_delay_rate,
            'ROUTE_DELAY_RATE': route_delay_rate,
        }])

        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        delay_prob = float(probability[1])

        # Risk level
        if delay_prob < 0.15: risk = "LOW"
        elif delay_prob < 0.30: risk = "MODERATE"
        elif delay_prob < 0.50: risk = "ELEVATED"
        else: risk = "HIGH"

        return PredictionResult(
            carrier=carrier, origin=origin, dest=dest,
            date=date_str, dep_time=dep_time,
            is_delayed=bool(prediction),
            delay_probability=delay_prob,
            on_time_probability=float(probability[0]),
            risk_level=risk, distance=distance,
            elapsed_time=elapsed_time,
        )

    def batch_predict(self, flights: list) -> list:
        """Predict delays for multiple flights."""
        return [
            self.predict(
                f['carrier'], f['origin'], f['dest'],
                f['date'], f['dep_time']
            ).to_dict()
            for f in flights
        ]
