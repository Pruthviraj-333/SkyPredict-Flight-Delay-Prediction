"""
Predict flight delay using the trained Fallback Model.

Usage:
    python predict.py
    
    Then enter flight details when prompted, or use command-line arguments:
    python predict.py --carrier AA --origin JFK --dest LAX --date 2026-02-22 --dep_time 0800

The fallback model uses only historical flight patterns (no weather data).
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_model_and_config():
    """Load the trained model, encoders, aggregate stats, and config."""
    model_path = os.path.join(MODELS_DIR, "fallback_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, "encoders.pkl")
    stats_path = os.path.join(MODELS_DIR, "aggregate_stats.pkl")
    config_path = os.path.join(MODELS_DIR, "model_config.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoders = pickle.load(f)
    with open(stats_path, 'rb') as f:
        agg_stats = pickle.load(f)
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    return model, encoders, agg_stats, config


def get_carrier_list(encoders):
    """Get list of known carriers."""
    return list(encoders['CARRIER'].classes_)


def get_airport_list(encoders):
    """Get list of known airports."""
    return list(encoders['ORIGIN'].classes_)


def prepare_features(carrier, origin, dest, date_str, dep_time, encoders, agg_stats):
    """
    Prepare feature vector from flight details.
    
    Args:
        carrier: Airline code (e.g., 'AA', 'DL', 'UA')
        origin: Origin airport code (e.g., 'JFK', 'LAX')
        dest: Destination airport code (e.g., 'SFO', 'ORD')
        date_str: Flight date as 'YYYY-MM-DD'
        dep_time: Scheduled departure time as 'HHMM' string (e.g., '0800', '1430')
        encoders: Label encoders dict
        agg_stats: Aggregate statistics dict
    
    Returns:
        DataFrame with feature values
    """
    # Parse date
    date = datetime.strptime(date_str, '%Y-%m-%d')
    month = date.month
    day_of_month = date.day
    day_of_week = date.isoweekday()  # 1=Monday ... 7=Sunday
    is_weekend = 1 if day_of_week in [6, 7] else 0
    
    # Parse departure time
    dep_hour = int(dep_time[:2]) if len(dep_time) >= 2 else 0
    dep_hour = min(max(dep_hour, 0), 23)
    
    # Time block
    time_blocks = [(-1, 5, 0), (5, 9, 1), (9, 13, 2), (13, 17, 3), (17, 21, 4), (21, 24, 5)]
    time_block = 0
    for low, high, block in time_blocks:
        if low < dep_hour <= high:
            time_block = block
            break
    
    # Encode carrier
    carrier = carrier.upper()
    if carrier in encoders['CARRIER'].classes_:
        carrier_encoded = encoders['CARRIER'].transform([carrier])[0]
    else:
        print(f"  [WARN] Unknown carrier '{carrier}', using default encoding")
        carrier_encoded = 0
    
    # Encode origin
    origin = origin.upper()
    if origin in encoders['ORIGIN'].classes_:
        origin_encoded = encoders['ORIGIN'].transform([origin])[0]
    else:
        print(f"  [WARN] Unknown origin '{origin}', using default encoding")
        origin_encoded = 0
    
    # Encode destination
    dest = dest.upper()
    if dest in encoders['DEST'].classes_:
        dest_encoded = encoders['DEST'].transform([dest])[0]
    else:
        print(f"  [WARN] Unknown destination '{dest}', using default encoding")
        dest_encoded = 0
    
    # Get aggregate delay rates (with defaults for unknown values)
    default_delay_rate = 0.203  # overall average from training data
    
    carrier_delay_rate = agg_stats['carrier_delay_rate'].get(carrier, default_delay_rate)
    origin_delay_rate = agg_stats['origin_delay_rate'].get(origin, default_delay_rate)
    dest_delay_rate = agg_stats['dest_delay_rate'].get(dest, default_delay_rate)
    hour_delay_rate = agg_stats['hour_delay_rate'].get(dep_hour, default_delay_rate)
    dow_delay_rate = agg_stats['dow_delay_rate'].get(day_of_week, default_delay_rate)
    
    route = f"{origin}_{dest}"
    route_delay_rate = agg_stats['route_delay_rate'].get(route, default_delay_rate)
    
    # Estimate distance and elapsed time from route stats (or use defaults)
    # For now, use a reasonable default. In production, these would come from a route lookup.
    distance = 0
    elapsed_time = 0
    
    # Try to estimate from training data route averages
    try:
        processed_path = os.path.join(BASE_DIR, "data", "processed", "fallback_dataset.csv")
        route_df = pd.read_csv(processed_path, usecols=['ORIGIN', 'DEST', 'DISTANCE', 'CRS_ELAPSED_TIME'])
        route_match = route_df[(route_df['ORIGIN'] == origin) & (route_df['DEST'] == dest)]
        if len(route_match) > 0:
            distance = route_match['DISTANCE'].mean()
            elapsed_time = route_match['CRS_ELAPSED_TIME'].mean()
    except:
        pass
    
    # Distance group
    if distance <= 250:
        distance_group = 0
    elif distance <= 500:
        distance_group = 1
    elif distance <= 1000:
        distance_group = 2
    elif distance <= 2000:
        distance_group = 3
    else:
        distance_group = 4
    
    # Estimate arrival hour
    arr_hour = (dep_hour + int(elapsed_time / 60)) % 24 if elapsed_time > 0 else (dep_hour + 2) % 24
    
    # Build feature dict (must match FEATURE_COLUMNS order from training)
    features = {
        'MONTH': month,
        'DAY_OF_MONTH': day_of_month,
        'DAY_OF_WEEK': day_of_week,
        'DEP_HOUR': dep_hour,
        'ARR_HOUR': arr_hour,
        'IS_WEEKEND': is_weekend,
        'TIME_BLOCK': time_block,
        'DISTANCE': distance,
        'CRS_ELAPSED_TIME': elapsed_time,
        'DISTANCE_GROUP': distance_group,
        'CARRIER_ENCODED': carrier_encoded,
        'ORIGIN_ENCODED': origin_encoded,
        'DEST_ENCODED': dest_encoded,
        'CARRIER_DELAY_RATE': carrier_delay_rate,
        'ORIGIN_DELAY_RATE': origin_delay_rate,
        'DEST_DELAY_RATE': dest_delay_rate,
        'HOUR_DELAY_RATE': hour_delay_rate,
        'DOW_DELAY_RATE': dow_delay_rate,
        'ROUTE_DELAY_RATE': route_delay_rate,
    }
    
    return pd.DataFrame([features])


def predict_delay(carrier, origin, dest, date_str, dep_time):
    """
    Predict whether a flight will be delayed.
    
    Returns:
        dict with prediction, probability, and details
    """
    model, encoders, agg_stats, config = load_model_and_config()
    
    # Prepare features
    X = prepare_features(carrier, origin, dest, date_str, dep_time, encoders, agg_stats)
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    result = {
        'carrier': carrier.upper(),
        'origin': origin.upper(),
        'dest': dest.upper(),
        'date': date_str,
        'dep_time': dep_time,
        'is_delayed': bool(prediction),
        'delay_probability': float(probability[1]),
        'on_time_probability': float(probability[0]),
    }
    
    return result


def print_prediction(result):
    """Pretty-print prediction result."""
    print("\n" + "=" * 55)
    print("  FLIGHT DELAY PREDICTION (Fallback Model)")
    print("=" * 55)
    print(f"  Carrier:    {result['carrier']}")
    print(f"  Route:      {result['origin']} → {result['dest']}")
    print(f"  Date:       {result['date']}")
    print(f"  Departure:  {result['dep_time']}")
    print("-" * 55)
    
    if result['is_delayed']:
        print(f"  ⚠️  PREDICTION: LIKELY DELAYED")
    else:
        print(f"  ✅ PREDICTION: LIKELY ON-TIME")
    
    print(f"  Delay probability:   {result['delay_probability']*100:.1f}%")
    print(f"  On-time probability: {result['on_time_probability']*100:.1f}%")
    print("=" * 55)
    
    # Risk level
    prob = result['delay_probability']
    if prob < 0.15:
        risk = "LOW RISK"
    elif prob < 0.30:
        risk = "MODERATE RISK"
    elif prob < 0.50:
        risk = "ELEVATED RISK"
    else:
        risk = "HIGH RISK"
    print(f"  Risk Level: {risk}")
    print("=" * 55)


def interactive_mode():
    """Run in interactive mode, asking user for flight details."""
    model, encoders, agg_stats, config = load_model_and_config()
    
    print("\n" + "=" * 55)
    print("  FLIGHT DELAY PREDICTOR — Interactive Mode")
    print("=" * 55)
    print(f"\n  Known carriers: {', '.join(sorted(get_carrier_list(encoders)))}")
    print(f"  Example airports: ATL, LAX, ORD, DFW, DEN, JFK, SFO, SEA, LAS, MCO")
    
    while True:
        print("\n" + "-" * 55)
        carrier = input("  Carrier code (e.g., AA, DL, UA) [or 'quit']: ").strip()
        if carrier.lower() in ['quit', 'exit', 'q']:
            break
        
        origin = input("  Origin airport (e.g., JFK): ").strip()
        dest = input("  Destination airport (e.g., LAX): ").strip()
        date_str = input("  Flight date (YYYY-MM-DD): ").strip()
        dep_time = input("  Scheduled departure time (HHMM, e.g., 0800): ").strip()
        
        try:
            result = predict_delay(carrier, origin, dest, date_str, dep_time)
            print_prediction(result)
        except Exception as e:
            print(f"\n  [ERROR] Failed to predict: {e}")


def main():
    parser = argparse.ArgumentParser(description="Predict flight delay using the Fallback Model")
    parser.add_argument('--carrier', type=str, help='Airline carrier code (e.g., AA, DL, UA)')
    parser.add_argument('--origin', type=str, help='Origin airport code (e.g., JFK)')
    parser.add_argument('--dest', type=str, help='Destination airport code (e.g., LAX)')
    parser.add_argument('--date', type=str, help='Flight date (YYYY-MM-DD)')
    parser.add_argument('--dep_time', type=str, help='Scheduled departure time (HHMM)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or not all([args.carrier, args.origin, args.dest, args.date, args.dep_time]):
        if all([args.carrier, args.origin, args.dest, args.date, args.dep_time]):
            result = predict_delay(args.carrier, args.origin, args.dest, args.date, args.dep_time)
            print_prediction(result)
        else:
            interactive_mode()
    else:
        result = predict_delay(args.carrier, args.origin, args.dest, args.date, args.dep_time)
        print_prediction(result)


if __name__ == "__main__":
    main()
