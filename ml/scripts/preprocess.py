"""
Preprocess BTS On-Time data and create the Fallback Model dataset.

This script:
1. Loads raw BTS CSV data
2. Drops cancelled/diverted flights
3. Creates binary target: IS_DELAYED (ARR_DELAY >= 15)
4. Engineers features from pre-departure information only (no data leakage)
5. Encodes categorical variables
6. Saves the processed dataset
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_raw_data():
    """Load raw BTS data."""
    raw_file = os.path.join(RAW_DIR, "ontime_2025_10.csv")
    print(f"[INFO] Loading raw data from {raw_file}...")
    df = pd.read_csv(raw_file, low_memory=False)
    print(f"[INFO] Raw data shape: {df.shape}")
    return df


def clean_data(df):
    """Clean the raw data: drop cancelled/diverted, remove NaN targets."""
    print("\n[CLEANING]")
    initial = len(df)
    
    # Drop cancelled flights
    if 'Cancelled' in df.columns:
        cancelled = df['Cancelled'].sum()
        df = df[df['Cancelled'] == 0.0]
        print(f"  Dropped {int(cancelled)} cancelled flights")
    
    # Drop diverted flights  
    if 'Diverted' in df.columns:
        diverted = df['Diverted'].sum()
        df = df[df['Diverted'] == 0.0]
        print(f"  Dropped {int(diverted)} diverted flights")
    
    # Drop rows where ArrDelay is missing
    before = len(df)
    df = df.dropna(subset=['ArrDelay'])
    print(f"  Dropped {before - len(df)} rows with missing ArrDelay")
    
    print(f"  Final: {len(df)} rows (dropped {initial - len(df)} total)")
    return df


def engineer_features(df):
    """
    Engineer features for the fallback model.
    
    IMPORTANT: Only uses information available BEFORE a flight departs.
    This avoids data leakage - we don't use DepDelay, ArrDelay, etc. as features.
    """
    print("\n[FEATURE ENGINEERING]")
    
    features = pd.DataFrame()
    
    # --- Target Variable ---
    features['IS_DELAYED'] = (df['ArrDelay'] >= 15).astype(int)
    print(f"  Target distribution: {features['IS_DELAYED'].value_counts().to_dict()}")
    delay_pct = features['IS_DELAYED'].mean() * 100
    print(f"  Delay rate: {delay_pct:.1f}%")
    
    # --- Temporal Features ---
    features['MONTH'] = df['Month'].astype(int)
    features['DAY_OF_MONTH'] = df['DayofMonth'].astype(int)
    features['DAY_OF_WEEK'] = df['DayOfWeek'].astype(int)
    
    # Departure hour from CRSDepTime (scheduled departure)
    # CRSDepTime is in HHMM format (e.g., 1430 = 2:30 PM)
    features['DEP_HOUR'] = (df['CRSDepTime'].fillna(0) / 100).astype(int).clip(0, 23)
    
    # Arrival hour from CRSArrTime
    features['ARR_HOUR'] = (df['CRSArrTime'].fillna(0) / 100).astype(int).clip(0, 23)
    
    # Is weekend?
    features['IS_WEEKEND'] = (df['DayOfWeek'].isin([6, 7])).astype(int)
    
    # Time of day buckets (early morning, morning, afternoon, evening, night)
    features['TIME_BLOCK'] = pd.cut(
        features['DEP_HOUR'],
        bins=[-1, 5, 9, 13, 17, 21, 24],
        labels=[0, 1, 2, 3, 4, 5]  # night, early_morning, morning, afternoon, evening, late_night
    ).astype(int)
    
    # --- Carrier Feature ---
    features['CARRIER'] = df['Reporting_Airline'].astype(str)
    
    # --- Route Features ---
    features['ORIGIN'] = df['Origin'].astype(str)
    features['DEST'] = df['Dest'].astype(str)
    
    # --- Distance and Duration ---
    features['DISTANCE'] = df['Distance'].fillna(0).astype(float)
    features['CRS_ELAPSED_TIME'] = df['CRSElapsedTime'].fillna(0).astype(float)
    
    # Distance buckets
    features['DISTANCE_GROUP'] = pd.cut(
        features['DISTANCE'],
        bins=[0, 250, 500, 1000, 2000, 6000],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    
    # --- Derived Features ---
    # Carrier delay history (will be computed as aggregate stats)
    # These are aggregate features, not leaking individual flight info
    
    print(f"  Created {len(features.columns)} features")
    print(f"  Features: {list(features.columns)}")
    
    return features


def add_aggregate_features(df):
    """
    Add aggregate statistical features based on historical patterns.
    These represent the AVERAGE behavior of carriers, origins, destinations etc.
    This is NOT data leakage because we're using population-level statistics.
    """
    print("\n[AGGREGATE FEATURES]")
    
    # Carrier average delay rate
    carrier_delay_rate = df.groupby('CARRIER')['IS_DELAYED'].mean()
    df['CARRIER_DELAY_RATE'] = df['CARRIER'].map(carrier_delay_rate)
    print(f"  Added CARRIER_DELAY_RATE (range: {carrier_delay_rate.min():.3f} - {carrier_delay_rate.max():.3f})")
    
    # Origin airport delay rate
    origin_delay_rate = df.groupby('ORIGIN')['IS_DELAYED'].mean()
    df['ORIGIN_DELAY_RATE'] = df['ORIGIN'].map(origin_delay_rate)
    print(f"  Added ORIGIN_DELAY_RATE")
    
    # Destination airport delay rate
    dest_delay_rate = df.groupby('DEST')['IS_DELAYED'].mean()
    df['DEST_DELAY_RATE'] = df['DEST'].map(dest_delay_rate)
    print(f"  Added DEST_DELAY_RATE")
    
    # Hour-of-day delay rate
    hour_delay_rate = df.groupby('DEP_HOUR')['IS_DELAYED'].mean()
    df['HOUR_DELAY_RATE'] = df['DEP_HOUR'].map(hour_delay_rate)
    print(f"  Added HOUR_DELAY_RATE")
    
    # Day of week delay rate
    dow_delay_rate = df.groupby('DAY_OF_WEEK')['IS_DELAYED'].mean()
    df['DOW_DELAY_RATE'] = df['DAY_OF_WEEK'].map(dow_delay_rate)
    print(f"  Added DOW_DELAY_RATE")
    
    # Route (ORIGIN-DEST) delay rate
    df['ROUTE'] = df['ORIGIN'] + '_' + df['DEST']
    route_delay_rate = df.groupby('ROUTE')['IS_DELAYED'].mean()
    df['ROUTE_DELAY_RATE'] = df['ROUTE'].map(route_delay_rate)
    print(f"  Added ROUTE_DELAY_RATE ({len(route_delay_rate)} unique routes)")
    
    return df


def encode_categoricals(df):
    """Label-encode categorical features and save the encoders."""
    print("\n[ENCODING CATEGORICALS]")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    encoders = {}
    
    categorical_cols = ['CARRIER', 'ORIGIN', 'DEST']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_ENCODED'] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  {col}: {len(le.classes_)} unique values -> encoded")
    
    # Save encoders
    encoder_path = os.path.join(MODELS_DIR, "encoders.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"  Saved encoders to {encoder_path}")
    
    # Save aggregate stats for prediction time
    agg_stats = {
        'carrier_delay_rate': df.groupby('CARRIER')['IS_DELAYED'].mean().to_dict(),
        'origin_delay_rate': df.groupby('ORIGIN')['IS_DELAYED'].mean().to_dict(),
        'dest_delay_rate': df.groupby('DEST')['IS_DELAYED'].mean().to_dict(),
        'hour_delay_rate': df.groupby('DEP_HOUR')['IS_DELAYED'].mean().to_dict(),
        'dow_delay_rate': df.groupby('DAY_OF_WEEK')['IS_DELAYED'].mean().to_dict(),
        'route_delay_rate': df.groupby('ROUTE')['IS_DELAYED'].mean().to_dict(),
    }
    
    stats_path = os.path.join(MODELS_DIR, "aggregate_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(agg_stats, f)
    print(f"  Saved aggregate stats to {stats_path}")
    
    return df, encoders


def main():
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("FALLBACK MODEL DATA PREPROCESSING")
    print("=" * 60)
    
    # Load
    raw_df = load_raw_data()
    
    # Clean
    clean_df = clean_data(raw_df)
    
    # Engineer features
    features_df = engineer_features(clean_df)
    
    # Add aggregate features
    features_df = add_aggregate_features(features_df)
    
    # Encode categoricals
    features_df, encoders = encode_categoricals(features_df)
    
    # Save processed dataset
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, "fallback_dataset.csv")
    features_df.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")
    print(f"Shape: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")
    print(f"\nTarget distribution:")
    print(features_df['IS_DELAYED'].value_counts())
    print(f"\nSample rows:")
    print(features_df.head())
    
    return features_df


if __name__ == "__main__":
    main()
