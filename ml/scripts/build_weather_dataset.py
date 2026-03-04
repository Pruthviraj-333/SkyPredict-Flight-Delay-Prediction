"""
Step 3: Merge BTS flight data with weather data + train primary model.

Combines the existing fallback features with 8 new weather features:
  Origin: temperature, wind_speed, precipitation, visibility
  Dest:   temperature, wind_speed, precipitation, visibility

Then trains a new XGBoost model and compares against the fallback.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

WEATHER_FILE = os.path.join(DATA_DIR, "weather", "airport_weather_oct2025.csv")
RAW_FILE = os.path.join(DATA_DIR, "raw", "ontime_2025_10.csv")
FALLBACK_FILE = os.path.join(DATA_DIR, "processed", "fallback_dataset.csv")
OUTPUT_DATASET = os.path.join(DATA_DIR, "processed", "weather_dataset.csv")
OUTPUT_MODEL = os.path.join(MODELS_DIR, "primary_model.pkl")
OUTPUT_CONFIG = os.path.join(MODELS_DIR, "primary_model_config.pkl")


def load_weather_data():
    """Load weather data and create lookup: (airport, date, hour) -> weather."""
    print("[1/5] Loading weather data...")
    wx = pd.read_csv(WEATHER_FILE)
    wx["date"] = pd.to_datetime(wx["date"]).dt.date

    # Keep key weather columns
    wx_cols = ["airport", "date", "hour", "temperature_2m", "wind_speed_10m",
               "precipitation", "visibility", "cloud_cover", "weather_code"]
    wx = wx[wx_cols].copy()

    # Fill missing values with median
    for col in ["temperature_2m", "wind_speed_10m", "precipitation", "visibility", "cloud_cover"]:
        wx[col] = wx[col].fillna(wx[col].median())
    wx["weather_code"] = wx["weather_code"].fillna(0)

    print(f"       Weather records: {len(wx):,}")
    print(f"       Airports with weather: {wx['airport'].nunique()}")
    return wx


def load_flight_data():
    """Load raw BTS data and extract needed columns for weather matching."""
    print("[2/5] Loading flight data...")
    cols = ["FlightDate", "Reporting_Airline", "Origin", "Dest",
            "DepTime", "ArrDelay", "Cancelled", "Diverted",
            "Distance", "AirTime", "DayOfWeek", "DayofMonth"]

    df = pd.read_csv(RAW_FILE, usecols=cols)

    # Clean: remove cancelled/diverted
    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)].copy()
    df = df.dropna(subset=["DepTime", "ArrDelay", "Origin", "Dest"])

    # Parse date and hour
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["date"] = df["FlightDate"].dt.date
    df["dep_hour"] = (df["DepTime"] // 100).astype(int).clip(0, 23)

    # Estimate arrival hour (dep_hour + air_time_hours)
    df["air_time_hr"] = (df["AirTime"].fillna(120) / 60).clip(0, 12)
    df["arr_hour"] = ((df["dep_hour"] + df["air_time_hr"]) % 24).astype(int)

    # Target
    df["IS_DELAYED"] = (df["ArrDelay"] >= 15).astype(int)

    print(f"       Clean flights: {len(df):,}")
    print(f"       Delay rate: {df['IS_DELAYED'].mean()*100:.1f}%")
    return df


def merge_weather(flights, weather):
    """Merge weather at origin (departure) and destination (arrival)."""
    print("[3/5] Merging flight data with weather...")

    # Merge origin weather
    origin_wx = weather.rename(columns={
        "temperature_2m": "origin_temp", "wind_speed_10m": "origin_wind",
        "precipitation": "origin_precip", "visibility": "origin_visibility",
        "cloud_cover": "origin_clouds", "weather_code": "origin_wx_code",
    })
    flights = flights.merge(
        origin_wx[["airport", "date", "hour",
                    "origin_temp", "origin_wind", "origin_precip",
                    "origin_visibility", "origin_clouds", "origin_wx_code"]],
        left_on=["Origin", "date", "dep_hour"],
        right_on=["airport", "date", "hour"],
        how="left",
    ).drop(columns=["airport", "hour"])

    # Merge destination weather
    dest_wx = weather.rename(columns={
        "temperature_2m": "dest_temp", "wind_speed_10m": "dest_wind",
        "precipitation": "dest_precip", "visibility": "dest_visibility",
        "cloud_cover": "dest_clouds", "weather_code": "dest_wx_code",
    })
    flights = flights.merge(
        dest_wx[["airport", "date", "hour",
                 "dest_temp", "dest_wind", "dest_precip",
                 "dest_visibility", "dest_clouds", "dest_wx_code"]],
        left_on=["Dest", "date", "arr_hour"],
        right_on=["airport", "date", "hour"],
        how="left",
    ).drop(columns=["airport", "hour"])

    # Fill any missing weather data with overall medians
    wx_features = ["origin_temp", "origin_wind", "origin_precip", "origin_visibility",
                    "origin_clouds", "origin_wx_code",
                    "dest_temp", "dest_wind", "dest_precip", "dest_visibility",
                    "dest_clouds", "dest_wx_code"]
    for col in wx_features:
        flights[col] = flights[col].fillna(flights[col].median())

    matched = flights[wx_features[0]].notna().sum()
    print(f"       Weather matched: {matched:,}/{len(flights):,} ({matched/len(flights)*100:.1f}%)")
    return flights, wx_features


def build_features(flights, wx_features):
    """Build the full feature set: fallback features + weather features."""
    print("[4/5] Building feature set...")

    # Load fallback model's aggregate stats for historical features
    agg_stats = pickle.load(open(os.path.join(MODELS_DIR, "aggregate_stats.pkl"), "rb"))
    encoders = pickle.load(open(os.path.join(MODELS_DIR, "encoders.pkl"), "rb"))

    # Temporal features
    flights["MONTH"] = flights["FlightDate"].dt.month
    flights["DAY_OF_WEEK"] = flights["DayOfWeek"]
    flights["DAY_OF_MONTH"] = flights["DayofMonth"]
    flights["DEP_HOUR"] = flights["dep_hour"]
    flights["IS_WEEKEND"] = flights["DAY_OF_WEEK"].isin([6, 7]).astype(int)
    flights["TIME_BLOCK"] = pd.cut(flights["DEP_HOUR"],
        bins=[-1, 5, 8, 11, 14, 17, 20, 24],
        labels=[0, 1, 2, 3, 4, 5, 6]).astype(float).fillna(3)

    # Route features
    flights["DISTANCE"] = flights["Distance"].fillna(flights["Distance"].median())
    flights["DISTANCE_GROUP"] = pd.cut(flights["DISTANCE"],
        bins=[0, 250, 500, 1000, 1500, 2000, 3000, 6000],
        labels=[0, 1, 2, 3, 4, 5, 6]).astype(float).fillna(2)
    flights["FLIGHT_DURATION"] = flights["AirTime"].fillna(flights["AirTime"].median())

    # Carrier/airport encoding (encoders are sklearn LabelEncoder objects)
    carrier_le = encoders.get("CARRIER")
    origin_le = encoders.get("ORIGIN")
    dest_le = encoders.get("DEST")

    # Create dict mappings from LabelEncoder classes
    carrier_map = {c: i for i, c in enumerate(carrier_le.classes_)} if carrier_le else {}
    origin_map = {c: i for i, c in enumerate(origin_le.classes_)} if origin_le else {}
    dest_map = {c: i for i, c in enumerate(dest_le.classes_)} if dest_le else {}

    flights["CARRIER_ENC"] = flights["Reporting_Airline"].map(carrier_map).fillna(-1).astype(int)
    flights["ORIGIN_ENC"] = flights["Origin"].map(origin_map).fillna(-1).astype(int)
    flights["DEST_ENC"] = flights["Dest"].map(dest_map).fillna(-1).astype(int)

    # Historical aggregate delay rates
    carrier_rates = agg_stats.get("carrier_delay_rate", {})
    origin_rates = agg_stats.get("origin_delay_rate", {})
    dest_rates = agg_stats.get("dest_delay_rate", {})
    route_rates = agg_stats.get("route_delay_rate", {})
    hour_rates = agg_stats.get("hour_delay_rate", {})
    dow_rates = agg_stats.get("dow_delay_rate", {})
    default_rate = agg_stats.get("overall_delay_rate", 0.2)

    flights["CARRIER_DELAY_RATE"] = flights["Reporting_Airline"].map(carrier_rates).fillna(default_rate)
    flights["ORIGIN_DELAY_RATE"] = flights["Origin"].map(origin_rates).fillna(default_rate)
    flights["DEST_DELAY_RATE"] = flights["Dest"].map(dest_rates).fillna(default_rate)
    flights["ROUTE"] = flights["Origin"] + "_" + flights["Dest"]
    flights["ROUTE_DELAY_RATE"] = flights["ROUTE"].map(route_rates).fillna(default_rate)
    flights["HOUR_DELAY_RATE"] = flights["DEP_HOUR"].map(hour_rates).fillna(default_rate)
    flights["DOW_DELAY_RATE"] = flights["DAY_OF_WEEK"].map(dow_rates).fillna(default_rate)

    # Derived weather features
    flights["bad_wx_origin"] = ((flights["origin_precip"] > 0.5) | (flights["origin_visibility"] < 5000) | (flights["origin_wind"] > 40)).astype(int)
    flights["bad_wx_dest"] = ((flights["dest_precip"] > 0.5) | (flights["dest_visibility"] < 5000) | (flights["dest_wind"] > 40)).astype(int)

    # Final feature list
    base_features = [
        "MONTH", "DAY_OF_WEEK", "DAY_OF_MONTH", "DEP_HOUR", "IS_WEEKEND", "TIME_BLOCK",
        "DISTANCE", "DISTANCE_GROUP", "FLIGHT_DURATION",
        "CARRIER_ENC", "ORIGIN_ENC", "DEST_ENC",
        "CARRIER_DELAY_RATE", "ORIGIN_DELAY_RATE", "DEST_DELAY_RATE",
        "ROUTE_DELAY_RATE", "HOUR_DELAY_RATE", "DOW_DELAY_RATE",
    ]
    all_features = base_features + wx_features + ["bad_wx_origin", "bad_wx_dest"]

    X = flights[all_features].copy()
    y = flights["IS_DELAYED"].copy()

    # Fill any remaining NaN
    X = X.fillna(0)

    print(f"       Total features: {len(all_features)} ({len(base_features)} base + {len(wx_features)} weather + 2 derived)")
    print(f"       Samples: {len(X):,}")

    # Save dataset
    save_df = pd.concat([X, y], axis=1)
    save_df.to_csv(OUTPUT_DATASET, index=False)
    print(f"       Saved dataset: {OUTPUT_DATASET}")

    return X, y, all_features


def train_and_evaluate(X, y, feature_names):
    """Train XGBoost and compare with fallback model."""
    print("[5/5] Training primary model...")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Scale pos weight for class imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    print(f"       Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"       Delay rate: {y_train.mean()*100:.1f}% | scale_pos_weight: {spw:.2f}")

    # Train
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │   PRIMARY MODEL RESULTS (with weather)  │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Accuracy:     {acc*100:6.1f}%                  │")
    print(f"  │  F1 Score:     {f1:.4f}                  │")
    print(f"  │  Precision:    {prec*100:6.1f}%                  │")
    print(f"  │  Recall:       {rec*100:6.1f}%                  │")
    print(f"  │  ROC-AUC:      {auc:.4f}                  │")
    print(f"  └─────────────────────────────────────────┘")
    print(f"\n  Confusion Matrix:")
    print(f"    Predicted:   On-Time  Delayed")
    print(f"    On-Time:     {cm[0][0]:7,}  {cm[0][1]:7,}")
    print(f"    Delayed:     {cm[1][0]:7,}  {cm[1][1]:7,}")

    # Classification report
    print(f"\n{classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed'])}")

    # Feature importance
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("  Top 15 Features:")
    print("  " + "-" * 40)
    for _, row in imp_df.head(15).iterrows():
        bar = "█" * int(row["importance"] / imp_df["importance"].max() * 25)
        print(f"  {row['feature']:25s} {row['importance']:.4f}  {bar}")

    # Compare with fallback
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │      COMPARISON: Primary vs Fallback    │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Metric       Primary    Fallback       │")
    print(f"  │  Accuracy:    {acc*100:5.1f}%     72.5%         │")
    print(f"  │  ROC-AUC:     {auc:.3f}      0.772         │")
    print(f"  │  Recall:      {rec*100:5.1f}%     66.2%         │")
    print(f"  │  Features:    {len(feature_names):3d}        19            │")
    print(f"  └─────────────────────────────────────────┘")

    # Save model
    pickle.dump(model, open(OUTPUT_MODEL, "wb"))
    pickle.dump({
        "features": feature_names,
        "accuracy": acc,
        "roc_auc": auc,
        "recall": rec,
        "f1": f1,
    }, open(OUTPUT_CONFIG, "wb"))

    print(f"\n  Model saved: {OUTPUT_MODEL}")
    print(f"  Config saved: {OUTPUT_CONFIG}")

    # Save comparison result
    comparison = pd.DataFrame([
        {"Model": "Primary (Weather)", "Accuracy": f"{acc*100:.1f}", "ROC-AUC": f"{auc:.4f}",
         "Recall": f"{rec*100:.1f}", "F1": f"{f1:.4f}", "Features": len(feature_names)},
        {"Model": "Fallback (No Weather)", "Accuracy": "72.5", "ROC-AUC": "0.7716",
         "Recall": "66.2", "F1": "0.5839", "Features": 19},
    ])
    comp_file = os.path.join(RESULTS_DIR, "primary_vs_fallback.csv")
    comparison.to_csv(comp_file, index=False)
    print(f"  Comparison: {comp_file}")

    return model, acc, auc


def main():
    print("=" * 55)
    print("  PRIMARY MODEL — WEATHER-ENHANCED TRAINING")
    print("=" * 55)

    weather = load_weather_data()
    flights = load_flight_data()
    merged, wx_features = merge_weather(flights, weather)
    X, y, feature_names = build_features(merged, wx_features)
    model, acc, auc = train_and_evaluate(X, y, feature_names)

    print(f"\n  Done! Primary model: {acc*100:.1f}% accuracy, {auc:.4f} AUC")


if __name__ == "__main__":
    main()
