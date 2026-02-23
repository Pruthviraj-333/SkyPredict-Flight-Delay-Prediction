"""
Test the Fallback Model on 200 real flights from November 2025 (out-of-sample).

Compares predicted delay vs actual delay for each flight, and prints
summary metrics + a detailed flight-by-flight comparison table.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

N_SAMPLES = 200
RANDOM_SEED = 42


def load_november_data():
    """Load and prepare November 2025 data."""
    path = os.path.join(RAW_DIR, "ontime_2025_11.csv")
    print(f"[INFO] Loading November 2025 data from {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw shape: {df.shape}")
    
    # Filter: non-cancelled, non-diverted, with valid ArrDelay
    df = df[df['Cancelled'] == 0.0]
    df = df[df['Diverted'] == 0.0]
    df = df.dropna(subset=['ArrDelay'])
    print(f"  After cleaning: {df.shape[0]} flights")
    
    return df


def sample_flights(df, n=200, seed=42):
    """
    Sample N flights with a mix of delayed and on-time 
    to get a representative sample.
    """
    # Create target
    df = df.copy()
    df['IS_DELAYED'] = (df['ArrDelay'] >= 15).astype(int)
    
    delay_rate = df['IS_DELAYED'].mean()
    n_delayed = int(n * delay_rate)
    n_ontime = n - n_delayed
    
    delayed = df[df['IS_DELAYED'] == 1].sample(n=n_delayed, random_state=seed)
    ontime = df[df['IS_DELAYED'] == 0].sample(n=n_ontime, random_state=seed)
    
    sample = pd.concat([delayed, ontime]).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"  Sampled {len(sample)} flights: {n_ontime} on-time, {n_delayed} delayed ({delay_rate*100:.1f}% delay rate)")
    return sample


def prepare_features_for_batch(sample_df, encoders, agg_stats):
    """Prepare features for a batch of flights."""
    
    features = pd.DataFrame()
    
    # Temporal
    features['MONTH'] = sample_df['Month'].astype(int)
    features['DAY_OF_MONTH'] = sample_df['DayofMonth'].astype(int)
    features['DAY_OF_WEEK'] = sample_df['DayOfWeek'].astype(int)
    features['DEP_HOUR'] = (sample_df['CRSDepTime'].fillna(0) / 100).astype(int).clip(0, 23)
    features['ARR_HOUR'] = (sample_df['CRSArrTime'].fillna(0) / 100).astype(int).clip(0, 23)
    features['IS_WEEKEND'] = (sample_df['DayOfWeek'].isin([6, 7])).astype(int)
    
    # Time block
    features['TIME_BLOCK'] = pd.cut(
        features['DEP_HOUR'],
        bins=[-1, 5, 9, 13, 17, 21, 24],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)
    
    # Distance and duration
    features['DISTANCE'] = sample_df['Distance'].fillna(0).astype(float)
    features['CRS_ELAPSED_TIME'] = sample_df['CRSElapsedTime'].fillna(0).astype(float)
    features['DISTANCE_GROUP'] = pd.cut(
        features['DISTANCE'],
        bins=[0, 250, 500, 1000, 2000, 6000],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    
    # Encode carrier
    carrier_col = sample_df['Reporting_Airline'].astype(str)
    features['CARRIER_ENCODED'] = carrier_col.apply(
        lambda x: encoders['CARRIER'].transform([x])[0] if x in encoders['CARRIER'].classes_ else 0
    )
    
    # Encode origin
    origin_col = sample_df['Origin'].astype(str)
    features['ORIGIN_ENCODED'] = origin_col.apply(
        lambda x: encoders['ORIGIN'].transform([x])[0] if x in encoders['ORIGIN'].classes_ else 0
    )
    
    # Encode dest
    dest_col = sample_df['Dest'].astype(str)
    features['DEST_ENCODED'] = dest_col.apply(
        lambda x: encoders['DEST'].transform([x])[0] if x in encoders['DEST'].classes_ else 0
    )
    
    # Aggregate delay rates
    default_rate = 0.203
    features['CARRIER_DELAY_RATE'] = carrier_col.map(agg_stats['carrier_delay_rate']).fillna(default_rate)
    features['ORIGIN_DELAY_RATE'] = origin_col.map(agg_stats['origin_delay_rate']).fillna(default_rate)
    features['DEST_DELAY_RATE'] = dest_col.map(agg_stats['dest_delay_rate']).fillna(default_rate)
    features['HOUR_DELAY_RATE'] = features['DEP_HOUR'].map(agg_stats['hour_delay_rate']).fillna(default_rate)
    features['DOW_DELAY_RATE'] = features['DAY_OF_WEEK'].map(agg_stats['dow_delay_rate']).fillna(default_rate)
    
    route = origin_col + '_' + dest_col
    features['ROUTE_DELAY_RATE'] = route.map(agg_stats['route_delay_rate']).fillna(default_rate)
    
    return features


def run_test():
    """Run the full test pipeline."""
    print("=" * 70)
    print("  FALLBACK MODEL — OUT-OF-SAMPLE TEST (November 2025)")
    print("=" * 70)
    
    # Load model and artifacts
    print("\n[1/5] Loading model and artifacts...")
    with open(os.path.join(MODELS_DIR, "fallback_model.pkl"), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "encoders.pkl"), 'rb') as f:
        encoders = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "aggregate_stats.pkl"), 'rb') as f:
        agg_stats = pickle.load(f)
    print("  ✓ Model, encoders, and aggregate stats loaded")
    
    # Load November data
    print("\n[2/5] Loading November 2025 data...")
    nov_df = load_november_data()
    
    # Sample 200 flights
    print("\n[3/5] Sampling flights...")
    sample = sample_flights(nov_df, n=N_SAMPLES, seed=RANDOM_SEED)
    
    # Prepare ground truth
    y_actual = (sample['ArrDelay'] >= 15).astype(int).values
    actual_delays = sample['ArrDelay'].values
    
    # Prepare features
    print("\n[4/5] Preparing features...")
    X = prepare_features_for_batch(sample, encoders, agg_stats)
    X = X.fillna(0)
    print(f"  Feature matrix: {X.shape}")
    
    # Predict
    print("\n[5/5] Running predictions...")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # ===== METRICS =====
    print("\n" + "=" * 70)
    print("  OVERALL METRICS")
    print("=" * 70)
    
    acc = accuracy_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
    prec = precision_score(y_actual, y_pred)
    rec = recall_score(y_actual, y_pred)
    auc = roc_auc_score(y_actual, y_proba)
    
    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*37}")
    print(f"  {'Accuracy':<25} {acc:>10.2%}")
    print(f"  {'F1 Score':<25} {f1:>10.4f}")
    print(f"  {'Precision':<25} {prec:>10.2%}")
    print(f"  {'Recall':<25} {rec:>10.2%}")
    print(f"  {'ROC-AUC':<25} {auc:>10.4f}")
    
    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_actual, y_pred, target_names=['On-Time', 'Delayed'], digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(y_actual, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {'':>20}  Predicted")
    print(f"  {'':>20}  On-Time  Delayed")
    print(f"  {'Actual On-Time':>20}  {cm[0][0]:>7}  {cm[0][1]:>7}")
    print(f"  {'Actual Delayed':>20}  {cm[1][0]:>7}  {cm[1][1]:>7}")
    
    # ===== DETAILED FLIGHT-BY-FLIGHT COMPARISON =====
    print("\n" + "=" * 70)
    print("  FLIGHT-BY-FLIGHT COMPARISON (200 flights)")
    print("=" * 70)
    
    results = pd.DataFrame({
        'Flight#': range(1, N_SAMPLES + 1),
        'Carrier': sample['Reporting_Airline'].values,
        'Origin': sample['Origin'].values,
        'Dest': sample['Dest'].values,
        'Date': sample['FlightDate'].values,
        'DepTime': sample['CRSDepTime'].values.astype(int),
        'ActualDelay': sample['ArrDelay'].values.astype(int),
        'ActualStatus': ['DELAYED' if d >= 15 else 'ON-TIME' for d in actual_delays],
        'PredProb': [f"{p:.1%}" for p in y_proba],
        'PredStatus': ['DELAYED' if p == 1 else 'ON-TIME' for p in y_pred],
        'Correct': ['✓' if a == p else '✗' for a, p in zip(y_actual, y_pred)],
    })
    
    # Print as formatted table
    pd.set_option('display.max_rows', 250)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 15)
    print(results.to_string(index=False))
    
    # ===== SUMMARY STATISTICS =====
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    correct = (y_actual == y_pred).sum()
    wrong = (y_actual != y_pred).sum()
    
    # True positives, etc.
    tp = ((y_actual == 1) & (y_pred == 1)).sum()
    tn = ((y_actual == 0) & (y_pred == 0)).sum()
    fp = ((y_actual == 0) & (y_pred == 1)).sum()
    fn = ((y_actual == 1) & (y_pred == 0)).sum()
    
    print(f"\n  Total flights tested:          {N_SAMPLES}")
    print(f"  Correctly predicted:           {correct} ({correct/N_SAMPLES*100:.1f}%)")
    print(f"  Incorrectly predicted:         {wrong} ({wrong/N_SAMPLES*100:.1f}%)")
    print(f"  ---")
    print(f"  Delayed flights caught (TP):   {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)" if (tp+fn)>0 else "  No delayed flights")
    print(f"  On-time correctly (TN):        {tn}/{tn+fp} ({tn/(tn+fp)*100:.1f}%)" if (tn+fp)>0 else "  No on-time flights")
    print(f"  False alarms (FP):             {fp} (predicted delayed, was on-time)")
    print(f"  Missed delays (FN):            {fn} (predicted on-time, was delayed)")
    
    # Save results to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "november_2025_test_results.csv")
    results.to_csv(results_path, index=False)
    print(f"\n  Results saved to: {results_path}")
    
    print("\n" + "=" * 70)
    print(f"  TEST COMPLETE — Accuracy: {acc:.1%} on {N_SAMPLES} unseen November flights")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
