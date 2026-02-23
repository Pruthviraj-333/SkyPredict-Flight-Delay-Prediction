"""
Compare multiple ML algorithms for the Fallback Flight Delay Model.

Tests the following algorithms on the same train/test split:
1. XGBoost
2. LightGBM
3. Random Forest
4. Gradient Boosting (sklearn)
5. Logistic Regression
6. K-Nearest Neighbors
7. Decision Tree
8. AdaBoost
9. Extra Trees
10. Support Vector Machine (Linear)

Also runs an out-of-sample test on November 2025 data for each model.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FEATURE_COLUMNS = [
    'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'ARR_HOUR',
    'IS_WEEKEND', 'TIME_BLOCK', 'DISTANCE', 'CRS_ELAPSED_TIME', 'DISTANCE_GROUP',
    'CARRIER_ENCODED', 'ORIGIN_ENCODED', 'DEST_ENCODED',
    'CARRIER_DELAY_RATE', 'ORIGIN_DELAY_RATE', 'DEST_DELAY_RATE',
    'HOUR_DELAY_RATE', 'DOW_DELAY_RATE', 'ROUTE_DELAY_RATE',
]
TARGET = 'IS_DELAYED'


def load_october_data():
    """Load processed October dataset."""
    path = os.path.join(PROCESSED_DIR, "fallback_dataset.csv")
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df[TARGET]
    return X, y


def load_november_test_data():
    """Load November 2025 data and prepare features for out-of-sample test."""
    raw_path = os.path.join(RAW_DIR, "ontime_2025_11.csv")
    df = pd.read_csv(raw_path, low_memory=False)
    
    # Clean
    df = df[df['Cancelled'] == 0.0]
    df = df[df['Diverted'] == 0.0]
    df = df.dropna(subset=['ArrDelay'])
    
    # Sample 200 flights (same seed as before for consistency)
    df = df.copy()
    df['IS_DELAYED'] = (df['ArrDelay'] >= 15).astype(int)
    delay_rate = df['IS_DELAYED'].mean()
    n_delayed = int(200 * delay_rate)
    n_ontime = 200 - n_delayed
    delayed = df[df['IS_DELAYED'] == 1].sample(n=n_delayed, random_state=42)
    ontime = df[df['IS_DELAYED'] == 0].sample(n=n_ontime, random_state=42)
    sample = pd.concat([delayed, ontime]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Load encoders and agg stats
    with open(os.path.join(MODELS_DIR, "encoders.pkl"), 'rb') as f:
        encoders = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "aggregate_stats.pkl"), 'rb') as f:
        agg_stats = pickle.load(f)
    
    # Prepare features
    features = pd.DataFrame()
    features['MONTH'] = sample['Month'].astype(int)
    features['DAY_OF_MONTH'] = sample['DayofMonth'].astype(int)
    features['DAY_OF_WEEK'] = sample['DayOfWeek'].astype(int)
    features['DEP_HOUR'] = (sample['CRSDepTime'].fillna(0) / 100).astype(int).clip(0, 23)
    features['ARR_HOUR'] = (sample['CRSArrTime'].fillna(0) / 100).astype(int).clip(0, 23)
    features['IS_WEEKEND'] = (sample['DayOfWeek'].isin([6, 7])).astype(int)
    features['TIME_BLOCK'] = pd.cut(features['DEP_HOUR'], bins=[-1,5,9,13,17,21,24], labels=[0,1,2,3,4,5]).astype(int)
    features['DISTANCE'] = sample['Distance'].fillna(0).astype(float)
    features['CRS_ELAPSED_TIME'] = sample['CRSElapsedTime'].fillna(0).astype(float)
    features['DISTANCE_GROUP'] = pd.cut(features['DISTANCE'], bins=[0,250,500,1000,2000,6000], labels=[0,1,2,3,4]).astype(int)
    
    carrier_col = sample['Reporting_Airline'].astype(str)
    origin_col = sample['Origin'].astype(str)
    dest_col = sample['Dest'].astype(str)
    
    features['CARRIER_ENCODED'] = carrier_col.apply(lambda x: encoders['CARRIER'].transform([x])[0] if x in encoders['CARRIER'].classes_ else 0)
    features['ORIGIN_ENCODED'] = origin_col.apply(lambda x: encoders['ORIGIN'].transform([x])[0] if x in encoders['ORIGIN'].classes_ else 0)
    features['DEST_ENCODED'] = dest_col.apply(lambda x: encoders['DEST'].transform([x])[0] if x in encoders['DEST'].classes_ else 0)
    
    default_rate = 0.203
    features['CARRIER_DELAY_RATE'] = carrier_col.map(agg_stats['carrier_delay_rate']).fillna(default_rate)
    features['ORIGIN_DELAY_RATE'] = origin_col.map(agg_stats['origin_delay_rate']).fillna(default_rate)
    features['DEST_DELAY_RATE'] = dest_col.map(agg_stats['dest_delay_rate']).fillna(default_rate)
    features['HOUR_DELAY_RATE'] = features['DEP_HOUR'].map(agg_stats['hour_delay_rate']).fillna(default_rate)
    features['DOW_DELAY_RATE'] = features['DAY_OF_WEEK'].map(agg_stats['dow_delay_rate']).fillna(default_rate)
    route = origin_col + '_' + dest_col
    features['ROUTE_DELAY_RATE'] = route.map(agg_stats['route_delay_rate']).fillna(default_rate)
    
    X_nov = features.fillna(0)
    y_nov = sample['IS_DELAYED'].values
    
    return X_nov, y_nov


def get_models():
    """Return dict of model name -> model instance."""
    from xgboost import XGBClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        AdaBoostClassifier, ExtraTreesClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import LinearSVC
    
    try:
        from lightgbm import LGBMClassifier
        has_lgbm = True
    except ImportError:
        has_lgbm = False
    
    models = {}
    
    models['XGBoost'] = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=3.92, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    
    if has_lgbm:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=3.92, random_state=42, n_jobs=-1, verbose=-1,
        )
    
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=20, random_state=42,
    )
    
    models['Extra Trees'] = ExtraTreesClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    
    models['AdaBoost'] = AdaBoostClassifier(
        n_estimators=200, learning_rate=0.1, random_state=42,
    )
    
    models['Decision Tree'] = DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=20,
        class_weight='balanced', random_state=42,
    )
    
    models['Logistic Regression'] = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1,
    )
    
    models['KNN (k=15)'] = KNeighborsClassifier(
        n_neighbors=15, n_jobs=-1,
    )
    
    models['Linear SVM'] = LinearSVC(
        max_iter=2000, class_weight='balanced', random_state=42,
    )
    
    return models


def main():
    print("=" * 90)
    print("  FALLBACK MODEL — MULTI-ALGORITHM COMPARISON")
    print("=" * 90)
    
    # Load data
    print("\n[1/4] Loading October 2025 data (train/test)...")
    X, y = load_october_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"  Delay rate: {y.mean()*100:.1f}%")
    
    # Scale for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Load November data
    print("\n[2/4] Loading November 2025 data (out-of-sample)...")
    X_nov, y_nov = load_november_test_data()
    X_nov_scaled = scaler.transform(X_nov)
    print(f"  November samples: {X_nov.shape[0]}  (delay rate: {y_nov.mean()*100:.1f}%)")
    
    # Get models
    print("\n[3/4] Initializing models...")
    models = get_models()
    print(f"  {len(models)} algorithms to compare")
    
    # Models that need scaled data
    needs_scaling = {'Logistic Regression', 'KNN (k=15)', 'Linear SVM'}
    
    # Train and evaluate each
    print("\n[4/4] Training and evaluating...\n")
    results = []
    
    for name, model in models.items():
        print(f"  Training {name}...", end="", flush=True)
        start = time.time()
        
        # Select scaled or unscaled data
        if name in needs_scaling:
            Xtr, Xte, Xnv = X_train_scaled, X_test_scaled, X_nov_scaled
        else:
            Xtr, Xte, Xnv = X_train.values, X_test.values, X_nov.values
        
        try:
            model.fit(Xtr, y_train)
            train_time = time.time() - start
            
            # October test predictions
            y_train_pred = model.predict(Xtr)
            y_test_pred = model.predict(Xte)
            
            # Handle models without predict_proba (LinearSVC)
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(Xte)[:, 1]
                y_nov_proba = model.predict_proba(Xnv)[:, 1]
                oct_auc = roc_auc_score(y_test, y_test_proba)
                nov_auc = roc_auc_score(y_nov, y_nov_proba)
            elif hasattr(model, 'decision_function'):
                y_test_scores = model.decision_function(Xte)
                y_nov_scores = model.decision_function(Xnv)
                oct_auc = roc_auc_score(y_test, y_test_scores)
                nov_auc = roc_auc_score(y_nov, y_nov_scores)
            else:
                oct_auc = 0
                nov_auc = 0
            
            # November predictions
            y_nov_pred = model.predict(Xnv)
            
            result = {
                'Model': name,
                'Train Acc': accuracy_score(y_train, y_train_pred),
                'Test Acc': accuracy_score(y_test, y_test_pred),
                'Test F1': f1_score(y_test, y_test_pred),
                'Test Precision': precision_score(y_test, y_test_pred),
                'Test Recall': recall_score(y_test, y_test_pred),
                'Oct ROC-AUC': oct_auc,
                'Nov Acc': accuracy_score(y_nov, y_nov_pred),
                'Nov F1': f1_score(y_nov, y_nov_pred),
                'Nov Recall': recall_score(y_nov, y_nov_pred),
                'Nov ROC-AUC': nov_auc,
                'Train Time (s)': train_time,
            }
            results.append(result)
            
            print(f"  done ({train_time:.1f}s) | Oct Test: {result['Test Acc']:.1%} | Nov: {result['Nov Acc']:.1%}")
            
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'Model': name,
                'Train Acc': 0, 'Test Acc': 0, 'Test F1': 0,
                'Test Precision': 0, 'Test Recall': 0, 'Oct ROC-AUC': 0,
                'Nov Acc': 0, 'Nov F1': 0, 'Nov Recall': 0, 'Nov ROC-AUC': 0,
                'Train Time (s)': 0,
            })
    
    # ===== COMPARISON TABLE =====
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Oct ROC-AUC', ascending=False).reset_index(drop=True)
    results_df.index += 1  # 1-indexed rank
    
    print("\n" + "=" * 90)
    print("  RESULTS COMPARISON — Sorted by October ROC-AUC (best overall metric)")
    print("=" * 90)
    
    # Format the comparison table
    print(f"\n{'Rank':<5} {'Model':<22} {'Train':>7} {'Oct Test':>8} {'Oct F1':>7} {'Oct AUC':>8} {'Nov Test':>8} {'Nov F1':>7} {'Nov AUC':>8} {'Time':>6}")
    print("-" * 90)
    
    for idx, row in results_df.iterrows():
        print(f"{idx:<5} {row['Model']:<22} {row['Train Acc']:>7.1%} {row['Test Acc']:>8.1%} {row['Test F1']:>7.3f} {row['Oct ROC-AUC']:>8.4f} {row['Nov Acc']:>8.1%} {row['Nov F1']:>7.3f} {row['Nov ROC-AUC']:>8.4f} {row['Train Time (s)']:>5.1f}s")
    
    # Best model
    best_idx = results_df['Oct ROC-AUC'].idxmax()
    best = results_df.loc[best_idx]
    
    print(f"\n{'=' * 90}")
    print(f"  🏆 BEST MODEL: {best['Model']}")
    print(f"     October Test Accuracy: {best['Test Acc']:.1%}  |  ROC-AUC: {best['Oct ROC-AUC']:.4f}")
    print(f"     November Out-of-Sample Accuracy: {best['Nov Acc']:.1%}  |  ROC-AUC: {best['Nov ROC-AUC']:.4f}")
    print(f"{'=' * 90}")
    
    # Detailed comparison: accuracy vs recall tradeoff
    print(f"\n{'=' * 90}")
    print("  DETAILED ANALYSIS — Accuracy vs Delay Detection")
    print(f"{'=' * 90}")
    print(f"\n{'Model':<22} {'Oct Acc':>8} {'Oct Recall':>10} {'Nov Acc':>8} {'Nov Recall':>10} {'Verdict':<20}")
    print("-" * 80)
    
    for idx, row in results_df.iterrows():
        # Determine verdict
        if row['Oct ROC-AUC'] >= 0.77:
            verdict = "⭐ Excellent"
        elif row['Oct ROC-AUC'] >= 0.72:
            verdict = "✅ Good"
        elif row['Oct ROC-AUC'] >= 0.65:
            verdict = "⚠️ Fair"
        else:
            verdict = "❌ Poor"
        
        print(f"{row['Model']:<22} {row['Test Acc']:>8.1%} {row['Test Recall']:>10.1%} {row['Nov Acc']:>8.1%} {row['Nov Recall']:>10.1%} {verdict:<20}")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index_label='Rank')
    print(f"\n  Results saved to: {results_path}")
    
    print(f"\n{'=' * 90}")
    print("  COMPARISON COMPLETE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
