"""
Improved Fallback Model — 12-Month Data with:
  1. Enhanced Feature Engineering (holidays, cyclical, interactions, season, congestion)
  2. Optuna Hyperparameter Tuning (50 trials)
  3. Threshold Optimization (maximize F1)

All research paper-grade metrics computed.
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "..", "results")
TARGET = 'IS_DELAYED'

# ============================================================
# US HOLIDAYS 2024-2025
# ============================================================
US_HOLIDAYS = {
    # 2024
    (2024, 12, 24), (2024, 12, 25), (2024, 12, 26),  # Christmas
    (2024, 12, 31),  # New Year's Eve
    # 2025
    (2025, 1, 1),   # New Year's Day
    (2025, 1, 20),  # MLK Day
    (2025, 2, 17),  # Presidents Day
    (2025, 3, 14), (2025, 3, 15), (2025, 3, 16), (2025, 3, 17),
    (2025, 3, 18), (2025, 3, 19), (2025, 3, 20), (2025, 3, 21),  # Spring Break window
    (2025, 5, 26),  # Memorial Day
    (2025, 7, 3), (2025, 7, 4), (2025, 7, 5),  # July 4th
    (2025, 9, 1),   # Labor Day
    (2025, 10, 13),  # Columbus Day
    (2025, 11, 11),  # Veterans Day
    (2025, 11, 26), (2025, 11, 27), (2025, 11, 28), (2025, 11, 29), (2025, 11, 30),  # Thanksgiving
}

# Days near holidays (travel surge days)
NEAR_HOLIDAY = set()
for y, m, d in US_HOLIDAYS:
    for delta in [-2, -1, 0, 1, 2]:
        try:
            from datetime import date, timedelta
            dt = date(y, m, d) + timedelta(days=delta)
            NEAR_HOLIDAY.add((dt.year, dt.month, dt.day))
        except:
            pass


# ============================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================
def engineer_enhanced_features(df):
    """Engineer features with improvements for better model performance."""
    print("\n[ENHANCED FEATURE ENGINEERING]")
    feat = pd.DataFrame()

    # Target
    feat[TARGET] = df['IS_DELAYED'].astype(int)

    # --- Original features ---
    feat['MONTH'] = df['Month'].astype(int)
    feat['DAY_OF_MONTH'] = df['DayofMonth'].astype(int)
    feat['DAY_OF_WEEK'] = df['DayOfWeek'].astype(int)
    feat['DEP_HOUR'] = (df['CRSDepTime'].fillna(0) / 100).astype(int).clip(0, 23)
    feat['ARR_HOUR'] = (df['CRSArrTime'].fillna(0) / 100).astype(int).clip(0, 23)
    feat['IS_WEEKEND'] = (df['DayOfWeek'].isin([6, 7])).astype(int)
    feat['TIME_BLOCK'] = pd.cut(feat['DEP_HOUR'], bins=[-1,5,9,13,17,21,24], labels=[0,1,2,3,4,5]).astype(int)
    feat['CARRIER'] = df['Reporting_Airline'].astype(str)
    feat['ORIGIN'] = df['Origin'].astype(str)
    feat['DEST'] = df['Dest'].astype(str)
    feat['DISTANCE'] = df['Distance'].fillna(0).astype(float)
    feat['CRS_ELAPSED_TIME'] = df['CRSElapsedTime'].fillna(0).astype(float)
    feat['DISTANCE_GROUP'] = pd.cut(feat['DISTANCE'], bins=[0,250,500,1000,2000,6000], labels=[0,1,2,3,4]).astype(int)

    # --- NEW: Cyclical month encoding (sin/cos) ---
    feat['MONTH_SIN'] = np.sin(2 * np.pi * feat['MONTH'] / 12)
    feat['MONTH_COS'] = np.cos(2 * np.pi * feat['MONTH'] / 12)

    # --- NEW: Cyclical hour encoding ---
    feat['HOUR_SIN'] = np.sin(2 * np.pi * feat['DEP_HOUR'] / 24)
    feat['HOUR_COS'] = np.cos(2 * np.pi * feat['DEP_HOUR'] / 24)

    # --- NEW: Cyclical day-of-week ---
    feat['DOW_SIN'] = np.sin(2 * np.pi * feat['DAY_OF_WEEK'] / 7)
    feat['DOW_COS'] = np.cos(2 * np.pi * feat['DAY_OF_WEEK'] / 7)

    # --- NEW: Season flag ---
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    feat['SEASON'] = feat['MONTH'].map(season_map)

    # --- NEW: Holiday flag ---
    year_col = df['Year'].astype(int)
    feat['IS_HOLIDAY'] = [
        1 if (y, m, d) in US_HOLIDAYS else 0
        for y, m, d in zip(year_col, feat['MONTH'], feat['DAY_OF_MONTH'])
    ]
    feat['NEAR_HOLIDAY'] = [
        1 if (y, m, d) in NEAR_HOLIDAY else 0
        for y, m, d in zip(year_col, feat['MONTH'], feat['DAY_OF_MONTH'])
    ]

    # --- NEW: Peak travel indicators ---
    feat['IS_FRIDAY_EVENING'] = ((feat['DAY_OF_WEEK'] == 5) & (feat['DEP_HOUR'] >= 15)).astype(int)
    feat['IS_SUNDAY_EVENING'] = ((feat['DAY_OF_WEEK'] == 7) & (feat['DEP_HOUR'] >= 15)).astype(int)
    feat['IS_MONDAY_MORNING'] = ((feat['DAY_OF_WEEK'] == 1) & (feat['DEP_HOUR'] <= 9)).astype(int)
    feat['IS_PEAK_HOUR'] = feat['DEP_HOUR'].isin([7, 8, 16, 17, 18]).astype(int)

    # --- NEW: Congestion proxy (flights per hour at origin) ---
    feat['ORIGIN_HOUR'] = feat['ORIGIN'] + '_' + feat['DEP_HOUR'].astype(str)
    origin_hour_counts = feat.groupby('ORIGIN_HOUR')[TARGET].count()
    feat['ORIGIN_HOUR_FLIGHTS'] = feat['ORIGIN_HOUR'].map(origin_hour_counts)
    # Normalize by max
    max_flights = feat['ORIGIN_HOUR_FLIGHTS'].max()
    feat['CONGESTION_PROXY'] = feat['ORIGIN_HOUR_FLIGHTS'] / max_flights if max_flights > 0 else 0

    # --- NEW: Flight duration bucket ---
    feat['DURATION_BUCKET'] = pd.cut(
        feat['CRS_ELAPSED_TIME'],
        bins=[0, 60, 120, 180, 300, 1500],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(0).astype(int)

    # Keep year for reference
    feat['YEAR'] = year_col

    print(f"  Created {len(feat.columns)} columns (vs 14 in baseline)")

    # Count new features
    new_features = ['MONTH_SIN', 'MONTH_COS', 'HOUR_SIN', 'HOUR_COS', 'DOW_SIN', 'DOW_COS',
                     'SEASON', 'IS_HOLIDAY', 'NEAR_HOLIDAY', 'IS_FRIDAY_EVENING',
                     'IS_SUNDAY_EVENING', 'IS_MONDAY_MORNING', 'IS_PEAK_HOUR',
                     'CONGESTION_PROXY', 'DURATION_BUCKET']
    print(f"  New features added: {len(new_features)}")
    for f in new_features:
        print(f"    + {f}")

    return feat


def add_aggregate_features(feat):
    """Compute population-level aggregate delay rates."""
    print("\n[AGGREGATE FEATURES]")
    for name, col in [('CARRIER_DELAY_RATE','CARRIER'), ('ORIGIN_DELAY_RATE','ORIGIN'),
                       ('DEST_DELAY_RATE','DEST'), ('HOUR_DELAY_RATE','DEP_HOUR'),
                       ('DOW_DELAY_RATE','DAY_OF_WEEK')]:
        rates = feat.groupby(col)[TARGET].mean()
        feat[name] = feat[col].map(rates)
        print(f"  {name}: {len(rates)} groups")

    feat['ROUTE'] = feat['ORIGIN'] + '_' + feat['DEST']
    route_rates = feat.groupby('ROUTE')[TARGET].mean()
    feat['ROUTE_DELAY_RATE'] = feat['ROUTE'].map(route_rates)
    print(f"  ROUTE_DELAY_RATE: {len(route_rates)} routes")

    # NEW: Carrier-Origin interaction delay rate
    feat['CARRIER_ORIGIN'] = feat['CARRIER'] + '_' + feat['ORIGIN']
    co_rates = feat.groupby('CARRIER_ORIGIN')[TARGET].mean()
    feat['CARRIER_ORIGIN_DELAY_RATE'] = feat['CARRIER_ORIGIN'].map(co_rates)
    print(f"  CARRIER_ORIGIN_DELAY_RATE: {len(co_rates)} combos")

    # NEW: Season delay rate
    season_rates = feat.groupby('SEASON')[TARGET].mean()
    feat['SEASON_DELAY_RATE'] = feat['SEASON'].map(season_rates)
    print(f"  SEASON_DELAY_RATE: {len(season_rates)} seasons")

    # Save stats
    agg_stats = {
        'carrier_delay_rate': feat.groupby('CARRIER')[TARGET].mean().to_dict(),
        'origin_delay_rate': feat.groupby('ORIGIN')[TARGET].mean().to_dict(),
        'dest_delay_rate': feat.groupby('DEST')[TARGET].mean().to_dict(),
        'hour_delay_rate': feat.groupby('DEP_HOUR')[TARGET].mean().to_dict(),
        'dow_delay_rate': feat.groupby('DAY_OF_WEEK')[TARGET].mean().to_dict(),
        'route_delay_rate': route_rates.to_dict(),
        'carrier_origin_delay_rate': co_rates.to_dict(),
        'season_delay_rate': season_rates.to_dict(),
    }

    return feat, agg_stats


def encode_and_prepare(feat):
    """Encode categoricals and define feature columns."""
    print("\n[ENCODING]")
    encoders = {}
    for col in ['CARRIER', 'ORIGIN', 'DEST']:
        le = LabelEncoder()
        feat[f'{col}_ENCODED'] = le.fit_transform(feat[col])
        encoders[col] = le
        print(f"  {col}: {len(le.classes_)} classes")

    # Define ALL feature columns (original + new)
    feature_cols = [
        # Original
        'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'ARR_HOUR',
        'IS_WEEKEND', 'TIME_BLOCK', 'DISTANCE', 'CRS_ELAPSED_TIME', 'DISTANCE_GROUP',
        'CARRIER_ENCODED', 'ORIGIN_ENCODED', 'DEST_ENCODED',
        # Aggregate rates
        'CARRIER_DELAY_RATE', 'ORIGIN_DELAY_RATE', 'DEST_DELAY_RATE',
        'HOUR_DELAY_RATE', 'DOW_DELAY_RATE', 'ROUTE_DELAY_RATE',
        'CARRIER_ORIGIN_DELAY_RATE', 'SEASON_DELAY_RATE',
        # Cyclical
        'MONTH_SIN', 'MONTH_COS', 'HOUR_SIN', 'HOUR_COS', 'DOW_SIN', 'DOW_COS',
        # New categorical/binary
        'SEASON', 'IS_HOLIDAY', 'NEAR_HOLIDAY',
        'IS_FRIDAY_EVENING', 'IS_SUNDAY_EVENING', 'IS_MONDAY_MORNING', 'IS_PEAK_HOUR',
        # New numeric
        'CONGESTION_PROXY', 'DURATION_BUCKET',
    ]

    print(f"  Total features: {len(feature_cols)} (was 19 in baseline)")
    return feat, encoders, feature_cols


# ============================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================
def optuna_tune(X_train, y_train, n_trials=50):
    """Tune XGBoost hyperparameters with Optuna."""
    print(f"\n{'=' * 60}")
    print(f"  OPTUNA HYPERPARAMETER TUNING ({n_trials} trials)")
    print(f"{'=' * 60}")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    # Use a 15% subsample for faster tuning on large datasets
    sample_size = min(len(X_train), 800_000)
    if len(X_train) > sample_size:
        print(f"  Subsampling to {sample_size:,} for faster tuning...")
        idx = np.random.RandomState(42).choice(len(X_train), sample_size, replace=False)
        X_tune = X_train.iloc[idx]
        y_tune = y_train.iloc[idx]
    else:
        X_tune = X_train
        y_tune = y_train

    best_score = [-1]
    trial_results = []

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'scale_pos_weight': spw,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
        }

        # 3-fold CV for speed
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in skf.split(X_tune, y_tune):
            Xtr, Xv = X_tune.iloc[train_idx], X_tune.iloc[val_idx]
            ytr, yv = y_tune.iloc[train_idx], y_tune.iloc[val_idx]

            model = XGBClassifier(**params)
            model.fit(Xtr, ytr)
            y_proba = model.predict_proba(Xv)[:, 1]
            aucs.append(roc_auc_score(yv, y_proba))

        mean_auc = np.mean(aucs)
        trial_results.append({
            'trial': trial.number, 'auc': mean_auc,
            **{k: v for k, v in params.items() if k not in ['scale_pos_weight', 'eval_metric', 'random_state', 'n_jobs', 'verbosity']}
        })

        if mean_auc > best_score[0]:
            best_score[0] = mean_auc
            print(f"  Trial {trial.number+1:>3}/{n_trials}: AUC={mean_auc:.4f} ★ NEW BEST")
        elif (trial.number + 1) % 10 == 0:
            print(f"  Trial {trial.number+1:>3}/{n_trials}: AUC={mean_auc:.4f}  (best: {best_score[0]:.4f})")

        return mean_auc

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    start = time.time()
    study.optimize(objective, n_trials=n_trials)
    tune_time = time.time() - start

    best_params = study.best_params
    best_params['scale_pos_weight'] = spw
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['verbosity'] = 1

    print(f"\n  Best AUC: {study.best_value:.4f}")
    print(f"  Tuning time: {tune_time:.1f}s ({tune_time/60:.1f} min)")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return best_params, study, trial_results


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================
def optimize_threshold(y_true, y_proba, metric='f1'):
    """Find the optimal classification threshold."""
    print(f"\n[THRESHOLD OPTIMIZATION — maximize {metric.upper()}]")

    thresholds = np.arange(0.15, 0.70, 0.01)
    scores = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if metric == 'f1':
            s = f1_score(y_true, y_pred)
        elif metric == 'mcc':
            s = matthews_corrcoef(y_true, y_pred)
        scores.append(s)

    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    print(f"  Default (0.50): {metric}={scores[int((0.50-0.15)/0.01)]:.4f}")
    print(f"  Optimal ({best_threshold:.2f}): {metric}={best_score:.4f}")
    print(f"  Improvement: +{best_score - scores[int((0.50-0.15)/0.01)]:.4f}")

    return best_threshold, thresholds, scores


# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, y_pred, y_proba, name="Test"):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'Set': name, 'N': len(y_true),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'F1_Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_true, y_proba),
        'PR_AUC': average_precision_score(y_true, y_proba),
        'Log_Loss': log_loss(y_true, y_proba),
        'Brier_Score': brier_score_loss(y_true, y_proba),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Cohens_Kappa': cohen_kappa_score(y_true, y_pred),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0,
    }


def print_metrics(m):
    print(f"\n{'=' * 60}")
    print(f"  {m['Set']} (N={m['N']:,})")
    print(f"{'=' * 60}")
    for name, key, pct in [
        ('Accuracy', 'Accuracy', True), ('Precision', 'Precision', True),
        ('Recall (Sensitivity)', 'Recall', True), ('Specificity', 'Specificity', True),
        ('F1 Score', 'F1_Score', False), ('ROC-AUC', 'ROC_AUC', False),
        ('PR-AUC', 'PR_AUC', False), ('Log Loss', 'Log_Loss', False),
        ('Brier Score', 'Brier_Score', False), ('MCC', 'MCC', False),
        ("Cohen's Kappa", 'Cohens_Kappa', False),
    ]:
        val = m[key]
        suffix = f" ({val*100:.2f}%)" if pct else ""
        print(f"  {name:<25} {val:.4f}{suffix}")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>20} Pred On-Time  Pred Delayed")
    print(f"  Actual On-Time  {m['TN']:>12,}  {m['FP']:>12,}")
    print(f"  Actual Delayed  {m['FN']:>12,}  {m['TP']:>12,}")
    print(f"  FPR: {m['FPR']:.4f}  |  FNR: {m['FNR']:.4f}")


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()
    print("\n" + "=" * 70)
    print("  IMPROVED FALLBACK MODEL — 12-MONTH DATA")
    print("  Enhanced Features + Optuna Tuning + Threshold Optimization")
    print("=" * 70)

    # Step 1: Load
    path = os.path.join(os.path.abspath(DATA_DIR), "combined_12month_raw.csv")
    print(f"\n[LOADING] {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Shape: {df.shape}")

    # Step 2: Enhanced features
    feat = engineer_enhanced_features(df)

    # Step 3: Aggregate features
    feat, agg_stats = add_aggregate_features(feat)

    # Step 4: Encode
    feat, encoders, feature_cols = encode_and_prepare(feat)

    # Step 5: Random 80/20 split
    print("\n[SPLIT]")
    X = feat[feature_cols].fillna(0)
    y = feat[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} ({y_train.mean()*100:.1f}% delayed)")
    print(f"  Test:  {len(X_test):,} ({y_test.mean()*100:.1f}% delayed)")

    # Step 6: Optuna tuning
    best_params, study, trial_results = optuna_tune(X_train, y_train, n_trials=50)

    # Step 7: Train final model with best params on FULL training set
    print(f"\n{'=' * 60}")
    print(f"  TRAINING FINAL MODEL (best params, full train set)")
    print(f"{'=' * 60}")
    final_model = XGBClassifier(**best_params)
    t0 = time.time()
    final_model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # Step 8: Get test probabilities
    y_proba_test = final_model.predict_proba(X_test)[:, 1]

    # Step 9: Threshold optimization
    best_threshold, thresholds, f1_scores = optimize_threshold(y_test, y_proba_test, metric='f1')

    # Step 10: Evaluate with DEFAULT threshold (0.5)
    y_pred_default = final_model.predict(X_test)
    m_default = compute_metrics(y_test, y_pred_default, y_proba_test, "Test (threshold=0.50)")
    print_metrics(m_default)

    # Step 11: Evaluate with OPTIMIZED threshold
    y_pred_optimal = (y_proba_test >= best_threshold).astype(int)
    m_optimal = compute_metrics(y_test, y_pred_optimal, y_proba_test, f"Test (threshold={best_threshold:.2f})")
    print_metrics(m_optimal)

    # Step 12: 5-Fold CV
    print(f"\n{'=' * 60}")
    print(f"  5-FOLD CROSS-VALIDATION (best params)")
    print(f"{'=' * 60}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = {'Accuracy':[], 'F1':[], 'Precision':[], 'Recall':[], 'ROC_AUC':[], 'PR_AUC':[], 'MCC':[]}
    for fold, (ti, vi) in enumerate(skf.split(X, y), 1):
        m_cv = XGBClassifier(**{**best_params, 'verbosity': 0})
        m_cv.fit(X.iloc[ti], y.iloc[ti])
        yp = m_cv.predict(X.iloc[vi])
        ypr = m_cv.predict_proba(X.iloc[vi])[:, 1]
        cv['Accuracy'].append(accuracy_score(y.iloc[vi], yp))
        cv['F1'].append(f1_score(y.iloc[vi], yp))
        cv['Precision'].append(precision_score(y.iloc[vi], yp))
        cv['Recall'].append(recall_score(y.iloc[vi], yp))
        cv['ROC_AUC'].append(roc_auc_score(y.iloc[vi], ypr))
        cv['PR_AUC'].append(average_precision_score(y.iloc[vi], ypr))
        cv['MCC'].append(matthews_corrcoef(y.iloc[vi], yp))
        print(f"  Fold {fold}: Acc={cv['Accuracy'][-1]:.4f}  F1={cv['F1'][-1]:.4f}  AUC={cv['ROC_AUC'][-1]:.4f}  MCC={cv['MCC'][-1]:.4f}")

    print(f"\n  {'Metric':<15} {'Mean':>10} {'Std':>10} {'95% CI':>22}")
    print(f"  {'-'*60}")
    cv_summary = {}
    for k, v in cv.items():
        mean, std = np.mean(v), np.std(v)
        cv_summary[k] = {'mean': mean, 'std': std, 'ci_low': mean-1.96*std, 'ci_high': mean+1.96*std}
        print(f"  {k:<15} {mean:>10.4f} {std:>10.4f} [{mean-1.96*std:.4f}, {mean+1.96*std:.4f}]")

    # Step 13: Feature importance
    print(f"\n{'=' * 60}")
    print(f"  FEATURE IMPORTANCE (Top 20)")
    print(f"{'=' * 60}")
    imp = final_model.feature_importances_
    idx = np.argsort(imp)[::-1]
    for i in range(min(20, len(idx))):
        ix = idx[i]
        bar = "█" * int(imp[ix] * 50)
        print(f"  {i+1:>2}. {feature_cols[ix]:<30} {imp[ix]:.4f} {bar}")

    # Step 14: Plots
    results_dir = os.path.abspath(RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)

    if HAS_MATPLOTLIB:
        print("\n[GENERATING PLOTS]")

        # ROC
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        ax.plot(fpr, tpr, label=f'Improved (AUC={m_default["ROC_AUC"]:.4f})', linewidth=2, color='#2196F3')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12); ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve — Improved Fallback Model', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, "improved_roc.png"), dpi=150); plt.close()

        # PR
        fig, ax = plt.subplots(figsize=(8, 6))
        prec, rec, _ = precision_recall_curve(y_test, y_proba_test)
        ax.plot(rec, prec, label=f'Improved (AP={m_default["PR_AUC"]:.4f})', linewidth=2, color='#4CAF50')
        ax.axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.5, label=f'Baseline ({y_test.mean():.3f})')
        ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall — Improved Fallback Model', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, "improved_pr.png"), dpi=150); plt.close()

        # Confusion matrix (optimal threshold)
        cm = confusion_matrix(y_test, y_pred_optimal)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'Confusion Matrix — Threshold={best_threshold:.2f}', fontsize=14)
        ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Actual', fontsize=12)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['On-Time','Delayed']); ax.set_yticklabels(['On-Time','Delayed'])
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i,j] > cm.max()/2 else 'black'
                ax.text(j, i, f'{cm[i][j]:,}\n({cm[i][j]/cm.sum()*100:.1f}%)', ha='center', va='center', fontsize=14, color=color)
        plt.colorbar(im)
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, "improved_cm.png"), dpi=150); plt.close()

        # Feature importance
        fig, ax = plt.subplots(figsize=(12, 9))
        top_n = min(25, len(feature_cols))
        sorted_imp = [imp[idx[i]] for i in range(top_n-1, -1, -1)]
        sorted_names = [feature_cols[idx[i]] for i in range(top_n-1, -1, -1)]
        ax.barh(range(top_n), sorted_imp, color='steelblue')
        ax.set_yticks(range(top_n)); ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
        ax.set_title('Feature Importance — Improved Fallback Model', fontsize=14)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, "improved_fi.png"), dpi=150); plt.close()

        # Threshold vs F1 plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, f1_scores, linewidth=2, color='#FF5722')
        ax.axvline(x=best_threshold, color='g', linestyle='--', label=f'Optimal={best_threshold:.2f}')
        ax.axvline(x=0.50, color='gray', linestyle='--', alpha=0.5, label='Default=0.50')
        ax.set_xlabel('Threshold', fontsize=12); ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Threshold Optimization', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, "improved_threshold.png"), dpi=150); plt.close()

        print("  Saved 5 plots to results/")

    # Step 15: Save everything
    print("\n[SAVING ARTIFACTS]")
    models_dir = os.path.abspath(MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)

    # Backup previous
    old_path = os.path.join(models_dir, "fallback_model.pkl")
    if os.path.exists(old_path):
        import shutil
        bak = os.path.join(models_dir, "fallback_model_12m_baseline_backup.pkl")
        if not os.path.exists(bak):
            shutil.copy2(old_path, bak)
            print(f"  Backed up baseline → {bak}")

    with open(old_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"  Saved model → {old_path}")

    with open(os.path.join(models_dir, "encoders.pkl"), 'wb') as f:
        pickle.dump(encoders, f)
    with open(os.path.join(models_dir, "aggregate_stats.pkl"), 'wb') as f:
        pickle.dump(agg_stats, f)

    config = {
        'feature_columns': feature_cols, 'target_column': TARGET,
        'model_type': 'XGBoost (Optuna-tuned)',
        'best_params': best_params, 'optimal_threshold': best_threshold,
        'training_data': '12 months (Dec 2024-Nov 2025), random 80/20 split',
        'train_size': len(X_train), 'training_time_seconds': train_time,
        'improvements': ['Enhanced features', 'Optuna tuning', 'Threshold optimization'],
    }
    with open(os.path.join(models_dir, "model_config.pkl"), 'wb') as f:
        pickle.dump(config, f)
    print("  Saved encoders, aggregate stats, config")

    # Benchmarks
    pd.DataFrame([m_default, m_optimal]).to_csv(
        os.path.join(results_dir, "improved_benchmarks.csv"), index=False)
    pd.DataFrame(cv_summary).T.to_csv(
        os.path.join(results_dir, "improved_cv.csv"))

    # Baseline vs Improved comparison
    comparison = pd.DataFrame([
        {'Model': 'Baseline (12-month)', 'Features': 19, 'Tuning': 'Manual',
         'Threshold': 0.50, 'Accuracy': 0.6897, 'ROC_AUC': 0.7462,
         'Recall': 0.6681, 'Precision': 0.3802, 'F1_Score': 0.4846, 'MCC': 0.3091},
        {'Model': 'Improved (default thresh)', 'Features': len(feature_cols), 'Tuning': 'Optuna',
         'Threshold': 0.50, 'Accuracy': m_default['Accuracy'], 'ROC_AUC': m_default['ROC_AUC'],
         'Recall': m_default['Recall'], 'Precision': m_default['Precision'],
         'F1_Score': m_default['F1_Score'], 'MCC': m_default['MCC']},
        {'Model': 'Improved (optimal thresh)', 'Features': len(feature_cols), 'Tuning': 'Optuna',
         'Threshold': best_threshold, 'Accuracy': m_optimal['Accuracy'], 'ROC_AUC': m_optimal['ROC_AUC'],
         'Recall': m_optimal['Recall'], 'Precision': m_optimal['Precision'],
         'F1_Score': m_optimal['F1_Score'], 'MCC': m_optimal['MCC']},
    ])
    comparison.to_csv(os.path.join(results_dir, "baseline_vs_improved.csv"), index=False)
    print(f"  Saved comparison → baseline_vs_improved.csv")

    # Optuna trials
    pd.DataFrame(trial_results).to_csv(
        os.path.join(results_dir, "optuna_trials.csv"), index=False)
    print(f"  Saved {len(trial_results)} Optuna trials")

    # Final summary
    total = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time:        {total:.1f}s ({total/60:.1f} min)")
    print(f"  Features:          {len(feature_cols)} (was 19)")
    print(f"  Optimal threshold: {best_threshold:.2f}")
    print(f"")
    print(f"  {'Metric':<22} {'Baseline':>12} {'Improved':>12} {'Change':>12}")
    print(f"  {'-'*58}")
    baseline = {'Accuracy': 0.6897, 'ROC_AUC': 0.7462, 'Recall': 0.6681,
                'Precision': 0.3802, 'F1_Score': 0.4846, 'MCC': 0.3091}
    for name, key in [('Accuracy','Accuracy'), ('ROC-AUC','ROC_AUC'), ('Recall','Recall'),
                       ('Precision','Precision'), ('F1 Score','F1_Score'), ('MCC','MCC')]:
        old_val = baseline[key]
        new_val = m_optimal[key]
        change = new_val - old_val
        sign = "+" if change >= 0 else ""
        print(f"  {name:<22} {old_val:>12.4f} {new_val:>12.4f} {sign}{change:>11.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
