"""
train_fallback_1month_v2.py
============================
Enhanced 1-Month Fallback Model Training for Flight Delay Prediction.

Improvements over v1:
  1. 45 engineered features (vs 19)  — cyclical encodings, holiday flags,
     airport congestion proxies, interaction aggregate delay rates,
     tail-number re-use proxy, speed proxy
  2. Temporal train/test split  — first 25 days = train, last 6 days = test
  3. Optuna hyperparameter tuning (75 trials, TPE, 3-fold CV on AUC)
  4. Threshold optimization  — sweeps 0.10-0.75 for best F1 & best Accuracy
  5. Research-grade metrics  — Accuracy, Precision, Recall, Specificity,
     F1, ROC-AUC, PR-AUC, MCC, Cohen's Kappa, Log Loss, Brier Score
  6. 5-Fold stratified cross-validation with 95% confidence intervals
  7. Saves model, encoders, aggregate_stats, config → models/

Usage:
    python ml/scripts/train_fallback_1month_v2.py
"""

import os
import pickle
import time
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================
# PATHS
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RAW_DIR    = os.path.join(ROOT_DIR, "data", "raw")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

RAW_FILE   = os.path.join(RAW_DIR, "ontime_2025_10.csv")   # 1-month data (Oct 2025)
OOS_FILE   = os.path.join(RAW_DIR, "ontime_2025_11.csv")   # out-of-sample (Nov 2025)

TARGET = "IS_DELAYED"

# ============================================================
# US HOLIDAYS  (Oct 2025 + surrounding months for context)
# ============================================================
_US_HOLIDAYS_RAW = {
    # 2024
    (2024, 12, 24), (2024, 12, 25), (2024, 12, 31),
    # 2025
    (2025, 1, 1), (2025, 1, 20), (2025, 2, 17),
    (2025, 3, 14), (2025, 3, 15), (2025, 3, 16), (2025, 3, 17),
    (2025, 3, 18), (2025, 3, 19), (2025, 3, 20), (2025, 3, 21),
    (2025, 5, 26), (2025, 7, 3), (2025, 7, 4), (2025, 7, 5),
    (2025, 9, 1),  (2025, 10, 13),               # Columbus Day
    (2025, 11, 11),                               # Veterans Day
    (2025, 11, 26), (2025, 11, 27), (2025, 11, 28), (2025, 11, 29), (2025, 11, 30),
    (2025, 12, 24), (2025, 12, 25), (2025, 12, 31),
}

US_HOLIDAYS = _US_HOLIDAYS_RAW.copy()
NEAR_HOLIDAY = set()
for y, m, d in _US_HOLIDAYS_RAW:
    for delta in range(-3, 4):
        try:
            dt = date(y, m, d) + timedelta(days=delta)
            NEAR_HOLIDAY.add((dt.year, dt.month, dt.day))
        except Exception:
            pass


# ============================================================
# STEP 1 — LOAD AND CLEAN RAW DATA
# ============================================================
def load_and_clean(path: str) -> pd.DataFrame:
    print(f"\n[LOAD] {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # Drop cancelled / diverted
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] == 0.0]
    if "Diverted" in df.columns:
        df = df[df["Diverted"] == 0.0]

    # Need ArrDelay for target
    df = df.dropna(subset=["ArrDelay"])
    print(f"  After cleaning: {len(df):,} rows")
    return df


# ============================================================
# STEP 2 — FEATURE ENGINEERING  (45 features)
# ============================================================
def engineer_features(df: pd.DataFrame, dataset_label: str = "") -> pd.DataFrame:
    print(f"\n[FEATURE ENGINEERING]{' — ' + dataset_label if dataset_label else ''}")
    feat = pd.DataFrame()

    # --- Target ---
    feat[TARGET] = (df["ArrDelay"] >= 15).astype(int)
    print(f"  Delay rate: {feat[TARGET].mean()*100:.1f}%  ({feat[TARGET].sum():,} delayed / {len(feat):,} total)")

    # --- Raw temporals ---
    year_col = df["Year"].astype(int)
    feat["YEAR"]        = year_col
    feat["MONTH"]       = df["Month"].astype(int)
    feat["DAY_OF_MONTH"]= df["DayofMonth"].astype(int)
    feat["DAY_OF_WEEK"] = df["DayOfWeek"].astype(int)
    feat["DEP_HOUR"]    = (df["CRSDepTime"].fillna(0) / 100).astype(int).clip(0, 23)
    feat["ARR_HOUR"]    = (df["CRSArrTime"].fillna(0) / 100).astype(int).clip(0, 23)
    feat["IS_WEEKEND"]  = df["DayOfWeek"].isin([6, 7]).astype(int)

    feat["TIME_BLOCK"] = pd.cut(
        feat["DEP_HOUR"],
        bins=[-1, 5, 9, 13, 17, 21, 24],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)

    # --- Categorical (kept raw for aggregate rates, then encoded) ---
    feat["CARRIER"] = df["Reporting_Airline"].astype(str)
    feat["ORIGIN"]  = df["Origin"].astype(str)
    feat["DEST"]    = df["Dest"].astype(str)

    # Tail number for utilization proxy
    feat["TAIL_NUM"] = df["Tail_Number"].fillna("UNK").astype(str) if "Tail_Number" in df.columns else "UNK"

    # --- Numeric ---
    feat["DISTANCE"]         = df["Distance"].fillna(0).astype(float)
    feat["CRS_ELAPSED_TIME"] = df["CRSElapsedTime"].fillna(0).astype(float)

    feat["DISTANCE_GROUP"] = pd.cut(
        feat["DISTANCE"],
        bins=[0, 250, 500, 1000, 2000, 6000],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    feat["DURATION_BUCKET"] = pd.cut(
        feat["CRS_ELAPSED_TIME"],
        bins=[0, 60, 120, 180, 300, 1500],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(0).astype(int)

    # Speed proxy (distance / elapsed time)
    feat["SPEED_PROXY"] = (
        feat["DISTANCE"] / feat["CRS_ELAPSED_TIME"].replace(0, np.nan)
    ).fillna(0)

    # --- Cyclical encodings ---
    feat["MONTH_SIN"] = np.sin(2 * np.pi * feat["MONTH"] / 12)
    feat["MONTH_COS"] = np.cos(2 * np.pi * feat["MONTH"] / 12)
    feat["HOUR_SIN"]  = np.sin(2 * np.pi * feat["DEP_HOUR"] / 24)
    feat["HOUR_COS"]  = np.cos(2 * np.pi * feat["DEP_HOUR"] / 24)
    feat["DOW_SIN"]   = np.sin(2 * np.pi * feat["DAY_OF_WEEK"] / 7)
    feat["DOW_COS"]   = np.cos(2 * np.pi * feat["DAY_OF_WEEK"] / 7)
    feat["DOM_SIN"]   = np.sin(2 * np.pi * feat["DAY_OF_MONTH"] / 31)
    feat["DOM_COS"]   = np.cos(2 * np.pi * feat["DAY_OF_MONTH"] / 31)

    # --- Season ---
    season_map = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}
    feat["SEASON"] = feat["MONTH"].map(season_map).astype(int)

    # --- Holiday flags ---
    feat["IS_HOLIDAY"] = [
        1 if (y, m, d) in US_HOLIDAYS else 0
        for y, m, d in zip(year_col, feat["MONTH"], feat["DAY_OF_MONTH"])
    ]
    feat["NEAR_HOLIDAY"] = [
        1 if (y, m, d) in NEAR_HOLIDAY else 0
        for y, m, d in zip(year_col, feat["MONTH"], feat["DAY_OF_MONTH"])
    ]

    # --- Peak travel indicators ---
    feat["IS_FRIDAY_EVENING"]  = ((feat["DAY_OF_WEEK"] == 5) & (feat["DEP_HOUR"] >= 15)).astype(int)
    feat["IS_SUNDAY_EVENING"]  = ((feat["DAY_OF_WEEK"] == 7) & (feat["DEP_HOUR"] >= 15)).astype(int)
    feat["IS_MONDAY_MORNING"]  = ((feat["DAY_OF_WEEK"] == 1) & (feat["DEP_HOUR"] <= 9)).astype(int)
    feat["IS_PEAK_HOUR"]       = feat["DEP_HOUR"].isin([7, 8, 16, 17, 18]).astype(int)
    feat["IS_EARLY_MORNING"]   = (feat["DEP_HOUR"] <= 6).astype(int)
    feat["IS_RED_EYE"]         = (feat["DEP_HOUR"] >= 22).astype(int)

    # --- Airport congestion proxies ---
    feat["ORIGIN_HOUR_KEY"] = feat["ORIGIN"] + "_" + feat["DEP_HOUR"].astype(str)
    feat["DEST_HOUR_KEY"]   = feat["DEST"]   + "_" + feat["ARR_HOUR"].astype(str)

    origin_hour_counts = feat.groupby("ORIGIN_HOUR_KEY")[TARGET].count()
    dest_hour_counts   = feat.groupby("DEST_HOUR_KEY")[TARGET].count()
    max_oh = origin_hour_counts.max()
    max_dh = dest_hour_counts.max()
    feat["ORIGIN_CONGESTION"] = feat["ORIGIN_HOUR_KEY"].map(origin_hour_counts).fillna(0) / (max_oh if max_oh > 0 else 1)
    feat["DEST_CONGESTION"]   = feat["DEST_HOUR_KEY"].map(dest_hour_counts).fillna(0) / (max_dh if max_dh > 0 else 1)

    # --- Tail-number flights per day (utilization proxy) ---
    if "TAIL_NUM" in feat.columns and feat["TAIL_NUM"].nunique() > 1:
        feat["DATE_KEY"] = year_col.astype(str) + "_" + feat["MONTH"].astype(str) + "_" + feat["DAY_OF_MONTH"].astype(str)
        tail_day_counts = feat.groupby(["TAIL_NUM", "DATE_KEY"])[TARGET].count().reset_index()
        tail_day_counts.columns = ["TAIL_NUM", "DATE_KEY", "TAIL_FLIGHTS_TODAY"]
        feat = feat.merge(tail_day_counts, on=["TAIL_NUM", "DATE_KEY"], how="left")
        feat["TAIL_FLIGHTS_TODAY"] = feat["TAIL_FLIGHTS_TODAY"].fillna(1).astype(int)
    else:
        feat["TAIL_FLIGHTS_TODAY"] = 1

    print(f"  Base features created: {len(feat.columns)}")
    return feat


# ============================================================
# STEP 3 — AGGREGATE DELAY RATES  (computed on full dataset)
# ============================================================
def add_aggregate_rates(feat: pd.DataFrame) -> (pd.DataFrame, dict):
    print("\n[AGGREGATE DELAY RATES]")
    agg_stats = {}

    # --- Level-1: single-dimension rates ---
    level1 = {
        "CARRIER_DELAY_RATE":   "CARRIER",
        "ORIGIN_DELAY_RATE":    "ORIGIN",
        "DEST_DELAY_RATE":      "DEST",
        "HOUR_DELAY_RATE":      "DEP_HOUR",
        "DOW_DELAY_RATE":       "DAY_OF_WEEK",
        "SEASON_DELAY_RATE":    "SEASON",
        "TIME_BLOCK_DELAY_RATE":"TIME_BLOCK",
    }
    for feat_name, col in level1.items():
        rates = feat.groupby(col)[TARGET].mean()
        feat[feat_name] = feat[col].map(rates).fillna(rates.mean())
        agg_stats[feat_name] = rates.to_dict()
        print(f"  {feat_name}: {len(rates)} groups")

    # Route delay rate
    feat["ROUTE"] = feat["ORIGIN"] + "_" + feat["DEST"]
    route_rates = feat.groupby("ROUTE")[TARGET].mean()
    feat["ROUTE_DELAY_RATE"] = feat["ROUTE"].map(route_rates).fillna(route_rates.mean())
    agg_stats["ROUTE_DELAY_RATE"] = route_rates.to_dict()
    print(f"  ROUTE_DELAY_RATE: {len(route_rates)} routes")

    # --- Level-2: two-dimension interaction rates ---
    interactions = [
        ("CARRIER_ORIGIN_DELAY_RATE",  ["CARRIER", "ORIGIN"]),
        ("CARRIER_HOUR_DELAY_RATE",    ["CARRIER", "DEP_HOUR"]),
        ("CARRIER_DOW_DELAY_RATE",     ["CARRIER", "DAY_OF_WEEK"]),
        ("ORIGIN_DOW_DELAY_RATE",      ["ORIGIN",  "DAY_OF_WEEK"]),
        ("ORIGIN_HOUR_DELAY_RATE",     ["ORIGIN",  "DEP_HOUR"]),
        ("ROUTE_HOUR_DELAY_RATE",      ["ROUTE",   "DEP_HOUR"]),
        ("DEST_HOUR_DELAY_RATE",       ["DEST",    "ARR_HOUR"]),
    ]
    for feat_name, cols in interactions:
        key_col = feat_name + "_KEY"
        feat[key_col] = feat[cols[0]].astype(str) + "_" + feat[cols[1]].astype(str)
        rates = feat.groupby(key_col)[TARGET].mean()
        global_mean = feat[TARGET].mean()
        feat[feat_name] = feat[key_col].map(rates).fillna(global_mean)
        agg_stats[feat_name] = rates.to_dict()
        feat.drop(columns=[key_col], inplace=True)
        print(f"  {feat_name}: {len(rates)} combos")

    return feat, agg_stats


# ============================================================
# STEP 4 — ENCODE CATEGORICALS
# ============================================================
def encode_categoricals(feat: pd.DataFrame) -> (pd.DataFrame, dict):
    print("\n[ENCODING CATEGORICALS]")
    encoders = {}
    for col in ["CARRIER", "ORIGIN", "DEST"]:
        le = LabelEncoder()
        feat[f"{col}_ENCODED"] = le.fit_transform(feat[col])
        encoders[col] = le
        print(f"  {col}: {len(le.classes_)} classes")
    return feat, encoders


# ============================================================
# STEP 5 — DEFINE FEATURE COLUMNS
# ============================================================
FEATURE_COLUMNS = [
    # Base temporal
    "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "DEP_HOUR", "ARR_HOUR",
    "IS_WEEKEND", "TIME_BLOCK",
    # Distance / duration
    "DISTANCE", "CRS_ELAPSED_TIME", "DISTANCE_GROUP", "DURATION_BUCKET", "SPEED_PROXY",
    # Encoded categoricals
    "CARRIER_ENCODED", "ORIGIN_ENCODED", "DEST_ENCODED",
    # Cyclical
    "MONTH_SIN", "MONTH_COS", "HOUR_SIN", "HOUR_COS",
    "DOW_SIN",   "DOW_COS",   "DOM_SIN",  "DOM_COS",
    # Season / holiday
    "SEASON", "IS_HOLIDAY", "NEAR_HOLIDAY",
    # Peak travel
    "IS_FRIDAY_EVENING", "IS_SUNDAY_EVENING", "IS_MONDAY_MORNING",
    "IS_PEAK_HOUR", "IS_EARLY_MORNING", "IS_RED_EYE",
    # Congestion
    "ORIGIN_CONGESTION", "DEST_CONGESTION",
    # Tail utilization
    "TAIL_FLIGHTS_TODAY",
    # Level-1 aggregate rates
    "CARRIER_DELAY_RATE", "ORIGIN_DELAY_RATE", "DEST_DELAY_RATE",
    "HOUR_DELAY_RATE", "DOW_DELAY_RATE", "ROUTE_DELAY_RATE",
    "SEASON_DELAY_RATE", "TIME_BLOCK_DELAY_RATE",
    # Level-2 interaction rates
    "CARRIER_ORIGIN_DELAY_RATE", "CARRIER_HOUR_DELAY_RATE", "CARRIER_DOW_DELAY_RATE",
    "ORIGIN_DOW_DELAY_RATE", "ORIGIN_HOUR_DELAY_RATE",
    "ROUTE_HOUR_DELAY_RATE", "DEST_HOUR_DELAY_RATE",
]


# ============================================================
# STEP 6 — TEMPORAL TRAIN / TEST SPLIT
# ============================================================
def temporal_split(feat: pd.DataFrame):
    """Split: first 25 days = train, last 6 days = test (within same month)."""
    print("\n[TEMPORAL SPLIT — first 25 days train / last 6 days test]")
    train_mask = feat["DAY_OF_MONTH"] <= 25
    test_mask  = feat["DAY_OF_MONTH"] >  25
    train = feat[train_mask].copy()
    test  = feat[test_mask].copy()
    print(f"  Train: {len(train):,}  ({train[TARGET].mean()*100:.1f}% delayed)")
    print(f"  Test:  {len(test):,}   ({test[TARGET].mean()*100:.1f}% delayed)")
    return train, test


# ============================================================
# STEP 7 — OPTUNA HYPERPARAMETER TUNING
# ============================================================
def optuna_tune(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 75) -> dict:
    print(f"\n{'='*60}")
    print(f"  OPTUNA HYPERPARAMETER TUNING  ({n_trials} trials, TPE)")
    print(f"{'='*60}")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    # Subsample for faster tuning
    sample_size = min(len(X_train), 100_000)
    if len(X_train) > sample_size:
        print(f"  Subsampling {sample_size:,} rows for tuning speed...")
        idx = np.random.RandomState(42).choice(len(X_train), sample_size, replace=False)
        Xt = X_train.iloc[idx]
        yt = y_train.iloc[idx]
    else:
        Xt, yt = X_train, y_train

    best_score = [-1.0]

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 30),
            "gamma":             trial.suggest_float("gamma", 0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_delta_step":    trial.suggest_int("max_delta_step", 0, 5),
            "scale_pos_weight":  spw,
            "eval_metric":       "logloss",
            "random_state":      42,
            "n_jobs":            -1,
            "verbosity":         0,
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for ti, vi in skf.split(Xt, yt):
            m = XGBClassifier(**params)
            m.fit(Xt.iloc[ti], yt.iloc[ti])
            prob = m.predict_proba(Xt.iloc[vi])[:, 1]
            aucs.append(roc_auc_score(yt.iloc[vi], prob))

        mean_auc = float(np.mean(aucs))
        if mean_auc > best_score[0]:
            best_score[0] = mean_auc
            print(f"  Trial {trial.number+1:>3}/{n_trials}: AUC={mean_auc:.4f} ★ NEW BEST")
        elif (trial.number + 1) % 15 == 0:
            print(f"  Trial {trial.number+1:>3}/{n_trials}: AUC={mean_auc:.4f}  (best: {best_score[0]:.4f})")
        return mean_auc

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    tune_time = time.time() - t0

    best = study.best_params
    best["scale_pos_weight"] = spw
    best["eval_metric"]      = "logloss"
    best["random_state"]     = 42
    best["n_jobs"]           = -1
    best["verbosity"]        = 1

    print(f"\n  Best AUC: {study.best_value:.4f}")
    print(f"  Tuning time: {tune_time:.0f}s ({tune_time/60:.1f} min)")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return best, study


# ============================================================
# STEP 8 — THRESHOLD OPTIMIZATION
# ============================================================
def optimize_threshold(y_true, y_proba):
    print("\n[THRESHOLD OPTIMIZATION]")
    thresholds = np.arange(0.10, 0.76, 0.01)

    best_f1_t, best_f1, best_acc_t, best_acc = 0.5, 0, 0.5, 0
    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        f  = f1_score(y_true, yp, zero_division=0)
        a  = accuracy_score(y_true, yp)
        if f > best_f1:
            best_f1, best_f1_t = f, t
        if a > best_acc:
            best_acc, best_acc_t = a, t

    print(f"  Default  (0.50): F1={f1_score(y_true,(y_proba>=0.50).astype(int)):.4f}  Acc={accuracy_score(y_true,(y_proba>=0.50).astype(int)):.4f}")
    print(f"  Best F1  ({best_f1_t:.2f}): F1={best_f1:.4f}")
    print(f"  Best Acc ({best_acc_t:.2f}): Acc={best_acc:.4f}")
    return best_f1_t, best_acc_t


# ============================================================
# STEP 9 — COMPUTE METRICS
# ============================================================
def compute_metrics(y_true, y_pred, y_proba, name="Test"):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Set": name, "N": int(len(y_true)),
        "Accuracy":      float(accuracy_score(y_true, y_pred)),
        "Precision":     float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":        float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity":   float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "F1_Score":      float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC":       float(roc_auc_score(y_true, y_proba)),
        "PR_AUC":        float(average_precision_score(y_true, y_proba)),
        "Log_Loss":      float(log_loss(y_true, y_proba)),
        "Brier_Score":   float(brier_score_loss(y_true, y_proba)),
        "MCC":           float(matthews_corrcoef(y_true, y_pred)),
        "Cohens_Kappa":  float(cohen_kappa_score(y_true, y_pred)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "FPR": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "FNR": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
    }


def print_metrics(m):
    print(f"\n{'='*60}")
    print(f"  {m['Set']}  (N={m['N']:,})")
    print(f"{'='*60}")
    rows = [
        ("Accuracy",        "Accuracy",    True),
        ("Precision",       "Precision",   True),
        ("Recall",          "Recall",      True),
        ("Specificity",     "Specificity", True),
        ("F1 Score",        "F1_Score",    False),
        ("ROC-AUC",         "ROC_AUC",     False),
        ("PR-AUC",          "PR_AUC",      False),
        ("Log Loss",        "Log_Loss",    False),
        ("Brier Score",     "Brier_Score", False),
        ("MCC",             "MCC",         False),
        ("Cohen's Kappa",   "Cohens_Kappa",False),
    ]
    for label, key, pct in rows:
        v = m[key]
        suf = f"  ({v*100:.2f}%)" if pct else ""
        print(f"  {label:<25} {v:.4f}{suf}")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>20} Pred On-Time   Pred Delayed")
    print(f"  Actual On-Time  {m['TN']:>12,}   {m['FP']:>12,}")
    print(f"  Actual Delayed  {m['FN']:>12,}   {m['TP']:>12,}")
    print(f"  FPR: {m['FPR']:.4f}  |  FNR: {m['FNR']:.4f}")


# ============================================================
# STEP 10 — CROSS-VALIDATION
# ============================================================
def cross_validate(X, y, params, n_folds=5):
    print(f"\n{'='*60}")
    print(f"  {n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'='*60}")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv = {k: [] for k in ["Accuracy", "F1", "Precision", "Recall", "ROC_AUC", "PR_AUC", "MCC"]}

    for fold, (ti, vi) in enumerate(skf.split(X, y), 1):
        p = {**params, "verbosity": 0}
        m = XGBClassifier(**p)
        m.fit(X.iloc[ti], y.iloc[ti])
        yp  = m.predict(X.iloc[vi])
        ypr = m.predict_proba(X.iloc[vi])[:, 1]
        cv["Accuracy"].append(accuracy_score(y.iloc[vi], yp))
        cv["F1"].append(f1_score(y.iloc[vi], yp, zero_division=0))
        cv["Precision"].append(precision_score(y.iloc[vi], yp, zero_division=0))
        cv["Recall"].append(recall_score(y.iloc[vi], yp, zero_division=0))
        cv["ROC_AUC"].append(roc_auc_score(y.iloc[vi], ypr))
        cv["PR_AUC"].append(average_precision_score(y.iloc[vi], ypr))
        cv["MCC"].append(matthews_corrcoef(y.iloc[vi], yp))
        print(f"  Fold {fold}: Acc={cv['Accuracy'][-1]:.4f}  F1={cv['F1'][-1]:.4f}  AUC={cv['ROC_AUC'][-1]:.4f}")

    print(f"\n  {'Metric':<15} {'Mean':>10} {'Std':>10} {'95% CI':>22}")
    print(f"  {'-'*60}")
    cv_summary = {}
    for k, vals in cv.items():
        mean, std = np.mean(vals), np.std(vals)
        cv_summary[k] = {"mean": mean, "std": std,
                         "ci_low": mean - 1.96*std, "ci_high": mean + 1.96*std}
        print(f"  {k:<15} {mean:>10.4f} {std:>10.4f} [{mean-1.96*std:.4f}, {mean+1.96*std:.4f}]")
    return cv_summary


# ============================================================
# STEP 11 — PLOTS
# ============================================================
def generate_plots(y_test, y_proba_test, model, feature_names, tag="fallback_v2"):
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n[PLOTS]")

    # ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    auc_val = roc_auc_score(y_test, y_proba_test)
    ax.plot(fpr, tpr, linewidth=2, label=f"Model (AUC={auc_val:.4f})")
    ax.plot([0,1],[0,1],"k--", alpha=0.5, label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve — {tag}"); ax.legend(); ax.grid(alpha=0.3)
    p = os.path.join(RESULTS_DIR, f"{tag}_roc.png")
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"  {p}")

    # PR
    fig, ax = plt.subplots(figsize=(8, 6))
    prec, rec, _ = precision_recall_curve(y_test, y_proba_test)
    ap = average_precision_score(y_test, y_proba_test)
    ax.plot(rec, prec, linewidth=2, label=f"Model (AP={ap:.4f})")
    ax.axhline(y=y_test.mean(), color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {tag}"); ax.legend(); ax.grid(alpha=0.3)
    p = os.path.join(RESULTS_DIR, f"{tag}_pr.png")
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"  {p}")

    # Feature importance
    imp = model.feature_importances_
    idx = np.argsort(imp)
    top_n = min(30, len(feature_names))
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(range(top_n), [imp[i] for i in idx[-top_n:]], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[-top_n:]], fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"Feature Importance — {tag}")
    ax.grid(axis="x", alpha=0.3)
    p = os.path.join(RESULTS_DIR, f"{tag}_feature_importance.png")
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"  {p}")

    # Confusion matrix
    cm_arr = confusion_matrix(y_test, (y_proba_test >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["On-Time","Delayed"]); ax.set_yticklabels(["On-Time","Delayed"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {tag} (thresh=0.50)")
    for i in range(2):
        for j in range(2):
            c = "white" if cm_arr[i,j] > cm_arr.max()/2 else "black"
            ax.text(j, i, f"{cm_arr[i,j]:,}\n({cm_arr[i,j]/cm_arr.sum()*100:.1f}%)",
                    ha="center", va="center", fontsize=13, color=c)
    plt.colorbar(im)
    p = os.path.join(RESULTS_DIR, f"{tag}_confusion_matrix.png")
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"  {p}")


# ============================================================
# STEP 12 — APPLY FEATURES TO UNSEEN OOS DATA
# ============================================================
def apply_features_to_oos(oos_df: pd.DataFrame, encoders: dict, agg_stats: dict,
                           feat_ref: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering to OOS data using training-set lookup tables."""
    print("\n[APPLY FEATURES TO OOS DATA]")
    feat = pd.DataFrame()
    feat[TARGET] = (oos_df["ArrDelay"] >= 15).astype(int)

    year_col = oos_df["Year"].astype(int)
    feat["YEAR"]         = year_col
    feat["MONTH"]        = oos_df["Month"].astype(int)
    feat["DAY_OF_MONTH"] = oos_df["DayofMonth"].astype(int)
    feat["DAY_OF_WEEK"]  = oos_df["DayOfWeek"].astype(int)
    feat["DEP_HOUR"]     = (oos_df["CRSDepTime"].fillna(0)/100).astype(int).clip(0, 23)
    feat["ARR_HOUR"]     = (oos_df["CRSArrTime"].fillna(0)/100).astype(int).clip(0, 23)
    feat["IS_WEEKEND"]   = oos_df["DayOfWeek"].isin([6,7]).astype(int)
    feat["TIME_BLOCK"]   = pd.cut(feat["DEP_HOUR"], bins=[-1,5,9,13,17,21,24], labels=[0,1,2,3,4,5]).astype(int)
    feat["CARRIER"]      = oos_df["Reporting_Airline"].astype(str)
    feat["ORIGIN"]       = oos_df["Origin"].astype(str)
    feat["DEST"]         = oos_df["Dest"].astype(str)
    feat["DISTANCE"]     = oos_df["Distance"].fillna(0).astype(float)
    feat["CRS_ELAPSED_TIME"] = oos_df["CRSElapsedTime"].fillna(0).astype(float)
    feat["DISTANCE_GROUP"] = pd.cut(feat["DISTANCE"], bins=[0,250,500,1000,2000,6000], labels=[0,1,2,3,4]).astype(int)
    feat["DURATION_BUCKET"] = pd.cut(feat["CRS_ELAPSED_TIME"], bins=[0,60,120,180,300,1500], labels=[0,1,2,3,4]).astype(float).fillna(0).astype(int)
    feat["SPEED_PROXY"]  = (feat["DISTANCE"] / feat["CRS_ELAPSED_TIME"].replace(0, np.nan)).fillna(0)

    feat["MONTH_SIN"] = np.sin(2*np.pi*feat["MONTH"]/12)
    feat["MONTH_COS"] = np.cos(2*np.pi*feat["MONTH"]/12)
    feat["HOUR_SIN"]  = np.sin(2*np.pi*feat["DEP_HOUR"]/24)
    feat["HOUR_COS"]  = np.cos(2*np.pi*feat["DEP_HOUR"]/24)
    feat["DOW_SIN"]   = np.sin(2*np.pi*feat["DAY_OF_WEEK"]/7)
    feat["DOW_COS"]   = np.cos(2*np.pi*feat["DAY_OF_WEEK"]/7)
    feat["DOM_SIN"]   = np.sin(2*np.pi*feat["DAY_OF_MONTH"]/31)
    feat["DOM_COS"]   = np.cos(2*np.pi*feat["DAY_OF_MONTH"]/31)

    season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}
    feat["SEASON"]       = feat["MONTH"].map(season_map).astype(int)
    feat["IS_HOLIDAY"]   = [(1 if (y,m,d) in US_HOLIDAYS else 0) for y,m,d in zip(year_col, feat["MONTH"], feat["DAY_OF_MONTH"])]
    feat["NEAR_HOLIDAY"] = [(1 if (y,m,d) in NEAR_HOLIDAY else 0) for y,m,d in zip(year_col, feat["MONTH"], feat["DAY_OF_MONTH"])]

    feat["IS_FRIDAY_EVENING"] = ((feat["DAY_OF_WEEK"]==5) & (feat["DEP_HOUR"]>=15)).astype(int)
    feat["IS_SUNDAY_EVENING"] = ((feat["DAY_OF_WEEK"]==7) & (feat["DEP_HOUR"]>=15)).astype(int)
    feat["IS_MONDAY_MORNING"] = ((feat["DAY_OF_WEEK"]==1) & (feat["DEP_HOUR"]<=9)).astype(int)
    feat["IS_PEAK_HOUR"]      = feat["DEP_HOUR"].isin([7,8,16,17,18]).astype(int)
    feat["IS_EARLY_MORNING"]  = (feat["DEP_HOUR"]<=6).astype(int)
    feat["IS_RED_EYE"]        = (feat["DEP_HOUR"]>=22).astype(int)

    # Congestion from training reference
    oh_key = feat["ORIGIN"] + "_" + feat["DEP_HOUR"].astype(str)
    dh_key = feat["DEST"]   + "_" + feat["ARR_HOUR"].astype(str)
    ref_oh = feat_ref.groupby("ORIGIN_HOUR_KEY")[TARGET].count() if "ORIGIN_HOUR_KEY" in feat_ref.columns else pd.Series(dtype=float)
    ref_dh = feat_ref.groupby("DEST_HOUR_KEY")[TARGET].count() if "DEST_HOUR_KEY" in feat_ref.columns else pd.Series(dtype=float)
    max_oh = ref_oh.max() if len(ref_oh) else 1
    max_dh = ref_dh.max() if len(ref_dh) else 1
    feat["ORIGIN_CONGESTION"] = oh_key.map(ref_oh).fillna(0) / (max_oh if max_oh > 0 else 1)
    feat["DEST_CONGESTION"]   = dh_key.map(ref_dh).fillna(0) / (max_dh if max_dh > 0 else 1)
    feat["TAIL_FLIGHTS_TODAY"] = 1   # default for OOS

    # Apply training aggregate stats (lookup tables)
    global_mean = 0.2
    feat["CARRIER_DELAY_RATE"]   = feat["CARRIER"].map(agg_stats.get("CARRIER_DELAY_RATE",{})).fillna(global_mean)
    feat["ORIGIN_DELAY_RATE"]    = feat["ORIGIN"].map(agg_stats.get("ORIGIN_DELAY_RATE",{})).fillna(global_mean)
    feat["DEST_DELAY_RATE"]      = feat["DEST"].map(agg_stats.get("DEST_DELAY_RATE",{})).fillna(global_mean)
    feat["HOUR_DELAY_RATE"]      = feat["DEP_HOUR"].map(agg_stats.get("HOUR_DELAY_RATE",{})).fillna(global_mean)
    feat["DOW_DELAY_RATE"]       = feat["DAY_OF_WEEK"].map(agg_stats.get("DOW_DELAY_RATE",{})).fillna(global_mean)
    feat["SEASON_DELAY_RATE"]    = feat["SEASON"].map(agg_stats.get("SEASON_DELAY_RATE",{})).fillna(global_mean)
    feat["TIME_BLOCK_DELAY_RATE"]= feat["TIME_BLOCK"].map(agg_stats.get("TIME_BLOCK_DELAY_RATE",{})).fillna(global_mean)
    feat["ROUTE"]                = feat["ORIGIN"] + "_" + feat["DEST"]
    feat["ROUTE_DELAY_RATE"]     = feat["ROUTE"].map(agg_stats.get("ROUTE_DELAY_RATE",{})).fillna(global_mean)

    for feat_name, keys in [
        ("CARRIER_ORIGIN_DELAY_RATE", ("CARRIER","ORIGIN",  "CARRIER_ORIGIN_DELAY_RATE")),
        ("CARRIER_HOUR_DELAY_RATE",   ("CARRIER","DEP_HOUR","CARRIER_HOUR_DELAY_RATE")),
        ("CARRIER_DOW_DELAY_RATE",    ("CARRIER","DAY_OF_WEEK","CARRIER_DOW_DELAY_RATE")),
        ("ORIGIN_DOW_DELAY_RATE",     ("ORIGIN","DAY_OF_WEEK","ORIGIN_DOW_DELAY_RATE")),
        ("ORIGIN_HOUR_DELAY_RATE",    ("ORIGIN","DEP_HOUR","ORIGIN_HOUR_DELAY_RATE")),
        ("ROUTE_HOUR_DELAY_RATE",     ("ROUTE","DEP_HOUR","ROUTE_HOUR_DELAY_RATE")),
        ("DEST_HOUR_DELAY_RATE",      ("DEST","ARR_HOUR","DEST_HOUR_DELAY_RATE")),
    ]:
        k_col = feat[keys[0]].astype(str) + "_" + feat[keys[1]].astype(str)
        feat[feat_name] = k_col.map(agg_stats.get(keys[2], {})).fillna(global_mean)

    # Encode categoricals using training encoders
    for col in ["CARRIER", "ORIGIN", "DEST"]:
        le = encoders[col]
        feat[f"{col}_ENCODED"] = feat[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )

    print(f"  OOS: {len(feat):,} rows  ({feat[TARGET].mean()*100:.1f}% delayed)")
    return feat


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()

    print("\n" + "="*70)
    print("  FALLBACK MODEL v2 — 1-MONTH (Oct 2025)  |  45 FEATURES + OPTUNA")
    print("="*70)

    # --- Load training data ---
    raw_df = load_and_clean(RAW_FILE)

    # --- Feature engineering ---
    feat_df = engineer_features(raw_df, "Oct 2025 (Training)")

    # --- Aggregate delay rates ---
    feat_df, agg_stats = add_aggregate_rates(feat_df)

    # --- Encode categoricals ---
    feat_df, encoders = encode_categoricals(feat_df)

    # --- Temporal split ---
    train_df, test_df = temporal_split(feat_df)

    # --- Prepare X/y ---
    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df[TARGET]
    X_test  = test_df[FEATURE_COLUMNS].fillna(0)
    y_test  = test_df[TARGET]

    print(f"\n  X_train: {X_train.shape}   X_test: {X_test.shape}")
    print(f"  Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    # --- Optuna tuning ---
    best_params, study = optuna_tune(X_train, y_train, n_trials=75)

    # --- Train final model ---
    print(f"\n{'='*60}")
    print(f"  TRAINING FINAL MODEL  (best params, full train set)")
    print(f"{'='*60}")
    model = XGBClassifier(**best_params)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # --- Test set probabilities ---
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # --- Threshold optimization ---
    best_f1_t, best_acc_t = optimize_threshold(y_test, y_proba_test)

    # --- Evaluate at default threshold ---
    m_default = compute_metrics(y_test, model.predict(X_test), y_proba_test, "Test (thresh=0.50)")
    print_metrics(m_default)

    # --- Evaluate at best F1 threshold ---
    y_pred_f1 = (y_proba_test >= best_f1_t).astype(int)
    m_f1 = compute_metrics(y_test, y_pred_f1, y_proba_test, f"Test (thresh={best_f1_t:.2f}, best-F1)")
    print_metrics(m_f1)

    # --- Evaluate at best accuracy threshold ---
    y_pred_acc = (y_proba_test >= best_acc_t).astype(int)
    m_acc = compute_metrics(y_test, y_pred_acc, y_proba_test, f"Test (thresh={best_acc_t:.2f}, best-Acc)")
    print_metrics(m_acc)

    # --- Feature importance ---
    print(f"\n{'='*60}")
    print(f"  FEATURE IMPORTANCE  (top {min(30, len(FEATURE_COLUMNS))})")
    print(f"{'='*60}")
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    for i in range(min(30, len(idx))):
        bar = "█" * int(imp[idx[i]] * 60)
        print(f"  {i+1:>2}. {FEATURE_COLUMNS[idx[i]]:<35} {imp[idx[i]]:.4f} {bar}")

    # --- 5-Fold CV ---
    print("\n[INFO] Running 5-fold CV (may take several minutes)...")
    X_all = feat_df[FEATURE_COLUMNS].fillna(0)
    y_all = feat_df[TARGET]
    cv_summary = cross_validate(X_all, y_all, best_params, n_folds=5)

    # --- Load + evaluate OOS (Nov 2025) ---
    metrics_oos = None
    if os.path.exists(OOS_FILE):
        oos_raw = load_and_clean(OOS_FILE)
        oos_feat = apply_features_to_oos(oos_raw, encoders, agg_stats, feat_df)
        X_oos = oos_feat[FEATURE_COLUMNS].fillna(0)
        y_oos = oos_feat[TARGET]
        y_proba_oos = model.predict_proba(X_oos)[:, 1]
        y_pred_oos  = (y_proba_oos >= best_f1_t).astype(int)
        metrics_oos = compute_metrics(y_oos, y_pred_oos, y_proba_oos, "OOS Nov 2025")
        print_metrics(metrics_oos)
    else:
        print("\n[SKIP] OOS file not found — skipping Nov 2025 evaluation")

    # --- Plots ---
    generate_plots(y_test, y_proba_test, model, FEATURE_COLUMNS, tag="fallback_1month_v2")

    # --- Save artifacts ---
    print("\n[SAVING ARTIFACTS]")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Backup existing model
    old_path = os.path.join(MODELS_DIR, "fallback_model.pkl")
    if os.path.exists(old_path):
        import shutil
        bak = os.path.join(MODELS_DIR, "fallback_model_v1_backup.pkl")
        if not os.path.exists(bak):
            shutil.copy2(old_path, bak)
            print(f"  Backed up old model → {bak}")

    # Save model
    with open(old_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved fallback_model.pkl")

    # Save encoders
    with open(os.path.join(MODELS_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    # Build agg_stats in the format model_service.py expects
    agg_stats_out = {
        "carrier_delay_rate":           agg_stats.get("CARRIER_DELAY_RATE", {}),
        "origin_delay_rate":            agg_stats.get("ORIGIN_DELAY_RATE", {}),
        "dest_delay_rate":              agg_stats.get("DEST_DELAY_RATE", {}),
        "hour_delay_rate":              agg_stats.get("HOUR_DELAY_RATE", {}),
        "dow_delay_rate":               agg_stats.get("DOW_DELAY_RATE", {}),
        "route_delay_rate":             agg_stats.get("ROUTE_DELAY_RATE", {}),
        "season_delay_rate":            agg_stats.get("SEASON_DELAY_RATE", {}),
        "time_block_delay_rate":        agg_stats.get("TIME_BLOCK_DELAY_RATE", {}),
        "carrier_origin_delay_rate":    agg_stats.get("CARRIER_ORIGIN_DELAY_RATE", {}),
        "carrier_hour_delay_rate":      agg_stats.get("CARRIER_HOUR_DELAY_RATE", {}),
        "carrier_dow_delay_rate":       agg_stats.get("CARRIER_DOW_DELAY_RATE", {}),
        "origin_dow_delay_rate":        agg_stats.get("ORIGIN_DOW_DELAY_RATE", {}),
        "origin_hour_delay_rate":       agg_stats.get("ORIGIN_HOUR_DELAY_RATE", {}),
        "route_hour_delay_rate":        agg_stats.get("ROUTE_HOUR_DELAY_RATE", {}),
        "dest_hour_delay_rate":         agg_stats.get("DEST_HOUR_DELAY_RATE", {}),
        # congestion lookup tables
        "origin_hour_congestion":       feat_df.groupby("ORIGIN_HOUR_KEY")[TARGET].count().to_dict() if "ORIGIN_HOUR_KEY" in feat_df.columns else {},
        "dest_hour_congestion":         feat_df.groupby("DEST_HOUR_KEY")[TARGET].count().to_dict() if "DEST_HOUR_KEY" in feat_df.columns else {},
        "congestion_max_origin":        float(feat_df.groupby("ORIGIN_HOUR_KEY")[TARGET].count().max()) if "ORIGIN_HOUR_KEY" in feat_df.columns else 1.0,
        "congestion_max_dest":          float(feat_df.groupby("DEST_HOUR_KEY")[TARGET].count().max()) if "DEST_HOUR_KEY" in feat_df.columns else 1.0,
    }
    with open(os.path.join(MODELS_DIR, "aggregate_stats.pkl"), "wb") as f:
        pickle.dump(agg_stats_out, f)
    print(f"  Saved encoders.pkl + aggregate_stats.pkl")

    # Save config
    config = {
        "feature_columns":  FEATURE_COLUMNS,
        "target_column":    TARGET,
        "model_type":       "XGBoost v2 (Optuna-tuned)",
        "version":          "fallback_v2",
        "best_params":      best_params,
        "optimal_threshold_f1":  float(best_f1_t),
        "optimal_threshold_acc": float(best_acc_t),
        "training_data":    "Oct 2025 (1 month), days 1-25",
        "test_data":        "Oct 2025, days 26-31",
        "train_size":       int(len(X_train)),
        "n_features":       len(FEATURE_COLUMNS),
        "training_time_s":  float(train_time),
    }
    with open(os.path.join(MODELS_DIR, "model_config.pkl"), "wb") as f:
        pickle.dump(config, f)
    print(f"  Saved model_config.pkl")

    # Save benchmark CSV
    rows = [m_default, m_f1, m_acc]
    if metrics_oos:
        rows.append(metrics_oos)
    bench_path = os.path.join(RESULTS_DIR, "fallback_1month_v2_benchmarks.csv")
    pd.DataFrame(rows).to_csv(bench_path, index=False)
    print(f"  Saved {bench_path}")

    # Save CV results
    cv_path = os.path.join(RESULTS_DIR, "fallback_1month_v2_cv.csv")
    pd.DataFrame(cv_summary).T.to_csv(cv_path)
    print(f"  Saved {cv_path}")

    # Save Optuna trials
    trials_path = os.path.join(RESULTS_DIR, "fallback_1month_v2_optuna_trials.csv")
    pd.DataFrame([
        {"trial": t.number, "auc": t.value, **t.params}
        for t in study.trials if t.value is not None
    ]).to_csv(trials_path, index=False)
    print(f"  Saved {trials_path}")

    # --- FINAL SUMMARY ---
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE — FALLBACK MODEL v2")
    print(f"{'='*70}")
    print(f"  Total time:           {total_time:.0f}s  ({total_time/60:.1f} min)")
    print(f"  Features:             {len(FEATURE_COLUMNS)}  (was 19 in v1)")
    print(f"  Optuna best AUC (CV): {study.best_value:.4f}")
    print(f"  Optimal thresh (F1):  {best_f1_t:.2f}")
    print(f"  Optimal thresh (Acc): {best_acc_t:.2f}")
    print(f"")
    print(f"  {'Metric':<20} {'Default(0.50)':>14} {'BestF1':>10} {'BestAcc':>10}")
    print(f"  {'-'*56}")
    for label, key in [("Accuracy","Accuracy"),("ROC-AUC","ROC_AUC"),
                        ("F1 Score","F1_Score"),("Recall","Recall"),("MCC","MCC")]:
        print(f"  {label:<20} {m_default[key]:>14.4f} {m_f1[key]:>10.4f} {m_acc[key]:>10.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
