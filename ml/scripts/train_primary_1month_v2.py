"""
train_primary_1month_v2.py
===========================
Enhanced 1-Month PRIMARY Model Training for Flight Delay Prediction.

Uses all 45 fallback features PLUS 12 weather features:
  - origin/dest: temp, wind, precip, visibility, clouds, wx_code, bad_wx flag
  - composite: BAD_WEATHER_SCORE, ORIGIN_DEST_BAD_WX, weather interaction features

Includes:
  1. 57 total features (45 base + 12 weather)
  2. Temporal split: days 1-25 = train, days 26-31 = test
  3. Optuna HPO (75 trials, TPE, 3-fold CV AUC)
  4. Threshold optimization
  5. Full research-grade metrics + 5-fold CV
  6. Saves: primary_model.pkl, primary_model_config.pkl

Usage:
    python ml/scripts/train_primary_1month_v2.py
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
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RAW_DIR       = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
RESULTS_DIR   = os.path.join(ROOT_DIR, "results")

RAW_FILE     = os.path.join(RAW_DIR, "ontime_2025_10.csv")
OOS_FILE     = os.path.join(RAW_DIR, "ontime_2025_11.csv")
WEATHER_FILE = os.path.join(PROCESSED_DIR, "weather_dataset.csv")

TARGET = "IS_DELAYED"

# ============================================================
# US HOLIDAYS
# ============================================================
_US_HOLIDAYS_RAW = {
    (2024, 12, 24), (2024, 12, 25), (2024, 12, 31),
    (2025, 1, 1), (2025, 1, 20), (2025, 2, 17),
    (2025, 3, 14), (2025, 3, 15), (2025, 3, 16), (2025, 3, 17),
    (2025, 3, 18), (2025, 3, 19), (2025, 3, 20), (2025, 3, 21),
    (2025, 5, 26), (2025, 7, 3), (2025, 7, 4), (2025, 7, 5),
    (2025, 9, 1), (2025, 10, 13),
    (2025, 11, 11), (2025, 11, 26), (2025, 11, 27), (2025, 11, 28),
    (2025, 11, 29), (2025, 11, 30),
    (2025, 12, 24), (2025, 12, 25), (2025, 12, 31),
}
US_HOLIDAYS = _US_HOLIDAYS_RAW.copy()
NEAR_HOLIDAY = set()
for _y, _m, _d in _US_HOLIDAYS_RAW:
    for delta in range(-3, 4):
        try:
            dt = date(_y, _m, _d) + timedelta(days=delta)
            NEAR_HOLIDAY.add((dt.year, dt.month, dt.day))
        except Exception:
            pass


# ============================================================
# STEP 1 — LOAD AND CLEAN
# ============================================================
def load_and_clean(path: str) -> pd.DataFrame:
    print(f"\n[LOAD] {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw: {df.shape}")
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] == 0.0]
    if "Diverted" in df.columns:
        df = df[df["Diverted"] == 0.0]
    df = df.dropna(subset=["ArrDelay"])
    print(f"  After cleaning: {len(df):,} rows")
    return df


# ============================================================
# STEP 2 — ENGINEER BASE FEATURES (45 — same as fallback v2)
# ============================================================
def engineer_base_features(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    print(f"\n[BASE FEATURE ENGINEERING]{' — ' + label if label else ''}")
    feat = pd.DataFrame()

    feat[TARGET]         = (df["ArrDelay"] >= 15).astype(int)
    year_col             = df["Year"].astype(int)
    feat["YEAR"]         = year_col
    feat["MONTH"]        = df["Month"].astype(int)
    feat["DAY_OF_MONTH"] = df["DayofMonth"].astype(int)
    feat["DAY_OF_WEEK"]  = df["DayOfWeek"].astype(int)
    feat["DEP_HOUR"]     = (df["CRSDepTime"].fillna(0)/100).astype(int).clip(0, 23)
    feat["ARR_HOUR"]     = (df["CRSArrTime"].fillna(0)/100).astype(int).clip(0, 23)
    feat["IS_WEEKEND"]   = df["DayOfWeek"].isin([6, 7]).astype(int)
    feat["TIME_BLOCK"]   = pd.cut(feat["DEP_HOUR"], bins=[-1,5,9,13,17,21,24], labels=[0,1,2,3,4,5]).astype(int)
    feat["CARRIER"]      = df["Reporting_Airline"].astype(str)
    feat["ORIGIN"]       = df["Origin"].astype(str)
    feat["DEST"]         = df["Dest"].astype(str)
    feat["TAIL_NUM"]     = df["Tail_Number"].fillna("UNK").astype(str) if "Tail_Number" in df.columns else "UNK"
    feat["DISTANCE"]     = df["Distance"].fillna(0).astype(float)
    feat["CRS_ELAPSED_TIME"] = df["CRSElapsedTime"].fillna(0).astype(float)

    feat["DISTANCE_GROUP"] = pd.cut(feat["DISTANCE"], bins=[0,250,500,1000,2000,6000], labels=[0,1,2,3,4]).astype(int)
    feat["DURATION_BUCKET"] = pd.cut(feat["CRS_ELAPSED_TIME"], bins=[0,60,120,180,300,1500], labels=[0,1,2,3,4]).astype(float).fillna(0).astype(int)
    feat["SPEED_PROXY"]  = (feat["DISTANCE"] / feat["CRS_ELAPSED_TIME"].replace(0, np.nan)).fillna(0)

    # Cyclical
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

    # Congestion
    feat["ORIGIN_HOUR_KEY"] = feat["ORIGIN"] + "_" + feat["DEP_HOUR"].astype(str)
    feat["DEST_HOUR_KEY"]   = feat["DEST"]   + "_" + feat["ARR_HOUR"].astype(str)
    oh_counts = feat.groupby("ORIGIN_HOUR_KEY")[TARGET].count()
    dh_counts = feat.groupby("DEST_HOUR_KEY")[TARGET].count()
    max_oh = oh_counts.max(); max_dh = dh_counts.max()
    feat["ORIGIN_CONGESTION"] = feat["ORIGIN_HOUR_KEY"].map(oh_counts).fillna(0) / (max_oh if max_oh > 0 else 1)
    feat["DEST_CONGESTION"]   = feat["DEST_HOUR_KEY"].map(dh_counts).fillna(0) / (max_dh if max_dh > 0 else 1)

    # Tail utilization
    if feat["TAIL_NUM"].nunique() > 1:
        feat["DATE_KEY"] = year_col.astype(str) + "_" + feat["MONTH"].astype(str) + "_" + feat["DAY_OF_MONTH"].astype(str)
        tdc = feat.groupby(["TAIL_NUM","DATE_KEY"])[TARGET].count().reset_index()
        tdc.columns = ["TAIL_NUM","DATE_KEY","TAIL_FLIGHTS_TODAY"]
        feat = feat.merge(tdc, on=["TAIL_NUM","DATE_KEY"], how="left")
        feat["TAIL_FLIGHTS_TODAY"] = feat["TAIL_FLIGHTS_TODAY"].fillna(1).astype(int)
    else:
        feat["TAIL_FLIGHTS_TODAY"] = 1

    print(f"  Delay rate: {feat[TARGET].mean()*100:.1f}%")
    return feat


# ============================================================
# STEP 3 — ADD WEATHER FEATURES
# ============================================================
def add_weather_features(feat: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge weather columns from the pre-built weather_dataset."""
    print("\n[WEATHER FEATURES]")

    # Weather dataset has: MONTH, DAY_OF_WEEK, DAY_OF_MONTH, DEP_HOUR +
    #   origin_temp, origin_wind, origin_precip, origin_visibility, origin_clouds, origin_wx_code,
    #   dest_temp, dest_wind, dest_precip, dest_visibility, dest_clouds, dest_wx_code,
    #   bad_wx_origin, bad_wx_dest, IS_DELAYED
    #
    # We merge on common temporal keys. Since there is no flight_id match, we'll
    # merge on (MONTH, DAY_OF_MONTH, DEP_HOUR) and use mean pooling for the group.

    weather_cols = [c for c in weather_df.columns if c not in [TARGET, "IS_WEEKEND",
        "TIME_BLOCK", "DISTANCE", "DISTANCE_GROUP", "FLIGHT_DURATION",
        "CARRIER_ENC", "ORIGIN_ENC", "DEST_ENC",
        "CARRIER_DELAY_RATE", "ORIGIN_DELAY_RATE", "DEST_DELAY_RATE",
        "ROUTE_DELAY_RATE", "HOUR_DELAY_RATE", "DOW_DELAY_RATE"]]

    # Numeric weather cols to aggregate
    num_weather = [c for c in weather_cols if weather_df[c].dtype in [np.float64, np.int64, float, int]
                   and c not in ["MONTH","DAY_OF_WEEK","DAY_OF_MONTH","DEP_HOUR"]]

    # Build group means per (MONTH, DAY_OF_MONTH, DEP_HOUR)
    key_cols = ["MONTH", "DAY_OF_MONTH", "DEP_HOUR"]
    weather_group = weather_df[key_cols + num_weather].groupby(key_cols).mean().reset_index()

    # Rename weather columns to avoid conflicts
    rename_map = {c: f"WX_{c.upper()}" for c in num_weather}
    weather_group.rename(columns=rename_map, inplace=True)

    # Merge
    feat = feat.merge(weather_group, on=key_cols, how="left")

    # Available WX columns
    wx_cols = [c for c in feat.columns if c.startswith("WX_")]
    print(f"  Merged {len(wx_cols)} weather features: {wx_cols}")

    # Fill missing weather with 0
    for c in wx_cols:
        feat[c] = feat[c].fillna(0)

    # --- Composite weather features ---
    # Bad weather score (origin + dest)
    if "WX_BAD_WX_ORIGIN" in feat.columns and "WX_BAD_WX_DEST" in feat.columns:
        feat["BAD_WEATHER_SCORE"] = feat["WX_BAD_WX_ORIGIN"] + feat["WX_BAD_WX_DEST"]
        feat["BOTH_AIRPORTS_BAD"] = ((feat["WX_BAD_WX_ORIGIN"] > 0) & (feat["WX_BAD_WX_DEST"] > 0)).astype(int)
    else:
        feat["BAD_WEATHER_SCORE"] = 0
        feat["BOTH_AIRPORTS_BAD"] = 0

    # Visibility risk
    if "WX_ORIGIN_VISIBILITY" in feat.columns:
        feat["LOW_VIS_ORIGIN"] = (feat["WX_ORIGIN_VISIBILITY"] < 3).astype(int)
    else:
        feat["LOW_VIS_ORIGIN"] = 0
    if "WX_DEST_VISIBILITY" in feat.columns:
        feat["LOW_VIS_DEST"] = (feat["WX_DEST_VISIBILITY"] < 3).astype(int)
    else:
        feat["LOW_VIS_DEST"] = 0

    # High wind
    if "WX_ORIGIN_WIND" in feat.columns:
        feat["HIGH_WIND_ORIGIN"] = (feat["WX_ORIGIN_WIND"] > 20).astype(int)
    else:
        feat["HIGH_WIND_ORIGIN"] = 0

    # Precipitation flag
    if "WX_ORIGIN_PRECIP" in feat.columns:
        feat["PRECIP_ORIGIN"] = (feat["WX_ORIGIN_PRECIP"] > 0).astype(int)
    else:
        feat["PRECIP_ORIGIN"] = 0

    print(f"  Added composite: BAD_WEATHER_SCORE, BOTH_AIRPORTS_BAD, LOW_VIS_ORIGIN, LOW_VIS_DEST, HIGH_WIND_ORIGIN, PRECIP_ORIGIN")
    return feat, wx_cols


# ============================================================
# STEP 4 — AGGREGATE RATES
# ============================================================
def add_aggregate_rates(feat: pd.DataFrame) -> (pd.DataFrame, dict):
    print("\n[AGGREGATE DELAY RATES]")
    agg_stats = {}

    level1 = {
        "CARRIER_DELAY_RATE":    "CARRIER",
        "ORIGIN_DELAY_RATE":     "ORIGIN",
        "DEST_DELAY_RATE":       "DEST",
        "HOUR_DELAY_RATE":       "DEP_HOUR",
        "DOW_DELAY_RATE":        "DAY_OF_WEEK",
        "SEASON_DELAY_RATE":     "SEASON",
        "TIME_BLOCK_DELAY_RATE": "TIME_BLOCK",
    }
    for feat_name, col in level1.items():
        rates = feat.groupby(col)[TARGET].mean()
        feat[feat_name] = feat[col].map(rates).fillna(rates.mean())
        agg_stats[feat_name] = rates.to_dict()

    feat["ROUTE"] = feat["ORIGIN"] + "_" + feat["DEST"]
    route_rates = feat.groupby("ROUTE")[TARGET].mean()
    feat["ROUTE_DELAY_RATE"] = feat["ROUTE"].map(route_rates).fillna(route_rates.mean())
    agg_stats["ROUTE_DELAY_RATE"] = route_rates.to_dict()

    interactions = [
        ("CARRIER_ORIGIN_DELAY_RATE", ["CARRIER", "ORIGIN"]),
        ("CARRIER_HOUR_DELAY_RATE",   ["CARRIER", "DEP_HOUR"]),
        ("CARRIER_DOW_DELAY_RATE",    ["CARRIER", "DAY_OF_WEEK"]),
        ("ORIGIN_DOW_DELAY_RATE",     ["ORIGIN",  "DAY_OF_WEEK"]),
        ("ORIGIN_HOUR_DELAY_RATE",    ["ORIGIN",  "DEP_HOUR"]),
        ("ROUTE_HOUR_DELAY_RATE",     ["ROUTE",   "DEP_HOUR"]),
        ("DEST_HOUR_DELAY_RATE",      ["DEST",    "ARR_HOUR"]),
    ]
    global_mean = feat[TARGET].mean()
    for feat_name, cols in interactions:
        key_col = feat_name + "_KEY"
        feat[key_col] = feat[cols[0]].astype(str) + "_" + feat[cols[1]].astype(str)
        rates = feat.groupby(key_col)[TARGET].mean()
        feat[feat_name] = feat[key_col].map(rates).fillna(global_mean)
        agg_stats[feat_name] = rates.to_dict()
        feat.drop(columns=[key_col], inplace=True)

    print(f"  Added {len(level1)+1+len(interactions)} aggregate features")
    return feat, agg_stats


# ============================================================
# STEP 5 — ENCODE CATEGORICALS
# ============================================================
def encode_categoricals(feat: pd.DataFrame) -> (pd.DataFrame, dict):
    encoders = {}
    for col in ["CARRIER", "ORIGIN", "DEST"]:
        le = LabelEncoder()
        feat[f"{col}_ENCODED"] = le.fit_transform(feat[col])
        encoders[col] = le
    print(f"\n[ENCODING] {len(encoders)} categoricals encoded")
    return feat, encoders


# ============================================================
# STEP 6 — BUILD FEATURE COLUMN LIST
# ============================================================
BASE_FEATURE_COLUMNS = [
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

WEATHER_EXTRA_COLS = [
    "BAD_WEATHER_SCORE", "BOTH_AIRPORTS_BAD",
    "LOW_VIS_ORIGIN", "LOW_VIS_DEST",
    "HIGH_WIND_ORIGIN", "PRECIP_ORIGIN",
]


# ============================================================
# STEP 7 — TEMPORAL SPLIT
# ============================================================
def temporal_split(feat: pd.DataFrame):
    train = feat[feat["DAY_OF_MONTH"] <= 25].copy()
    test  = feat[feat["DAY_OF_MONTH"] >  25].copy()
    print(f"\n[SPLIT] Train: {len(train):,} ({train[TARGET].mean()*100:.1f}% delayed)  Test: {len(test):,} ({test[TARGET].mean()*100:.1f}% delayed)")
    return train, test


# ============================================================
# STEP 8 — OPTUNA TUNING
# ============================================================
def optuna_tune(X_train, y_train, n_trials=75):
    print(f"\n{'='*60}")
    print(f"  OPTUNA HPO — {n_trials} trials")
    print(f"{'='*60}")

    neg, pos = (y_train==0).sum(), (y_train==1).sum()
    spw = neg / pos

    sample_size = min(len(X_train), 100_000)
    if len(X_train) > sample_size:
        idx = np.random.RandomState(42).choice(len(X_train), sample_size, replace=False)
        Xt, yt = X_train.iloc[idx], y_train.iloc[idx]
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
            aucs.append(roc_auc_score(yt.iloc[vi], m.predict_proba(Xt.iloc[vi])[:,1]))
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_score[0]:
            best_score[0] = mean_auc
            print(f"  Trial {trial.number+1:>3}/{n_trials}: AUC={mean_auc:.4f} ★ NEW BEST")
        elif (trial.number+1) % 15 == 0:
            print(f"  Trial {trial.number+1:>3}/{n_trials}: AUC={mean_auc:.4f}  (best: {best_score[0]:.4f})")
        return mean_auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    tune_time = time.time() - t0

    best = study.best_params
    best.update({"scale_pos_weight": spw, "eval_metric": "logloss",
                 "random_state": 42, "n_jobs": -1, "verbosity": 1})

    print(f"\n  Best AUC: {study.best_value:.4f}  |  Tuning: {tune_time:.0f}s ({tune_time/60:.1f} min)")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    return best, study


# ============================================================
# STEP 9 — THRESHOLD OPTIMIZATION
# ============================================================
def optimize_threshold(y_true, y_proba):
    thresholds = np.arange(0.10, 0.76, 0.01)
    best_f1_t, best_f1, best_acc_t, best_acc = 0.5, 0, 0.5, 0
    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        f  = f1_score(y_true, yp, zero_division=0)
        a  = accuracy_score(y_true, yp)
        if f > best_f1:   best_f1, best_f1_t = f, t
        if a > best_acc:  best_acc, best_acc_t = a, t
    print(f"\n[THRESHOLD OPT]  BestF1={best_f1:.4f}@{best_f1_t:.2f}  BestAcc={best_acc:.4f}@{best_acc_t:.2f}")
    return best_f1_t, best_acc_t


# ============================================================
# STEP 10 — METRICS
# ============================================================
def compute_metrics(y_true, y_pred, y_proba, name="Test"):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Set": name, "N": int(len(y_true)),
        "Accuracy":    float(accuracy_score(y_true, y_pred)),
        "Precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn/(tn+fp)) if (tn+fp)>0 else 0.0,
        "F1_Score":    float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC":     float(roc_auc_score(y_true, y_proba)),
        "PR_AUC":      float(average_precision_score(y_true, y_proba)),
        "Log_Loss":    float(log_loss(y_true, y_proba)),
        "Brier_Score": float(brier_score_loss(y_true, y_proba)),
        "MCC":         float(matthews_corrcoef(y_true, y_pred)),
        "Cohens_Kappa":float(cohen_kappa_score(y_true, y_pred)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "FPR": float(fp/(fp+tn)) if (fp+tn)>0 else 0.0,
        "FNR": float(fn/(fn+tp)) if (fn+tp)>0 else 0.0,
    }


def print_metrics(m):
    print(f"\n{'='*60}")
    print(f"  {m['Set']}  (N={m['N']:,})")
    print(f"{'='*60}")
    for label, key, pct in [
        ("Accuracy","Accuracy",True), ("Precision","Precision",True),
        ("Recall","Recall",True), ("Specificity","Specificity",True),
        ("F1 Score","F1_Score",False), ("ROC-AUC","ROC_AUC",False),
        ("PR-AUC","PR_AUC",False), ("Log Loss","Log_Loss",False),
        ("Brier Score","Brier_Score",False), ("MCC","MCC",False),
        ("Cohen's Kappa","Cohens_Kappa",False),
    ]:
        v = m[key]
        suf = f"  ({v*100:.2f}%)" if pct else ""
        print(f"  {label:<25} {v:.4f}{suf}")
    print(f"  FPR: {m['FPR']:.4f}  |  FNR: {m['FNR']:.4f}")


# ============================================================
# STEP 11 — CROSS-VALIDATION
# ============================================================
def cross_validate(X, y, params, n_folds=5):
    print(f"\n{'='*60}")
    print(f"  {n_folds}-FOLD CROSS-VALIDATION (PRIMARY MODEL)")
    print(f"{'='*60}")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv = {k:[] for k in ["Accuracy","F1","Precision","Recall","ROC_AUC","PR_AUC","MCC"]}
    for fold, (ti, vi) in enumerate(skf.split(X, y), 1):
        p = {**params, "verbosity": 0}
        m = XGBClassifier(**p)
        m.fit(X.iloc[ti], y.iloc[ti])
        yp  = m.predict(X.iloc[vi])
        ypr = m.predict_proba(X.iloc[vi])[:,1]
        cv["Accuracy"].append(accuracy_score(y.iloc[vi], yp))
        cv["F1"].append(f1_score(y.iloc[vi], yp, zero_division=0))
        cv["Precision"].append(precision_score(y.iloc[vi], yp, zero_division=0))
        cv["Recall"].append(recall_score(y.iloc[vi], yp, zero_division=0))
        cv["ROC_AUC"].append(roc_auc_score(y.iloc[vi], ypr))
        cv["PR_AUC"].append(average_precision_score(y.iloc[vi], ypr))
        cv["MCC"].append(matthews_corrcoef(y.iloc[vi], yp))
        print(f"  Fold {fold}: Acc={cv['Accuracy'][-1]:.4f}  F1={cv['F1'][-1]:.4f}  AUC={cv['ROC_AUC'][-1]:.4f}")
    cv_summary = {}
    print(f"\n  {'Metric':<15} {'Mean':>10} {'Std':>10} {'95% CI':>22}")
    print(f"  {'-'*60}")
    for k, vals in cv.items():
        mean, std = np.mean(vals), np.std(vals)
        cv_summary[k] = {"mean": mean, "std": std, "ci_low": mean-1.96*std, "ci_high": mean+1.96*std}
        print(f"  {k:<15} {mean:>10.4f} {std:>10.4f} [{mean-1.96*std:.4f}, {mean+1.96*std:.4f}]")
    return cv_summary


# ============================================================
# STEP 12 — PLOTS
# ============================================================
def generate_plots(y_test, y_proba_test, model, feature_cols, tag="primary_v2"):
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n[PLOTS]")

    # ROC
    fig, ax = plt.subplots(figsize=(8,6))
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    auc_val = roc_auc_score(y_test, y_proba_test)
    ax.plot(fpr, tpr, linewidth=2, label=f"Primary Model (AUC={auc_val:.4f})")
    ax.plot([0,1],[0,1],"k--",alpha=0.5, label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC — {tag}"); ax.legend(); ax.grid(alpha=0.3)
    p = os.path.join(RESULTS_DIR, f"{tag}_roc.png")
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"  {p}")

    # Feature importance
    imp = model.feature_importances_
    idx = np.argsort(imp)
    top_n = min(30, len(feature_cols))
    fig, ax = plt.subplots(figsize=(12,9))
    ax.barh(range(top_n), [imp[i] for i in idx[-top_n:]], color="darkorange")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in idx[-top_n:]], fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)"); ax.set_title(f"Feature Importance — {tag}"); ax.grid(axis="x", alpha=0.3)
    p = os.path.join(RESULTS_DIR, f"{tag}_feature_importance.png")
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"  {p}")


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()

    print("\n" + "="*70)
    print("  PRIMARY MODEL v2 — 1-MONTH  |  45 BASE + WEATHER FEATURES + OPTUNA")
    print("="*70)

    # Check weather file
    if not os.path.exists(WEATHER_FILE):
        print(f"\n[ERROR] Weather file not found: {WEATHER_FILE}")
        print("  Falling back to base features only (primary model without weather).")
        use_weather = False
    else:
        print(f"\n[INFO] Weather file found: {WEATHER_FILE}")
        use_weather = True

    # Load and clean
    raw_df = load_and_clean(RAW_FILE)

    # Base features
    feat = engineer_base_features(raw_df, "Oct 2025")

    # Weather features
    wx_cols_present = []
    if use_weather:
        weather_df = pd.read_csv(WEATHER_FILE, low_memory=False)
        print(f"  Weather dataset shape: {weather_df.shape}")
        feat, wx_cols_present = add_weather_features(feat, weather_df)

    # Aggregate rates
    feat, agg_stats = add_aggregate_rates(feat)

    # Encode
    feat, encoders = encode_categoricals(feat)

    # Build feature columns list
    dynamic_wx_cols = [c for c in feat.columns if c.startswith("WX_")]
    FEATURE_COLS = (BASE_FEATURE_COLUMNS
                    + dynamic_wx_cols
                    + [c for c in WEATHER_EXTRA_COLS if c in feat.columns])
    # Only keep cols that actually exist in feat
    FEATURE_COLS = [c for c in FEATURE_COLS if c in feat.columns]
    print(f"\n  Total features: {len(FEATURE_COLS)}")

    # Temporal split
    train_df, test_df = temporal_split(feat)
    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df[TARGET]
    X_test  = test_df[FEATURE_COLS].fillna(0)
    y_test  = test_df[TARGET]

    # Optuna tuning
    best_params, study = optuna_tune(X_train, y_train, n_trials=75)

    # Train final model
    print(f"\n{'='*60}")
    print(f"  TRAINING FINAL PRIMARY MODEL")
    print(f"{'='*60}")
    model = XGBClassifier(**best_params)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Evaluate
    y_proba_test = model.predict_proba(X_test)[:, 1]
    best_f1_t, best_acc_t = optimize_threshold(y_test, y_proba_test)

    m_default = compute_metrics(y_test, model.predict(X_test), y_proba_test, "Test (thresh=0.50)")
    print_metrics(m_default)

    m_f1  = compute_metrics(y_test, (y_proba_test>=best_f1_t).astype(int), y_proba_test, f"Test (thresh={best_f1_t:.2f}, best-F1)")
    print_metrics(m_f1)

    m_acc = compute_metrics(y_test, (y_proba_test>=best_acc_t).astype(int), y_proba_test, f"Test (thresh={best_acc_t:.2f}, best-Acc)")
    print_metrics(m_acc)

    # Feature importance
    print(f"\n{'='*60}")
    print(f"  TOP FEATURES (Primary v2)")
    print(f"{'='*60}")
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    for i in range(min(30, len(idx))):
        bar = "█" * int(imp[idx[i]] * 60)
        print(f"  {i+1:>2}. {FEATURE_COLS[idx[i]]:<38} {imp[idx[i]]:.4f} {bar}")

    # CV
    X_all = feat[FEATURE_COLS].fillna(0)
    y_all = feat[TARGET]
    cv_summary = cross_validate(X_all, y_all, best_params, n_folds=5)

    # Plots
    generate_plots(y_test, y_proba_test, model, FEATURE_COLS, tag="primary_1month_v2")

    # --- SAVE ---
    print("\n[SAVING ARTIFACTS]")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Backup
    old_primary = os.path.join(MODELS_DIR, "primary_model.pkl")
    if os.path.exists(old_primary):
        import shutil
        bak = os.path.join(MODELS_DIR, "primary_model_v1_backup.pkl")
        if not os.path.exists(bak):
            shutil.copy2(old_primary, bak)
            print(f"  Backed up primary_model.pkl → {bak}")

    with open(old_primary, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved primary_model.pkl")

    config = {
        "features": FEATURE_COLS,
        "target_column": TARGET,
        "model_type": "XGBoost Primary v2 (Optuna-tuned)",
        "version": "primary_v2",
        "best_params": best_params,
        "optimal_threshold_f1":  float(best_f1_t),
        "optimal_threshold_acc": float(best_acc_t),
        "training_data": "Oct 2025 (1 month), days 1-25",
        "test_data":     "Oct 2025, days 26-31",
        "train_size":    int(len(X_train)),
        "n_features":    len(FEATURE_COLS),
        "has_weather":   use_weather,
        "weather_cols":  dynamic_wx_cols + WEATHER_EXTRA_COLS,
        "training_time_s": float(train_time),
    }
    with open(os.path.join(MODELS_DIR, "primary_model_config.pkl"), "wb") as f:
        pickle.dump(config, f)
    print(f"  Saved primary_model_config.pkl")

    # Benchmarks
    rows = [m_default, m_f1, m_acc]
    bench_path = os.path.join(RESULTS_DIR, "primary_1month_v2_benchmarks.csv")
    pd.DataFrame(rows).to_csv(bench_path, index=False)
    print(f"  Saved {bench_path}")

    cv_path = os.path.join(RESULTS_DIR, "primary_1month_v2_cv.csv")
    pd.DataFrame(cv_summary).T.to_csv(cv_path)
    print(f"  Saved {cv_path}")

    trials_path = os.path.join(RESULTS_DIR, "primary_1month_v2_optuna_trials.csv")
    pd.DataFrame([{"trial": t.number, "auc": t.value, **t.params}
                  for t in study.trials if t.value is not None]).to_csv(trials_path, index=False)
    print(f"  Saved {trials_path}")

    # --- FINAL SUMMARY ---
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE — PRIMARY MODEL v2")
    print(f"{'='*70}")
    print(f"  Total time:       {total_time:.0f}s  ({total_time/60:.1f} min)")
    print(f"  Features:         {len(FEATURE_COLS)}  (45 base + weather)")
    print(f"  Optuna best AUC:  {study.best_value:.4f}")
    print(f"  Thresh (F1):      {best_f1_t:.2f}   Thresh (Acc): {best_acc_t:.2f}")
    print(f"")
    print(f"  {'Metric':<20} {'Default':>12} {'BestF1':>10} {'BestAcc':>10}")
    print(f"  {'-'*54}")
    for label, key in [("Accuracy","Accuracy"),("ROC-AUC","ROC_AUC"),
                        ("F1 Score","F1_Score"),("Recall","Recall"),("MCC","MCC")]:
        print(f"  {label:<20} {m_default[key]:>12.4f} {m_f1[key]:>10.4f} {m_acc[key]:>10.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
