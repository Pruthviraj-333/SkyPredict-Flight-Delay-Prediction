"""
train_primary_regression.py
============================
XGBoost Regression Model — Predicts EXACT DELAY MINUTES (WITH weather features).

Target: ARR_DELAY_MINUTES = ArrDelay.clip(lower=0)
Features: Same 45 base features + weather features (same as primary classifier v2).

Includes:
  1. 45 base features + dynamic weather features (WX_*) + composite weather
  2. Temporal split: days 1-25 = train, days 26-31 = test
  3. Optuna HPO (50 trials, minimize RMSE, 3-fold KFold)
  4. Regression metrics: RMSE, MAE, R², Within-15min/30min accuracy
  5. 5-Fold cross-validation
  6. Saves: primary_reg_model.pkl, primary_reg_config.pkl

Usage:
    python ml/scripts/train_primary_regression.py
"""

import os
import pickle
import time
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
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

TARGET = "ARR_DELAY_MINUTES"

# ============================================================
# US HOLIDAYS
# ============================================================
_US_HOLIDAYS_RAW = {
    (2024, 12, 24), (2024, 12, 25), (2024, 12, 31),
    (2025, 1, 1),  (2025, 1, 20),  (2025, 2, 17),
    (2025, 3, 14), (2025, 3, 15),  (2025, 3, 16),  (2025, 3, 17),
    (2025, 3, 18), (2025, 3, 19),  (2025, 3, 20),  (2025, 3, 21),
    (2025, 5, 26), (2025, 7, 3),   (2025, 7, 4),   (2025, 7, 5),
    (2025, 9, 1),  (2025, 10, 13),
    (2025, 11, 11),(2025, 11, 26), (2025, 11, 27), (2025, 11, 28),
    (2025, 11, 29),(2025, 11, 30),
    (2025, 12, 24),(2025, 12, 25), (2025, 12, 31),
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
    print(f"  Raw shape: {df.shape}")
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] == 0.0]
    if "Diverted" in df.columns:
        df = df[df["Diverted"] == 0.0]
    df = df.dropna(subset=["ArrDelay"])
    print(f"  After cleaning: {len(df):,} rows")
    return df


# ============================================================
# STEP 2 — BASE FEATURE ENGINEERING (45 features)
# ============================================================
def engineer_base_features(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    print(f"\n[BASE FEATURE ENGINEERING]{' — ' + label if label else ''}")
    feat = pd.DataFrame()

    # Regression target
    feat[TARGET] = df["ArrDelay"].clip(lower=0).astype(float)
    delayed_mask = feat[TARGET] > 0
    print(f"  Mean delay (all):      {feat[TARGET].mean():.2f} min")
    print(f"  Mean delay (delayed):  {feat.loc[delayed_mask, TARGET].mean():.2f} min")

    year_col = df["Year"].astype(int)
    feat["YEAR"]         = year_col
    feat["MONTH"]        = df["Month"].astype(int)
    feat["DAY_OF_MONTH"] = df["DayofMonth"].astype(int)
    feat["DAY_OF_WEEK"]  = df["DayOfWeek"].astype(int)
    feat["DEP_HOUR"]     = (df["CRSDepTime"].fillna(0) / 100).astype(int).clip(0, 23)
    feat["ARR_HOUR"]     = (df["CRSArrTime"].fillna(0) / 100).astype(int).clip(0, 23)
    feat["IS_WEEKEND"]   = df["DayOfWeek"].isin([6, 7]).astype(int)
    feat["TIME_BLOCK"]   = pd.cut(
        feat["DEP_HOUR"], bins=[-1, 5, 9, 13, 17, 21, 24], labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)

    feat["CARRIER"] = df["Reporting_Airline"].astype(str)
    feat["ORIGIN"]  = df["Origin"].astype(str)
    feat["DEST"]    = df["Dest"].astype(str)
    feat["TAIL_NUM"] = df["Tail_Number"].fillna("UNK").astype(str) if "Tail_Number" in df.columns else "UNK"

    feat["DISTANCE"]         = df["Distance"].fillna(0).astype(float)
    feat["CRS_ELAPSED_TIME"] = df["CRSElapsedTime"].fillna(0).astype(float)
    feat["DISTANCE_GROUP"]   = pd.cut(feat["DISTANCE"], bins=[0, 250, 500, 1000, 2000, 6000], labels=[0, 1, 2, 3, 4]).astype(int)
    feat["DURATION_BUCKET"]  = pd.cut(feat["CRS_ELAPSED_TIME"], bins=[0, 60, 120, 180, 300, 1500], labels=[0, 1, 2, 3, 4]).astype(float).fillna(0).astype(int)
    feat["SPEED_PROXY"]      = (feat["DISTANCE"] / feat["CRS_ELAPSED_TIME"].replace(0, np.nan)).fillna(0)

    feat["MONTH_SIN"] = np.sin(2 * np.pi * feat["MONTH"] / 12)
    feat["MONTH_COS"] = np.cos(2 * np.pi * feat["MONTH"] / 12)
    feat["HOUR_SIN"]  = np.sin(2 * np.pi * feat["DEP_HOUR"] / 24)
    feat["HOUR_COS"]  = np.cos(2 * np.pi * feat["DEP_HOUR"] / 24)
    feat["DOW_SIN"]   = np.sin(2 * np.pi * feat["DAY_OF_WEEK"] / 7)
    feat["DOW_COS"]   = np.cos(2 * np.pi * feat["DAY_OF_WEEK"] / 7)
    feat["DOM_SIN"]   = np.sin(2 * np.pi * feat["DAY_OF_MONTH"] / 31)
    feat["DOM_COS"]   = np.cos(2 * np.pi * feat["DAY_OF_MONTH"] / 31)

    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    feat["SEASON"] = feat["MONTH"].map(season_map).astype(int)

    feat["IS_HOLIDAY"]   = [1 if (y, m, d) in US_HOLIDAYS else 0
                            for y, m, d in zip(year_col, feat["MONTH"], feat["DAY_OF_MONTH"])]
    feat["NEAR_HOLIDAY"] = [1 if (y, m, d) in NEAR_HOLIDAY else 0
                            for y, m, d in zip(year_col, feat["MONTH"], feat["DAY_OF_MONTH"])]

    feat["IS_FRIDAY_EVENING"] = ((feat["DAY_OF_WEEK"] == 5) & (feat["DEP_HOUR"] >= 15)).astype(int)
    feat["IS_SUNDAY_EVENING"] = ((feat["DAY_OF_WEEK"] == 7) & (feat["DEP_HOUR"] >= 15)).astype(int)
    feat["IS_MONDAY_MORNING"] = ((feat["DAY_OF_WEEK"] == 1) & (feat["DEP_HOUR"] <= 9)).astype(int)
    feat["IS_PEAK_HOUR"]      = feat["DEP_HOUR"].isin([7, 8, 16, 17, 18]).astype(int)
    feat["IS_EARLY_MORNING"]  = (feat["DEP_HOUR"] <= 6).astype(int)
    feat["IS_RED_EYE"]        = (feat["DEP_HOUR"] >= 22).astype(int)

    feat["ORIGIN_HOUR_KEY"] = feat["ORIGIN"] + "_" + feat["DEP_HOUR"].astype(str)
    feat["DEST_HOUR_KEY"]   = feat["DEST"]   + "_" + feat["ARR_HOUR"].astype(str)
    oh_counts = feat.groupby("ORIGIN_HOUR_KEY")[TARGET].count()
    dh_counts = feat.groupby("DEST_HOUR_KEY")[TARGET].count()
    max_oh = oh_counts.max(); max_dh = dh_counts.max()
    feat["ORIGIN_CONGESTION"] = feat["ORIGIN_HOUR_KEY"].map(oh_counts).fillna(0) / (max_oh if max_oh > 0 else 1)
    feat["DEST_CONGESTION"]   = feat["DEST_HOUR_KEY"].map(dh_counts).fillna(0)   / (max_dh if max_dh > 0 else 1)

    if feat["TAIL_NUM"].nunique() > 1:
        feat["DATE_KEY"] = year_col.astype(str) + "_" + feat["MONTH"].astype(str) + "_" + feat["DAY_OF_MONTH"].astype(str)
        tdc = feat.groupby(["TAIL_NUM", "DATE_KEY"])[TARGET].count().reset_index()
        tdc.columns = ["TAIL_NUM", "DATE_KEY", "TAIL_FLIGHTS_TODAY"]
        feat = feat.merge(tdc, on=["TAIL_NUM", "DATE_KEY"], how="left")
        feat["TAIL_FLIGHTS_TODAY"] = feat["TAIL_FLIGHTS_TODAY"].fillna(1).astype(int)
    else:
        feat["TAIL_FLIGHTS_TODAY"] = 1

    return feat


# ============================================================
# STEP 3 — ADD WEATHER FEATURES (same as primary classifier v2)
# ============================================================
def add_weather_features(feat: pd.DataFrame, weather_df: pd.DataFrame) -> tuple:
    print("\n[WEATHER FEATURES]")
    weather_cols = [c for c in weather_df.columns if c not in [
        "IS_DELAYED", "IS_WEEKEND", "TIME_BLOCK", "DISTANCE", "DISTANCE_GROUP",
        "FLIGHT_DURATION", "CARRIER_ENC", "ORIGIN_ENC", "DEST_ENC",
        "CARRIER_DELAY_RATE", "ORIGIN_DELAY_RATE", "DEST_DELAY_RATE",
        "ROUTE_DELAY_RATE", "HOUR_DELAY_RATE", "DOW_DELAY_RATE"
    ]]
    num_weather = [c for c in weather_cols
                   if weather_df[c].dtype in [np.float64, np.int64, float, int]
                   and c not in ["MONTH", "DAY_OF_WEEK", "DAY_OF_MONTH", "DEP_HOUR"]]
    key_cols = ["MONTH", "DAY_OF_MONTH", "DEP_HOUR"]
    weather_group = weather_df[key_cols + num_weather].groupby(key_cols).mean().reset_index()
    rename_map = {c: f"WX_{c.upper()}" for c in num_weather}
    weather_group.rename(columns=rename_map, inplace=True)
    feat = feat.merge(weather_group, on=key_cols, how="left")

    wx_cols = [c for c in feat.columns if c.startswith("WX_")]
    print(f"  Merged {len(wx_cols)} weather features: {wx_cols}")
    for c in wx_cols:
        feat[c] = feat[c].fillna(0)

    # Composite weather features
    if "WX_BAD_WX_ORIGIN" in feat.columns and "WX_BAD_WX_DEST" in feat.columns:
        feat["BAD_WEATHER_SCORE"] = feat["WX_BAD_WX_ORIGIN"] + feat["WX_BAD_WX_DEST"]
        feat["BOTH_AIRPORTS_BAD"] = ((feat["WX_BAD_WX_ORIGIN"] > 0) & (feat["WX_BAD_WX_DEST"] > 0)).astype(int)
    else:
        feat["BAD_WEATHER_SCORE"] = 0; feat["BOTH_AIRPORTS_BAD"] = 0

    feat["LOW_VIS_ORIGIN"]  = (feat["WX_ORIGIN_VISIBILITY"] < 3).astype(int) if "WX_ORIGIN_VISIBILITY" in feat.columns else 0
    feat["LOW_VIS_DEST"]    = (feat["WX_DEST_VISIBILITY"] < 3).astype(int)   if "WX_DEST_VISIBILITY"   in feat.columns else 0
    feat["HIGH_WIND_ORIGIN"]= (feat["WX_ORIGIN_WIND"] > 20).astype(int)      if "WX_ORIGIN_WIND"       in feat.columns else 0
    feat["PRECIP_ORIGIN"]   = (feat["WX_ORIGIN_PRECIP"] > 0).astype(int)     if "WX_ORIGIN_PRECIP"     in feat.columns else 0

    print("  Added composite: BAD_WEATHER_SCORE, BOTH_AIRPORTS_BAD, LOW_VIS_ORIGIN, LOW_VIS_DEST, HIGH_WIND_ORIGIN, PRECIP_ORIGIN")
    return feat, wx_cols


# ============================================================
# STEP 4 — AGGREGATE DELAY RATES
# ============================================================
def add_aggregate_rates(feat: pd.DataFrame) -> tuple:
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
        rates = feat.groupby(col)[TARGET].apply(lambda x: (x > 0).mean())
        feat[feat_name] = feat[col].map(rates).fillna(rates.mean())
        agg_stats[feat_name] = rates.to_dict()

    feat["ROUTE"] = feat["ORIGIN"] + "_" + feat["DEST"]
    route_rates = feat.groupby("ROUTE")[TARGET].apply(lambda x: (x > 0).mean())
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
    global_mean = (feat[TARGET] > 0).mean()
    for feat_name, cols in interactions:
        key_col = feat_name + "_KEY"
        feat[key_col] = feat[cols[0]].astype(str) + "_" + feat[cols[1]].astype(str)
        rates = feat.groupby(key_col)[TARGET].apply(lambda x: (x > 0).mean())
        feat[feat_name] = feat[key_col].map(rates).fillna(global_mean)
        agg_stats[feat_name] = rates.to_dict()
        feat.drop(columns=[key_col], inplace=True)
    return feat, agg_stats


# ============================================================
# STEP 5 — ENCODE CATEGORICALS
# ============================================================
def encode_categoricals(feat: pd.DataFrame) -> tuple:
    encoders = {}
    for col in ["CARRIER", "ORIGIN", "DEST"]:
        le = LabelEncoder()
        feat[f"{col}_ENCODED"] = le.fit_transform(feat[col])
        encoders[col] = le
    print(f"\n[ENCODING] 3 categoricals encoded")
    return feat, encoders


# ============================================================
# STEP 6 — FEATURE COLUMN DEFINITIONS
# ============================================================
BASE_FEATURE_COLUMNS = [
    "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "DEP_HOUR", "ARR_HOUR",
    "IS_WEEKEND", "TIME_BLOCK",
    "DISTANCE", "CRS_ELAPSED_TIME", "DISTANCE_GROUP", "DURATION_BUCKET", "SPEED_PROXY",
    "CARRIER_ENCODED", "ORIGIN_ENCODED", "DEST_ENCODED",
    "MONTH_SIN", "MONTH_COS", "HOUR_SIN", "HOUR_COS",
    "DOW_SIN", "DOW_COS", "DOM_SIN", "DOM_COS",
    "SEASON", "IS_HOLIDAY", "NEAR_HOLIDAY",
    "IS_FRIDAY_EVENING", "IS_SUNDAY_EVENING", "IS_MONDAY_MORNING",
    "IS_PEAK_HOUR", "IS_EARLY_MORNING", "IS_RED_EYE",
    "ORIGIN_CONGESTION", "DEST_CONGESTION",
    "TAIL_FLIGHTS_TODAY",
    "CARRIER_DELAY_RATE", "ORIGIN_DELAY_RATE", "DEST_DELAY_RATE",
    "HOUR_DELAY_RATE", "DOW_DELAY_RATE", "ROUTE_DELAY_RATE",
    "SEASON_DELAY_RATE", "TIME_BLOCK_DELAY_RATE",
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
def temporal_split(feat: pd.DataFrame) -> tuple:
    train = feat[feat["DAY_OF_MONTH"] <= 25].copy()
    test  = feat[feat["DAY_OF_MONTH"] >  25].copy()
    print(f"\n[SPLIT] Train: {len(train):,}  |  Test: {len(test):,}")
    print(f"  Train mean delay: {train[TARGET].mean():.2f} min  |  Test: {test[TARGET].mean():.2f} min")
    return train, test


# ============================================================
# STEP 8 — OPTUNA TUNING
# ============================================================
def optuna_tune(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50) -> tuple:
    print(f"\n{'='*60}")
    print(f"  OPTUNA HPO — {n_trials} trials (minimize RMSE, PRIMARY REGRESSOR)")
    print(f"{'='*60}")

    sample_size = min(len(X_train), 80_000)
    if len(X_train) > sample_size:
        idx = np.random.RandomState(42).choice(len(X_train), sample_size, replace=False)
        Xt, yt = X_train.iloc[idx], y_train.iloc[idx]
    else:
        Xt, yt = X_train, y_train

    best_score = [float("inf")]

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth":        trial.suggest_int("max_depth", 4, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma":            trial.suggest_float("gamma", 0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_delta_step":   trial.suggest_int("max_delta_step", 0, 5),
            "eval_metric":      "rmse",
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        0,
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        rmses = []
        for ti, vi in kf.split(Xt):
            m = XGBRegressor(**params)
            m.fit(Xt.iloc[ti], yt.iloc[ti])
            preds = m.predict(Xt.iloc[vi]).clip(min=0)
            rmses.append(np.sqrt(mean_squared_error(yt.iloc[vi], preds)))
        mean_rmse = float(np.mean(rmses))
        if mean_rmse < best_score[0]:
            best_score[0] = mean_rmse
            print(f"  Trial {trial.number+1:>3}/{n_trials}: RMSE={mean_rmse:.3f} ★ NEW BEST")
        elif (trial.number + 1) % 10 == 0:
            print(f"  Trial {trial.number+1:>3}/{n_trials}: RMSE={mean_rmse:.3f}  (best: {best_score[0]:.3f})")
        return mean_rmse

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    tune_time = time.time() - t0

    best = study.best_params
    best.update({"eval_metric": "rmse", "random_state": 42, "n_jobs": -1, "verbosity": 1})
    print(f"\n  Best RMSE: {study.best_value:.3f}  |  Tuning: {tune_time:.0f}s ({tune_time/60:.1f} min)")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    return best, study


# ============================================================
# STEP 9 — REGRESSION METRICS
# ============================================================
def compute_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = "Test") -> dict:
    y_pred_c = y_pred.clip(min=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_c))
    mae  = mean_absolute_error(y_true, y_pred_c)
    r2   = r2_score(y_true, y_pred_c)
    within_15 = float((np.abs(y_true - y_pred_c) <= 15).mean())
    within_30 = float((np.abs(y_true - y_pred_c) <= 30).mean())
    mask = y_true > 0
    rmse_del = np.sqrt(mean_squared_error(y_true[mask], y_pred_c[mask])) if mask.sum() > 0 else 0.0
    mae_del  = mean_absolute_error(y_true[mask], y_pred_c[mask]) if mask.sum() > 0 else 0.0
    return {
        "Set": name, "N": int(len(y_true)),
        "RMSE": round(float(rmse), 3), "MAE": round(float(mae), 3),
        "R2":   round(float(r2), 4),
        "Within_15min": round(within_15, 4),
        "Within_30min": round(within_30, 4),
        "RMSE_delayed_only": round(float(rmse_del), 3),
        "MAE_delayed_only":  round(float(mae_del), 3),
        "Mean_Actual_min":   round(float(y_true.mean()), 2),
        "Mean_Pred_min":     round(float(y_pred_c.mean()), 2),
    }


def print_reg_metrics(m: dict):
    print(f"\n{'='*60}")
    print(f"  {m['Set']}  (N={m['N']:,})")
    print(f"{'='*60}")
    print(f"  {'RMSE':<30} {m['RMSE']:.3f} min")
    print(f"  {'MAE':<30} {m['MAE']:.3f} min")
    print(f"  {'R²':<30} {m['R2']:.4f}")
    print(f"  {'Within ±15 min':<30} {m['Within_15min']*100:.2f}%")
    print(f"  {'Within ±30 min':<30} {m['Within_30min']*100:.2f}%")
    print(f"  {'RMSE (delayed only)':<30} {m['RMSE_delayed_only']:.3f} min")
    print(f"  {'MAE (delayed only)':<30} {m['MAE_delayed_only']:.3f} min")
    print(f"  {'Mean Actual':<30} {m['Mean_Actual_min']:.2f} min")
    print(f"  {'Mean Predicted':<30} {m['Mean_Pred_min']:.2f} min")


# ============================================================
# STEP 10 — CROSS-VALIDATION
# ============================================================
def cross_validate(X: pd.DataFrame, y: pd.Series, params: dict, n_folds: int = 5) -> dict:
    print(f"\n{'='*60}")
    print(f"  {n_folds}-FOLD CROSS-VALIDATION (Primary Regressor)")
    print(f"{'='*60}")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv = {k: [] for k in ["RMSE", "MAE", "R2", "Within_15min"]}
    for fold, (ti, vi) in enumerate(kf.split(X), 1):
        p = {**params, "verbosity": 0}
        m = XGBRegressor(**p)
        m.fit(X.iloc[ti], y.iloc[ti])
        preds = m.predict(X.iloc[vi]).clip(min=0)
        cv["RMSE"].append(np.sqrt(mean_squared_error(y.iloc[vi], preds)))
        cv["MAE"].append(mean_absolute_error(y.iloc[vi], preds))
        cv["R2"].append(r2_score(y.iloc[vi], preds))
        cv["Within_15min"].append(float((np.abs(y.iloc[vi].values - preds) <= 15).mean()))
        print(f"  Fold {fold}: RMSE={cv['RMSE'][-1]:.3f}  MAE={cv['MAE'][-1]:.3f}  R²={cv['R2'][-1]:.4f}")
    cv_summary = {}
    print(f"\n  {'Metric':<15} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*38}")
    for k, vals in cv.items():
        mean, std = np.mean(vals), np.std(vals)
        cv_summary[k] = {"mean": mean, "std": std,
                         "ci_low": mean - 1.96 * std, "ci_high": mean + 1.96 * std}
        print(f"  {k:<15} {mean:>10.4f} {std:>10.4f}")
    return cv_summary


# ============================================================
# STEP 11 — PLOTS
# ============================================================
def generate_plots(y_test: np.ndarray, y_pred: np.ndarray,
                   model, feature_names: list, tag: str = "primary_reg_v1"):
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    y_pred_c = y_pred.clip(min=0)
    print("\n[PLOTS]")

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred_c, alpha=0.2, s=5, color="darkorange")
    lim = max(y_test.max(), y_pred_c.max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Delay (min)"); ax.set_ylabel("Predicted Delay (min)")
    ax.set_title(f"Actual vs Predicted — {tag}"); ax.legend(); ax.grid(alpha=0.3)
    p = os.path.join(RESULTS_DIR, f"{tag}_actual_vs_pred.png")
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"  {p}")

    # Residuals
    residuals = y_test - y_pred_c
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Residual (Actual − Predicted, min)")
    ax.set_ylabel("Count"); ax.set_title(f"Residual Distribution — {tag}"); ax.grid(axis="y", alpha=0.3)
    p = os.path.join(RESULTS_DIR, f"{tag}_residuals.png")
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"  {p}")

    # Feature importance
    imp = model.feature_importances_
    idx = np.argsort(imp)
    top_n = min(30, len(feature_names))
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(range(top_n), [imp[i] for i in idx[-top_n:]], color="darkorange")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[-top_n:]], fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)"); ax.set_title(f"Feature Importance — {tag}"); ax.grid(axis="x", alpha=0.3)
    p = os.path.join(RESULTS_DIR, f"{tag}_feature_importance.png")
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"  {p}")


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()
    print("\n" + "=" * 70)
    print("  PRIMARY REGRESSOR v1 — Predict Exact Delay Minutes (WITH weather)")
    print("=" * 70)

    # Check weather file
    if not os.path.exists(WEATHER_FILE):
        print(f"\n[WARN] Weather file not found: {WEATHER_FILE}")
        print("  Falling back to base features only (no weather).")
        use_weather = False
    else:
        print(f"\n[INFO] Weather file found: {WEATHER_FILE}")
        use_weather = True

    raw_df = load_and_clean(RAW_FILE)
    feat = engineer_base_features(raw_df, "Oct 2025")

    wx_cols_present = []
    if use_weather:
        weather_df = pd.read_csv(WEATHER_FILE, low_memory=False)
        print(f"  Weather dataset shape: {weather_df.shape}")
        feat, wx_cols_present = add_weather_features(feat, weather_df)

    feat, agg_stats = add_aggregate_rates(feat)
    feat, encoders  = encode_categoricals(feat)

    # Build feature columns
    dynamic_wx_cols = [c for c in feat.columns if c.startswith("WX_")]
    FEATURE_COLS = (BASE_FEATURE_COLUMNS
                    + dynamic_wx_cols
                    + [c for c in WEATHER_EXTRA_COLS if c in feat.columns])
    FEATURE_COLS = [c for c in FEATURE_COLS if c in feat.columns]
    print(f"\n  Total features: {len(FEATURE_COLS)}")

    train_df, test_df = temporal_split(feat)
    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df[TARGET]
    X_test  = test_df[FEATURE_COLS].fillna(0)
    y_test  = test_df[TARGET]

    best_params, study = optuna_tune(X_train, y_train, n_trials=50)

    print(f"\n{'='*60}")
    print(f"  TRAINING FINAL PRIMARY REGRESSION MODEL")
    print(f"{'='*60}")
    model = XGBRegressor(**best_params)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    y_pred_test = model.predict(X_test)
    m_test = compute_reg_metrics(y_test.values, y_pred_test, "Test (days 26-31)")
    print_reg_metrics(m_test)

    # Feature importance
    print(f"\n{'='*60}")
    print(f"  TOP 20 FEATURES")
    print(f"{'='*60}")
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    for i in range(min(20, len(idx))):
        bar = "█" * int(imp[idx[i]] * 60)
        print(f"  {i+1:>2}. {FEATURE_COLS[idx[i]]:<40} {imp[idx[i]]:.4f} {bar}")

    # Cross-validation
    print("\n[INFO] Running 5-fold CV...")
    X_all = feat[FEATURE_COLS].fillna(0)
    y_all = feat[TARGET]
    cv_summary = cross_validate(X_all, y_all, best_params, n_folds=5)

    generate_plots(y_test.values, y_pred_test, model, FEATURE_COLS, tag="primary_reg_v1")

    # ─── Save Artifacts ───
    print("\n[SAVING ARTIFACTS]")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "primary_reg_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved {model_path}")

    reg_config = {
        "features":         FEATURE_COLS,
        "target_column":    TARGET,
        "model_type":       "XGBoost Regressor Primary v1",
        "version":          "primary_reg_v1",
        "best_params":      best_params,
        "optuna_best_rmse": float(study.best_value),
        "training_data":    "Oct 2025 (days 1-25)",
        "test_data":        "Oct 2025 (days 26-31)",
        "train_size":       int(len(X_train)),
        "n_features":       len(FEATURE_COLS),
        "has_weather":      use_weather,
        "weather_cols":     dynamic_wx_cols + WEATHER_EXTRA_COLS,
        "training_time_s":  float(train_time),
    }
    config_path = os.path.join(MODELS_DIR, "primary_reg_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(reg_config, f)
    print(f"  Saved {config_path}")

    bench_path = os.path.join(RESULTS_DIR, "primary_reg_v1_benchmarks.csv")
    pd.DataFrame([m_test]).to_csv(bench_path, index=False)
    print(f"  Saved {bench_path}")

    cv_path = os.path.join(RESULTS_DIR, "primary_reg_v1_cv.csv")
    pd.DataFrame(cv_summary).T.to_csv(cv_path)
    print(f"  Saved {cv_path}")

    trials_path = os.path.join(RESULTS_DIR, "primary_reg_v1_optuna_trials.csv")
    pd.DataFrame([{"trial": t.number, "rmse": t.value, **t.params}
                  for t in study.trials if t.value is not None]).to_csv(trials_path, index=False)
    print(f"  Saved {trials_path}")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE — Primary Regressor v1")
    print(f"{'='*70}")
    print(f"  Total time:        {total_time:.0f}s  ({total_time/60:.1f} min)")
    print(f"  Features:          {len(FEATURE_COLS)}  (45 base + weather)")
    print(f"  Optuna best RMSE:  {study.best_value:.3f} min")
    print(f"\n  Test Set Results:")
    print(f"    RMSE:            {m_test['RMSE']:.3f} min")
    print(f"    MAE:             {m_test['MAE']:.3f} min")
    print(f"    R²:              {m_test['R2']:.4f}")
    print(f"    Within ±15 min:  {m_test['Within_15min']*100:.2f}%")
    print(f"    Within ±30 min:  {m_test['Within_30min']*100:.2f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
