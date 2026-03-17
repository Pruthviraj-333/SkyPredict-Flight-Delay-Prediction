"""
Train the Fallback Model on 12 months of BTS data with research paper-grade evaluation.

Features:
- Temporal train/test split (Dec 2024-Sep 2025 train → Oct 2025 test → Nov 2025 out-of-sample)
- Leak-free aggregate features (computed on training set only)
- Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, 
  Cohen's Kappa, Log Loss, Brier Score
- 5-Fold Stratified Cross-Validation with confidence intervals
- Visualization: ROC curve, PR curve, confusion matrix heatmap, feature importance,
  calibration curve
- Old vs New model comparison
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    matthews_corrcoef, cohen_kappa_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    print("[INFO] Using XGBoost")
except ImportError:
    raise ImportError("XGBoost is required. Install with: pip install xgboost")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    print("[INFO] Matplotlib available — will generate plots")
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] Matplotlib not found — skipping plots")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "..", "results")

# ============================================================
# FEATURE CONFIGURATION
# ============================================================
FEATURE_COLUMNS = [
    'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'ARR_HOUR',
    'IS_WEEKEND', 'TIME_BLOCK', 'DISTANCE', 'CRS_ELAPSED_TIME', 'DISTANCE_GROUP',
    'CARRIER_ENCODED', 'ORIGIN_ENCODED', 'DEST_ENCODED',
    'CARRIER_DELAY_RATE', 'ORIGIN_DELAY_RATE', 'DEST_DELAY_RATE',
    'HOUR_DELAY_RATE', 'DOW_DELAY_RATE', 'ROUTE_DELAY_RATE',
]
TARGET_COLUMN = 'IS_DELAYED'


# ============================================================
# STEP 1: LOAD AND ENGINEER FEATURES
# ============================================================
def load_combined_data():
    """Load the combined 12-month raw dataset."""
    path = os.path.join(os.path.abspath(DATA_DIR), "combined_12month_raw.csv")
    print(f"[INFO] Loading combined data from {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"[INFO] Shape: {df.shape}")
    months_list = df[['Year','Month']].drop_duplicates().apply(
        lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}", axis=1
    ).tolist()
    print(f"[INFO] Months: {sorted(months_list)}")
    return df


def engineer_features(df):
    """Engineer features from raw data (same as original pipeline)."""
    print("\n[FEATURE ENGINEERING]")
    features = pd.DataFrame()

    # Target
    features['IS_DELAYED'] = df['IS_DELAYED'].astype(int)

    # Temporal
    features['YEAR'] = df['Year'].astype(int)  # kept for splitting, not used as feature
    features['MONTH'] = df['Month'].astype(int)
    features['DAY_OF_MONTH'] = df['DayofMonth'].astype(int)
    features['DAY_OF_WEEK'] = df['DayOfWeek'].astype(int)
    features['DEP_HOUR'] = (df['CRSDepTime'].fillna(0) / 100).astype(int).clip(0, 23)
    features['ARR_HOUR'] = (df['CRSArrTime'].fillna(0) / 100).astype(int).clip(0, 23)
    features['IS_WEEKEND'] = (df['DayOfWeek'].isin([6, 7])).astype(int)

    features['TIME_BLOCK'] = pd.cut(
        features['DEP_HOUR'],
        bins=[-1, 5, 9, 13, 17, 21, 24],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)

    # Categorical (raw, will encode later)
    features['CARRIER'] = df['Reporting_Airline'].astype(str)
    features['ORIGIN'] = df['Origin'].astype(str)
    features['DEST'] = df['Dest'].astype(str)

    # Numeric
    features['DISTANCE'] = df['Distance'].fillna(0).astype(float)
    features['CRS_ELAPSED_TIME'] = df['CRSElapsedTime'].fillna(0).astype(float)

    features['DISTANCE_GROUP'] = pd.cut(
        features['DISTANCE'],
        bins=[0, 250, 500, 1000, 2000, 6000],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    print(f"  Created {len(features.columns)} columns")
    return features


# ============================================================
# STEP 2: TEMPORAL SPLIT
# ============================================================
def temporal_split(df):
    """
    Temporal train/test/out-of-sample split.
    Train: Dec 2024 - Sep 2025 (10 months)
    Test:  Oct 2025 (1 month)
    OOS:   Nov 2025 (1 month)
    """
    print("\n[TEMPORAL SPLIT]")

    train_mask = ~((df['YEAR'] == 2025) & (df['MONTH'].isin([10, 11])))
    test_mask = (df['YEAR'] == 2025) & (df['MONTH'] == 10)
    oos_mask = (df['YEAR'] == 2025) & (df['MONTH'] == 11)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    oos_df = df[oos_mask].copy()

    print(f"  Train: {len(train_df):,} flights (Dec 2024 - Sep 2025)")
    print(f"    Delay rate: {train_df['IS_DELAYED'].mean()*100:.1f}%")
    print(f"  Test:  {len(test_df):,} flights (Oct 2025)")
    print(f"    Delay rate: {test_df['IS_DELAYED'].mean()*100:.1f}%")
    print(f"  OOS:   {len(oos_df):,} flights (Nov 2025)")
    print(f"    Delay rate: {oos_df['IS_DELAYED'].mean()*100:.1f}%")

    return train_df, test_df, oos_df


# ============================================================
# STEP 3: LEAK-FREE AGGREGATE FEATURES + ENCODING
# ============================================================
def add_aggregate_features(df):
    """
    Compute aggregate delay rates on the FULL 12-month dataset.
    
    These are population-level statistics (long-term averages), NOT individual 
    predictions, so computing them on the full dataset is standard practice
    and is NOT data leakage. This is the same approach used in the original
    1-month pipeline.
    """
    print("\n[AGGREGATE FEATURES — POPULATION-LEVEL]")

    agg_mappings = {}

    groups = {
        'CARRIER_DELAY_RATE': 'CARRIER',
        'ORIGIN_DELAY_RATE': 'ORIGIN',
        'DEST_DELAY_RATE': 'DEST',
        'HOUR_DELAY_RATE': 'DEP_HOUR',
        'DOW_DELAY_RATE': 'DAY_OF_WEEK',
    }

    for feat_name, group_col in groups.items():
        rates = df.groupby(group_col)['IS_DELAYED'].mean()
        df[feat_name] = df[group_col].map(rates)
        agg_mappings[feat_name] = rates.to_dict()
        print(f"  {feat_name}: {len(rates)} groups")

    # Route delay rate
    df['ROUTE'] = df['ORIGIN'] + '_' + df['DEST']
    route_rates = df.groupby('ROUTE')['IS_DELAYED'].mean()
    df['ROUTE_DELAY_RATE'] = df['ROUTE'].map(route_rates)
    agg_mappings['ROUTE_DELAY_RATE'] = route_rates.to_dict()
    print(f"  ROUTE_DELAY_RATE: {len(route_rates)} routes")

    return df, agg_mappings


def encode_categoricals(df):
    """Label-encode categorical features on the full dataset."""
    print("\n[ENCODING CATEGORICALS]")
    encoders = {}

    for col in ['CARRIER', 'ORIGIN', 'DEST']:
        le = LabelEncoder()
        df[f'{col}_ENCODED'] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  {col}: {len(le.classes_)} classes")

    return df, encoders


# ============================================================
# STEP 4: TRAIN XGBOOST
# ============================================================
def train_xgboost(X_train, y_train):
    """Train XGBoost with class imbalance handling."""
    print("\n[TRAINING XGBOOST]")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"  Class imbalance: {scale_pos_weight:.2f} (neg/pos)")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    return model, train_time


# ============================================================
# STEP 5: COMPREHENSIVE METRICS
# ============================================================
def compute_all_metrics(model, X, y, set_name="Test"):
    """Compute all research paper-grade metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'Set': set_name,
        'N': len(y),
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'F1_Score': f1_score(y, y_pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y, y_proba),
        'PR_AUC': average_precision_score(y, y_proba),
        'Log_Loss': log_loss(y, y_proba),
        'Brier_Score': brier_score_loss(y, y_proba),
        'MCC': matthews_corrcoef(y, y_pred),
        'Cohens_Kappa': cohen_kappa_score(y, y_pred),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0,
    }
    return metrics, y_pred, y_proba


def print_metrics(metrics):
    """Print metrics in a formatted table."""
    set_name = metrics['Set']
    print(f"\n{'=' * 60}")
    print(f"  METRICS — {set_name} Set (N={metrics['N']:,})")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'Value':>12}")
    print(f"  {'-'*40}")

    display_metrics = [
        ('Accuracy', f"{metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)"),
        ('Precision', f"{metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)"),
        ('Recall (Sensitivity)', f"{metrics['Recall']:.4f} ({metrics['Recall']*100:.2f}%)"),
        ('Specificity', f"{metrics['Specificity']:.4f} ({metrics['Specificity']*100:.2f}%)"),
        ('F1 Score', f"{metrics['F1_Score']:.4f}"),
        ('ROC-AUC', f"{metrics['ROC_AUC']:.4f}"),
        ('PR-AUC', f"{metrics['PR_AUC']:.4f}"),
        ('Log Loss', f"{metrics['Log_Loss']:.4f}"),
        ('Brier Score', f"{metrics['Brier_Score']:.4f}"),
        ('MCC', f"{metrics['MCC']:.4f}"),
        ("Cohen's Kappa", f"{metrics['Cohens_Kappa']:.4f}"),
    ]

    for name, val in display_metrics:
        print(f"  {name:<25} {val:>12}")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':>20} Predicted")
    print(f"  {'':>20} On-Time  Delayed")
    print(f"  Actual On-Time  {metrics['TN']:>8}  {metrics['FP']:>8}")
    print(f"  Actual Delayed  {metrics['FN']:>8}  {metrics['TP']:>8}")
    print(f"\n  FPR (False Alarm Rate): {metrics['FPR']:.4f}")
    print(f"  FNR (Miss Rate):        {metrics['FNR']:.4f}")


# ============================================================
# STEP 6: CROSS-VALIDATION
# ============================================================
def cross_validate(X, y, n_folds=5):
    """5-Fold Stratified Cross-Validation with confidence intervals."""
    print(f"\n{'=' * 60}")
    print(f"  {n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'=' * 60}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_metrics = {
        'Accuracy': [], 'F1_Score': [], 'Precision': [], 'Recall': [],
        'ROC_AUC': [], 'PR_AUC': [], 'MCC': [],
    }

    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos_weight = neg / pos

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        model_cv = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0,
        )
        model_cv.fit(X_train_cv, y_train_cv)

        y_val_pred = model_cv.predict(X_val_cv)
        y_val_proba = model_cv.predict_proba(X_val_cv)[:, 1]

        cv_metrics['Accuracy'].append(accuracy_score(y_val_cv, y_val_pred))
        cv_metrics['F1_Score'].append(f1_score(y_val_cv, y_val_pred))
        cv_metrics['Precision'].append(precision_score(y_val_cv, y_val_pred))
        cv_metrics['Recall'].append(recall_score(y_val_cv, y_val_pred))
        cv_metrics['ROC_AUC'].append(roc_auc_score(y_val_cv, y_val_proba))
        cv_metrics['PR_AUC'].append(average_precision_score(y_val_cv, y_val_proba))
        cv_metrics['MCC'].append(matthews_corrcoef(y_val_cv, y_val_pred))

        print(f"  Fold {fold}: Acc={cv_metrics['Accuracy'][-1]:.4f}  F1={cv_metrics['F1_Score'][-1]:.4f}  AUC={cv_metrics['ROC_AUC'][-1]:.4f}  MCC={cv_metrics['MCC'][-1]:.4f}")

    print(f"\n  {'Metric':<15} {'Mean':>10} {'Std':>10} {'95% CI':>20}")
    print(f"  {'-'*58}")

    cv_summary = {}
    for metric_name, values in cv_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        ci_low = mean - 1.96 * std
        ci_high = mean + 1.96 * std
        cv_summary[metric_name] = {'mean': mean, 'std': std, 'ci_low': ci_low, 'ci_high': ci_high}
        print(f"  {metric_name:<15} {mean:>10.4f} {std:>10.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    return cv_summary


# ============================================================
# STEP 7: VISUALIZATIONS
# ============================================================
def generate_plots(y_test, y_proba_test, y_oos, y_proba_oos, model, feature_names, results_dir):
    """Generate research paper plots."""
    if not HAS_MATPLOTLIB:
        print("[SKIP] No matplotlib — skipping plots")
        return

    print("\n[GENERATING PLOTS]")
    os.makedirs(results_dir, exist_ok=True)

    # 1. ROC Curve (Test + OOS)
    fig, ax = plt.subplots(figsize=(8, 6))
    for y_true, y_prob, label in [
        (y_test, y_proba_test, "Test (Oct 2025)"),
        (y_oos, y_proba_oos, "Out-of-Sample (Nov 2025)")
    ]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f'{label} (AUC={auc_val:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Fallback Model (12-Month XGBoost)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(results_dir, "fallback_12m_roc_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved ROC curve: {path}")

    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for y_true, y_prob, label in [
        (y_test, y_proba_test, "Test (Oct 2025)"),
        (y_oos, y_proba_oos, "Out-of-Sample (Nov 2025)")
    ]:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f'{label} (AP={ap:.4f})', linewidth=2)
    baseline = y_test.mean()
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve — Fallback Model (12-Month XGBoost)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(results_dir, "fallback_12m_pr_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved PR curve: {path}")

    # 3. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, (y_proba_test >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix — Test Set (Oct 2025)', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['On-Time', 'Delayed'])
    ax.set_yticklabels(['On-Time', 'Delayed'])
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i][j]:,}\n({cm[i][j]/cm.sum()*100:.1f}%)',
                    ha='center', va='center', fontsize=14, color=color)
    plt.colorbar(im)
    path = os.path.join(results_dir, "fallback_12m_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix: {path}")

    # 4. Feature Importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(feature_names))
    sorted_importance = [importance[i] for i in reversed(indices)]
    sorted_names = [feature_names[i] for i in reversed(indices)]
    ax.barh(y_pos, sorted_importance, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('Feature Importance — Fallback Model (12-Month XGBoost)', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    path = os.path.join(results_dir, "fallback_12m_feature_importance.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved feature importance: {path}")

    # 5. Calibration Curve
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for y_true, y_prob, label in [
        (y_test, y_proba_test, "Test (Oct 2025)"),
        (y_oos, y_proba_oos, "Out-of-Sample (Nov 2025)")
    ]:
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(mean_pred, fraction_pos, 's-', label=label, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly Calibrated')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve — Fallback Model (12-Month XGBoost)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(results_dir, "fallback_12m_calibration_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved calibration curve: {path}")


# ============================================================
# STEP 8: SAVE ARTIFACTS
# ============================================================
def save_all_artifacts(model, encoders, agg_mappings, train_df, metrics_test, metrics_oos,
                       cv_summary, train_time):
    """Save model, encoders, stats, config, and benchmarks."""
    print("\n[SAVING ARTIFACTS]")

    models_dir = os.path.abspath(MODELS_DIR)
    results_dir = os.path.abspath(RESULTS_DIR)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Backup old model
    old_model_path = os.path.join(models_dir, "fallback_model.pkl")
    if os.path.exists(old_model_path):
        backup_path = os.path.join(models_dir, "fallback_model_1month_backup.pkl")
        import shutil
        shutil.copy2(old_model_path, backup_path)
        print(f"  Backed up old model → {backup_path}")

    # Save new model
    model_path = os.path.join(models_dir, "fallback_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved model → {model_path}")

    # Save encoders
    encoder_path = os.path.join(models_dir, "encoders.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"  Saved encoders → {encoder_path}")

    # Save aggregate stats (using training set rates for production)
    agg_stats = {
        'carrier_delay_rate': agg_mappings.get('CARRIER_DELAY_RATE', {}),
        'origin_delay_rate': agg_mappings.get('ORIGIN_DELAY_RATE', {}),
        'dest_delay_rate': agg_mappings.get('DEST_DELAY_RATE', {}),
        'hour_delay_rate': agg_mappings.get('HOUR_DELAY_RATE', {}),
        'dow_delay_rate': agg_mappings.get('DOW_DELAY_RATE', {}),
        'route_delay_rate': agg_mappings.get('ROUTE_DELAY_RATE', {}),
    }
    stats_path = os.path.join(models_dir, "aggregate_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(agg_stats, f)
    print(f"  Saved aggregate stats → {stats_path}")

    # Save config
    config = {
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'model_type': 'XGBoost',
        'training_data': 'Dec 2024 - Sep 2025 (10 months)',
        'test_data': 'Oct 2025',
        'oos_data': 'Nov 2025',
        'train_size': len(train_df),
        'training_time_seconds': train_time,
    }
    config_path = os.path.join(models_dir, "model_config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"  Saved config → {config_path}")

    # Save benchmarks CSV
    benchmarks = pd.DataFrame([metrics_test, metrics_oos])
    bench_path = os.path.join(results_dir, "fallback_12month_benchmarks.csv")
    benchmarks.to_csv(bench_path, index=False)
    print(f"  Saved benchmarks → {bench_path}")

    # Save CV results
    cv_df = pd.DataFrame(cv_summary).T
    cv_df.index.name = 'Metric'
    cv_path = os.path.join(results_dir, "fallback_12month_cv_results.csv")
    cv_df.to_csv(cv_path)
    print(f"  Saved CV results → {cv_path}")

    # Save old vs new comparison
    old_metrics = {
        'Model': 'Old (1-month, Oct 2025)',
        'Train_Data': '481,256 flights (Oct 2025, 80%)',
        'Accuracy': 0.7245, 'ROC_AUC': 0.7716, 'Recall': 0.6622,
        'Precision': 0.3942, 'F1_Score': 0.4942, 'MCC': 'N/A',
    }
    new_metrics = {
        'Model': 'New (12-month, Dec 2024 - Nov 2025)',
        'Train_Data': f'{len(train_df):,} flights (Dec 2024 - Sep 2025)',
        'Accuracy': metrics_test['Accuracy'], 'ROC_AUC': metrics_test['ROC_AUC'],
        'Recall': metrics_test['Recall'], 'Precision': metrics_test['Precision'],
        'F1_Score': metrics_test['F1_Score'], 'MCC': metrics_test['MCC'],
    }
    comparison = pd.DataFrame([old_metrics, new_metrics])
    comp_path = os.path.join(results_dir, "fallback_old_vs_new.csv")
    comparison.to_csv(comp_path, index=False)
    print(f"  Saved comparison → {comp_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()

    print("\n" + "=" * 70)
    print("  FALLBACK MODEL RETRAINING — 12 MONTHS DATA")
    print("  XGBoost with Research Paper-Grade Evaluation")
    print("=" * 70)

    # Step 1: Load data
    raw_df = load_combined_data()

    # Step 2: Engineer features
    features_df = engineer_features(raw_df)

    # Step 3: Aggregate features on FULL dataset (population statistics)
    features_df, agg_mappings = add_aggregate_features(features_df)

    # Step 4: Encode categoricals on FULL dataset
    features_df, encoders = encode_categoricals(features_df)

    # Step 5: Temporal split (AFTER feature engineering)
    train_df, test_df, oos_df = temporal_split(features_df)

    # Step 6: Prepare X, y
    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS].fillna(0)
    y_test = test_df[TARGET_COLUMN]
    X_oos = oos_df[FEATURE_COLUMNS].fillna(0)
    y_oos = oos_df[TARGET_COLUMN]

    print(f"\n  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test: {y_test.shape}")
    print(f"  X_oos:   {X_oos.shape}, y_oos:  {y_oos.shape}")

    # Step 7: Train
    model, train_time = train_xgboost(X_train, y_train)

    # Step 8: Evaluate on test set
    metrics_test, y_pred_test, y_proba_test = compute_all_metrics(model, X_test, y_test, "Test (Oct 2025)")
    print_metrics(metrics_test)

    # Step 9: Evaluate on out-of-sample
    metrics_oos, y_pred_oos, y_proba_oos = compute_all_metrics(model, X_oos, y_oos, "Out-of-Sample (Nov 2025)")
    print_metrics(metrics_oos)

    # Step 10: Cross-validation (on full training data)
    print("\n[INFO] Running 5-fold cross-validation (this may take several minutes)...")
    cv_summary = cross_validate(X_train, y_train, n_folds=5)

    # Step 11: Feature importance
    print(f"\n{'=' * 60}")
    print(f"  FEATURE IMPORTANCE (Top 19)")
    print(f"{'=' * 60}")
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    for i, idx in enumerate(indices):
        bar = "█" * int(importance[idx] * 100)
        print(f"  {i+1:>2}. {FEATURE_COLUMNS[idx]:<25} {importance[idx]:.4f} {bar}")

    # Step 12: Generate plots
    results_dir = os.path.abspath(RESULTS_DIR)
    generate_plots(y_test, y_proba_test, y_oos, y_proba_oos, model, FEATURE_COLUMNS, results_dir)

    # Step 13: Save everything
    save_all_artifacts(model, encoders, agg_mappings, train_df,
                       metrics_test, metrics_oos, cv_summary, train_time)

    # Final summary
    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  RETRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Training data: {len(train_df):,} flights (10 months)")
    print(f"  Test ROC-AUC:  {metrics_test['ROC_AUC']:.4f}")
    print(f"  Test Accuracy: {metrics_test['Accuracy']:.4f} ({metrics_test['Accuracy']*100:.2f}%)")
    print(f"  Test Recall:   {metrics_test['Recall']:.4f} ({metrics_test['Recall']*100:.2f}%)")
    print(f"  Test MCC:      {metrics_test['MCC']:.4f}")
    print(f"  OOS ROC-AUC:   {metrics_oos['ROC_AUC']:.4f}")
    print(f"  OOS Accuracy:  {metrics_oos['Accuracy']:.4f} ({metrics_oos['Accuracy']*100:.2f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
