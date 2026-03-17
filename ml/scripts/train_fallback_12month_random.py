"""
Train Fallback Model on 12-month data using RANDOM 80/20 stratified split.
This mirrors the old 1-month model's splitting strategy for apples-to-apples comparison.
All research paper-grade metrics are computed.
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
print("[INFO] Using XGBoost")

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

FEATURE_COLUMNS = [
    'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'ARR_HOUR',
    'IS_WEEKEND', 'TIME_BLOCK', 'DISTANCE', 'CRS_ELAPSED_TIME', 'DISTANCE_GROUP',
    'CARRIER_ENCODED', 'ORIGIN_ENCODED', 'DEST_ENCODED',
    'CARRIER_DELAY_RATE', 'ORIGIN_DELAY_RATE', 'DEST_DELAY_RATE',
    'HOUR_DELAY_RATE', 'DOW_DELAY_RATE', 'ROUTE_DELAY_RATE',
]
TARGET = 'IS_DELAYED'


def compute_metrics(model, X, y, name="Test"):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'Set': name, 'N': len(y),
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
    }, y_pred, y_proba


def print_metrics(m):
    print(f"\n{'=' * 60}")
    print(f"  METRICS — {m['Set']} (N={m['N']:,})")
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
    print(f"  {'':>20} Predicted On-Time  Delayed")
    print(f"  Actual On-Time  {m['TN']:>12,}  {m['FP']:>8,}")
    print(f"  Actual Delayed  {m['FN']:>12,}  {m['TP']:>8,}")
    print(f"  FPR: {m['FPR']:.4f}  |  FNR: {m['FNR']:.4f}")


def main():
    total_start = time.time()
    print("\n" + "=" * 70)
    print("  FALLBACK MODEL — 12-MONTH DATA, RANDOM 80/20 SPLIT")
    print("=" * 70)

    # Load
    path = os.path.join(os.path.abspath(DATA_DIR), "combined_12month_raw.csv")
    print(f"\n[LOADING] {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Shape: {df.shape}")

    # Feature engineering
    print("\n[FEATURE ENGINEERING]")
    feat = pd.DataFrame()
    feat[TARGET] = df['IS_DELAYED'].astype(int)
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
    print(f"  Created {len(feat.columns)} columns")

    # Aggregate features on FULL dataset
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

    # Save aggregate stats for production
    agg_stats = {
        'carrier_delay_rate': feat.groupby('CARRIER')[TARGET].mean().to_dict(),
        'origin_delay_rate': feat.groupby('ORIGIN')[TARGET].mean().to_dict(),
        'dest_delay_rate': feat.groupby('DEST')[TARGET].mean().to_dict(),
        'hour_delay_rate': feat.groupby('DEP_HOUR')[TARGET].mean().to_dict(),
        'dow_delay_rate': feat.groupby('DAY_OF_WEEK')[TARGET].mean().to_dict(),
        'route_delay_rate': route_rates.to_dict(),
    }

    # Encode categoricals
    print("\n[ENCODING]")
    encoders = {}
    for col in ['CARRIER', 'ORIGIN', 'DEST']:
        le = LabelEncoder()
        feat[f'{col}_ENCODED'] = le.fit_transform(feat[col])
        encoders[col] = le
        print(f"  {col}: {len(le.classes_)} classes")

    # RANDOM 80/20 stratified split
    print("\n[RANDOM SPLIT]")
    X = feat[FEATURE_COLUMNS].fillna(0)
    y = feat[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} ({y_train.mean()*100:.1f}% delayed)")
    print(f"  Test:  {len(X_test):,} ({y_test.mean()*100:.1f}% delayed)")

    # Train XGBoost
    print("\n[TRAINING]")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"  Class imbalance: {spw:.2f}")

    model = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # Train metrics
    m_train, _, _ = compute_metrics(model, X_train, y_train, "Train")
    print_metrics(m_train)

    # Test metrics
    m_test, y_pred_test, y_proba_test = compute_metrics(model, X_test, y_test, "Test (Random 20%)")
    print_metrics(m_test)

    # 5-Fold CV
    print(f"\n{'=' * 60}")
    print(f"  5-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'=' * 60}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = {'Accuracy':[], 'F1':[], 'Precision':[], 'Recall':[], 'ROC_AUC':[], 'PR_AUC':[], 'MCC':[]}
    for fold, (ti, vi) in enumerate(skf.split(X, y), 1):
        Xtr, Xv = X.iloc[ti], X.iloc[vi]
        ytr, yv = y.iloc[ti], y.iloc[vi]
        m_cv = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0,
        )
        m_cv.fit(Xtr, ytr)
        yp = m_cv.predict(Xv)
        ypr = m_cv.predict_proba(Xv)[:, 1]
        cv['Accuracy'].append(accuracy_score(yv, yp))
        cv['F1'].append(f1_score(yv, yp))
        cv['Precision'].append(precision_score(yv, yp))
        cv['Recall'].append(recall_score(yv, yp))
        cv['ROC_AUC'].append(roc_auc_score(yv, ypr))
        cv['PR_AUC'].append(average_precision_score(yv, ypr))
        cv['MCC'].append(matthews_corrcoef(yv, yp))
        print(f"  Fold {fold}: Acc={cv['Accuracy'][-1]:.4f}  F1={cv['F1'][-1]:.4f}  AUC={cv['ROC_AUC'][-1]:.4f}  MCC={cv['MCC'][-1]:.4f}")

    print(f"\n  {'Metric':<15} {'Mean':>10} {'Std':>10} {'95% CI':>22}")
    print(f"  {'-'*60}")
    cv_summary = {}
    for k, v in cv.items():
        mean, std = np.mean(v), np.std(v)
        cv_summary[k] = {'mean': mean, 'std': std, 'ci_low': mean-1.96*std, 'ci_high': mean+1.96*std}
        print(f"  {k:<15} {mean:>10.4f} {std:>10.4f} [{mean-1.96*std:.4f}, {mean+1.96*std:.4f}]")

    # Feature importance
    print(f"\n{'=' * 60}")
    print(f"  FEATURE IMPORTANCE")
    print(f"{'=' * 60}")
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    for i, ix in enumerate(idx):
        bar = "█" * int(imp[ix] * 100)
        print(f"  {i+1:>2}. {FEATURE_COLUMNS[ix]:<25} {imp[ix]:.4f} {bar}")

    # Plots
    if HAS_MATPLOTLIB:
        results_dir = os.path.abspath(RESULTS_DIR)
        os.makedirs(results_dir, exist_ok=True)

        # ROC
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        ax.plot(fpr, tpr, label=f'XGBoost 12-month (AUC={m_test["ROC_AUC"]:.4f})', linewidth=2, color='#2196F3')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve — Fallback Model (12-Month, Random Split)', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, "fallback_12m_random_roc.png"), dpi=150); plt.close()

        # PR
        fig, ax = plt.subplots(figsize=(8, 6))
        prec, rec, _ = precision_recall_curve(y_test, y_proba_test)
        ax.plot(rec, prec, label=f'XGBoost 12-month (AP={m_test["PR_AUC"]:.4f})', linewidth=2, color='#4CAF50')
        ax.axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.5, label=f'Baseline ({y_test.mean():.3f})')
        ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall — Fallback Model (12-Month, Random Split)', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, "fallback_12m_random_pr.png"), dpi=150); plt.close()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion Matrix — 12-Month Random Split', fontsize=14)
        ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Actual', fontsize=12)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['On-Time','Delayed']); ax.set_yticklabels(['On-Time','Delayed'])
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i,j] > cm.max()/2 else 'black'
                ax.text(j, i, f'{cm[i][j]:,}\n({cm[i][j]/cm.sum()*100:.1f}%)', ha='center', va='center', fontsize=14, color=color)
        plt.colorbar(im)
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, "fallback_12m_random_cm.png"), dpi=150); plt.close()

        # Feature importance
        fig, ax = plt.subplots(figsize=(10, 7))
        sorted_imp = [imp[i] for i in reversed(idx)]
        sorted_names = [FEATURE_COLUMNS[i] for i in reversed(idx)]
        ax.barh(range(len(sorted_names)), sorted_imp, color='steelblue')
        ax.set_yticks(range(len(sorted_names))); ax.set_yticklabels(sorted_names, fontsize=10)
        ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
        ax.set_title('Feature Importance — 12-Month Random Split', fontsize=14)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, "fallback_12m_random_fi.png"), dpi=150); plt.close()

        print("\n[PLOTS] Saved ROC, PR, CM, FI to results/")

    # Save artifacts
    print("\n[SAVING]")
    models_dir = os.path.abspath(MODELS_DIR)
    results_dir = os.path.abspath(RESULTS_DIR)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Backup and save model
    old_path = os.path.join(models_dir, "fallback_model.pkl")
    if os.path.exists(old_path):
        import shutil
        bak = os.path.join(models_dir, "fallback_model_temporal_backup.pkl")
        if not os.path.exists(bak):
            shutil.copy2(old_path, bak)
            print(f"  Backed up temporal model → {bak}")

    with open(old_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved model → {old_path}")

    with open(os.path.join(models_dir, "encoders.pkl"), 'wb') as f:
        pickle.dump(encoders, f)
    with open(os.path.join(models_dir, "aggregate_stats.pkl"), 'wb') as f:
        pickle.dump(agg_stats, f)

    config = {
        'feature_columns': FEATURE_COLUMNS, 'target_column': TARGET,
        'model_type': 'XGBoost', 'training_data': '12 months (Dec 2024-Nov 2025), random 80/20 split',
        'train_size': len(X_train), 'training_time_seconds': train_time,
    }
    with open(os.path.join(models_dir, "model_config.pkl"), 'wb') as f:
        pickle.dump(config, f)
    print("  Saved encoders, aggregate stats, config")

    # Benchmarks
    pd.DataFrame([m_train, m_test]).to_csv(
        os.path.join(results_dir, "fallback_12m_random_benchmarks.csv"), index=False)
    pd.DataFrame(cv_summary).T.to_csv(
        os.path.join(results_dir, "fallback_12m_random_cv.csv"))

    # Old vs new comparison
    comparison = pd.DataFrame([
        {'Model': 'Old (1-month Oct 2025)', 'Split': 'Random 80/20', 'Train_Size': '481,256',
         'Accuracy': 0.7245, 'ROC_AUC': 0.7716, 'Recall': 0.6622, 'Precision': 0.3942,
         'F1_Score': 0.4942, 'MCC': 'N/A'},
        {'Model': 'New (12-month)', 'Split': 'Random 80/20', 'Train_Size': f'{len(X_train):,}',
         'Accuracy': m_test['Accuracy'], 'ROC_AUC': m_test['ROC_AUC'],
         'Recall': m_test['Recall'], 'Precision': m_test['Precision'],
         'F1_Score': m_test['F1_Score'], 'MCC': m_test['MCC']},
    ])
    comp_path = os.path.join(results_dir, "fallback_old_vs_new_random.csv")
    comparison.to_csv(comp_path, index=False)
    print(f"  Saved comparison → {comp_path}")

    total = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE — Random 80/20 Split")
    print(f"{'=' * 70}")
    print(f"  Total time:    {total:.1f}s ({total/60:.1f} min)")
    print(f"  Train size:    {len(X_train):,} flights")
    print(f"  Test size:     {len(X_test):,} flights")
    print(f"  Test Accuracy: {m_test['Accuracy']:.4f} ({m_test['Accuracy']*100:.2f}%)")
    print(f"  Test ROC-AUC:  {m_test['ROC_AUC']:.4f}")
    print(f"  Test Recall:   {m_test['Recall']:.4f} ({m_test['Recall']*100:.2f}%)")
    print(f"  Test F1:       {m_test['F1_Score']:.4f}")
    print(f"  Test MCC:      {m_test['MCC']:.4f}")
    print(f"  Test Precision:{m_test['Precision']:.4f} ({m_test['Precision']*100:.2f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
