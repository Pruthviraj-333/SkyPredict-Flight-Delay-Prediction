"""
Train the Fallback Model for Flight Delay Prediction.

Uses XGBoost classifier trained on historical flight patterns ONLY.
No weather data, no actual departure delay — avoids data leakage.

Features used:
- Carrier, Origin, Destination (label-encoded)
- Departure Hour, Day of Week, Month
- Distance, Scheduled Elapsed Time
- Aggregate delay rates (carrier, origin, dest, route, hour, day-of-week)
- Weekend flag, time block, distance group

Outputs:
- Trained model saved to models/fallback_model.pkl
- Prints train/test accuracy, classification report, confusion matrix, ROC-AUC
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# Try XGBoost first, fall back to sklearn GBM
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
    print("[INFO] Using XGBoost")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGBOOST = False
    print("[INFO] XGBoost not found, using sklearn GradientBoosting")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Features to use for training (no leakage — all pre-departure info)
FEATURE_COLUMNS = [
    'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'ARR_HOUR',
    'IS_WEEKEND', 'TIME_BLOCK', 'DISTANCE', 'CRS_ELAPSED_TIME', 'DISTANCE_GROUP',
    'CARRIER_ENCODED', 'ORIGIN_ENCODED', 'DEST_ENCODED',
    'CARRIER_DELAY_RATE', 'ORIGIN_DELAY_RATE', 'DEST_DELAY_RATE',
    'HOUR_DELAY_RATE', 'DOW_DELAY_RATE', 'ROUTE_DELAY_RATE',
]

TARGET_COLUMN = 'IS_DELAYED'


def load_data():
    """Load processed dataset."""
    path = os.path.join(PROCESSED_DIR, "fallback_dataset.csv")
    print(f"[INFO] Loading processed data from {path}...")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Target distribution:\n{df[TARGET_COLUMN].value_counts()}")
    print(f"[INFO] Delay rate: {df[TARGET_COLUMN].mean()*100:.1f}%")
    return df


def train_model(df):
    """Train XGBoost model with stratified train/test split."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # Handle any NaN in features
    X = X.fillna(0)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Stratified split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples ({y_train.mean()*100:.1f}% delayed)")
    print(f"Test set:  {X_test.shape[0]} samples ({y_test.mean()*100:.1f}% delayed)")
    
    # Calculate scale_pos_weight for class imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Class imbalance ratio: {scale_pos_weight:.2f}")
    
    # Train model
    if USE_XGBOOST:
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
    else:
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
    
    print("\n[TRAINING] Fitting model...")
    model.fit(X_train, y_train)
    print("[TRAINING] Model trained successfully!")
    
    # Evaluate
    evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Feature importance
    print_feature_importance(model, FEATURE_COLUMNS)
    
    # Save model
    save_model(model)
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Print comprehensive evaluation metrics."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n{'Metric':<25} {'Train':>10} {'Test':>10}")
    print("-" * 47)
    print(f"{'Accuracy':<25} {train_acc:>10.4f} {test_acc:>10.4f}")
    print(f"{'F1 Score':<25} {f1_score(y_train, y_train_pred):>10.4f} {f1_score(y_test, y_test_pred):>10.4f}")
    print(f"{'Precision':<25} {precision_score(y_train, y_train_pred):>10.4f} {precision_score(y_test, y_test_pred):>10.4f}")
    print(f"{'Recall':<25} {recall_score(y_train, y_train_pred):>10.4f} {recall_score(y_test, y_test_pred):>10.4f}")
    print(f"{'ROC-AUC':<25} {'':>10} {roc_auc_score(y_test, y_test_proba):>10.4f}")
    
    # Classification Report
    print(f"\n--- Classification Report (Test Set) ---")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['On-Time', 'Delayed']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"--- Confusion Matrix (Test Set) ---")
    print(f"                  Predicted")
    print(f"                  On-Time  Delayed")
    print(f"  Actual On-Time  {cm[0][0]:>7}  {cm[0][1]:>7}")
    print(f"  Actual Delayed  {cm[1][0]:>7}  {cm[1][1]:>7}")
    
    # Additional analysis
    print(f"\n--- Error Analysis ---")
    total = cm.sum()
    print(f"  True Positives (correctly predicted delayed):  {cm[1][1]} ({cm[1][1]/total*100:.1f}%)")
    print(f"  True Negatives (correctly predicted on-time):  {cm[0][0]} ({cm[0][0]/total*100:.1f}%)")
    print(f"  False Positives (predicted delayed, was on-time): {cm[0][1]} ({cm[0][1]/total*100:.1f}%)")
    print(f"  False Negatives (predicted on-time, was delayed): {cm[1][0]} ({cm[1][0]/total*100:.1f}%)")


def print_feature_importance(model, feature_names):
    """Print feature importance ranking."""
    print(f"\n--- Feature Importance ---")
    
    if USE_XGBOOST:
        importance = model.feature_importances_
    else:
        importance = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importance)[::-1]
    
    for i, idx in enumerate(indices):
        bar = "█" * int(importance[idx] * 50)
        print(f"  {i+1:>2}. {feature_names[idx]:<25} {importance[idx]:.4f} {bar}")


def save_model(model):
    """Save the trained model."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "fallback_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n[SAVED] Model saved to {model_path}")
    
    # Also save the feature column list
    config_path = os.path.join(MODELS_DIR, "model_config.pkl")
    config = {
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'model_type': 'XGBoost' if USE_XGBOOST else 'GradientBoosting',
    }
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"[SAVED] Model config saved to {config_path}")


def main():
    df = load_data()
    model = train_model(df)
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — Model ready for predictions!")
    print("=" * 60)


if __name__ == "__main__":
    main()
