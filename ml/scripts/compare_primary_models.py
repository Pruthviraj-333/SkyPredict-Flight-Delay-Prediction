"""
Compare 10 ML algorithms on the weather-enhanced dataset (32 features).
Same algorithms as the fallback comparison, but with weather data included.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATASET = os.path.join(DATA_DIR, "processed", "weather_dataset.csv")


def get_models():
    """Return dict of all models to compare."""
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbose=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7, n_jobs=-1,
        ),
        "Linear SVM": LinearSVC(
            max_iter=2000, random_state=42,
        ),
    }


# Models that need feature scaling
NEEDS_SCALING = {"Logistic Regression", "KNN", "Linear SVM"}


def main():
    print("=" * 65)
    print("  PRIMARY MODEL — 10 ALGORITHM COMPARISON (Weather Dataset)")
    print("=" * 65)

    # Load dataset
    print("\n[1/3] Loading weather-enhanced dataset...")
    df = pd.read_csv(DATASET)
    y = df["IS_DELAYED"]
    X = df.drop("IS_DELAYED", axis=1)
    print(f"       Samples: {len(df):,} | Features: {X.shape[1]} | Delay rate: {y.mean()*100:.1f}%")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"       Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Scale for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    # Train and evaluate each model
    print(f"\n[2/3] Training 10 algorithms...\n")
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"  Training {name:25s} ...", end=" ", flush=True)

        try:
            # Set class weight where possible
            if hasattr(model, "scale_pos_weight"):
                model.set_params(scale_pos_weight=spw)
            elif hasattr(model, "class_weight"):
                model.set_params(class_weight="balanced")

            # Use scaled data for models that need it
            if name in NEEDS_SCALING:
                Xtr, Xte = X_train_scaled, X_test_scaled
            else:
                Xtr, Xte = X_train, X_test

            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)

            # ROC-AUC
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(Xte)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            elif hasattr(model, "decision_function"):
                y_scores = model.decision_function(Xte)
                auc = roc_auc_score(y_test, y_scores)
            else:
                auc = 0.0

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            results.append({
                "Algorithm": name,
                "Accuracy": round(acc * 100, 1),
                "ROC-AUC": round(auc, 4),
                "Delay_Recall": round(rec * 100, 1),
                "F1_Score": round(f1, 4),
            })

            print(f"Acc: {acc*100:5.1f}% | AUC: {auc:.4f} | Recall: {rec*100:5.1f}%")

        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "Algorithm": name, "Accuracy": 0, "ROC-AUC": 0,
                "Delay_Recall": 0, "F1_Score": 0,
            })

    # Sort by ROC-AUC (best overall metric)
    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    results_df.index = results_df.index + 1  # rank from 1

    # Display results
    print(f"\n[3/3] Results (ranked by ROC-AUC):\n")
    print("  " + "=" * 62)
    print(f"  {'Rank':>4}  {'Algorithm':<22} {'Acc':>6} {'AUC':>7} {'Recall':>7} {'F1':>7}")
    print("  " + "-" * 62)
    for i, row in results_df.iterrows():
        marker = " ★" if i == 1 else "  "
        print(f"  {i:>4}  {row['Algorithm']:<22} {row['Accuracy']:>5.1f}% {row['ROC-AUC']:>7.4f} {row['Delay_Recall']:>6.1f}% {row['F1_Score']:>7.4f}{marker}")
    print("  " + "=" * 62)

    # Winner
    winner = results_df.iloc[0]
    print(f"\n  🏆 BEST: {winner['Algorithm']}")
    print(f"     Accuracy: {winner['Accuracy']}% | ROC-AUC: {winner['ROC-AUC']} | Delay Recall: {winner['Delay_Recall']}%")

    # Analysis
    print(f"\n  Analysis:")
    print(f"  -" * 32)

    # High accuracy but low recall trap
    acc_trap = results_df[(results_df["Accuracy"] > 75) & (results_df["Delay_Recall"] < 25)]
    if len(acc_trap) > 0:
        print(f"  ⚠ Accuracy trap (high acc, low recall): {', '.join(acc_trap['Algorithm'].tolist())}")
        print(f"    These predict 'on-time' for everything → useless for delay detection")

    # Best balanced models
    balanced = results_df[(results_df["Delay_Recall"] > 50) & (results_df["ROC-AUC"] > 0.75)]
    if len(balanced) > 0:
        print(f"  ✓ Best balanced models: {', '.join(balanced['Algorithm'].tolist())}")

    # Save
    output = os.path.join(RESULTS_DIR, "primary_model_comparison.csv")
    results_df.to_csv(output, index=True, index_label="Rank")
    print(f"\n  Saved: {output}")


if __name__ == "__main__":
    main()
