# Model Evolution: Classification to Regression

SkyPredict employs a **Dual-Modeling Strategy** that separates the "Risk of Delay" from the "Magnitude of Delay."

## 1. Initial Phase: Binary Classification
The project began by predicting if a flight would be delayed by more than 15 minutes (`is_delayed`).
- **Algorithm**: XGBoost (Classifier).
- **Performance**: 80%–92% Accuracy / 0.85+ F1-Score.
- **Role**: Serves as the first "gate" in the prediction pipeline. If the probability of delay is low, the system reports an on-time estimate.

## 2. Advanced Phase: Minute-Level Regression
The recent upgrade introduced regression models to predict exact minutes.
- **Algorithm**: XGBoost (Regressor).
- **Metrics**: 17.2 min MAE / 86% within ±30 min.
- **Role**: Activated when the classifier suggests a high risk of delay, providing actionable time estimates for passengers.

## 3. Why Two Models?
Academic research suggests that a single model often struggles to differentiate between "No Delay" and "Small Delay." By using a **Log-Normal or Hurdle-Style architecture** (Classification then Regression), we achieve:
1.  **Lower False Positives**: The classifier handles the "zero-inflated" nature of the data.
2.  **Higher Tail Precision**: The regressor can focus specifically on the intensity of delays without being biased by thousands of on-time records.

## 4. Hyperparameter Optimization Tuning Evolution
- **Baseline**: Manual grid search (limited scope).
- **Current**: Bayesian optimization via **Optuna**.
    - Optimization focuses on RMSE for regression to minimize high-cost large errors.
    - Achieved a ~15% reduction in error magnitude compared to the baseline classifier-only estimates.
