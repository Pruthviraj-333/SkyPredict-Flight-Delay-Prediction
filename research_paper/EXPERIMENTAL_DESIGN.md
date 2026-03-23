# Experimental Design

## 1. Data Splitting (Temporal Split)
Unlike random partitioning, this research uses a **Temporal Split** to mirror real-world deployment:
- **Training Set**: First 25 days of the month (~480,000 records).
- **Testing Set**: Last 6 days of the month (~115,000 records).
- *Rationale*: Avoids "future look-ahead" bias where models learn from future data to predict past flights.

## 2. Hyperparameter Optimization (HPO)
- **Framework**: Optuna (Bayesian Optimization).
- **Trials**: 50 independent trials per model.
- **Search Space**:
    - `n_estimators`: 200–800
    - `max_depth`: 3–10
    - `learning_rate`: 0.01–0.1
    - `subsample`: 0.5–1.0
    - `colsample_bytree`: 0.5–1.0
    - `gamma`: 0–5
    - `min_child_weight`: 1–10

## 3. Evaluation Metrics
1. **Primary**: **RMSE** (penalizes large errors heavily; crucial for airline scheduling).
2. **Secondary**: **MAE** (provides an intuitive "average error" in minutes).
3. **R² Score**: Measures the variance explained by the model.
4. **Operational Benchmark**: Accuracy @ ±15 min and Accuracy @ ±30 min (practical utility metrics).

## 4. Hardware/Software Setup
- **Environment**: Python 3.11, XGBoost 2.0+, Pandas, Scikit-Learn 1.8.0.
- **Hardware**: Multi-core CPU training (Optuna parallelization).
