# Flight Delay Regression Metrics

This document provides the performance benchmarks for the regression models integrated into SkyPredict. These models predict the number of minutes a flight will be delayed.

## 1. Model Summary

| Model Type | Primary (Weather-Aware) | Fallback (Base) |
| :--- | :--- | :--- |
| **Algorithm** | XGBoost Regressor | XGBoost Regressor |
| **Target Variable** | `ARR_DELAY_MINUTES` | `ARR_DELAY_MINUTES` |
| **Features** | 45 Base + 25 Weather | 45 Base |
| **Optimization** | Optuna (RMSE) | Optuna (RMSE) |

## 2. Performance Metrics (5-Fold CV)

| Metric | Primary | Fallback |
| :--- | :--- | :--- |
| **Root Mean Squared Error (RMSE)** | 43.92 min | 43.90 min |
| **Mean Absolute Error (MAE)** | 17.20 min | 17.21 min |
| **R² Score** | 0.114 | 0.116 |
| **Within ±15 Minutes** | 67.3% | 67.3% |
| **Within ±30 Minutes** | 86.5% | 86.5% |

## 3. Artifact Details

Artifacts are located in the `models/` directory:
- `primary_reg_model.pkl` (32.6 MB)
- `primary_reg_config.pkl`
- `fallback_reg_model.pkl` (12.8 MB)
- `fallback_reg_config.pkl`

## 4. Usage

The `ModelService` in `backend/model_service.py` automatically selects the Primary model if weather data is available and the Fallback model otherwise. Negative predictions are clipped to 0.
