# Project Abstract: SkyPredict

## Problem Statement
Flight delays cause billions of dollars in economic loss annually. While binary classification (predicting IF a delay will occur) is a common baseline, predicting the **exact magnitude of a delay** and providing **live tracking context** is critical for operational resilience.

## Research Objective
To develop a holistic framework—SkyPredict—that combines dual-model machine learning (Classification + Regression) with real-time weather integration and live flight tracking to provide comprehensive delay intelligence.

## Contributions
1. **Holistic Architecture**: A three-tier system integrating 12 months of historical BTS data with real-time METAR weather.
2. **Dual-Model Strategy**: Separation of delay risk (Classifier) and delay magnitude (Regressor).
3. **Operational Dashboards**: Real-time visualization for passenger and dispatcher use-cases.

## Outcomes
The system achieves **86.5% accuracy within a 30-minute window** for regression magnitude, powered by an optimized XGBoost pipeline. This confirms that modern GBTs, when combined with high-resolution historical features, provide a robust benchmark for aviation logistics.
