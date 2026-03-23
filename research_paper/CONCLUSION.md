# CONCLUSION

## 1. Research Impact
The SkyPredict project demonstrates that a high-precision, minute-level flight delay prediction system can be built using standard tabular data augmented with real-time weather information.

## 2. Key Takeaways
- **RMSE Optimization**: By focusing on RMSE during HPO, we significantly improved the model's reliability for "large delay" events, which are the most costly for airlines.
- **The Power of History**: Daily carrier and route delay rates remain the strongest baseline features, often outperforming complex weather signals for minor delays.
- **Operational Feasibility**: An accuracy of 86.5% within 30 minutes proves that machine learning can provide actionable intelligence for travelers and dispatchers.

## 3. Academic Value
This implementation provides a benchmark for:
- **Bayesian HPO (Optuna)** applied to large-scale aviation data.
- **Dual-Model** approaches for zero-inflated target variables.
- **Real-time API integration** (FastAPI) as a bridge between theoretical ML and production deployment.

## 4. Final Statement
While a "perfect" prediction is impossible due to the stochastic nature of flight operations, SkyPredict provides a statistically robust and operationally useful tool that pushes the boundaries of time-of-arrival estimation.
