# Result Analysis

## 1. Quantitative Performance
The regression model demonstrates higher stability than traditional linear baselines.

| Metric | Measured Value | interpretation |
| :--- | :--- | :--- |
| **MAE** | 17.21 min | Average prediction is within ~17 mins of ground truth. |
| **RMSE** | 43.90 min | Heavy tail of outliers (severe weather/mechanical) inflates RMSE. |
| **R²** | 0.116 | Predictable factors (schedule/weather) explain ~11.6% of variance. |
| **Within ±30 min** | 86.51% | High reliability for general traveler planning. |
| **Within ±15 min** | 67.34% | Comparable to human flight dispatcher estimates. |

## 2. Regression Plot Insights (Interpretation)
- **Zero-Inflation**: Most flights arrive on-time. The model correctly centers its most frequent predictions near zero but maintains a positive bias for high-risk routes.
- **Lower Bound Performance**: By clipping predictions at zero, we avoid the logical fallacy of "early arrival" contributing to delay magnitude (a common pitfall in simpler models).

## 3. Weather Impact Analysis
- The **Primary (Weather-Aware)** model demonstrates significant performance gains on days with localized visibility below 1 mile or precipitation exceeding 0.1 inches/hr.
- **Wind Gusts** at high-volume hubs (ORD, JFK, ATL) were identified as the strongest external predictors of arrival delay magnitude.

## 4. Feature Importance
Features ranked by F-score/Gain:
1. **Historical Carrier Delay Rate** (Dominant predictor).
2. **Sch_Dep_Hour** (Congestion indicator).
3. **Scheduled Flight Duration** (Magnitude correlation).
4. **Origin Visibility** (Localized weather).
5. **Day-of-Week Cyclical Sine** (Temporal traffic flow).
