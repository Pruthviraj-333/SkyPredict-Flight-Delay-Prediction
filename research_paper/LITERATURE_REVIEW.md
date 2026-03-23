# Literature Review & Academic Context

SkyPredict sits within a mature field of aviation logistics research. Below are the key scientific orientations of the project.

## 1. Classical Baselines
Historically, flight delay prediction relied on **Multivariate Linear Regression** or **Logit Models**.
- **Ref**: Sternberg et al. (2017) noted that linear models often fail to capture the non-linear "ripple effects" of airport congestion.
- **SkyPredict Context**: We utilize XGBoost precisely because gradient-boosted trees can model these non-linear interactions without requiring explicit basis functions.

## 2. The Network Effect
Delays are rarely isolated events; they propagate through the "Tail Number chain."
- **Ref**: Rebollo & Laporte (2014) demonstrated that predicting delays at a network level requires a high-resolution feature set.
- **SkyPredict Context**: Our inclusion of `Carrier_Delay_Rate` and `Tail_Number` utilization proxies aligns with the "Network propagation" theory.

## 3. Weather as a Stochastic Driver
Meteorological impact is the most significant "external" cause of delays.
- **Ref**: Klein et al. (2010) identified weather as responsible for 70% of delays in the US National Airspace System.
- **SkyPredict Context**: Our Dual-Model approach (Primary with Weather vs Fallback) replicates the findings from many research papers that show localized weather (visibility/wind) is a top-5 predictor of delay magnitude.

## 4. Machine Learning vs Deep Learning
There is an ongoing debate about using RNNs/LSTMs vs. GBTs for this task.
- **Ref**: Recent studies often show that for **tabular** data (like BTS records), Gradient Boosting (XGBoost/LightGBM) outperforms Deep Learning in both accuracy and training efficiency.
- **SkyPredict Context**: Our selection of XGBoost + Bayesian HPO represents the current **Practical State of the Art** for tabular transportation data.

## 5. Potential Paper Citations
- Sternberg, A., et al. (2017). "Machine Learning for Airline Flight Delay Predictions."
- Rebollo, J. J., & Laporte, G. (2014). "A network-based model for predicting flight delays."
- Klein, A., et al. (2010). "Weather-related airport delay and capacity analysis."
