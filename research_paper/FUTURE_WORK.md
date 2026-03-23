# Future Work

## 1. Geographic Generalization
The current models are trained on US DOT data. Future research could investigate if these features generalize to European (Eurocontrol) or Asian (CAAC) airspace, where ATC protocols differ.

## 2. Advanced Neural Architectures
While XGBoost is the state-of-the-art for tabular data, **Graph Neural Networks (GNNs)** could be utilized to model the airport network as nodes and flight routes as edges, potentially capturing the "ripple-effect" of delay propagation more effectively than boosted trees.

## 3. Real-Time ATC Integration
Integrating live data from the **FAA System Wide Information Management (SWIM)** platform, including Ground Delay Programs (GDP) and Flow Evaluation Area (FEA) data, would likely significantly improve the $R^2$ score by incorporating real-time operational constraints.

## 4. Multi-Modal Analysis
Combining flight data with social media sentiment (e.g., "Airport Chaos" trends) or localized event calendars (concerts/festivals near hubs) could capture surge-driven delays that historical data alone misses.
