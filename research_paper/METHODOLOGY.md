# Methodology

## 1. Data Source
- **Primary Dataset**: United States Department of Transportation (DOT) Bureau of Transportation Statistics (BTS) "On-Time Performance" dataset.
- **Scale**: ~600,000 records (e.g., Oct 2025).
- **External Data**: NOAA/AviationWeather.gov integration for real-time METAR-based weather features at 346 US airports.

## 2. Feature Engineering (110 Total)
- **Temporal (15)**: Cyclical Day-of-Week/Hour-of-Day (Sine/Cosine), Month, Holiday flags (Near-Holiday window ±3 days).
- **Categorical (10)**: Carrier, Origin, Destination, Tail Number (Carrier-encoded).
- **Historical Aggregates (45)**: Mean delay rates by Hour, Day, Carrier, Origin, and Route (Interaction Terms).
- **Weather (25)**: Visibility, Temperature, Wind Speed, Precipitation intensity, and specific flags (Snow/Fog/Thunderstorm) at both Origin and Destination.
- **Complexity Proxies (15)**: Airport daily flight volume, individual plane (Tail Number) utilization rate.

## 3. Model Architecture
- **Algorithm**: XGBoost Regressor (e-Xtreme Gradient Boosting).
- **Loss Function**: `reg:squarederror`.
- **Optimization Metric**: Root Mean Squared Error (RMSE).
- **Implementation**: Scikit-Learn wrapper with Early Stopping to prevent overfitting.

## 4. Pipeline Logic
1. **Missing Data Imputation**: Median-based imputation for numerical weather gaps.
2. **Encoding**: Target-mean encoding for high-cardinality categorical variables (Carrier/Airport).
3. **Clipping**: The target variable `ARR_DELAY_MINUTES` is clipped at [0, 1440] to focus on positive delays and remove outliers.
