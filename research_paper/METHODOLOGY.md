# Methodology

## 1. Data Source
- **Primary Dataset**: United States Department of Transportation (DOT) Bureau of Transportation Statistics (BTS) "On-Time Performance" dataset.
- **Scale**: ~600,000 records (e.g., Oct 2025).
- **External Data**: Open-Meteo Forecast API integration for real-time weather features at US airports.

## 2. Feature Engineering
The framework utilizes a Dual-Model strategy where the feature matrix strictly aligns to the deployed model signatures (`.pkl`):
- **Fallback Model**: 50 Base Features.
- **Primary Model**: 70 Total Features (50 Base + 20 Weather).

Feature breakdown:
- **Base Temporal & Cyclical (15)**: Cyclical Day-of-Week/Hour-of-Day (Sine/Cosine), Month, Time Blocks.
- **Flight Characteristics & Categorical (8)**: Distance, Elapsed Time, Target-Encoded Carrier/Origin/Dest.
- **Peak Travel & Holidays (9)**: Near-Holiday window, Weekend flags, Red-eye markers, Peak-hour indicators.
- **Congestion & Utilization (3)**: Airport daily flight volume, Tail Number operational proxy.
- **Historical Aggregates (15)**: Multi-level mean delay rates (L1 and L2 interactions) by Hour, Day, Carrier, Origin, and specific Routes.
- **Weather Metrics (20)**: Origin/Dest Temperature, Visibility, Wind Speed, Precipitation, Cloud Cover, WMO Weather Codes, and binary "Bad Weather" flags.

## 3. Model Architecture
- **Algorithm**: XGBoost Regressor (e-Xtreme Gradient Boosting).
- **Loss Function**: `reg:squarederror`.
- **Optimization Metric**: Root Mean Squared Error (RMSE).
- **Implementation**: Scikit-Learn wrapper with Early Stopping to prevent overfitting.

## 4. Pipeline Logic
1. **Missing Data Imputation**: Median-based imputation for numerical weather gaps.
2. **Encoding**: Target-mean encoding for high-cardinality categorical variables (Carrier/Airport).
3. **Clipping**: The target variable `ARR_DELAY_MINUTES` is clipped at [0, 1440] to focus on positive delays and remove outliers.
