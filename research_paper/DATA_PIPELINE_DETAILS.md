# Data Pipeline Details

The SkyPredict data pipeline is designed for high-throughput processing and feature consistency between training and inference environments.

## 1. Data Collection
- **Flight Data**: Sourced from the US Bureau of Transportation Statistics (BTS).
- **Scale**: The system handles individual months (~600k rows) or full-year datasets (~7M rows).
- **Format**: Large CSV files containing ~110 raw columns.

## 2. Preprocessing Logic
1.  **Imputation**: 
    - Flights with missing arrival/departure times are removed (cancelled flights handled separately).
    - Missing weather values (e.g., Wind Speed) are imputed with median-per-airport values.
2.  **Outlier Filtering**: Arrival delays are clipped to [0, 1440] minutes. Negative delays (early arrivals) are treated as 0-minute delays for the purpose of "Magnitude Analysis."
3.  **Normalization**: Minimal scaling is required as XGBoost is scale-invariant, but temporal features are transformed into cyclical sine/cosine pairs.

## 3. The 45-Feature Base Pipeline
Every flight is transformed into a vector containing:
- **Temporal context**: Day of week, hour, season, holiday proximity.
- **Congestion proxies**: Hourly traffic at origin, daily carrier volume.
- **Historical Bias**: The **Mean Delay Rate** ($DR$) for a specific carrier/route/hour over the training period. This is the most critical feature group.

## 4. Weather Merging (The "Primary" Path)
Real-time weather data is fetched from local METAR servers for 346 US airports.
- **Match Criteria**: Latitude/Longitude threshold (within 0.1 deg) and Time-window matching (closest record within 1 hour).
- **Feature Set**: Temperature, Visibility, Precipitation intensity, Wind Speed, and specific weather flags (e.g., TSRA - Thunderstorm with Rain).

## 5. Temporal Splitting Strategy
Crucial for research validity, the pipeline implements a **25/5 day split** for single-month studies:
- **Training**: Days 1–25.
- **Testing**: Days 26–31.
This preserves temporal causality and prevents the model from "cheating" by knowing monthly average trends before they occur.
