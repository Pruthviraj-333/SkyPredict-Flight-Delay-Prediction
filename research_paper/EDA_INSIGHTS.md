# EDA Insights (Exploratory Data Analysis)

Based on the 12-month historical analysis of the US DOT dataset (~7.2M flights), we identified several key trends that justify our modeling choices.

## 1. The "Carrier Delay Signature"
Different airlines exhibit distinct delay patterns:
- **Legacy Carriers (UA, AA, DL)**: Higher average delay minutes but more predictable patterns (hub-and-spoke effects).
- **Low-Cost Carriers (B6, NK)**: More sensitive to airport congestion and "ripple effects."
- **Insight**: Including `Carrier_Delay_Rate` as a feature accounts for ~35% of the model's gain.

## 2. Temporal Peaks (The Diurnal Cycle)
Delay magnitudes are NOT uniform throughout the day:
- **06:00 – 10:00**: Lowest delay risk (fresh crews, clear airspaces).
- **16:00 – 20:00**: Peak delay risk. Delays accumulate throughout the day, peaking in the evening commute period.
- **Insight**: Sine/Cosine hour encodings are mandatory to capture these cyclic fluctuations.

## 3. Geography of Congestion
- **Top Delay Hubs**: JFK, ORD, SFO, ATL (high volume, weather-sensitive).
- **Insight**: We implemented `Origin_Volume` and `Dest_Volume` metrics to capture the proxy of airport throughput stress.

## 4. Seasonal Variance
- **July & December**: Peak months due to summer storms and holiday travel volumes.
- **November**: Significantly more stable but highly sensitive to localized "Thanksgiving" travel spikes.
- **Insight**: Models trained on individual months (Oct/Nov/Dec) produce 10–12% better results than a single "one-size-fits-all" annual model.

## 5. Magnitude Distribution (The "Long Tail")
- **0–15 min**: 72% of flights.
- **15–60 min**: 18% of flights.
- **60+ min**: 10% of flights.
- **Insight**: The "Long Tail" is why we prioritize **RMSE** (which penalizes large errors) over MAE.
