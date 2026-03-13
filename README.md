# SkyPredict — Flight Delay Prediction System

A full-stack web application that predicts flight delays using machine learning, trained on 600K+ real U.S. domestic flights from the Bureau of Transportation Statistics. Features a **dual-model architecture** with real-time weather integration and live flight tracking.

## Features

- 🔮 **Delay Predictions** — ML-powered delay probability with risk levels
- 🌦️ **Weather-Enhanced** — Real-time weather data boosts accuracy via dual-model logic gate
- ✈️ **Live Flight Tracking** — Track any flight's status via AviationStack API
- 📊 **Staff Dashboard** — Analytics with charts, carrier rankings, and route insights

## Project Structure

```
forecasting-flight-delay/
│
├── backend/                        # FastAPI REST API
│   ├── main.py                     # Server entry point (12 endpoints)
│   ├── model_service.py            # Dual-model wrapper (primary + fallback)
│   ├── weather_service.py          # Open-Meteo weather forecast service
│   ├── flight_tracker.py           # AviationStack live flight tracking
│   ├── requirements.txt            # Python dependencies
│   └── __init__.py
│
├── frontend/                       # Next.js web application
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx            # Passenger prediction + flight tracking
│   │   │   ├── staff/page.tsx      # Staff analytics dashboard
│   │   │   ├── layout.tsx          # Root layout + metadata
│   │   │   └── globals.css         # Design system
│   │   ├── components/
│   │   │   └── Navbar.tsx          # Navigation bar
│   │   └── lib/
│   │       └── api.ts              # Type-safe API client
│   └── package.json
│
├── ml/                             # Machine learning pipeline
│   └── scripts/
│       ├── download_data.py        # BTS data downloader
│       ├── preprocess.py           # Feature engineering
│       ├── train_fallback_model.py # Fallback model training (19 features)
│       ├── compare_models.py       # 10-algorithm comparison (fallback)
│       ├── airport_coords.py       # Airport lat/lon lookup
│       ├── fetch_weather.py        # Open-Meteo historical weather fetch
│       ├── build_weather_dataset.py # Merge flights + weather + train primary
│       ├── compare_primary_models.py # 10-algorithm comparison (primary)
│       ├── predict.py              # CLI prediction tool
│       ├── test_november.py        # Out-of-sample test
│       └── test_logic_gate.py      # Dual-model logic gate test
│
├── data/
│   ├── raw/                        # Raw BTS CSV files
│   ├── processed/                  # Cleaned feature datasets
│   ├── weather/                    # Historical weather data
│   └── airport_coordinates.csv     # Airport lat/lon for weather
│
├── models/                         # Trained model artifacts
│   ├── primary_model.pkl           # XGBoost with weather (32 features)
│   ├── primary_model_config.pkl    # Primary model config
│   ├── fallback_model.pkl          # XGBoost without weather (19 features)
│   ├── encoders.pkl                # Label encoders
│   ├── aggregate_stats.pkl         # Historical delay rates
│   └── model_config.pkl            # Fallback model config
│
├── results/                        # Evaluation results
│   ├── primary_model_comparison.csv
│   ├── primary_vs_fallback.csv
│   ├── model_comparison.csv
│   └── november_2025_test_results.csv
│
├── docs/                           # Documentation
│   └── Seminar_Documentation_Flight_Delay_Prediction.md
│
├── .env                            # API keys (not committed)
├── requirements.txt                # Python dependencies
├── README.md
└── .gitignore
```

## Quick Start

### 0. Set Up Virtual Environment (one-time)
```bash
cd "d:\forecasting flight delay"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Start Backend
```bash
cd "d:\forecasting flight delay"
venv\Scripts\activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 2. Start Frontend
```bash
cd "d:\forecasting flight delay\frontend"
npm install
npm run dev
```

### 3. Open in Browser
| Page | URL |
|---|---|
| Passenger View | http://localhost:3000 |
| Staff Dashboard | http://localhost:3000/staff |
| API Docs | http://localhost:8000/docs |

## Dual-Model Architecture (v2)

The system utilizes a dual-model logic gate that selects the optimal model based on whether real-time weather data is available for the flight.

```
User Request → Fetch Weather (Open-Meteo) → Weather Available?
                                              ├─ YES → Primary Model v2 (~78.1% acc, 0.780 AUC)
                                              └─ NO  → Fallback Model v2 (~78.1% acc, 0.778 AUC)
```

| Model | Features | Accuracy | ROC-AUC | Delay Recall |
|---|---|---|---|---|
| **Primary v2** (weather) | 60+ (45+ base) | **78.1%** | 0.780 | 72.0% |
| **Fallback v2** (standard) | 45+ | **78.1%** | 0.778 | 69.1% |

### v2 Enhancements (March 2026)
- **45+ Features**: Added cyclical time encodings, US holidays, airport congestion proxies, and multi-level interaction delay rates.
- **Optuna Optimization**: Hyperparameters tuned via 75 TPE trials for maximum ROC-AUC.
- **Temporal Splitting**: Realistic training on first 25 days of the month, testing on the last 6 days.
- **Threshold Optimization**: Dynamically tuned classification thresholds for balanced Precision/Recall.

## ML Pipeline

### Model Training (v2)
```bash
# Recommended: Enhanced 1-month models with Optuna
python ml/scripts/train_fallback_1month_v2.py
python ml/scripts/train_primary_1month_v2.py
```

### Legacy Pipeline
```bash
# Fallback model (v1)
python ml/scripts/download_data.py
python ml/scripts/preprocess.py
python ml/scripts/train_fallback_model.py

# Primary model (v1)
python ml/scripts/airport_coords.py
python ml/scripts/fetch_weather.py
python ml/scripts/build_weather_dataset.py
```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Frontend | Next.js 16, TypeScript |
| Styling | Vanilla CSS (custom design system) |
| Charts | Recharts |
| ML Model | XGBoost (dual-model) |
| Weather | Open-Meteo API (free, no key) |
| Flight Tracking | AviationStack API |
| Data Source | BTS TranStats (U.S. Government) |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check + model status |
| GET | `/api/airlines` | List airlines |
| GET | `/api/airports` | List airports |
| POST | `/api/predict` | Predict flight delay (auto-selects model) |
| POST | `/api/batch-predict` | Batch predictions |
| GET | `/api/stats` | Aggregate statistics |
| GET | `/api/analytics/trends` | Delay by day of week |
| GET | `/api/analytics/routes` | Top delayed routes |
| GET | `/api/analytics/carriers` | Delay by carrier |
| GET | `/api/analytics/hours` | Delay by hour |
| GET | `/api/analytics/heatmap` | Hour × Day heatmap |
| GET | `/api/flight-status/{iata}` | Live flight tracking |

## License

MIT
