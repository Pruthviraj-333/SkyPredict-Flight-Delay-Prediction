# SkyPredict — Flight Delay Prediction System

A full-stack web application that predicts flight delays using machine learning, trained on 600K+ real U.S. domestic flights from the Bureau of Transportation Statistics.

## Project Structure

```
forecasting-flight-delay/
│
├── backend/                    # FastAPI REST API
│   ├── main.py                 # Server entry point (11 endpoints)
│   ├── model_service.py        # ML model wrapper service
│   ├── requirements.txt        # Python dependencies
│   └── __init__.py
│
├── frontend/                   # Next.js web application
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx      # Root layout + metadata
│   │   │   ├── globals.css     # Design system
│   │   │   ├── page.tsx        # Passenger prediction page
│   │   │   └── staff/
│   │   │       └── page.tsx    # Staff analytics dashboard
│   │   ├── components/
│   │   │   └── Navbar.tsx      # Navigation bar
│   │   └── lib/
│   │       └── api.ts          # Type-safe API client
│   └── package.json
│
├── ml/                         # Machine learning pipeline
│   └── scripts/
│       ├── download_data.py    # BTS data downloader
│       ├── preprocess.py       # Feature engineering
│       ├── train_fallback_model.py  # XGBoost model training
│       ├── compare_models.py   # 10-algorithm comparison
│       ├── predict.py          # CLI prediction tool
│       ├── test_november.py    # Out-of-sample test
│       └── test_app.py         # End-to-end app tests
│
├── data/
│   ├── raw/                    # Raw BTS CSV files
│   └── processed/              # Cleaned feature datasets
│
├── models/                     # Trained model artifacts
│   ├── fallback_model.pkl      # XGBoost model
│   ├── encoders.pkl            # Label encoders
│   ├── aggregate_stats.pkl     # Historical delay rates
│   └── model_config.pkl        # Feature configuration
│
├── results/                    # Evaluation results
│   ├── model_comparison.csv    # 10-algorithm comparison
│   └── november_2025_test_results.csv
│
├── docs/                       # Documentation
│   └── Seminar_Documentation_Flight_Delay_Prediction.md
│
├── venv/                       # Python virtual environment
├── requirements.txt            # Python dependencies
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

## ML Pipeline

Run these scripts in order from the `ml/scripts/` directory to rebuild the model:

```bash
python ml/scripts/download_data.py          # Download BTS flight data
python ml/scripts/preprocess.py             # Feature engineering
python ml/scripts/train_fallback_model.py   # Train XGBoost model
python ml/scripts/compare_models.py         # Compare 10 algorithms
python ml/scripts/test_november.py          # Out-of-sample test
```

## Model Performance

| Metric | October Test | November (Unseen) |
|---|---|---|
| Accuracy | 72.5% | 65.0% |
| ROC-AUC | 0.772 | 0.616 |
| Delay Recall | 66.2% | 43.9% |

XGBoost was selected as the best algorithm after comparing 10 ML models.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Frontend | Next.js 16, TypeScript |
| Styling | Tailwind CSS 4 |
| Charts | Recharts |
| ML Model | XGBoost |
| Data Source | BTS TranStats (U.S. Government) |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/airlines` | List airlines |
| GET | `/api/airports` | List airports |
| POST | `/api/predict` | Predict single flight |
| POST | `/api/batch-predict` | Batch predictions |
| GET | `/api/stats` | Aggregate statistics |
| GET | `/api/analytics/trends` | Delay by day of week |
| GET | `/api/analytics/routes` | Top delayed routes |
| GET | `/api/analytics/carriers` | Delay by carrier |
| GET | `/api/analytics/hours` | Delay by hour |
| GET | `/api/analytics/heatmap` | Hour × Day heatmap |
