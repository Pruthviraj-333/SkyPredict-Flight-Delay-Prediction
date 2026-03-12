# SkyPredict — Project Installation Guide

This guide provides step-by-step instructions for teammates to set up and run the **SkyPredict — Flight Delay Prediction System** on their local machines.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- **Git**: For version control.
- **Python (3.9+)**: The core language for the backend and ML pipeline.
- **Node.js (18+) & npm**: For the frontend application.

---

## 🚀 Getting Started

### 1. Clone the Repository
Open your terminal and run:
```bash
git clone https://github.com/Pruthviraj-333/SkyPredict-Flight-Delay-Prediction.git
cd SkyPredict-Flight-Delay-Prediction
```

### 2. Backend Setup (FastAPI & ML)

Create a virtual environment and install the required Python packages:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]  
> There is also a `requirements.txt` inside the `backend/` folder if you need specific backend dependencies later.

### 3. Frontend Setup (Next.js)

Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
cd ..
```

---

## ⚙️ Environment Configuration

Create a `.env` file in the root directory to store your API keys:

```bash
touch .env
```

Add your keys to the `.env` file:
```env
# Example .env content
AVIATIONSTACK_API_KEY=your_api_key_here
```

---

## 🧠 ML Pipeline & Model Training

To get the application running, you have two options:

### Option A: Replicate Model Training
If you want to train the models from scratch using real data, run these scripts in order from the root directory:

#### Phase 1: Fallback Model (No Weather)
```bash
python ml/scripts/download_data.py          # Download raw BTS data
python ml/scripts/preprocess.py             # Feature engineering
python ml/scripts/train_fallback_model.py  # Train XGBoost fallback model
```

#### Phase 2: Primary Model (Weather-Enhanced)
```bash
python ml/scripts/airport_coords.py        # Generate airport coordinates
python ml/scripts/fetch_weather.py         # Fetch historical weather data
python ml/scripts/build_weather_dataset.py # Merge flights + weather and train
```

---

### Option B: Use Pre-trained Models (Faster)
If you don't want to train the models, request these files from the project owner and place them in the specified directories:

**1. Essential Model Artifacts (Place in `models/`)**
- `fallback_model.pkl`
- `primary_model.pkl`
- `encoders.pkl`
- `aggregate_stats.pkl`
- `model_config.pkl`
- `primary_model_config.pkl`

**2. Essential Data Artifacts (Place in `data/`)**
- `airport_coordinates.csv` (Required for weather-related predictions)

> [!TIP]
> Using pre-trained models allows you to skip several hours of data processing and training.

---

## 💻 Running the Application

To run the full stack, you need to open two terminal windows (both with the virtual environment activated).

### Window 1: Start the Backend
```bash
venv\Scripts\activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Window 2: Start the Frontend
```bash
cd frontend
npm run dev
```

### Accessing the App
| Component | URL |
|---|---|
| **Passenger Interface** | [http://localhost:3000](http://localhost:3000) |
| **Staff Dashboard** | [http://localhost:3000/staff](http://localhost:3000/staff) |
| **API Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) |

---

## 🛠️ Project Structure Overview

- `backend/`: FastAPI REST API endpoints.
- `frontend/`: Next.js frontend with TypeScript and Vanilla CSS.
- `ml/`: Machine learning pipeline scripts.
- `models/`: Pre-trained model artifacts (`.pkl`).
- `data/`: Storage for raw and processed datasets.

---

## 🤝 Need Help?

Contact the repository owner or open an issue on GitHub if you encounter any setup problems!
