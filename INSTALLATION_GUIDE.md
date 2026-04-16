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

> [!NOTE]  
> If you encounter peer dependency conflicts (due to React 19), use `npm install --legacy-peer-deps` instead.

---

## ⚙️ Environment Configuration

### Backend `.env`
Create a `.env` file in the root directory:
```env
AVIATIONSTACK_API_KEY=your_api_key_here
FIREBASE_SERVICE_ACCOUNT_PATH=backend/firebase-service-account.json
```

### Firebase Authentication Setup
1. Go to [Firebase Console](https://console.firebase.google.com) and create a project
2. Enable **Google** sign-in provider under Authentication → Sign-in method
3. Download the **Service Account JSON** from Project Settings → Service Accounts → Generate New Private Key
4. Save it as `backend/firebase-service-account.json`
5. Go to Project Settings → General → Your apps → Add web app, then copy the config
6. Create `frontend/.env.local` with your Firebase config:
```env
NEXT_PUBLIC_FIREBASE_API_KEY=your-api-key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=000000000000
NEXT_PUBLIC_FIREBASE_APP_ID=1:000000000000:web:abcdef1234567890
```

> [!IMPORTANT]
> Both `backend/firebase-service-account.json` and `frontend/.env.local` contain secrets and are excluded from git. Each teammate must create their own from the Firebase Console.

---

## 🔄 Keeping Your Local Code Up-to-Date

If your teammates have already set up the project and want to pull the latest changes (including model updates and new features), they should run:

```bash
# 1. Pull the latest code from GitHub
git pull origin main

# 2. Update Python dependencies (if requirements.txt changed)
# Ensure your venv is activated first
pip install -r requirements.txt

# 3. Update Frontend dependencies
cd frontend
npm install --legacy-peer-deps
cd ..
```

---

## 🧠 ML Pipeline & Model Training (v3)

The system now uses an **Advanced Dual-Model Logic Gate (v3)**. It first classifies the risk of delay and then, if a delay is predicted, runs a high-precision **XGBoost Regressor** to estimate the exact number of minutes. Both models support a **Primary Weather-Enhanced** path and an automatic **High-Accuracy Fallback** path.

### Option A: Replicate Model Training
To train the latest **v2 models** (optimized with Optuna and enhanced feature engineering), run these scripts:

#### 1. Data Preparation
```bash
python ml/scripts/download_data.py          # Download raw BTS data
python ml/scripts/preprocess.py             # Basic cleaning
python ml/scripts/airport_coords.py        # Generate airport coordinates
```

#### 2. Train Optimized Models
```bash
# -- Classification Models --
# Train the 45-feature Fallback Classifier (v2)
python ml/scripts/train_fallback_1month_v2.py

# Train the Weather-Enhanced Primary Classifier (v2)
python ml/scripts/train_primary_1month_v2.py

# -- Regression Models --
# Train the 45-feature Fallback Regressor (v3)
python ml/scripts/train_fallback_regression.py

# Train the Weather-Enhanced Primary Regressor (v3)
python ml/scripts/train_primary_regression.py
```

---

### Option B: Use Pre-trained Models (Faster)
If you don't want to train the models, request these files from the project owner and place them in the specified directories:

**1. Essential Model Artifacts (Place in `models/`)**

| Logic Tier | Classifier (is_delayed) | Regressor (delay_minutes) |
|---|---|---|
| **Primary (Weather)** | `primary_model.pkl` | `primary_reg_model.pkl` |
| **Fallback (Base)** | `fallback_model.pkl` | `fallback_reg_model.pkl` |

**Support Files (Required):**
- `encoders.pkl`
- `aggregate_stats.pkl`
- `model_config.pkl`
- `primary_model_config.pkl`
- `fallback_reg_config.pkl`
- `primary_reg_config.pkl`

**2. Essential Data Artifacts (Place in `data/`)**
- `airport_coordinates.csv`

> [!TIP]
> The v2 models achieve significantly higher accuracy (90%+) by using advanced features like cyclical time encodings, holiday proxies, and airport congestion metrics.

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
| **Landing Page** | [http://localhost:3000](http://localhost:3000) |
| **Predict Delays** | [http://localhost:3000/predict](http://localhost:3000/predict) |
| **Analytics Dashboard** | [http://localhost:3000/dashboard](http://localhost:3000/dashboard) |

The **Analytics Dashboard** includes an interactive **U.S. Airport Delay Map** with color-coded markers for 338 airports, route arcs for the top 10 most delayed routes, and hover tooltips — all powered by `react-simple-maps`.

| **Sign In** | [http://localhost:3000/login](http://localhost:3000/login) |
| **API Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) |

---

## 🛠️ Project Structure Overview

- `backend/`: FastAPI REST API + Firebase auth endpoints.
- `frontend/`: Next.js frontend with TypeScript, Vanilla CSS, Firebase Auth, and `react-simple-maps` for map visualizations.
- `ml/`: Machine learning pipeline scripts.
- `models/`: Pre-trained model artifacts (`.pkl`).
- `data/`: Storage for raw and processed datasets.

---

## 🤝 Need Help?

Contact the repository owner or open an issue on GitHub if you encounter any setup problems!
