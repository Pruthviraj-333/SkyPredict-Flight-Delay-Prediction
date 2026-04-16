"""
SkyPredict Backend — FastAPI application for flight delay prediction.

Endpoints:
  GET  /api/health              → Service health check
  GET  /api/airlines            → List of known airlines
  GET  /api/airports            → List of known airports
  POST /api/predict             → Single flight prediction
  POST /api/batch-predict       → Batch flight predictions
  GET  /api/flight-status/{id}  → Live flight tracking
  GET  /api/stats               → Aggregate delay statistics
  GET  /api/analytics/trends    → Delay by day of week
  GET  /api/analytics/routes    → Top delayed routes
  GET  /api/analytics/heatmap   → Hour × Day heatmap
  GET  /api/analytics/carriers  → Delay by carrier
  GET  /api/analytics/hours     → Delay by hour of day
  POST /api/auth/session        → Create/update user session
  GET  /api/auth/me             → Current user profile
"""

import os
import sys
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Add parent directory to path so we can find the models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Load .env
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from backend.model_service import ModelService
from backend.flight_tracker import FlightTracker
from backend.auth import init_firebase, get_current_user, is_firebase_available
from backend.database import init_db, upsert_user, get_user

# Initialize app
app = FastAPI(
    title="SkyPredict API",
    description="Flight Delay Prediction using Machine Learning",
    version="1.0.0",
)

# CORS — allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model service at startup
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
model_service: Optional[ModelService] = None
flight_tracker: Optional[FlightTracker] = None


@app.on_event("startup")
async def startup():
    global model_service, flight_tracker
    print("[INFO] Loading ML model and data...")
    model_service = ModelService(MODELS_DIR, DATA_DIR)
    flight_tracker = FlightTracker()
    print(f"[INFO] Model loaded. {len(model_service.get_airlines())} airlines, {len(model_service.get_airports())} airports.")
    print(f"[INFO] Flight tracking: {'enabled' if flight_tracker.is_available() else 'disabled (no API key)'}")

    # Initialize authentication
    init_firebase()
    init_db()
    print(f"[INFO] Authentication: {'enabled' if is_firebase_available() else 'disabled (no service account)'}")


# ─── Request/Response Models ────────────────────────────────────

class PredictRequest(BaseModel):
    carrier: str = Field(..., description="Airline code (e.g., AA, DL, UA)", min_length=2, max_length=3)
    origin: str = Field(..., description="Origin airport code (e.g., JFK)", min_length=3, max_length=4)
    dest: str = Field(..., description="Destination airport code (e.g., LAX)", min_length=3, max_length=4)
    date: str = Field(..., description="Flight date (YYYY-MM-DD)")
    dep_time: str = Field(..., description="Scheduled departure time (HHMM, e.g., 0800)")

class BatchPredictRequest(BaseModel):
    flights: List[PredictRequest]


# ─── Auth Endpoints ──────────────────────────────────────────────

@app.post("/api/auth/session")
async def create_session(user: Dict[str, Any] = Depends(get_current_user)):
    """Register or update user in DB after Firebase login."""
    db_user = upsert_user(
        uid=user["uid"],
        email=user.get("email"),
        phone=user.get("phone"),
        display_name=user.get("name"),
        photo_url=user.get("picture"),
        provider=user.get("provider", "unknown"),
    )
    return {"user": db_user}


@app.get("/api/auth/me")
async def get_me(user: Dict[str, Any] = Depends(get_current_user)):
    """Get current authenticated user profile."""
    db_user = get_user(user["uid"])
    if db_user:
        return {"user": db_user}
    return {"user": user}


# ─── Endpoints ───────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_service is not None,
        "model_type": "XGBoost Dual-Model (Classifier + Regressor)",
        "regressor_loaded": model_service.fallback_reg_model is not None if model_service else False,
        "primary_regressor_loaded": model_service.primary_reg_model is not None if model_service else False,
        "flight_tracking": flight_tracker.is_available() if flight_tracker else False,
        "auth_enabled": is_firebase_available(),
    }


@app.get("/api/flight-status/{flight_iata}")
async def get_flight_status(flight_iata: str):
    """Get live flight status from AviationStack."""
    if not flight_tracker or not flight_tracker.is_available():
        raise HTTPException(status_code=503, detail="Flight tracking not available (no API key)")
    result = flight_tracker.get_flight_status(flight_iata)
    if result is None:
        raise HTTPException(status_code=404, detail="Flight not found")
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"flight_status": result}


@app.get("/api/flight-lookup/{flight_iata}")
async def flight_lookup(flight_iata: str):
    """Look up a flight by number and return form-ready data for prediction."""
    if not flight_tracker or not flight_tracker.is_available():
        raise HTTPException(status_code=503, detail="Flight tracking not available (no API key)")

    result = flight_tracker.get_flight_status(flight_iata)
    if result is None:
        raise HTTPException(status_code=404, detail="Flight not found")
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Extract data for the prediction form
    dep_iata = result.get("departure", {}).get("iata", "")
    arr_iata = result.get("arrival", {}).get("iata", "")
    airline_iata = result.get("airline_iata", "")
    airline_name = result.get("airline_name", "")
    scheduled = result.get("departure", {}).get("scheduled")

    # Parse scheduled datetime → date + dep_time
    flight_date = ""
    dep_time = ""
    if scheduled:
        try:
            from datetime import datetime as dt
            parsed = dt.fromisoformat(scheduled.replace("Z", "+00:00"))
            flight_date = parsed.strftime("%Y-%m-%d")
            dep_time = parsed.strftime("%H%M")
        except Exception:
            pass

    # Try to map airline IATA to our known carrier codes
    carrier = airline_iata.upper() if airline_iata else ""

    return {
        "lookup": {
            "carrier": carrier,
            "airline_name": airline_name,
            "origin": dep_iata.upper() if dep_iata else "",
            "dest": arr_iata.upper() if arr_iata else "",
            "date": flight_date,
            "dep_time": dep_time,
            "dep_airport_name": result.get("departure", {}).get("airport", ""),
            "arr_airport_name": result.get("arrival", {}).get("airport", ""),
            "flight_iata": result.get("flight_iata", flight_iata),
            "status": result.get("status", ""),
        }
    }


@app.get("/api/airlines")
async def get_airlines():
    names = model_service.get_airline_names()
    return {
        "airlines": [
            {"code": code, "name": name}
            for code, name in sorted(names.items(), key=lambda x: x[1])
        ]
    }


@app.get("/api/airports")
async def get_airports():
    airports = model_service.get_airports()
    names = model_service.get_airport_names()
    return {
        "airports": [
            {"code": code, "name": names.get(code, code)}
            for code in airports
        ]
    }


@app.post("/api/predict")
async def predict(req: PredictRequest):
    try:
        result = model_service.predict(
            carrier=req.carrier,
            origin=req.origin,
            dest=req.dest,
            date_str=req.date,
            dep_time=req.dep_time,
        )
        return {"prediction": result.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/batch-predict")
async def batch_predict(req: BatchPredictRequest):
    try:
        flights = [f.dict() for f in req.flights]
        results = model_service.batch_predict(flights)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/api/stats")
async def get_stats():
    return {"stats": model_service.get_stats()}


@app.get("/api/analytics/trends")
async def get_trends():
    return {"trends": model_service.get_analytics_trends()}


@app.get("/api/analytics/routes")
async def get_routes(top_n: int = 10):
    return {"routes": model_service.get_analytics_routes(top_n=top_n)}


@app.get("/api/analytics/heatmap")
async def get_heatmap():
    return {"heatmap": model_service.get_analytics_heatmap()}


@app.get("/api/analytics/carriers")
async def get_carriers():
    return {"carriers": model_service.get_analytics_carriers()}


@app.get("/api/analytics/hours")
async def get_hours():
    return {"hours": model_service.get_analytics_hours()}


@app.get("/api/analytics/airport-map")
async def get_airport_map():
    return model_service.get_airport_map_data()
