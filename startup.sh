#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SkyPredict — Azure App Service Startup Script
# ─────────────────────────────────────────────────────────────
# Used when deploying WITHOUT Docker (Python runtime on App Service).
# Azure App Service exposes PORT via environment variable.
# ─────────────────────────────────────────────────────────────

# Use Azure's PORT env var, default to 8000 for local testing
PORT="${PORT:-8000}"

echo "[STARTUP] Starting SkyPredict Backend on port $PORT..."

# Start FastAPI with uvicorn
# --workers 1: Free tier has limited CPU, single worker is safer
# --timeout-keep-alive 120: Prevent premature connection drops
uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --timeout-keep-alive 120
