# ─────────────────────────────────────────────────────────────
# SkyPredict Backend — Docker Image for Azure App Service
# ─────────────────────────────────────────────────────────────
# Build:  docker build -t skypredict-backend .
# Run:    docker run -p 8000:8000 --env-file .env skypredict-backend
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Prevent Python buffering (so logs appear in real-time on Azure)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install system dependencies ──────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ──────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────
COPY backend/ ./backend/

# ── Copy ML models & data ────────────────────────────────────
# NOTE: models/*.pkl and data/ are gitignored.
# You must have these files locally when building the Docker image.
# See AZURE_DEPLOYMENT.md for details on handling model files.
COPY models/ ./models/
COPY data/airport_coordinates.csv ./data/airport_coordinates.csv
COPY data/processed/ ./data/processed/

# ── Copy environment config ──────────────────────────────────
# .env is gitignored — set env vars via Azure Portal instead.
# This COPY is optional for local Docker testing only.
# COPY .env .env

# ── Expose port ──────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# ── Start the server ─────────────────────────────────────────
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
