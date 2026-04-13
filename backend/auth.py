"""
SkyPredict — Firebase Authentication middleware for FastAPI.

Verifies Firebase ID tokens from the frontend and extracts user info.
"""

import os
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

# ─── Firebase Admin Initialization ──────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_firebase_app: Optional[firebase_admin.App] = None
_firebase_available = False


def init_firebase():
    """Initialize Firebase Admin SDK. Call once at startup."""
    global _firebase_app, _firebase_available

    service_account_path = os.getenv(
        "FIREBASE_SERVICE_ACCOUNT_PATH",
        os.path.join(BASE_DIR, "backend", "firebase-service-account.json"),
    )

    if not os.path.exists(service_account_path):
        print(f"[WARN] Firebase service account not found at {service_account_path}")
        print("[WARN] Authentication will not work until you add the service account file.")
        _firebase_available = False
        return

    try:
        cred = credentials.Certificate(service_account_path)
        _firebase_app = firebase_admin.initialize_app(cred)
        _firebase_available = True
        print("[INFO] Firebase Admin SDK initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Firebase: {e}")
        _firebase_available = False


def is_firebase_available() -> bool:
    return _firebase_available


# ─── Token Verification ────────────────────────────────────────

security = HTTPBearer(auto_error=False)


def verify_firebase_token(token: str) -> Dict[str, Any]:
    """
    Verify a Firebase ID token and return the decoded claims.

    Returns dict with: uid, email, phone_number, name, picture, firebase.sign_in_provider, etc.
    Raises HTTPException 401 if invalid.
    """
    if not _firebase_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not configured. Add firebase-service-account.json.",
        )

    try:
        decoded = firebase_auth.verify_id_token(token, app=_firebase_app)
        return decoded
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired. Please sign in again.",
        )
    except firebase_auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
        )


# ─── FastAPI Dependencies ──────────────────────────────────────

async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """
    FastAPI dependency: extracts and verifies the Bearer token.

    Usage:
        @app.get("/protected")
        async def protected(user = Depends(get_current_user)):
            return {"uid": user["uid"]}
    """
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    decoded = verify_firebase_token(creds.credentials)

    # Normalize user info from Firebase claims
    provider_data = decoded.get("firebase", {})
    return {
        "uid": decoded["uid"],
        "email": decoded.get("email"),
        "phone": decoded.get("phone_number"),
        "name": decoded.get("name"),
        "picture": decoded.get("picture"),
        "provider": provider_data.get("sign_in_provider", "unknown"),
    }


async def get_optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency: returns user if token present, None otherwise.
    Use for endpoints that work for both authenticated and anonymous users.
    """
    if creds is None or not creds.credentials:
        return None

    try:
        decoded = verify_firebase_token(creds.credentials)
        provider_data = decoded.get("firebase", {})
        return {
            "uid": decoded["uid"],
            "email": decoded.get("email"),
            "phone": decoded.get("phone_number"),
            "name": decoded.get("name"),
            "picture": decoded.get("picture"),
            "provider": provider_data.get("sign_in_provider", "unknown"),
        }
    except HTTPException:
        return None
