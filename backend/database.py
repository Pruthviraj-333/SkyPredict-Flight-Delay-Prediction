"""
SkyPredict — Lightweight SQLite user store for authentication.

Tables:
  users: uid (PK), email, phone, display_name, photo_url, provider, created_at, last_login
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "skypredict_users.db")


def _get_conn() -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create users table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            uid            TEXT PRIMARY KEY,
            email          TEXT,
            phone          TEXT,
            display_name   TEXT,
            photo_url      TEXT,
            provider       TEXT NOT NULL DEFAULT 'unknown',
            created_at     TEXT NOT NULL,
            last_login     TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"[INFO] User database initialized at {DB_PATH}")


def upsert_user(
    uid: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    display_name: Optional[str] = None,
    photo_url: Optional[str] = None,
    provider: str = "unknown",
) -> Dict[str, Any]:
    """Create or update a user record. Returns the user dict."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()

    existing = conn.execute("SELECT * FROM users WHERE uid = ?", (uid,)).fetchone()

    if existing:
        conn.execute("""
            UPDATE users
            SET email = COALESCE(?, email),
                phone = COALESCE(?, phone),
                display_name = COALESCE(?, display_name),
                photo_url = COALESCE(?, photo_url),
                provider = ?,
                last_login = ?
            WHERE uid = ?
        """, (email, phone, display_name, photo_url, provider, now, uid))
    else:
        conn.execute("""
            INSERT INTO users (uid, email, phone, display_name, photo_url, provider, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (uid, email, phone, display_name, photo_url, provider, now, now))

    conn.commit()
    user = conn.execute("SELECT * FROM users WHERE uid = ?", (uid,)).fetchone()
    conn.close()
    return dict(user)


def get_user(uid: str) -> Optional[Dict[str, Any]]:
    """Retrieve a user by UID. Returns None if not found."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE uid = ?", (uid,)).fetchone()
    conn.close()
    return dict(row) if row else None
