"""
database.py – SQL Database Layer
=================================
Uses SQLite for development. Swap the connection string for
PostgreSQL / MySQL in production:

    # PostgreSQL via SQLAlchemy:
    from sqlalchemy import create_engine
    engine = create_engine("postgresql+psycopg2://user:pass@host/dbname")

    # Or use environment variable:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///healthscan.db")
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict

log = logging.getLogger(__name__)
DB_PATH = Path("healthscan.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                scan_id         TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                body_part       TEXT,
                language        TEXT,
                top_label       TEXT,
                confidence      REAL,
                category        TEXT,
                encrypted_path  TEXT,
                encryption_key_id TEXT,
                timestamp       REAL,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    log.info(f"Database initialized at {DB_PATH.resolve()}")


def save_scan_record(record: Dict):
    """Persist a scan metadata record."""
    try:
        with _connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scans
                (scan_id, user_id, body_part, language, top_label,
                 confidence, category, encrypted_path, encryption_key_id, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                record["scan_id"],
                record.get("user_id", "anonymous"),
                record.get("body_part"),
                record.get("language"),
                record.get("top_label"),
                record.get("confidence"),
                record.get("category"),
                record.get("encrypted_path"),
                record.get("encryption_key_id"),
                record.get("timestamp"),
            ))
            conn.commit()
        log.info(f"Scan record saved: {record['scan_id']}")
    except Exception as e:
        log.error(f"Failed to save scan record: {e}")


def get_scan_record(scan_id: str) -> Optional[Dict]:
    """Retrieve a single scan by ID."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM scans WHERE scan_id = ?", (scan_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_scans(user_id: str = "anonymous", limit: int = 20) -> List[Dict]:
    """Get recent scans for a user."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM scans WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    return [dict(r) for r in rows]
