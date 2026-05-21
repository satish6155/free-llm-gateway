"""Persistent request logging to SQLite.

Stores every routed request for analytics and debugging.
Complements the in-memory log in router.py (which keeps last 100)
with full history that survives restarts.

Thread-safe — uses threading.Lock matching the pattern in tracking.py.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DB_FILE = DATA_DIR / "requests.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_model TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    latency_ms REAL NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    attempt INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_provider ON requests(provider);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_success ON requests(success);
"""

_insert_sql = """
INSERT INTO requests (timestamp, model, provider, provider_model, success,
                      error, latency_ms, prompt_tokens, completion_tokens,
                      total_tokens, attempt)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class RequestDB:
    """Thread-safe SQLite request logger."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else DB_FILE
        self._lock = Lock()
        self._conn: sqlite3.Connection | None = None

    def init(self) -> None:
        """Create tables and indexes. Called once at startup."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()
        logger.info("Request DB initialized at %s", self._db_path)

    def close(self) -> None:
        """Close the DB connection. Called at shutdown."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def log_request(
        self,
        model: str,
        provider: str,
        provider_model: str,
        success: bool,
        error: str | None = None,
        latency_ms: float = 0.0,
        tokens: dict[str, int] | None = None,
        attempt: int = 1,
    ) -> None:
        """Insert a request record."""
        if not self._conn:
            return
        tok = tokens or {}
        with self._lock:
            try:
                self._conn.execute(
                    _insert_sql,
                    (
                        time.time(),
                        model,
                        provider,
                        provider_model,
                        1 if success else 0,
                        error,
                        latency_ms,
                        tok.get("prompt_tokens", 0),
                        tok.get("completion_tokens", 0),
                        tok.get("total_tokens", 0),
                        attempt,
                    ),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                logger.warning("Failed to log request to SQLite: %s", e)

    # ── Query helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _range_to_since(range_str: str) -> float:
        """Convert a range string ('24h', '7d', '30d') to a Unix timestamp."""
        now = time.time()
        mapping = {"24h": 86400, "7d": 86400 * 7, "30d": 86400 * 30}
        return now - mapping.get(range_str, 86400 * 7)

    def get_summary(self, range_str: str = "7d") -> dict[str, Any]:
        """Summary stats: total requests, success rate, tokens, avg latency."""
        if not self._conn:
            return {}
        since = self._range_to_since(range_str)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                    SUM(prompt_tokens) as total_input_tokens,
                    SUM(completion_tokens) as total_output_tokens,
                    AVG(latency_ms) as avg_latency_ms
                FROM requests WHERE timestamp >= ?
                """,
                (since,),
            ).fetchone()

        total = row[0] or 0
        success_count = row[1] or 0
        success_rate = round((success_count / total) * 100, 1) if total > 0 else 0.0
        total_input = row[2] or 0
        total_output = row[3] or 0

        # Estimated savings: GPT-4o pricing ($3/M input, $15/M output)
        input_cost = (total_input / 1_000_000) * 3
        output_cost = (total_output / 1_000_000) * 15

        return {
            "total_requests": total,
            "success_rate": success_rate,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "avg_latency_ms": round(row[4] or 0, 1),
            "estimated_cost_savings": round(input_cost + output_cost, 2),
        }

    def get_by_model(self, range_str: str = "7d") -> list[dict[str, Any]]:
        """Stats grouped by (provider, provider_model)."""
        if not self._conn:
            return []
        since = self._range_to_since(range_str)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    provider, provider_model, model,
                    COUNT(*) as requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(prompt_tokens) as total_input_tokens,
                    SUM(completion_tokens) as total_output_tokens
                FROM requests WHERE timestamp >= ?
                GROUP BY provider, provider_model
                ORDER BY requests DESC
                """,
                (since,),
            ).fetchall()

        return [
            {
                "provider": r[0],
                "provider_model": r[1],
                "model": r[2],
                "requests": r[3],
                "success_rate": round(r[4], 1),
                "avg_latency_ms": round(r[5] or 0, 1),
                "total_input_tokens": r[6] or 0,
                "total_output_tokens": r[7] or 0,
            }
            for r in rows
        ]

    def get_by_provider(self, range_str: str = "7d") -> list[dict[str, Any]]:
        """Stats grouped by provider."""
        if not self._conn:
            return []
        since = self._range_to_since(range_str)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    provider,
                    COUNT(*) as requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(prompt_tokens) as total_input_tokens,
                    SUM(completion_tokens) as total_output_tokens
                FROM requests WHERE timestamp >= ?
                GROUP BY provider
                ORDER BY requests DESC
                """,
                (since,),
            ).fetchall()

        return [
            {
                "provider": r[0],
                "requests": r[1],
                "success_rate": round(r[2], 1),
                "avg_latency_ms": round(r[3] or 0, 1),
                "total_input_tokens": r[4] or 0,
                "total_output_tokens": r[5] or 0,
            }
            for r in rows
        ]

    def get_timeline(
        self, range_str: str = "7d", interval: str = "day",
    ) -> list[dict[str, Any]]:
        """Time-bucketed request counts."""
        if not self._conn:
            return []
        since = self._range_to_since(range_str)
        fmt = "%Y-%m-%dT%H:00:00" if interval == "hour" else "%Y-%m-%d"
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT
                    strftime('{fmt}', timestamp, 'unixepoch') as ts,
                    COUNT(*) as requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failure_count
                FROM requests WHERE timestamp >= ?
                GROUP BY ts
                ORDER BY ts ASC
                """,
                (since,),
            ).fetchall()

        return [
            {
                "timestamp": r[0],
                "requests": r[1],
                "success_count": r[2],
                "failure_count": r[3],
            }
            for r in rows
        ]

    def get_errors(self, range_str: str = "7d") -> dict[str, Any]:
        """Error distribution + recent errors."""
        if not self._conn:
            return {"by_category": [], "by_provider": [], "recent": []}
        since = self._range_to_since(range_str)
        with self._lock:
            # Errors by category
            by_category = self._conn.execute(
                """
                SELECT
                    CASE
                        WHEN error LIKE '%429%' OR error LIKE '%rate limit%' OR error LIKE '%rate%'
                            THEN 'Rate Limited (429)'
                        WHEN error LIKE '%timeout%' OR error LIKE '%timed out%'
                            THEN 'Timeout'
                        WHEN error LIKE '%401%' OR error LIKE '%unauthorized%'
                            THEN 'Auth Error (401)'
                        WHEN error LIKE '%403%' OR error LIKE '%forbidden%'
                            THEN 'Forbidden (403)'
                        WHEN error LIKE '%500%' OR error LIKE '%internal%'
                            THEN 'Server Error (500)'
                        WHEN error LIKE '%502%' OR error LIKE '%503%' OR error LIKE '%unavailable%'
                            THEN 'Server Error (502/503)'
                        ELSE 'Other'
                    END as category,
                    COUNT(*) as count
                FROM requests
                WHERE success = 0 AND timestamp >= ?
                GROUP BY category
                ORDER BY count DESC
                """,
                (since,),
            ).fetchall()

            # Errors by provider
            by_provider = self._conn.execute(
                """
                SELECT provider, COUNT(*) as count
                FROM requests
                WHERE success = 0 AND timestamp >= ?
                GROUP BY provider
                ORDER BY count DESC
                """,
                (since,),
            ).fetchall()

            # Recent errors
            recent = self._conn.execute(
                """
                SELECT id, timestamp, provider, provider_model, model, error, latency_ms
                FROM requests
                WHERE success = 0 AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 50
                """,
                (since,),
            ).fetchall()

        return {
            "by_category": [{"category": r[0], "count": r[1]} for r in by_category],
            "by_provider": [{"provider": r[0], "count": r[1]} for r in by_provider],
            "recent": [
                {
                    "id": r[0],
                    "timestamp": r[1],
                    "provider": r[2],
                    "provider_model": r[3],
                    "model": r[4],
                    "error": r[5],
                    "latency_ms": round(r[6] or 0, 1),
                }
                for r in recent
            ],
        }

    def get_total_count(self) -> int:
        """Total rows in the requests table."""
        if not self._conn:
            return 0
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM requests").fetchone()
        return row[0] if row else 0

    def get_model_token_usage(self, days: int = 30) -> dict[str, dict[str, Any]]:
        """Get token usage grouped by model name for the last N days.

        Returns {model_name: {total_tokens, prompt_tokens, completion_tokens, requests}}.
        """
        if not self._conn:
            return {}
        since = time.time() - (days * 86400)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    model,
                    COUNT(*) as requests,
                    SUM(prompt_tokens) as total_input,
                    SUM(completion_tokens) as total_output,
                    SUM(total_tokens) as total_tokens
                FROM requests
                WHERE timestamp >= ? AND success = 1
                GROUP BY model
                ORDER BY total_tokens DESC
                """,
                (since,),
            ).fetchall()

        return {
            r[0]: {
                "requests": r[1],
                "prompt_tokens": r[2] or 0,
                "completion_tokens": r[3] or 0,
                "total_tokens": r[4] or 0,
            }
            for r in rows
        }


# Global singleton
request_db = RequestDB()
