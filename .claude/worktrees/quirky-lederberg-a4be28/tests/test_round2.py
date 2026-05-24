"""Tests for Round 2 features: SQLite request log, safe_stream, fallback editing."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import AppConfig, ModelFallback, ModelConfig, ModelCapabilities


# ── 1. SQLite Request DB ──────────────────────────────────────────────────────


class TestRequestDB:
    """Verify RequestDB SQLite persistence and queries."""

    def setup_method(self) -> None:
        from request_db import RequestDB
        self.db_path = Path(__file__).parent / "test_requests.db"
        self.db = RequestDB(db_path=self.db_path)
        self.db.init()

    def teardown_method(self) -> None:
        self.db.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def test_init_creates_table(self) -> None:
        """init() should create the requests table."""
        assert self.db.get_total_count() == 0

    def test_log_request_inserts_row(self) -> None:
        """log_request should insert a row."""
        self.db.log_request(
            model="gpt-4", provider="groq", provider_model="llama3",
            success=True, latency_ms=150.0,
            tokens={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        assert self.db.get_total_count() == 1

    def test_log_multiple_requests(self) -> None:
        """Multiple requests should all be persisted."""
        for i in range(10):
            self.db.log_request(
                model=f"model-{i}", provider="groq", provider_model="llama3",
                success=i % 2 == 0, latency_ms=100.0 + i,
            )
        assert self.db.get_total_count() == 10

    def test_summary_returns_correct_stats(self) -> None:
        """get_summary should return aggregated stats."""
        self.db.log_request("gpt-4", "groq", "llama3", True, latency_ms=100,
                            tokens={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        self.db.log_request("gpt-4", "cerebras", "llama3", False, error="timeout",
                            latency_ms=200)
        summary = self.db.get_summary("7d")
        assert summary["total_requests"] == 2
        assert summary["success_rate"] == 50.0
        assert summary["total_input_tokens"] == 100

    def test_by_model_groups_correctly(self) -> None:
        """get_by_model should group by (provider, provider_model)."""
        self.db.log_request("gpt-4", "groq", "llama3", True, latency_ms=100)
        self.db.log_request("gpt-4", "groq", "llama3", True, latency_ms=200)
        self.db.log_request("gpt-4", "cerebras", "llama3", True, latency_ms=150)
        rows = self.db.get_by_model("7d")
        assert len(rows) == 2
        groq_row = next(r for r in rows if r["provider"] == "groq")
        assert groq_row["requests"] == 2

    def test_by_provider_groups_correctly(self) -> None:
        """get_by_provider should group by provider."""
        self.db.log_request("gpt-4", "groq", "llama3", True, latency_ms=100)
        self.db.log_request("gpt-4", "cerebras", "llama3", False, error="429")
        rows = self.db.get_by_provider("7d")
        assert len(rows) == 2

    def test_timeline_returns_buckets(self) -> None:
        """get_timeline should return time-bucketed counts."""
        self.db.log_request("gpt-4", "groq", "llama3", True, latency_ms=100)
        self.db.log_request("gpt-4", "groq", "llama3", False, error="timeout")
        timeline = self.db.get_timeline("7d", interval="day")
        assert len(timeline) >= 1
        bucket = timeline[0]
        assert bucket["requests"] == 2
        assert bucket["success_count"] == 1
        assert bucket["failure_count"] == 1

    def test_errors_categorizes_correctly(self) -> None:
        """get_errors should categorize errors."""
        self.db.log_request("gpt-4", "groq", "llama3", False, error="rate limit exceeded")
        self.db.log_request("gpt-4", "cerebras", "llama3", False, error="timeout")
        self.db.log_request("gpt-4", "groq", "llama3", False, error="401 unauthorized")
        errors = self.db.get_errors("7d")
        assert len(errors["by_category"]) >= 2
        assert len(errors["recent"]) == 3

    def test_range_filtering(self) -> None:
        """Range filter should exclude old records."""
        # Insert an old record by manipulating timestamp
        self.db.log_request("gpt-4", "groq", "llama3", True, latency_ms=100)
        # Update its timestamp to 8 days ago
        with self.db._lock:
            self.db._conn.execute(
                "UPDATE requests SET timestamp = ? WHERE provider = 'groq'",
                (time.time() - 86400 * 8,),
            )
            self.db._conn.commit()

        summary = self.db.get_summary("7d")
        assert summary["total_requests"] == 0

    def test_log_request_handles_missing_tokens(self) -> None:
        """log_request should handle None tokens gracefully."""
        self.db.log_request("gpt-4", "groq", "llama3", True, tokens=None)
        assert self.db.get_total_count() == 1


# ── 2. Safe Stream ────────────────────────────────────────────────────────────


class TestSafeStream:
    """Verify safe_stream wraps provider streams with error handling."""

    def test_normal_stream_passes_through(self) -> None:
        """Normal chunks should pass through unchanged."""
        from routes.chat import safe_stream

        async def good_stream():
            yield b"data: hello\n\n"
            yield b"data: world\n\n"

        async def _run():
            chunks = []
            async for chunk in safe_stream(good_stream(), "groq", "llama3"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == 2
        assert chunks[0] == b"data: hello\n\n"

    def test_error_stream_emits_error_frame(self) -> None:
        """Mid-stream error should yield error SSE frame + [DONE]."""
        from routes.chat import safe_stream

        async def bad_stream():
            yield b"data: hello\n\n"
            raise RuntimeError("Connection reset")

        async def _run():
            chunks = []
            async for chunk in safe_stream(bad_stream(), "groq", "llama3"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == 3
        # First chunk is the normal one
        assert chunks[0] == b"data: hello\n\n"
        # Second chunk is the error frame
        error_text = chunks[1].decode()
        assert "stream_error" in error_text
        assert "stream interrupted" in error_text
        # Third chunk is [DONE]
        assert b"[DONE]" in chunks[2]


# ── 3. Fallback Editing ──────────────────────────────────────────────────────


class TestFallbackEditing:
    """Verify ModelFallback.enabled field and fallback editing logic."""

    def test_model_fallback_has_enabled_default(self) -> None:
        """ModelFallback should default to enabled=True."""
        fb = ModelFallback(provider="groq", model="llama3")
        assert fb.enabled is True

    def test_model_fallback_can_be_disabled(self) -> None:
        """ModelFallback can be set to enabled=False."""
        fb = ModelFallback(provider="groq", model="llama3", enabled=False)
        assert fb.enabled is False

    def test_router_skips_disabled_fallbacks(self) -> None:
        """Router._select_provider should skip disabled fallbacks."""
        from rate_limiter import RateLimiter
        from router import Router

        cfg = MagicMock(spec=AppConfig)
        cfg.providers = {}
        cfg.models = {}
        rl = RateLimiter()
        r = Router(cfg, rl)

        # Create fallbacks with one disabled
        fallbacks = [
            ModelFallback(provider="groq", model="llama3", enabled=False),
            ModelFallback(provider="cerebras", model="llama3", enabled=True),
        ]
        # Mock _get_provider to return None (no keys configured)
        r._get_provider = MagicMock(return_value=None)

        candidates = r._select_provider("test-model", fallbacks)
        # Both should be filtered: first by enabled=False, second by no provider
        # The enabled check happens before _get_provider
        assert r._get_provider.call_count == 1  # Only called for the enabled one
        r._get_provider.assert_called_once_with("cerebras")


# ── 4. Analytics Endpoints Exist ─────────────────────────────────────────────


class TestAnalyticsEndpoints:
    """Verify analytics endpoints are registered."""

    def test_analytics_summary_route_exists(self) -> None:
        """GET /api/analytics/summary should be registered."""
        import main
        routes = [r.path for r in main.app.routes if hasattr(r, "path")]
        assert "/api/analytics/summary" in routes

    def test_analytics_by_model_route_exists(self) -> None:
        """GET /api/analytics/by-model should be registered."""
        import main
        routes = [r.path for r in main.app.routes if hasattr(r, "path")]
        assert "/api/analytics/by-model" in routes

    def test_analytics_by_provider_route_exists(self) -> None:
        """GET /api/analytics/by-provider should be registered."""
        import main
        routes = [r.path for r in main.app.routes if hasattr(r, "path")]
        assert "/api/analytics/by-provider" in routes

    def test_analytics_timeline_route_exists(self) -> None:
        """GET /api/analytics/timeline should be registered."""
        import main
        routes = [r.path for r in main.app.routes if hasattr(r, "path")]
        assert "/api/analytics/timeline" in routes

    def test_analytics_errors_route_exists(self) -> None:
        """GET /api/analytics/errors should be registered."""
        import main
        routes = [r.path for r in main.app.routes if hasattr(r, "path")]
        assert "/api/analytics/errors" in routes


# ── 5. Fallback Endpoints Exist ──────────────────────────────────────────────


class TestFallbackEndpoints:
    """Verify fallback editing endpoints are registered."""

    def test_get_fallbacks_route_exists(self) -> None:
        """GET /api/fallbacks should be registered."""
        import main
        routes = [r.path for r in main.app.routes if hasattr(r, "path")]
        assert "/api/fallbacks" in routes

    def test_put_fallbacks_route_exists(self) -> None:
        """PUT /api/fallbacks/{model} should be registered."""
        import main
        routes = [r.path for r in main.app.routes if hasattr(r, "path")]
        assert "/api/fallbacks/{model}" in routes

    def test_sort_fallbacks_route_exists(self) -> None:
        """POST /api/fallbacks/{model}/sort/{preset} should be registered."""
        import main
        routes = [r.path for r in main.app.routes if hasattr(r, "path")]
        assert "/api/fallbacks/{model}/sort/{preset}" in routes
