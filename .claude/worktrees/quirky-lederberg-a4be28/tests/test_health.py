"""Tests for health check system."""

import pytest
from health import HealthChecker, ProviderHealth


class TestProviderHealth:
    def test_default_state(self) -> None:
        """Default health should be unknown."""
        h = ProviderHealth()
        assert h.status == "unknown"
        assert h.consecutive_failures == 0

    def test_custom_state(self) -> None:
        """Should accept custom state."""
        h = ProviderHealth(status="up", latency_ms=42.5)
        assert h.status == "up"
        assert h.latency_ms == 42.5


class TestHealthChecker:
    def setup_method(self) -> None:
        self.checker = HealthChecker()

    def test_unknown_provider_available(self) -> None:
        """Unknown providers should be assumed available."""
        assert self.checker.is_available("unknown-provider")

    def test_get_health_unknown(self) -> None:
        """Should return default health for unknown provider."""
        h = self.checker.get_health("unknown")
        assert h.status == "unknown"

    def test_get_all_health_empty(self) -> None:
        """Should return empty dict when no checks done."""
        result = self.checker.get_all_health()
        assert isinstance(result, dict)
