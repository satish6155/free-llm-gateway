"""Tests for per-key rate tracking (RPM/RPD/TPM/TPD)."""

import time
import pytest
from rate_tracker import PerKeyRateTracker, RateLimits, PROVIDER_FREE_LIMITS


class TestPerKeyRateTracker:
    def setup_method(self) -> None:
        self.tracker = PerKeyRateTracker()

    def test_record_request_no_limits(self) -> None:
        """Unlimited keys should never be rate limited."""
        limited, reason = self.tracker.is_limited("groq", "llama3", "test-key-1234")
        assert not limited
        assert reason == ""

    def test_record_and_check_rpm(self) -> None:
        """Should track RPM per key."""
        self.tracker.set_limits("groq", "llama3", "test-key-1234", rpm=3)
        assert not self.tracker.is_limited("groq", "llama3", "test-key-1234")[0]

        for _ in range(3):
            self.tracker.record_request("groq", "llama3", "test-key-1234")

        limited, reason = self.tracker.is_limited("groq", "llama3", "test-key-1234")
        assert limited
        assert "RPM" in reason

    def test_separate_keys_independent(self) -> None:
        """Different keys should have independent rate limits."""
        self.tracker.set_limits("groq", "llama3", "key-aaaa", rpm=2)
        self.tracker.set_limits("groq", "llama3", "key-bbbb", rpm=2)

        self.tracker.record_request("groq", "llama3", "key-aaaa")
        self.tracker.record_request("groq", "llama3", "key-aaaa")

        assert self.tracker.is_limited("groq", "llama3", "key-aaaa")[0]
        assert not self.tracker.is_limited("groq", "llama3", "key-bbbb")[0]

    def test_token_tracking_tpm(self) -> None:
        """Should track tokens per minute."""
        self.tracker.set_limits("groq", "llama3", "key-tok", tpm=100)

        self.tracker.record_request("groq", "llama3", "key-tok", tokens=60)
        assert not self.tracker.is_limited("groq", "llama3", "key-tok")[0]

        self.tracker.record_request("groq", "llama3", "key-tok", tokens=50)
        limited, reason = self.tracker.is_limited("groq", "llama3", "key-tok")
        assert limited
        assert "TPM" in reason

    def test_token_tracking_tpd(self) -> None:
        """Should track tokens per day."""
        self.tracker.set_limits("groq", "llama3", "key-tpd", tpd=200)

        self.tracker.record_request("groq", "llama3", "key-tpd", tokens=150)
        assert not self.tracker.is_limited("groq", "llama3", "key-tpd")[0]

        self.tracker.record_request("groq", "llama3", "key-tpd", tokens=60)
        limited, reason = self.tracker.is_limited("groq", "llama3", "key-tpd")
        assert limited
        assert "TPD" in reason

    def test_rpd_limit(self) -> None:
        """Should track requests per day."""
        self.tracker.set_limits("groq", "llama3", "key-rpd", rpd=2)

        self.tracker.record_request("groq", "llama3", "key-rpd")
        self.tracker.record_request("groq", "llama3", "key-rpd")

        limited, reason = self.tracker.is_limited("groq", "llama3", "key-rpd")
        assert limited
        assert "RPD" in reason

    def test_get_usage(self) -> None:
        """Should return detailed usage stats."""
        self.tracker.set_limits("groq", "llama3", "key-usage", rpm=30, tpm=5000)
        self.tracker.record_request("groq", "llama3", "key-usage", tokens=100)

        usage = self.tracker.get_usage("groq", "llama3", "key-usage")
        assert usage["rpm"]["used"] == 1
        assert usage["rpm"]["limit"] == 30
        assert usage["tpm"]["used"] == 100
        assert usage["tpm"]["limit"] == 5000

    def test_get_all_usage(self) -> None:
        """Should return stats for all tracked keys."""
        self.tracker.set_limits("groq", "llama3", "key-a", rpm=30)
        self.tracker.set_limits("cerebras", "llama3", "key-b", rpm=20)
        self.tracker.record_request("groq", "llama3", "key-a")
        self.tracker.record_request("cerebras", "llama3", "key-b")

        all_usage = self.tracker.get_all_usage()
        assert len(all_usage) >= 2

    def test_get_provider_usage(self) -> None:
        """Should filter usage by provider."""
        self.tracker.set_limits("groq", "llama3", "key-x", rpm=30)
        self.tracker.set_limits("cerebras", "llama3", "key-y", rpm=20)
        self.tracker.record_request("groq", "llama3", "key-x")

        groq_usage = self.tracker.get_provider_usage("groq")
        assert len(groq_usage) >= 1
        assert groq_usage[0]["provider"] == "groq"

    def test_cleanup_expired(self) -> None:
        """Should remove stale entries."""
        self.tracker.set_limits("groq", "llama3", "key-clean", rpm=30)
        self.tracker.record_request("groq", "llama3", "key-clean")
        # Manually age out the timestamps
        k = self.tracker._key("groq", "llama3", "key-clean")
        bucket = self.tracker._buckets[k]
        bucket.timestamps = [time.time() - 200000]

        removed = self.tracker.cleanup_expired()
        assert removed >= 1


class TestProviderFreeLimits:
    def test_groq_limits_defined(self) -> None:
        assert "groq" in PROVIDER_FREE_LIMITS
        groq = PROVIDER_FREE_LIMITS["groq"]
        assert groq["default"].rpm > 0

    def test_deepseek_limits_defined(self) -> None:
        assert "deepseek" in PROVIDER_FREE_LIMITS

    def test_together_limits_defined(self) -> None:
        assert "together" in PROVIDER_FREE_LIMITS
