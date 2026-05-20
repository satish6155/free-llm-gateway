"""Per-key rate tracking: RPM, RPD, TPM, TPD per (provider, model, key).

Tracks request counts and token usage within rolling time windows.
Thread-safe with fine-grained locking for high concurrency.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class RateBucket:
    """Tracks requests and tokens within a rolling time window."""

    requests: int = 0
    tokens: int = 0
    timestamps: list[float] = field(default_factory=list)
    token_timestamps: list[tuple[float, int]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def record_request(self, tokens: int = 0, now: float = 0.0) -> None:
        if not now:
            now = time.time()
        with self._lock:
            self.timestamps.append(now)
            self.requests += 1
            if tokens > 0:
                self.token_timestamps.append((now, tokens))
                self.tokens += tokens

    def prune(self, window_seconds: float, now: float = 0.0) -> None:
        if not now:
            now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            self.timestamps = [t for t in self.timestamps if t > cutoff]
            self.token_timestamps = [(t, tok) for t, tok in self.token_timestamps if t > cutoff]
            self.requests = len(self.timestamps)
            self.tokens = sum(tok for _, tok in self.token_timestamps)

    def request_count(self, window_seconds: float, now: float = 0.0) -> int:
        if not now:
            now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            return sum(1 for t in self.timestamps if t > cutoff)

    def token_count(self, window_seconds: float, now: float = 0.0) -> int:
        if not now:
            now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            return sum(tok for t, tok in self.token_timestamps if t > cutoff)


@dataclass
class RateLimits:
    """Rate limit configuration for a (provider, model, key) triple."""

    rpm: int = 0   # requests per minute, 0 = unlimited
    rpd: int = 0   # requests per day, 0 = unlimited
    tpm: int = 0   # tokens per minute, 0 = unlimited
    tpd: int = 0   # tokens per day, 0 = unlimited


class PerKeyRateTracker:
    """Tracks RPM, RPD, TPM, TPD per (provider, model, key) combination.

    Provides fine-grained rate tracking so the router can skip
    keys that have exhausted their quota and select the next available one.
    """

    MINUTE = 60
    DAY = 86400

    def __init__(self) -> None:
        self._buckets: dict[str, RateBucket] = {}
        self._limits: dict[str, RateLimits] = {}
        self._lock = Lock()

    @staticmethod
    def _key(provider: str, model: str, api_key: str) -> str:
        # Use last 8 chars of key for tracking to avoid storing full key
        key_suffix = api_key[-8:] if len(api_key) > 8 else api_key
        return f"{provider}:{model}:{key_suffix}"

    def set_limits(
        self, provider: str, model: str, api_key: str,
        rpm: int = 0, rpd: int = 0, tpm: int = 0, tpd: int = 0,
    ) -> None:
        """Set rate limits for a specific (provider, model, key) triple."""
        k = self._key(provider, model, api_key)
        with self._lock:
            self._limits[k] = RateLimits(rpm=rpm, rpd=rpd, tpm=tpm, tpd=tpd)
            if k not in self._buckets:
                self._buckets[k] = RateBucket()

    def record_request(
        self, provider: str, model: str, api_key: str, tokens: int = 0,
    ) -> None:
        """Record a request with optional token count."""
        k = self._key(provider, model, api_key)
        with self._lock:
            if k not in self._buckets:
                self._buckets[k] = RateBucket()
        self._buckets[k].record_request(tokens=tokens)

    def is_limited(
        self, provider: str, model: str, api_key: str,
    ) -> tuple[bool, str]:
        """Check if this (provider, model, key) is rate limited.

        Returns (is_limited, reason). reason is empty if not limited.
        """
        k = self._key(provider, model, api_key)
        limits = self._limits.get(k)
        bucket = self._buckets.get(k)
        if not limits or not bucket:
            return False, ""

        now = time.time()
        bucket.prune(self.DAY, now)

        # Check RPM
        if limits.rpm > 0:
            rpm_used = bucket.request_count(self.MINUTE, now)
            if rpm_used >= limits.rpm:
                return True, f"RPM limit reached ({rpm_used}/{limits.rpm})"

        # Check RPD
        if limits.rpd > 0:
            rpd_used = bucket.request_count(self.DAY, now)
            if rpd_used >= limits.rpd:
                return True, f"RPD limit reached ({rpd_used}/{limits.rpd})"

        # Check TPM
        if limits.tpm > 0:
            tpm_used = bucket.token_count(self.MINUTE, now)
            if tpm_used >= limits.tpm:
                return True, f"TPM limit reached ({tpm_used}/{limits.tpm})"

        # Check TPD
        if limits.tpd > 0:
            tpd_used = bucket.token_count(self.DAY, now)
            if tpd_used >= limits.tpd:
                return True, f"TPD limit reached ({tpd_used}/{limits.tpd})"

        return False, ""

    def get_usage(
        self, provider: str, model: str, api_key: str,
    ) -> dict[str, Any]:
        """Get current usage stats for a (provider, model, key)."""
        k = self._key(provider, model, api_key)
        limits = self._limits.get(k, RateLimits())
        bucket = self._buckets.get(k, RateBucket())
        now = time.time()
        bucket.prune(self.DAY, now)

        return {
            "provider": provider,
            "model": model,
            "key_suffix": api_key[-8:] if len(api_key) > 8 else api_key,
            "rpm": {"used": bucket.request_count(self.MINUTE, now), "limit": limits.rpm},
            "rpd": {"used": bucket.request_count(self.DAY, now), "limit": limits.rpd},
            "tpm": {"used": bucket.token_count(self.MINUTE, now), "limit": limits.tpm},
            "tpd": {"used": bucket.token_count(self.DAY, now), "limit": limits.tpd},
        }

    def get_all_usage(self) -> list[dict[str, Any]]:
        """Get usage stats for all tracked (provider, model, key) combos."""
        results: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        with self._lock:
            for k in self._buckets:
                parts = k.split(":", 2)
                if len(parts) == 3:
                    provider, model, key_suffix = parts
                    # Reconstruct a dummy key for get_usage
                    full_key = key_suffix  # We only track suffix
                    if k not in seen_keys:
                        seen_keys.add(k)
                        limits = self._limits.get(k, RateLimits())
                        bucket = self._buckets[k]
                        now = time.time()
                        bucket.prune(self.DAY, now)
                        results.append({
                            "provider": provider,
                            "model": model,
                            "key_suffix": key_suffix,
                            "rpm": {"used": bucket.request_count(self.MINUTE, now), "limit": limits.rpm},
                            "rpd": {"used": bucket.request_count(self.DAY, now), "limit": limits.rpd},
                            "tpm": {"used": bucket.token_count(self.MINUTE, now), "limit": limits.tpm},
                            "tpd": {"used": bucket.token_count(self.DAY, now), "limit": limits.tpd},
                        })
        return results

    def get_provider_usage(self, provider: str) -> list[dict[str, Any]]:
        """Get usage stats for all keys of a specific provider."""
        results: list[dict[str, Any]] = []
        prefix = f"{provider}:"
        with self._lock:
            for k in self._buckets:
                if k.startswith(prefix):
                    parts = k.split(":", 2)
                    if len(parts) == 3:
                        _, model, key_suffix = parts
                        limits = self._limits.get(k, RateLimits())
                        bucket = self._buckets[k]
                        now = time.time()
                        bucket.prune(self.DAY, now)
                        results.append({
                            "provider": provider,
                            "model": model,
                            "key_suffix": key_suffix,
                            "rpm": {"used": bucket.request_count(self.MINUTE, now), "limit": limits.rpm},
                            "rpd": {"used": bucket.request_count(self.DAY, now), "limit": limits.rpd},
                            "tpm": {"used": bucket.token_count(self.MINUTE, now), "limit": limits.tpm},
                            "tpd": {"used": bucket.token_count(self.DAY, now), "limit": limits.tpd},
                        })
        return results

    def cleanup_expired(self, max_age: float = 86400 * 2) -> int:
        """Remove buckets with no activity in the last max_age seconds."""
        now = time.time()
        cutoff = now - max_age
        removed = 0
        with self._lock:
            expired = []
            for k, bucket in self._buckets.items():
                bucket.prune(self.DAY, now)
                if not bucket.timestamps:
                    expired.append(k)
            for k in expired:
                del self._buckets[k]
                self._limits.pop(k, None)
                removed += 1
        return removed


# Known free tier limits per provider
PROVIDER_FREE_LIMITS: dict[str, dict[str, RateLimits]] = {
    "groq": {
        "default": RateLimits(rpm=30, rpd=14400, tpm=6000, tpd=0),
        "llama-3.3-70b-versatile": RateLimits(rpm=30, rpd=14400, tpm=6000, tpd=0),
        "mixtral-8x7b-32768": RateLimits(rpm=30, rpd=14400, tpm=5000, tpd=0),
    },
    "cerebras": {
        "default": RateLimits(rpm=30, rpd=0, tpm=0, tpd=0),
    },
    "github": {
        "default": RateLimits(rpm=15, rpd=0, tpm=0, tpd=0),
    },
    "openrouter": {
        "default": RateLimits(rpm=20, rpd=0, tpm=0, tpd=0),
    },
    "google_gemini": {
        "default": RateLimits(rpm=15, rpd=1500, tpm=1000000, tpd=0),
        "gemini-2.0-flash-lite": RateLimits(rpm=30, rpd=1500, tpm=1000000, tpd=0),
    },
    "cohere": {
        "default": RateLimits(rpm=20, rpd=0, tpm=0, tpd=0),
    },
    "mistral": {
        "default": RateLimits(rpm=10, rpd=0, tpm=0, tpd=0),
    },
    "nvidia": {
        "default": RateLimits(rpm=10, rpd=0, tpm=0, tpd=0),
    },
    "deepseek": {
        "default": RateLimits(rpm=30, rpd=0, tpm=0, tpd=0),
    },
    "together": {
        "default": RateLimits(rpm=10, rpd=0, tpm=0, tpd=0),
    },
    "fireworks": {
        "default": RateLimits(rpm=10, rpd=0, tpm=0, tpd=0),
    },
    "sambanova": {
        "default": RateLimits(rpm=10, rpd=0, tpm=0, tpd=0),
    },
    "siliconflow": {
        "default": RateLimits(rpm=10, rpd=0, tpm=0, tpd=0),
    },
}

# Global singleton
per_key_rate_tracker = PerKeyRateTracker()
