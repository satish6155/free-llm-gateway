"""Model routing with provider fallback logic, retry with backoff, and round-robin."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from config import AppConfig, ModelFallback, local_llm_enabled, local_llm_model
from providers import ProviderConfig, ProviderError, send_to_provider
from rate_limiter import RateLimiter
from rate_tracker import per_key_rate_tracker
from request_db import request_db
from token_estimator import estimate_request_tokens

if TYPE_CHECKING:
    from health import HealthChecker

logger = logging.getLogger(__name__)

MAX_RETRIES = 10  # max fallback providers to try per request
RETRY_MAX_ATTEMPTS = int(os.environ.get("RETRY_MAX_ATTEMPTS", "2"))
RETRY_BACKOFF_BASE = float(os.environ.get("RETRY_BACKOFF_BASE", "1.0"))


# ── Dynamic penalty: models that return 429s sink in priority ────────────────
PENALTY_PER_429 = 3
MAX_PENALTY = 10
DECAY_INTERVAL_S = 120  # 2 minutes
DECAY_AMOUNT = 1


class PenaltyTracker:
    """Track 429 penalties per provider so frequently rate-limited ones sink.

    Models that return 429 get +PENALTY_PER_429 penalty (max MAX_PENALTY).
    Successful requests reduce penalty by 1. Penalties decay over time
    (every DECAY_INTERVAL_S seconds, reduce by DECAY_AMOUNT) so models
    recover after their rate limits reset.
    """

    def __init__(self) -> None:
        self._penalties: dict[str, dict[str, int | float]] = {}

    def record_hit(self, provider: str, model: str) -> None:
        """Record a 429 rate limit hit — increase penalty."""
        key = f"{provider}:{model}"
        entry = self._penalties.get(key)
        now = time.time()
        if entry:
            entry["count"] = entry.get("count", 0) + 1
            entry["last_hit"] = now
            entry["penalty"] = min(entry.get("penalty", 0) + PENALTY_PER_429, MAX_PENALTY)
        else:
            self._penalties[key] = {"count": 1, "last_hit": now, "penalty": PENALTY_PER_429}

    def record_success(self, provider: str, model: str) -> None:
        """Record a successful request — reduce penalty."""
        key = f"{provider}:{model}"
        entry = self._penalties.get(key)
        if entry:
            entry["penalty"] = max(0, entry.get("penalty", 0) - 1)
            if entry["penalty"] == 0:
                del self._penalties[key]

    def get_penalty(self, provider: str, model: str) -> int:
        """Get current penalty for a provider+model, with time-based decay."""
        key = f"{provider}:{model}"
        entry = self._penalties.get(key)
        if not entry:
            return 0
        # Apply time-based decay
        now = time.time()
        elapsed = now - entry.get("last_hit", now)
        decay_steps = int(elapsed / DECAY_INTERVAL_S)
        if decay_steps > 0:
            entry["penalty"] = max(0, entry.get("penalty", 0) - (decay_steps * DECAY_AMOUNT))
            entry["last_hit"] = now
            if entry["penalty"] == 0:
                del self._penalties[key]
                return 0
        return int(entry["penalty"])

    def get_all_penalties(self) -> list[dict[str, Any]]:
        """Get all current penalties (for dashboard / debugging)."""
        result = []
        for key, entry in list(self._penalties.items()):
            parts = key.split(":", 1)
            penalty = self.get_penalty(parts[0], parts[1] if len(parts) > 1 else "")
            if penalty > 0:
                result.append({
                    "provider": parts[0],
                    "model": parts[1] if len(parts) > 1 else "",
                    "count": entry.get("count", 0),
                    "penalty": penalty,
                })
        return sorted(result, key=lambda x: x["penalty"], reverse=True)


@dataclass
class RequestLog:
    timestamp: float
    model: str
    provider: str
    provider_model: str
    success: bool
    error: str | None = None
    latency_ms: float = 0.0
    tokens: dict[str, int] | None = None
    attempt: int = 1


def _extract_usage(result: Any) -> dict[str, int] | None:
    """Extract token usage from a provider response."""
    if not isinstance(result, dict):
        return None
    usage = result.get("usage")
    if not usage or not isinstance(usage, dict):
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
        "completion_tokens": usage.get("completion_tokens", 0) or 0,
        "total_tokens": usage.get("total_tokens", 0) or 0,
    }


class AllRateLimitedError(RuntimeError):
    """All providers for a model returned 429 rate-limit errors."""


class Router:
    """Routes requests to providers with fallback logic and round-robin."""

    def __init__(
        self,
        config: AppConfig,
        rate_limiter: RateLimiter,
        health_checker: HealthChecker | None = None,
    ) -> None:
        self.config = config
        self.rate_limiter = rate_limiter
        self.health_checker = health_checker
        self.penalty_tracker = PenaltyTracker()
        self._logs: list[RequestLog] = []
        self._max_logs = 100
        self._rr_index: dict[str, int] = {}  # round-robin index per model

    def get_fallbacks(self, model: str) -> list[ModelFallback]:
        """Get the ordered fallback chain for a unified model name.

        When a local LLM is configured (LOCAL_LLM_MODEL), it is appended as the
        last-resort entry after all cloud providers. Use
        ``apply_preferred_connection`` to put a provider first when requested.
        """
        model_cfg = self.config.models.get(model)
        fallbacks = list(model_cfg.fallbacks) if model_cfg else []
        if local_llm_enabled() and self._get_provider("local"):
            local_model = local_llm_model() or "local"
            if not model_cfg:
                local_model = model or local_model
            if not any(fb.provider == "local" for fb in fallbacks):
                fallbacks.append(ModelFallback(provider="local", model=local_model))
        return fallbacks

    def apply_preferred_connection(
        self,
        fallbacks: list[ModelFallback],
        preferred: str | None,
    ) -> list[ModelFallback]:
        """Reorder fallbacks for a preferred provider, with fallback.

        * Default (no preference): cloud first, local Ollama last-resort.
          Local is only used after all free/cloud providers fail.
        * ``preferred_connection=local`` tries local first, then cloud
          (for async jobs like Mem0).
        * Any other preferred name is moved to the front of the chain.
        """
        preferred_name = (preferred or "").strip().lower()
        items = list(fallbacks)

        if preferred_name == "local" and not any(fb.provider == "local" for fb in items):
            if local_llm_enabled() and self._get_provider("local"):
                items.insert(0, ModelFallback(provider="local", model=local_llm_model() or "local"))

        if not preferred_name:
            # Keep local pinned last; do not drop it.
            non_local = [fb for fb in items if fb.provider != "local"]
            local_items = [fb for fb in items if fb.provider == "local"]
            return non_local + local_items

        preferred_items = [fb for fb in items if fb.provider == preferred_name]
        rest = [fb for fb in items if fb.provider != preferred_name]
        return preferred_items + rest

    def _get_provider(self, name: str) -> ProviderConfig | None:
        p = self.config.providers.get(name)
        if not p or not p.api_key:
            return None
        return p

    def _select_provider(
        self, model: str, fallbacks: list[ModelFallback],
        payload: dict[str, Any] | None = None,
    ) -> list[tuple[ProviderConfig, str]]:
        """Filter fallbacks to available providers with round-robin ordering.

        Providers are ordered: healthy first (round-robin rotated), then
        down-but-in-cooldown, skipping rate-limited ones. Providers with
        recent 429 penalties sink in priority so working ones are tried first.

        The local LLM provider is always kept at the end as a last resort and
        is excluded from round-robin rotation.

        If payload is provided, estimates token count and pre-checks TPM/TPD
        limits before adding a provider as a candidate.
        """
        # Estimate tokens from the request payload for TPM/TPD pre-checks
        estimated_tokens = estimate_request_tokens(payload) if payload else 0

        candidates: list[tuple[ProviderConfig, str]] = []
        for fb in fallbacks:
            if not fb.enabled:
                continue
            provider = self._get_provider(fb.provider)
            if not provider:
                continue
            state = self.rate_limiter.get_or_create(
                provider.name, provider.rpm_limit, provider.rpd_limit
            )
            if state.is_limited():
                logger.info("Provider %s is rate-limited, skipping", provider.name)
                continue
            # Skip providers marked as down (unless cooldown expired)
            if self.health_checker and not self.health_checker.is_available(provider.name):
                continue
            active_key = provider.api_key or ""
            # Register per-model rate limits (lazy init on first route)
            if fb.rpm_limit or fb.rpd_limit or fb.tpm_limit or fb.tpd_limit:
                per_key_rate_tracker.set_limits(
                    provider.name, fb.model, active_key,
                    rpm=fb.rpm_limit, rpd=fb.rpd_limit,
                    tpm=fb.tpm_limit, tpd=fb.tpd_limit,
                )
            # Check per-key rate limits (RPM/RPD/TPM/TPD)
            limited, reason = per_key_rate_tracker.is_limited(provider.name, fb.model, active_key)
            if limited:
                logger.debug(
                    "Provider %s/%s is per-model rate limited: %s",
                    provider.name, fb.model, reason,
                )
                continue
            # Pre-check TPM/TPD with estimated tokens for this request
            if estimated_tokens > 0:
                would_exceed, est_reason = per_key_rate_tracker.would_exceed_token_limit(
                    provider.name, fb.model, active_key, estimated_tokens,
                )
                if would_exceed:
                    logger.debug(
                        "Provider %s/%s would exceed token limit with %d estimated tokens: %s",
                        provider.name, fb.model, estimated_tokens, est_reason,
                    )
                    continue
            # Skip if this provider+model combo is on cooldown (recent 429)
            if per_key_rate_tracker.is_on_cooldown(provider.name, fb.model, active_key):
                logger.debug(
                    "Provider %s/%s is on cooldown, skipping", provider.name, fb.model,
                )
                continue
            candidates.append((provider, fb.model))

        # Keep local LLM pinned as last-resort; round-robin only cloud providers
        local_candidates = [c for c in candidates if c[0].name == "local"]
        candidates = [c for c in candidates if c[0].name != "local"]

        if len(candidates) > 1:
            # Sort by dynamic penalty: providers with more 429s sink lower
            def _sort_key(item: tuple[ProviderConfig, str]) -> int:
                provider, provider_model = item
                return self.penalty_tracker.get_penalty(provider.name, provider_model)

            candidates.sort(key=_sort_key)

            # Apply round-robin among providers with equal penalty
            idx = self._rr_index.get(model, 0) % len(candidates)
            self._rr_index[model] = idx + 1
            candidates = candidates[idx:] + candidates[:idx]

        return candidates + local_candidates

    def _log_request(self, log: RequestLog) -> None:
        self._logs.append(log)
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs:]
        # Persist to SQLite
        tokens = log.tokens or {}
        request_db.log_request(
            model=log.model,
            provider=log.provider,
            provider_model=log.provider_model,
            success=log.success,
            error=log.error,
            latency_ms=log.latency_ms,
            tokens=tokens,
            attempt=log.attempt,
        )

    def get_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        logs = self._logs[-limit:]
        return [
            {
                "timestamp": l.timestamp,
                "time_str": time.strftime("%H:%M:%S", time.localtime(l.timestamp)),
                "model": l.model,
                "provider": l.provider,
                "provider_model": l.provider_model,
                "success": l.success,
                "error": l.error,
                "latency_ms": round(l.latency_ms, 1),
                "tokens": l.tokens,
                "attempt": l.attempt,
            }
            for l in reversed(logs)
        ]

    async def route_request(
        self,
        model: str,
        payload: dict[str, Any],
        client: Any,
        preferred_connection: str | None = None,
    ) -> tuple[Any, str, str]:
        """Route a request through the fallback chain with retry and backoff.

        Returns (response, provider_name, provider_model_name).
        Raises if all providers fail.
        """
        preferred = preferred_connection or (payload or {}).get("preferred_connection")
        fallbacks = self.apply_preferred_connection(self.get_fallbacks(model), preferred)
        if not fallbacks:
            raise ValueError(f"Unknown model: {model}")

        upstream = dict(payload or {})
        upstream.pop("preferred_connection", None)

        candidates = self._select_provider(model, fallbacks, upstream)
        if not candidates:
            raise ValueError(
                f"No available providers for model '{model}'. "
                "Check API keys and rate limits."
            )

        # Cap attempts, but always keep the local last-resort provider if present
        to_try = candidates[:MAX_RETRIES]
        if candidates and candidates[-1][0].name == "local":
            if not any(p.name == "local" for p, _ in to_try):
                to_try = to_try + [candidates[-1]]

        errors: list[str] = []
        all_rate_limited = True
        for provider, provider_model in to_try:
            for attempt in range(RETRY_MAX_ATTEMPTS + 1):
                start = time.time()
                try:
                    logger.info(
                        "Routing %s -> %s/%s (attempt %d)",
                        model, provider.name, provider_model, attempt + 1,
                    )
                    self.rate_limiter.record_request(provider.name)
                    result = await send_to_provider(client, provider, provider_model, upstream)
                    latency = (time.time() - start) * 1000

                    tokens = _extract_usage(result)
                    self._log_request(RequestLog(
                        timestamp=start,
                        model=model,
                        provider=provider.name,
                        provider_model=provider_model,
                        success=True,
                        latency_ms=latency,
                        tokens=tokens,
                        attempt=attempt + 1,
                    ))
                    self.penalty_tracker.record_success(provider.name, provider_model)
                    return result, provider.name, provider_model

                except ProviderError as e:
                    latency = (time.time() - start) * 1000
                    error_msg = e.message[:200]

                    # Timeout (status 0): move to next provider immediately
                    if e.status == 0:
                        errors.append(f"{provider.name}: timeout")
                        self._log_request(RequestLog(
                            timestamp=start, model=model,
                            provider=provider.name,
                            provider_model=provider_model, success=False,
                            error="timeout", latency_ms=latency,
                            attempt=attempt + 1,
                        ))
                        break

                    # 429 rate limit
                    if e.status == 429:
                        # If Retry-After present, wait and retry same provider
                        if e.retry_after and attempt < RETRY_MAX_ATTEMPTS:
                            logger.info(
                                "Provider %s rate-limited, waiting %.1fs "
                                "(Retry-After), attempt %d/%d",
                                provider.name, e.retry_after,
                                attempt + 1, RETRY_MAX_ATTEMPTS + 1,
                            )
                            self._log_request(RequestLog(
                                timestamp=start, model=model,
                                provider=provider.name,
                                provider_model=provider_model, success=False,
                                error=f"rate limited (retry-after: {e.retry_after:.0f}s)",
                                latency_ms=latency, attempt=attempt + 1,
                            ))
                            await asyncio.sleep(e.retry_after)
                            continue
                        # Rotate key and move to next provider
                        if provider.total_keys > 1:
                            provider.rotate_key()
                            logger.info(
                                "Rotated key for %s to index %d",
                                provider.name, provider.active_key_index,
                            )
                        self.penalty_tracker.record_hit(provider.name, provider_model)
                        # Put this key on cooldown so subsequent requests skip it
                        active_key = provider.api_key or ""
                        per_key_rate_tracker.set_cooldown(provider.name, provider_model, active_key)
                        errors.append(f"{provider.name}: rate limited")
                        self._log_request(RequestLog(
                            timestamp=start, model=model,
                            provider=provider.name,
                            provider_model=provider_model, success=False,
                            error="rate limited", latency_ms=latency,
                            attempt=attempt + 1,
                        ))
                        logger.warning("Provider %s hit rate limit", provider.name)
                        break

                    # 500/502/503: retry with exponential backoff
                    if e.status in (500, 502, 503) and attempt < RETRY_MAX_ATTEMPTS:
                        backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                        logger.info(
                            "Provider %s error %d, retrying in %.1fs "
                            "(attempt %d/%d)",
                            provider.name, e.status, backoff,
                            attempt + 1, RETRY_MAX_ATTEMPTS + 1,
                        )
                        self._log_request(RequestLog(
                            timestamp=start, model=model,
                            provider=provider.name,
                            provider_model=provider_model, success=False,
                            error=f"server error {e.status}",
                            latency_ms=latency, attempt=attempt + 1,
                        ))
                        await asyncio.sleep(backoff)
                        continue

                    # Auth error: rotate key
                    if e.status in (401, 403) and provider.total_keys > 1:
                        provider.rotate_key()
                        logger.info(
                            "Rotated key for %s (auth error) to index %d",
                            provider.name, provider.active_key_index,
                        )

                    # All other errors: log and move to next provider
                    all_rate_limited = False
                    errors.append(f"{provider.name}: {error_msg}")
                    self._log_request(RequestLog(
                        timestamp=start, model=model,
                        provider=provider.name,
                        provider_model=provider_model, success=False,
                        error=error_msg, latency_ms=latency,
                        attempt=attempt + 1,
                    ))
                    logger.warning(
                        "Provider %s failed for %s: %s",
                        provider.name, model, e.message[:100],
                    )
                    break

        error_msg = f"All providers failed for model '{model}': {'; '.join(errors)}"
        if all_rate_limited and errors:
            raise AllRateLimitedError(error_msg)
        raise RuntimeError(error_msg)

    def get_model_status(self) -> list[dict[str, Any]]:
        """Get status overview of all configured models."""
        result = []
        for model_name, model_cfg in self.config.models.items():
            providers_info = []
            for fb in model_cfg.fallbacks:
                provider = self._get_provider(fb.provider)
                has_key = provider is not None
                limited = self.rate_limiter.is_limited(fb.provider) if has_key else False
                # Health status
                health_up = True
                health_status = "unknown"
                if self.health_checker:
                    h = self.health_checker.get_health(fb.provider)
                    health_status = h.status
                    health_up = self.health_checker.is_available(fb.provider)
                providers_info.append({
                    "provider": fb.provider,
                    "model": fb.model,
                    "available": has_key and not limited and health_up,
                    "has_key": has_key,
                    "rate_limited": limited,
                    "health": health_status,
                })
            active = next(
                (p for p in providers_info if p["available"]), None
            )
            result.append({
                "name": model_name,
                "providers": providers_info,
                "active_provider": active,
            })
        return result


def _is_rate_limited_error(err: ProviderError) -> bool:
    return err.status == 429
