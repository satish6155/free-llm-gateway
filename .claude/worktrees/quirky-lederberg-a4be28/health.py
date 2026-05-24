"""Background health checks for provider availability and per-key validation."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from config import ProviderConfig
from providers import get_provider

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 180  # 3 minutes
DOWN_THRESHOLD = 2  # consecutive failures before marking down
COOLDOWN_SECONDS = 300  # retry down provider after 5 minutes
CONSECUTIVE_FAILURES_TO_DISABLE = 3  # auto-disable key after this many failures


@dataclass
class ProviderHealth:
    status: str = "unknown"  # up, down, unknown
    last_check_time: float = 0.0
    last_error: str | None = None
    consecutive_failures: int = 0
    latency_ms: float = 0.0


@dataclass
class KeyHealth:
    """Health status for a single API key."""
    key_index: int = 0
    status: str = "unknown"  # healthy, invalid, error, rate_limited, unknown
    consecutive_failures: int = 0
    last_checked: float = 0.0
    last_error: str | None = None
    disabled: bool = False


class HealthChecker:
    """Runs periodic health checks on all configured providers and API keys."""

    def __init__(self) -> None:
        self._health: dict[str, ProviderHealth] = {}
        self._key_health: dict[str, list[KeyHealth]] = {}
        self._task: asyncio.Task | None = None
        self._check_cycle: int = 0  # alternates provider/key checks

    def get_health(self, provider: str) -> ProviderHealth:
        return self._health.get(provider, ProviderHealth())

    def get_all_health(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "status": h.status,
                "last_check_time": h.last_check_time,
                "last_error": h.last_error,
                "consecutive_failures": h.consecutive_failures,
                "latency_ms": round(h.latency_ms, 1),
            }
            for name, h in self._health.items()
        }

    def get_key_health(self, provider: str) -> list[dict[str, Any]]:
        """Get per-key health status for a provider."""
        keys = self._key_health.get(provider, [])
        return [
            {
                "key_index": kh.key_index,
                "status": kh.status,
                "consecutive_failures": kh.consecutive_failures,
                "last_checked": kh.last_checked,
                "last_error": kh.last_error,
                "disabled": kh.disabled,
            }
            for kh in keys
        ]

    def get_all_key_health(self) -> dict[str, list[dict[str, Any]]]:
        """Get per-key health status for all providers."""
        return {
            provider: self.get_key_health(provider)
            for provider in self._key_health
        }

    def is_available(self, provider: str) -> bool:
        h = self._health.get(provider)
        if not h or h.status == "unknown":
            return True  # assume available until first check
        if h.status == "up":
            return True
        # status == "down": allow retry after cooldown
        if time.time() - h.last_check_time > COOLDOWN_SECONDS:
            return True
        return False

    async def check_provider(
        self, client: httpx.AsyncClient, provider: ProviderConfig
    ) -> None:
        """Ping a single provider and update its health status."""
        if not provider.api_key:
            self._health[provider.name] = ProviderHealth(status="unknown")
            return

        start = time.time()
        try:
            ok = await self._ping(client, provider)
            latency = (time.time() - start) * 1000

            health = self._health.get(provider.name, ProviderHealth())
            health.last_check_time = time.time()
            health.latency_ms = latency

            if ok:
                health.status = "up"
                health.last_error = None
                health.consecutive_failures = 0
                logger.debug("Health check %s: UP (%.0fms)", provider.name, latency)
            else:
                health.consecutive_failures += 1
                health.last_error = "ping failed"
                if health.consecutive_failures >= DOWN_THRESHOLD:
                    health.status = "down"
                    logger.warning(
                        "Health check %s: DOWN (%d consecutive failures)",
                        provider.name,
                        health.consecutive_failures,
                    )
                else:
                    health.status = "up"  # still up, one failure is ok
                    logger.info(
                        "Health check %s: degraded (failure %d/%d)",
                        provider.name,
                        health.consecutive_failures,
                        DOWN_THRESHOLD,
                    )

            self._health[provider.name] = health

        except Exception as e:
            latency = (time.time() - start) * 1000
            health = self._health.get(provider.name, ProviderHealth())
            health.last_check_time = time.time()
            health.latency_ms = latency
            health.last_error = str(e)[:200]
            health.consecutive_failures += 1
            if health.consecutive_failures >= DOWN_THRESHOLD:
                health.status = "down"
            self._health[provider.name] = health
            logger.debug("Health check %s failed: %s", provider.name, e)

    async def check_key(
        self,
        client: httpx.AsyncClient,
        provider: ProviderConfig,
        key_index: int,
    ) -> KeyHealth:
        """Validate a single API key against the provider's /models endpoint.

        Updates KeyHealth status and auto-disables after
        CONSECUTIVE_FAILURES_TO_DISABLE confirmed auth failures.
        """
        key = provider.api_keys[key_index] if key_index < len(provider.api_keys) else ""
        if not key:
            return KeyHealth(key_index=key_index, status="unknown")

        kh = self._get_or_create_key_health(provider.name, key_index)
        kh.last_checked = time.time()

        try:
            status = await self._validate_key(client, provider, key)

            if status == 200:
                kh.status = "healthy"
                kh.consecutive_failures = 0
                kh.last_error = None
                # Re-enable key if it was disabled and is now healthy
                if kh.disabled and key in provider.disabled_keys:
                    provider.disabled_keys.remove(key)
                    kh.disabled = False
                    logger.info("Key %d for %s re-enabled (healthy)", key_index, provider.name)
            elif status in (401, 403):
                kh.status = "invalid"
                kh.consecutive_failures += 1
                kh.last_error = f"Auth error ({status})"
                if kh.consecutive_failures >= CONSECUTIVE_FAILURES_TO_DISABLE:
                    self._disable_key(provider, key_index, kh, "invalid")
            elif status == 429:
                kh.status = "rate_limited"
                kh.last_error = "Rate limited (429)"
                # Rate limits don't count toward disable threshold
            elif status >= 500:
                kh.status = "error"
                kh.last_error = f"Server error ({status})"
                # Transport/server errors don't count toward disable
            else:
                kh.status = "error"
                kh.last_error = f"Unexpected status ({status})"

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            kh.status = "error"
            kh.last_error = f"Connection error: {type(e).__name__}"
            # Transport errors don't count toward disable threshold
        except Exception as e:
            kh.status = "error"
            kh.last_error = str(e)[:200]

        return kh

    async def check_all_keys(
        self, client: httpx.AsyncClient, providers: dict[str, ProviderConfig]
    ) -> None:
        """Validate all individual API keys for all providers."""
        tasks = []
        for provider in providers.values():
            if not provider.api_keys:
                continue
            for idx in range(len(provider.api_keys)):
                tasks.append(self.check_key(client, provider, idx))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _get_or_create_key_health(self, provider: str, key_index: int) -> KeyHealth:
        """Get or create KeyHealth entry for a provider+key."""
        if provider not in self._key_health:
            self._key_health[provider] = []
        keys = self._key_health[provider]
        while len(keys) <= key_index:
            keys.append(KeyHealth(key_index=len(keys)))
        return keys[key_index]

    def _disable_key(
        self, provider: ProviderConfig, key_index: int, kh: KeyHealth, reason: str
    ) -> None:
        """Disable an API key after consecutive failures."""
        if kh.disabled:
            return
        key = provider.api_keys[key_index] if key_index < len(provider.api_keys) else ""
        if not key:
            return
        kh.disabled = True
        if key not in provider.disabled_keys:
            provider.disabled_keys.append(key)
        logger.warning(
            "Auto-disabled key %d for %s (reason: %s, %d consecutive failures)",
            key_index, provider.name, reason, kh.consecutive_failures,
        )

    async def _validate_key(
        self, client: httpx.AsyncClient, provider: ProviderConfig, api_key: str
    ) -> int:
        """Validate a specific API key against provider's models endpoint.

        Delegates URL and header construction to the provider adapter.
        Returns the HTTP status code.
        """
        try:
            p = get_provider(provider.name)
            url = p.validate_key_url(provider)
            headers = p.get_models_headers(provider, api_key=api_key)
            resp = await client.get(url, headers=headers, timeout=10.0)
            return resp.status_code
        except (httpx.TimeoutException, httpx.ConnectError):
            raise
        except Exception as e:
            logger.debug("Key validation error for %s: %s", provider.name, e)
            return 0

    async def _ping(self, client: httpx.AsyncClient, provider: ProviderConfig) -> bool:
        """Send a lightweight request to check if provider is alive."""
        try:
            p = get_provider(provider.name)
            url = p.get_models_url(provider)
            headers = p.get_models_headers(provider, api_key=provider.api_key)
            resp = await client.get(url, headers=headers, timeout=10.0)
            return resp.status_code < 500
        except (httpx.TimeoutException, httpx.ConnectError):
            return False

    async def check_all(
        self, client: httpx.AsyncClient, providers: dict[str, ProviderConfig]
    ) -> None:
        """Run health checks on all providers concurrently."""
        tasks = [
            self.check_provider(client, p)
            for p in providers.values()
            if p.api_key
        ]
        if tasks:
            await asyncio.gather(*tasks)

    def start(
        self, client: httpx.AsyncClient, providers: dict[str, ProviderConfig]
    ) -> None:
        """Start the background health check loop."""
        self._task = asyncio.create_task(
            self._loop(client, providers)
        )

    async def stop(self) -> None:
        """Cancel the background task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(
        self, client: httpx.AsyncClient, providers: dict[str, ProviderConfig]
    ) -> None:
        """Periodically check all providers and individual keys.

        Alternates between provider connectivity checks and per-key validation
        to spread the load.
        """
        while True:
            try:
                self._check_cycle += 1
                if self._check_cycle % 2 == 0:
                    # Even cycles: check individual keys
                    await self.check_all_keys(client, providers)
                else:
                    # Odd cycles: check provider connectivity
                    await self.check_all(client, providers)
            except Exception as e:
                logger.error("Health check loop error: %s", e)
            await asyncio.sleep(CHECK_INTERVAL)
