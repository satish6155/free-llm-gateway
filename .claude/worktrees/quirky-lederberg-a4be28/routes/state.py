"""Shared state for route modules — set during app lifespan.

All route modules import from here to access shared instances.
main.py sets these during startup via init_state().
"""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING

from fastapi import HTTPException
from gateway_auth import GatewayAuthManager

if TYPE_CHECKING:
    import httpx
    from benchmark import BenchmarkRunner
    from cache import ResponseCache
    from config import AppConfig
    from custom_combos import ComboManager
    from gateway_auth import GatewayAuthManager
    from health import HealthChecker
    from key_encryptor import EncryptedKeyStore
    from key_manager import KeyManager
    from oauth_manager import OAuthManager
    from quota_tracker import QuotaTracker
    from rate_limiter import RateLimiter
    from request_queue import RequestQueue
    from router import Router
    from smart_default import SmartDefault
    from smart_router import SmartRouter
    from sticky_sessions import StickySessionManager
    from tracking import UsageTracker

# ── Shared instances (set during lifespan) ──────────────────────────────────────

config: AppConfig | None = None
rate_limiter: RateLimiter | None = None
health_checker: HealthChecker | None = None
router: Router | None = None
key_manager: KeyManager | None = None
smart_router: SmartRouter | None = None
benchmark_runner: BenchmarkRunner | None = None
smart_default: SmartDefault | None = None
encrypted_key_store: EncryptedKeyStore | None = None
gateway_auth: GatewayAuthManager | None = None
client: httpx.AsyncClient | None = None

# Feature module singletons (imported from their modules)
# These are set so route modules don't need to re-import
response_cache: ResponseCache | None = None
request_queue: RequestQueue | None = None
usage_tracker: UsageTracker | None = None
combo_manager: ComboManager | None = None
quota_tracker: QuotaTracker | None = None
oauth_manager: OAuthManager | None = None
sticky_sessions: StickySessionManager | None = None

# Benchmark results (mutable dict)
benchmark_results: dict = {}


def init_state(**kwargs) -> None:
    """Initialize shared state from main.py lifespan."""
    for key, value in kwargs.items():
        if hasattr(type_globals := globals(), key):
            globals()[key] = value


def _sync_keys_to_config() -> None:
    """Push key_manager keys into config.providers so the router picks them up."""
    if not config or not key_manager:
        return
    for name, prov in config.providers.items():
        km_keys = key_manager.get_keys(name)
        if km_keys:
            prov.api_keys = km_keys


def verify_master_key(authorization: str | None) -> None:
    """Validate the Authorization header against master key or gateway keys.

    Uses constant-time comparison to prevent timing attacks.
    Raises HTTPException(401) on failure.
    """
    if not config:
        return  # no config loaded yet
    if not config.master_key and not gateway_auth:
        return  # no auth if nothing configured
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    token = authorization.removeprefix("Bearer ").strip()

    # Check unified gateway API key first
    if gateway_auth and token.startswith(GatewayAuthManager.GATEWAY_KEY_PREFIX):
        gw_key = gateway_auth.validate_key(token)
        if gw_key and gw_key.enabled:
            return
        raise HTTPException(401, "Invalid gateway API key")

    # Fall back to master key — constant-time comparison
    if config.master_key and hmac.compare_digest(token, config.master_key):
        return

    raise HTTPException(401, "Invalid API key")
