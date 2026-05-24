"""Health & liveness endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Header, HTTPException

from routes.state import config, health_checker, client, verify_master_key

router = APIRouter()


@router.get("/api/ping")
async def ping():
    """Lightweight liveness probe for health checks and uptime monitoring."""
    return {"status": "ok", "timestamp": time.time()}


@router.get("/api/health/keys")
async def api_health_keys(authorization: str | None = Header(None)):
    """Per-key health status for all providers."""
    verify_master_key(authorization)
    return health_checker.get_all_key_health()


@router.get("/api/health/keys/{provider}")
async def api_health_keys_provider(
    provider: str, authorization: str | None = Header(None),
):
    """Per-key health status for a specific provider."""
    verify_master_key(authorization)
    if provider not in config.providers:
        raise HTTPException(404, f"Provider '{provider}' not found")
    return health_checker.get_key_health(provider)


@router.post("/api/health/check")
async def api_health_check_now(authorization: str | None = Header(None)):
    """Trigger an immediate health check (providers + keys)."""
    verify_master_key(authorization)
    if not client:
        raise HTTPException(503, "Server not ready")
    await health_checker.check_all(client, config.providers)
    await health_checker.check_all_keys(client, config.providers)
    return {
        "providers": health_checker.get_all_health(),
        "keys": health_checker.get_all_key_health(),
    }
