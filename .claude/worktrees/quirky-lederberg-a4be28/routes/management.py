"""Management endpoints — rate tracking, sticky sessions, gateway keys, encrypted keys."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request

from rate_tracker import per_key_rate_tracker, PROVIDER_FREE_LIMITS
from routes.state import (
    config, encrypted_key_store, gateway_auth, sticky_sessions,
    verify_master_key,
)

router = APIRouter()


# ── Per-Key Rate Tracking API ────────────────────────────────────────────────


@router.get("/api/rate-tracking")
async def api_rate_tracking(authorization: str | None = Header(None)):
    """Get per-key rate tracking for all providers."""
    verify_master_key(authorization)
    return {
        "usage": per_key_rate_tracker.get_all_usage(),
        "known_limits": {
            prov: {mk: {"rpm": l.rpm, "rpd": l.rpd, "tpm": l.tpm, "tpd": l.tpd}
                   for mk, l in models.items()}
            for prov, models in PROVIDER_FREE_LIMITS.items()
        },
    }


@router.get("/api/rate-tracking/{provider}")
async def api_rate_tracking_provider(provider: str, authorization: str | None = Header(None)):
    """Get per-key rate tracking for a specific provider."""
    verify_master_key(authorization)
    return {"provider": provider, "usage": per_key_rate_tracker.get_provider_usage(provider)}


@router.post("/api/rate-tracking/set-limits")
async def api_set_rate_limits(request: Request, authorization: str | None = Header(None)):
    """Set rate limits for a (provider, model, key)."""
    verify_master_key(authorization)
    body = await request.json()
    provider = body.get("provider", "")
    model = body.get("model", "default")
    api_key = body.get("key", "")
    if not provider or not api_key:
        raise HTTPException(400, "Missing 'provider' or 'key'")
    per_key_rate_tracker.set_limits(
        provider, model, api_key,
        rpm=body.get("rpm", 0), rpd=body.get("rpd", 0),
        tpm=body.get("tpm", 0), tpd=body.get("tpd", 0),
    )
    return {"ok": True}


@router.post("/api/rate-tracking/cleanup")
async def api_rate_tracking_cleanup(authorization: str | None = Header(None)):
    """Clean up expired rate tracking entries."""
    verify_master_key(authorization)
    removed = per_key_rate_tracker.cleanup_expired()
    return {"removed": removed}


# ── Sticky Sessions API ──────────────────────────────────────────────────────


@router.get("/api/sessions")
async def api_sessions(authorization: str | None = Header(None)):
    """Get all active sticky sessions."""
    verify_master_key(authorization)
    return {
        "sessions": sticky_sessions.get_all(),
        "stats": sticky_sessions.get_stats(),
    }


@router.delete("/api/sessions/{conversation_id}")
async def api_session_remove(conversation_id: str, authorization: str | None = Header(None)):
    """Remove a sticky session."""
    verify_master_key(authorization)
    if not sticky_sessions.remove(conversation_id):
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@router.post("/api/sessions/cleanup")
async def api_sessions_cleanup(authorization: str | None = Header(None)):
    """Clean up expired sessions."""
    verify_master_key(authorization)
    removed = sticky_sessions.cleanup_expired()
    return {"removed": removed}


# ── Gateway Auth API ─────────────────────────────────────────────────────────


@router.get("/api/gateway-keys")
async def api_gateway_keys(authorization: str | None = Header(None)):
    """List all gateway API keys."""
    verify_master_key(authorization)
    if not gateway_auth:
        return {"keys": [], "stats": {"total_keys": 0}}
    return {"keys": gateway_auth.list_keys(), "stats": gateway_auth.get_stats()}


@router.post("/api/gateway-keys")
async def api_create_gateway_key(request: Request, authorization: str | None = Header(None)):
    """Create a new gateway API key."""
    verify_master_key(authorization)
    if not gateway_auth:
        raise HTTPException(400, "Gateway auth not configured (set MASTER_KEY)")
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Missing 'name' field")
    raw_key, gw_key = gateway_auth.create_key(
        name,
        allowed_models=body.get("allowed_models", []),
        allowed_providers=body.get("allowed_providers", []),
        max_rpm=body.get("max_rpm", 0),
        max_tpm=body.get("max_tpm", 0),
        is_admin=body.get("is_admin", False),
    )
    return {"ok": True, "key": raw_key, "name": gw_key.name, "warning": "Save this key — it won't be shown again!"}


@router.delete("/api/gateway-keys/{name}")
async def api_revoke_gateway_key(name: str, authorization: str | None = Header(None)):
    """Revoke a gateway API key."""
    verify_master_key(authorization)
    if not gateway_auth:
        raise HTTPException(400, "Gateway auth not configured")
    if not gateway_auth.revoke_key(name):
        raise HTTPException(404, f"Key '{name}' not found")
    return {"ok": True}


@router.put("/api/gateway-keys/{name}/toggle")
async def api_toggle_gateway_key(name: str, request: Request, authorization: str | None = Header(None)):
    """Enable or disable a gateway key."""
    verify_master_key(authorization)
    if not gateway_auth:
        raise HTTPException(400, "Gateway auth not configured")
    body = await request.json()
    enabled = body.get("enabled", True)
    if not gateway_auth.toggle_key(name, enabled):
        raise HTTPException(404, f"Key '{name}' not found")
    return {"ok": True, "name": name, "enabled": enabled}


# ── Encrypted Key Storage API ────────────────────────────────────────────────


@router.get("/api/encrypted-keys")
async def api_encrypted_keys(authorization: str | None = Header(None)):
    """List encrypted keys (masked)."""
    verify_master_key(authorization)
    return {"keys": encrypted_key_store.list_keys()}


@router.post("/api/encrypted-keys")
async def api_add_encrypted_key(request: Request, authorization: str | None = Header(None)):
    """Add a key to encrypted storage."""
    verify_master_key(authorization)
    body = await request.json()
    provider = body.get("provider", "").strip()
    key = body.get("key", "").strip()
    if not provider or not key:
        raise HTTPException(400, "Missing 'provider' or 'key'")
    index = encrypted_key_store.add_key(provider, key)
    prov = config.providers.get(provider)
    if prov:
        all_keys = encrypted_key_store.get_keys(provider)
        prov.api_keys = all_keys
    return {"ok": True, "provider": provider, "index": index}


@router.delete("/api/encrypted-keys/{provider}/{index}")
async def api_remove_encrypted_key(provider: str, index: int, authorization: str | None = Header(None)):
    """Remove a key from encrypted storage."""
    verify_master_key(authorization)
    if not encrypted_key_store.remove_key(provider, index):
        raise HTTPException(404, "Key not found")
    prov = config.providers.get(provider)
    if prov:
        remaining = encrypted_key_store.get_keys(provider)
        prov.api_keys = remaining
    return {"ok": True}
