"""Other endpoints — cache, queue, usage, combos, quotas, OAuth."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import HTMLResponse

from routes.state import (
    client, config, combo_manager, oauth_manager, quota_tracker,
    request_queue, response_cache, usage_tracker, verify_master_key,
)

router = APIRouter()


# ── Usage tracking API ───────────────────────────────────────────────────────


@router.get("/api/usage")
async def api_usage(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return usage_tracker.get_stats()


# ── Cache API ────────────────────────────────────────────────────────────────


@router.get("/api/cache")
async def api_cache_stats(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return response_cache.stats()


@router.delete("/api/cache")
async def api_cache_clear(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    cleared = response_cache.clear()
    return {"cleared": cleared}


@router.post("/api/cache/prune")
async def api_cache_prune(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    pruned = response_cache.prune_expired()
    return {"pruned": pruned}


# ── Queue API ────────────────────────────────────────────────────────────────


@router.get("/api/queue")
async def api_queue_stats(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return {
        **request_queue.stats(),
        "pending": request_queue.get_pending_list(),
    }


@router.get("/api/queue/{request_id}")
async def api_queue_poll(request_id: str, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    req = request_queue._pending.get(request_id)
    if not req:
        raise HTTPException(404, f"Request {request_id} not found")
    return {
        "id": req.id,
        "model": req.model,
        "status": req.status,
        "attempts": req.attempts,
        "result": req.result,
        "error": req.error,
    }


# ── Custom Combos API ────────────────────────────────────────────────────────


@router.get("/api/combos")
async def api_list_combos(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return {"combos": combo_manager.list_combos()}


@router.post("/api/combos")
async def api_create_combo(request: Request, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    body = await request.json()
    name = body.get("name", "").strip()
    entries = body.get("entries", [])
    description = body.get("description", "")

    if not name:
        raise HTTPException(400, "Missing 'name' field")
    if not entries:
        raise HTTPException(400, "Missing 'entries' field")
    for e in entries:
        if "provider" not in e or "model" not in e:
            raise HTTPException(400, "Each entry must have 'provider' and 'model'")

    combo = combo_manager.create_combo(name, entries, description)
    config.models[name] = combo.to_model_config()
    return {"ok": True, "combo": combo.name, "entries": len(combo.entries)}


@router.put("/api/combos/{name}")
async def api_update_combo(name: str, request: Request, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    body = await request.json()
    combo = combo_manager.update_combo(
        name,
        entries=body.get("entries"),
        description=body.get("description"),
    )
    if not combo:
        raise HTTPException(404, f"Combo '{name}' not found")
    config.models[name] = combo.to_model_config()
    return {"ok": True, "combo": combo.name}


@router.delete("/api/combos/{name}")
async def api_delete_combo(name: str, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    if not combo_manager.delete_combo(name):
        raise HTTPException(404, f"Combo '{name}' not found")
    config.models.pop(name, None)
    return {"ok": True}


# ── Quota Tracking API ────────────────────────────────────────────────────────


@router.get("/api/quotas")
async def api_quotas(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return {
        "providers": quota_tracker.get_all_quotas(),
        "dashboard": quota_tracker.get_dashboard_summary(),
    }


@router.get("/api/quotas/{provider}")
async def api_quota_provider(provider: str, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    quota = quota_tracker.get_quota(provider)
    if not quota:
        raise HTTPException(404, f"No quota data for provider: {provider}")
    return quota


@router.put("/api/quotas/{provider}")
async def api_set_quota_limits(
    provider: str,
    request: Request,
    authorization: str | None = Header(None),
):
    verify_master_key(authorization)
    body = await request.json()
    q = quota_tracker.set_provider_limits(
        provider,
        rpm=body.get("rpm"),
        rpd=body.get("rpd"),
        tokens_per_day=body.get("tokens_per_day"),
        spending_daily=body.get("spending_limit_daily"),
        spending_monthly=body.get("spending_limit_monthly"),
    )
    return {"ok": True, "provider": provider, "quota": q.to_dict()}


# ── OAuth API ─────────────────────────────────────────────────────────────────


@router.get("/api/oauth/providers")
async def api_oauth_providers(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return {"providers": oauth_manager.list_connections()}


@router.post("/api/oauth/authorize")
async def api_oauth_authorize(request: Request, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    body = await request.json()
    provider = body.get("provider", "").strip()
    client_id = body.get("client_id", "").strip()
    if not provider:
        raise HTTPException(400, "Missing 'provider' field")
    result = oauth_manager.get_authorize_url(provider, client_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.get("/api/oauth/callback")
async def api_oauth_callback(state: str, code: str):
    if not client:
        raise HTTPException(503, "Server not ready")
    result = await oauth_manager.handle_callback(state, code, client)
    if "error" in result:
        return HTMLResponse(content=f"<h3>OAuth Error</h3><p>{result['error']}</p>")
    return HTMLResponse(
        content=f"<h3>Connected!</h3><p>{result['provider']} is now connected.</p>"
        "<p>You can close this tab.</p>"
    )


@router.post("/api/oauth/refresh/{provider}")
async def api_oauth_refresh(provider: str, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    if not client:
        raise HTTPException(503, "Server not ready")
    result = await oauth_manager.refresh_token(provider, client)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.delete("/api/oauth/{provider}")
async def api_oauth_disconnect(provider: str, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    if not oauth_manager.remove_token(provider):
        raise HTTPException(404, f"No OAuth connection for: {provider}")
    return {"ok": True}
