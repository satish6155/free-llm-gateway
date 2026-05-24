"""API key management endpoints — CRUD, validation, toggle."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request

from config import PROVIDER_DEFS
from routes.state import (
    client, config, key_manager, verify_master_key, _sync_keys_to_config,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/keys")
async def add_key(request: Request, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    body = await request.json()
    provider = body.get("provider", "").strip()
    key = body.get("key", "").strip()
    if not provider or not key:
        raise HTTPException(400, "Missing 'provider' or 'key'")
    if provider not in PROVIDER_DEFS:
        raise HTTPException(400, f"Unknown provider: {provider}")

    # Validate key against provider's /models endpoint
    valid = False
    validation_error = None
    prov_def = PROVIDER_DEFS[provider]
    base_url = prov_def[1]
    if "{account_id}" in base_url:
        base_url = base_url.replace(
            "{account_id}", os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        )
    test_url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {key}"}
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/free-llm-gateway"
        headers["X-Title"] = "Free LLM Gateway"
    try:
        if client:
            resp = await client.get(test_url, headers=headers, timeout=15.0)
            valid = resp.status_code < 400
            if not valid:
                validation_error = f"HTTP {resp.status_code}"
    except Exception as e:
        validation_error = str(e)[:200]

    index = key_manager.add_key(provider, key)
    key_manager.set_validated(provider, index, valid)
    _sync_keys_to_config()
    return {
        "ok": True, "provider": provider, "index": index,
        "valid": valid, "validation_error": validation_error,
    }


@router.delete("/api/keys/{provider}/{index}")
async def remove_key(provider: str, index: int, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    if not key_manager.remove_key(provider, index):
        raise HTTPException(404, "Key not found")
    _sync_keys_to_config()
    return {"ok": True}


@router.get("/api/keys")
async def list_keys(authorization: str | None = Header(None)):
    """List all keys: .env-sourced (read-only) + runtime-added (deletable)."""
    verify_master_key(authorization)
    runtime_keys = key_manager.list_keys()
    env_keys: dict[str, list[dict[str, Any]]] = {}
    for name, prov in config.providers.items():
        if prov.api_keys:
            env_keys[name] = []
            for i, key in enumerate(prov.api_keys):
                masked = "****" + key[-4:] if len(key) > 4 else "****"
                runtime_entries = runtime_keys.get(name, [])
                is_runtime = any(
                    e.get("key_masked") == masked for e in runtime_entries
                )
                if not is_runtime:
                    env_keys[name].append({
                        "index": i,
                        "key_masked": masked,
                        "source": "env",
                        "validated": None,
                        "deletable": False,
                        "added_at": "",
                    })

    all_keys: dict[str, list[dict[str, Any]]] = {}
    all_providers = set(list(runtime_keys.keys()) + list(env_keys.keys()))
    for pname in all_providers:
        entries = []
        for e in env_keys.get(pname, []):
            entries.append(e)
        for e in runtime_keys.get(pname, []):
            entry = {**e, "source": "runtime", "deletable": True}
            entries.append(entry)
        if entries:
            all_keys[pname] = entries
    return {"keys": all_keys}


@router.post("/api/keys/{provider}/validate")
async def validate_provider_keys(provider: str, authorization: str | None = Header(None)):
    """Validate all keys for a provider by testing each against the /models endpoint."""
    verify_master_key(authorization)
    keys = key_manager.get_keys(provider)
    if not keys:
        raise HTTPException(404, f"No keys found for provider: {provider}")

    prov_def = PROVIDER_DEFS.get(provider)
    if not prov_def:
        raise HTTPException(400, f"Unknown provider: {provider}")

    results = []
    for i, api_key in enumerate(keys):
        base_url = prov_def[1]
        if "{account_id}" in base_url:
            base_url = base_url.replace(
                "{account_id}", os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
            )
        test_url = f"{base_url}/models"
        headers = {"Authorization": f"Bearer " + api_key}
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/free-llm-gateway"
            headers["X-Title"] = "Free LLM Gateway"
        try:
            resp = await client.get(test_url, headers=headers, timeout=15.0)
            valid = resp.status_code < 400
            key_manager.set_validated(provider, i, valid)
            results.append({"index": i, "valid": valid})
        except Exception as e:
            key_manager.set_validated(provider, i, False)
            results.append({"index": i, "valid": False, "error": str(e)[:200]})

    return {"provider": provider, "results": results}


@router.post("/api/keys/{provider}/{index}/validate")
async def validate_key(provider: str, index: int, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    keys = key_manager.get_keys(provider)
    if index < 0 or index >= len(keys):
        raise HTTPException(404, "Key not found")

    api_key = keys[index]
    prov_def = PROVIDER_DEFS.get(provider)
    if not prov_def:
        raise HTTPException(400, f"Unknown provider: {provider}")

    base_url = prov_def[1]
    if "{account_id}" in base_url:
        base_url = base_url.replace(
            "{account_id}", os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        )

    test_url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer " + api_key}
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/free-llm-gateway"
        headers["X-Title"] = "Free LLM Gateway"

    try:
        resp = await client.get(test_url, headers=headers, timeout=15.0)
        valid = resp.status_code < 400
        key_manager.set_validated(provider, index, valid)
        if valid:
            return {"valid": True}
        else:
            return {"valid": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        key_manager.set_validated(provider, index, False)
        return {"valid": False, "error": str(e)[:200]}


@router.patch("/api/keys/{provider}/{index}/toggle")
async def toggle_key(provider: str, index: int, request: Request, authorization: str | None = Header(None)):
    """Enable or disable a specific API key by index."""
    verify_master_key(authorization)
    body = await request.json()
    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(400, "Field 'enabled' must be a boolean")

    prov = config.providers.get(provider)
    if not prov:
        raise HTTPException(404, f"Provider '{provider}' not found")
    if index < 0 or index >= len(prov.api_keys):
        raise HTTPException(404, "Key index out of range")

    key = prov.api_keys[index]
    if enabled:
        if key in prov.disabled_keys:
            prov.disabled_keys.remove(key)
    else:
        if key not in prov.disabled_keys:
            prov.disabled_keys.append(key)

    return {
        "success": True,
        "provider": provider,
        "key_index": index,
        "enabled": enabled,
        "disabled_keys_count": len(prov.disabled_keys),
    }


@router.post("/api/keys/validate-all")
async def api_validate_all_keys(authorization: str | None = Header(None)):
    """Validate all keys for all providers."""
    verify_master_key(authorization)
    if not client:
        raise HTTPException(503, "Server not ready")

    results = {}
    for name, prov_def in PROVIDER_DEFS.items():
        keys = key_manager.get_keys(name)
        provider = config.providers.get(name)
        if not keys and provider and provider.api_keys:
            keys = provider.api_keys

        if not keys:
            results[name] = {"status": "no_key", "keys": []}
            continue

        base_url = prov_def[1]
        if "{account_id}" in base_url:
            base_url = base_url.replace(
                "{account_id}", os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
            )

        key_results = []
        for i, api_key in enumerate(keys):
            test_url = f"{base_url}/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            if name == "openrouter":
                headers["HTTP-Referer"] = "https://github.com/free-llm-gateway"
                headers["X-Title"] = "Free LLM Gateway"
            try:
                resp = await client.get(test_url, headers=headers, timeout=15.0)
                if resp.status_code < 400:
                    status = "valid"
                elif resp.status_code == 429:
                    status = "rate_limited"
                elif resp.status_code in (401, 403):
                    status = "invalid"
                else:
                    status = "error"
                key_results.append({"index": i, "status": status})
            except Exception:
                key_results.append({"index": i, "status": "error"})

        overall = "valid"
        if all(k["status"] in ("invalid", "error") for k in key_results):
            overall = "invalid"
        elif any(k["status"] == "rate_limited" for k in key_results):
            overall = "rate_limited"
        elif any(k["status"] == "error" for k in key_results):
            overall = "error"

        results[name] = {"status": overall, "keys": key_results}

    valid = sum(1 for r in results.values() if r["status"] == "valid")
    total_with_keys = sum(1 for r in results.values() if r["status"] != "no_key")
    return {
        "summary": {
            "total": total_with_keys,
            "valid": valid,
            "invalid": sum(1 for r in results.values() if r["status"] == "invalid"),
            "rate_limited": sum(1 for r in results.values() if r["status"] == "rate_limited"),
            "no_key": sum(1 for r in results.values() if r["status"] == "no_key"),
            "error": sum(1 for r in results.values() if r["status"] == "error"),
        },
        "results": results,
    }
