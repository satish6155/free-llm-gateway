"""Fallback chain management — list, update, sort, token usage, model limits."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request

from config import ModelFallback
from rate_tracker import per_key_rate_tracker
from request_db import request_db
from routes.state import config, router, verify_master_key

router = APIRouter()


@router.get("/api/fallbacks")
async def api_get_fallbacks(authorization: str | None = Header(None)):
    """Get current fallback chains for all models with penalty info."""
    verify_master_key(authorization)
    result = []
    for model_name, model_cfg in config.models.items():
        providers = []
        for fb in model_cfg.fallbacks:
            penalty = router.penalty_tracker.get_penalty(fb.provider, fb.model)
            providers.append({
                "provider": fb.provider,
                "model": fb.model,
                "enabled": fb.enabled,
                "penalty": penalty,
                "rpm_limit": fb.rpm_limit,
                "rpd_limit": fb.rpd_limit,
                "tpm_limit": fb.tpm_limit,
                "tpd_limit": fb.tpd_limit,
            })
        result.append({
            "model": model_name,
            "providers": providers,
            "intelligence_rank": model_cfg.intelligence_rank,
            "speed_rank": model_cfg.speed_rank,
            "size_label": model_cfg.size_label or None,
            "monthly_token_budget": model_cfg.monthly_token_budget,
        })
    return result


@router.put("/api/fallbacks/{model}")
async def api_update_fallbacks(
    model: str,
    request: Request,
    authorization: str | None = Header(None),
):
    """Reorder or enable/disable fallbacks for a specific model."""
    verify_master_key(authorization)
    model_cfg = config.models.get(model)
    if not model_cfg:
        raise HTTPException(404, f"Model '{model}' not found")

    body = await request.json()
    if not isinstance(body, list):
        raise HTTPException(400, "Body must be a list of fallback entries")

    new_fallbacks: list = []
    for entry in body:
        provider = entry.get("provider", "")
        provider_model = entry.get("model", "")
        enabled = entry.get("enabled", True)
        existing = next(
            (fb for fb in model_cfg.fallbacks
             if fb.provider == provider and fb.model == provider_model),
            None,
        )
        if existing:
            existing.enabled = enabled
            new_fallbacks.append(existing)
        else:
            new_fallbacks.append(ModelFallback(
                provider=provider, model=provider_model, enabled=enabled,
            ))

    model_cfg.fallbacks = new_fallbacks
    return {"success": True, "model": model, "fallbacks": len(new_fallbacks)}


@router.post("/api/fallbacks/{model}/sort/{preset}")
async def api_sort_fallbacks(
    model: str,
    preset: str,
    authorization: str | None = Header(None),
):
    """Sort fallbacks by preset: priority, penalty, health, intelligence, speed, budget."""
    verify_master_key(authorization)
    model_cfg = config.models.get(model)
    if not model_cfg:
        raise HTTPException(404, f"Model '{model}' not found")

    if preset == "penalty":
        model_cfg.fallbacks.sort(
            key=lambda fb: router.penalty_tracker.get_penalty(fb.provider, fb.model),
        )
    elif preset == "health":
        def _health_key(fb: Any) -> int:
            if not fb.enabled:
                return 1000
            if router.health_checker and not router.health_checker.is_available(fb.provider):
                return 500
            return router.penalty_tracker.get_penalty(fb.provider, fb.model)
        model_cfg.fallbacks.sort(key=_health_key)
    elif preset == "intelligence":
        model_cfg.fallbacks.sort(
            key=lambda fb: -_get_fallback_rank(fb, "intelligence"),
        )
    elif preset == "speed":
        model_cfg.fallbacks.sort(
            key=lambda fb: -_get_fallback_rank(fb, "speed"),
        )
    elif preset == "budget":
        usage = request_db.get_model_token_usage(30)
        def _budget_key(fb: Any) -> int:
            key = f"{fb.provider}:{fb.model}"
            return usage.get(key, {}).get("total_tokens", 0)
        model_cfg.fallbacks.sort(key=_budget_key)
    elif preset == "priority":
        pass  # Reset to original order
    else:
        raise HTTPException(
            400,
            f"Unknown preset '{preset}'. Use: priority, penalty, health, intelligence, speed, budget",
        )

    return {
        "success": True,
        "model": model,
        "preset": preset,
        "fallbacks": [
            {
                "provider": fb.provider,
                "model": fb.model,
                "enabled": fb.enabled,
                "penalty": router.penalty_tracker.get_penalty(fb.provider, fb.model),
            }
            for fb in model_cfg.fallbacks
        ],
    }


def _get_fallback_rank(fb: Any, rank_type: str) -> int:
    """Get intelligence or speed rank for a fallback's model."""
    for _model_name, model_cfg in config.models.items():
        for existing_fb in model_cfg.fallbacks:
            if existing_fb.provider == fb.provider and existing_fb.model == fb.model:
                if rank_type == "intelligence":
                    return model_cfg.intelligence_rank
                elif rank_type == "speed":
                    return model_cfg.speed_rank
    return 0


@router.get("/api/fallbacks/token-usage")
async def api_fallback_token_usage(authorization: str | None = Header(None)):
    """Per-model token usage vs monthly budget."""
    verify_master_key(authorization)
    usage = request_db.get_model_token_usage(30)
    result = []
    for model_name, model_cfg in config.models.items():
        model_usage = usage.get(model_name, {})
        total_tokens = model_usage.get("total_tokens", 0)
        budget = model_cfg.monthly_token_budget
        result.append({
            "model": model_name,
            "budget": budget,
            "used": total_tokens,
            "remaining": max(0, budget - total_tokens) if budget > 0 else -1,
            "unlimited": budget == 0,
            "requests": model_usage.get("requests", 0),
            "prompt_tokens": model_usage.get("prompt_tokens", 0),
            "completion_tokens": model_usage.get("completion_tokens", 0),
        })
    return result


@router.get("/api/models/{model}/limits")
async def api_model_limits(model: str, authorization: str | None = Header(None)):
    """Get per-model rate limits for all fallbacks."""
    verify_master_key(authorization)
    model_cfg = config.models.get(model)
    if not model_cfg:
        raise HTTPException(404, f"Model '{model}' not found")

    limits = []
    for fb in model_cfg.fallbacks:
        provider = config.providers.get(fb.provider)
        active_key = provider.api_key if provider and provider.api_key else ""
        key_usage = per_key_rate_tracker.get_usage(fb.provider, fb.model, active_key)
        limits.append({
            "provider": fb.provider,
            "model": fb.model,
            "rpm_limit": fb.rpm_limit,
            "rpd_limit": fb.rpd_limit,
            "tpm_limit": fb.tpm_limit,
            "tpd_limit": fb.tpd_limit,
            "current_usage": key_usage,
        })
    return {"model": model, "limits": limits}


@router.get("/api/models/resolve")
async def resolve_model(name: str, authorization: str | None = Header(None)):
    """Resolve a model name through aliases and equivalence mapping."""
    verify_master_key(authorization)
    if not name:
        raise HTTPException(400, "Missing 'name' query parameter")
    from routes.state import smart_router
    result = smart_router.resolve(name)
    model_cfg = config.models.get(result.resolved_name)
    capabilities = None
    if model_cfg:
        capabilities = {
            "supports_tools": model_cfg.capabilities.supports_tools,
            "supports_vision": model_cfg.capabilities.supports_vision,
            "supports_streaming": model_cfg.capabilities.supports_streaming,
        }
    return {
        "original_name": result.original_name,
        "resolved_name": result.resolved_name,
        "alias_used": result.alias_used,
        "substitution": result.substitution,
        "available": result.resolved_name in config.models,
        "capabilities": capabilities,
    }
