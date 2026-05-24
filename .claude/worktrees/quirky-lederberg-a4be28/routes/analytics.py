"""Analytics endpoints — usage stats, per-model, per-provider, timeline, errors."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header

from request_db import request_db
from routes.state import router, usage_tracker, verify_master_key

router = APIRouter()

VALID_RANGES = {"24h", "7d", "30d"}


@router.get("/api/analytics")
async def api_analytics(authorization: str | None = Header(None)):
    """Usage analytics with aggregated model/provider stats."""
    verify_master_key(authorization)
    stats = usage_tracker.get_stats()
    daily = stats.get("today", {})
    week = stats.get("week", {})
    all_time = stats.get("all_time", {})

    daily_data = usage_tracker._data.get("daily", {})
    model_totals: dict[str, dict[str, int]] = {}
    provider_totals: dict[str, dict[str, int]] = {}
    provider_success_map: dict[str, dict[str, int]] = {}

    for day_key, day_data in daily_data.items():
        for model, mdata in day_data.get("by_model", {}).items():
            if model not in model_totals:
                model_totals[model] = {"requests": 0, "total_tokens": 0}
            model_totals[model]["requests"] += mdata.get("requests", 0)
            model_totals[model]["total_tokens"] += mdata.get("total_tokens", 0)

        for provider, pdata in day_data.get("by_provider", {}).items():
            if provider not in provider_totals:
                provider_totals[provider] = {"requests": 0, "total_tokens": 0}
            provider_totals[provider]["requests"] += pdata.get("requests", 0)
            provider_totals[provider]["total_tokens"] += pdata.get("total_tokens", 0)

    logs = router.get_logs(200)
    for log_entry in logs:
        p = log_entry.get("provider", "unknown")
        if p not in provider_success_map:
            provider_success_map[p] = {"success": 0, "total": 0}
        provider_success_map[p]["total"] += 1
        if log_entry.get("success"):
            provider_success_map[p]["success"] += 1

    model_latency: dict[str, list[float]] = {}
    for log_entry in logs:
        m = log_entry.get("model", "")
        if m not in model_latency:
            model_latency[m] = []
        model_latency[m].append(log_entry.get("latency_ms", 0))
    avg_latency = {
        m: round(sum(lats) / len(lats), 1)
        for m, lats in model_latency.items() if lats
    }

    top_models = sorted(model_totals.items(), key=lambda x: x[1]["requests"], reverse=True)[:10]
    top_providers = sorted(provider_totals.items(), key=lambda x: x[1]["requests"], reverse=True)

    gpt4_input = 0.03 / 1000
    gpt4_output = 0.06 / 1000

    def calc_savings(data: dict) -> float:
        inp = data.get("prompt_tokens", 0)
        out = data.get("completion_tokens", 0)
        return round(inp * gpt4_input + out * gpt4_output, 4)

    return {
        "summary": {
            "total_requests": all_time.get("requests", 0),
            "total_tokens": all_time.get("total_tokens", 0),
            "total_prompt_tokens": all_time.get("prompt_tokens", 0),
            "total_completion_tokens": all_time.get("completion_tokens", 0),
            "today_requests": daily.get("requests", 0),
            "today_tokens": daily.get("total_tokens", 0),
            "week_requests": week.get("requests", 0),
            "week_tokens": week.get("total_tokens", 0),
        },
        "savings": {
            "today_usd": calc_savings(daily),
            "week_usd": calc_savings(week),
            "all_time_usd": calc_savings(all_time),
        },
        "top_models": [
            {"model": m, "requests": d["requests"], "tokens": d["total_tokens"]}
            for m, d in top_models
        ],
        "providers": [
            {
                "provider": p,
                "requests": d["requests"],
                "tokens": d["total_tokens"],
                "success_rate": round(
                    provider_success_map.get(p, {}).get("success", 0) /
                    max(provider_success_map.get(p, {}).get("total", 1), 1) * 100, 1
                ),
            }
            for p, d in top_providers
        ],
        "avg_latency_per_model": avg_latency,
        "daily_history": [
            {
                "date": dk,
                "requests": dd.get("requests", 0),
                "tokens": dd.get("total_tokens", 0),
            }
            for dk, dd in sorted(daily_data.items(), reverse=True)[:30]
        ],
    }


# ── Rich Analytics (SQLite-backed) ───────────────────────────────────────────


@router.get("/api/analytics/summary")
async def analytics_summary(
    range: str = "7d", authorization: str | None = Header(None),
):
    verify_master_key(authorization)
    r = range if range in VALID_RANGES else "7d"
    return request_db.get_summary(r)


@router.get("/api/analytics/by-model")
async def analytics_by_model(
    range: str = "7d", authorization: str | None = Header(None),
):
    verify_master_key(authorization)
    r = range if range in VALID_RANGES else "7d"
    return request_db.get_by_model(r)


@router.get("/api/analytics/by-provider")
async def analytics_by_provider(
    range: str = "7d", authorization: str | None = Header(None),
):
    verify_master_key(authorization)
    r = range if range in VALID_RANGES else "7d"
    return request_db.get_by_provider(r)


@router.get("/api/analytics/timeline")
async def analytics_timeline(
    range: str = "7d", interval: str = "day",
    authorization: str | None = Header(None),
):
    verify_master_key(authorization)
    r = range if range in VALID_RANGES else "7d"
    iv = interval if interval in ("hour", "day") else "day"
    return request_db.get_timeline(r, iv)


@router.get("/api/analytics/errors")
async def analytics_errors(
    range: str = "7d", authorization: str | None = Header(None),
):
    verify_master_key(authorization)
    r = range if range in VALID_RANGES else "7d"
    return request_db.get_errors(r)
