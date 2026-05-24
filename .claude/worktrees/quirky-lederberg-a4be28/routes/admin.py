"""Admin & configuration endpoints — status, discovery, sync, config export, benchmarks, playground."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from config import discover_models
from format_translator import detect_format, translate_to_openai
from provider_guides import get_all_guides, get_guide
from router import AllRateLimitedError
from routes.state import (
    benchmark_runner, client, config, health_checker,
    key_manager, quota_tracker, rate_limiter, request_queue, response_cache,
    router, smart_default, smart_router, usage_tracker, verify_master_key,
    combo_manager, gateway_auth, oauth_manager, sticky_sessions,
)
from token_compressor import compress_messages
from tool_call_translator import prepare_tools_for_provider

logger = logging.getLogger(__name__)

router = APIRouter()

TOP_RECOMMENDED_MODELS = [
    "nemotron-super-120b", "llama-3.3-70b", "deepseek-r1",
    "gemma-4-31b", "qwen3-coder", "mistral-large",
    "gpt-oss-120b", "hermes-3-405b", "minimax-m2.5", "qwen3-next-80b",
]


def _get_base_url() -> str:
    host = config.host if config.host != "0.0.0.0" else "localhost"
    return f"http://{host}:{config.port}/v1"


# ── System status ─────────────────────────────────────────────────────────────


@router.get("/api/status")
async def api_status():
    from rate_tracker import per_key_rate_tracker
    providers_info = {}
    for name, p in config.providers.items():
        providers_info[name] = {
            "has_key": bool(p.api_keys),
            "total_keys": p.total_keys,
            "active_key_index": p.active_key_index,
            "base_url": p.base_url,
        }
    return {
        "models": router.get_model_status(),
        "rate_limits": rate_limiter.get_all_status(),
        "health": health_checker.get_all_health(),
        "logs": router.get_logs(50),
        "usage": usage_tracker.get_stats(),
        "cache": response_cache.stats(),
        "queue": request_queue.stats(),
        "providers": providers_info,
        "quotas": quota_tracker.get_dashboard_summary(),
        "combos": combo_manager.list_combos(),
        "oauth": oauth_manager.list_connections(),
        "per_key_rates": per_key_rate_tracker.get_all_usage(),
        "sessions": sticky_sessions.get_stats(),
        "gateway_auth": gateway_auth.get_stats() if gateway_auth else {"total_keys": 0},
    }


# ── Dashboard ────────────────────────────────────────────────────────────────


@router.get("/", response_class=HTMLResponse)
async def dashboard():
    html = Path("templates/dashboard.html").read_text()
    return HTMLResponse(content=html)


# ── Model discovery ──────────────────────────────────────────────────────────


@router.post("/api/discover")
async def api_discover(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    if not client:
        raise HTTPException(503, "Server not ready")
    discovered = await discover_models(client, config)
    return {
        "discovered_count": len(discovered),
        "discovered_models": list(discovered.keys()),
        "total_models": len(config.models),
    }


# ── Sync providers ───────────────────────────────────────────────────────────


@router.post("/api/sync-providers")
async def api_sync_providers(authorization: str | None = Header(None)):
    """Sync providers from awesome-free-llm-apis upstream."""
    verify_master_key(authorization)
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "sync_providers.py")],
            capture_output=True, text=True, timeout=60,
        )
        from config import load_config
        from routes import state
        state.config = load_config()
        total = len(state.config.models)
        return {
            "ok": True,
            "output": result.stdout[-500:] if result.stdout else "",
            "new_models": 0,
            "providers": len(state.config.providers),
            "total_models": total,
        }
    except Exception as e:
        raise HTTPException(500, f"Sync failed: {e}")


@router.get("/api/auto-update")
async def api_auto_update(authorization: str | None = Header(None)):
    """Trigger auto-update: re-discover models from all providers."""
    verify_master_key(authorization)
    if not client:
        raise HTTPException(503, "Server not ready")

    old_count = len(config.models)
    discovered = await discover_models(client, config)
    new_count = len(config.models)
    added = new_count - old_count

    return {
        "ok": True,
        "previous_model_count": old_count,
        "current_model_count": new_count,
        "new_models_discovered": added,
        "discovered_models": list(discovered.keys())[:20] if discovered else [],
    }


# ── Connection info & Provider guides ────────────────────────────────────────


@router.get("/api/connection-info")
async def api_connection_info():
    base_url = _get_base_url()
    master_key = config.master_key or ""
    masked = ""
    if master_key:
        masked = ("*" * max(0, len(master_key) - 4)) + master_key[-4:]
    else:
        masked = "(not set)"
    available_top = [m for m in TOP_RECOMMENDED_MODELS if m in config.models][:10]
    guides = get_all_guides()
    provider_info = []
    for name, guide in guides.items():
        prov = config.providers.get(name)
        has_key = bool(prov and prov.api_keys) if prov else False
        provider_info.append({
            "id": name,
            "name": guide["name"],
            "has_key": has_key,
            "sign_in_url": guide["sign_in_url"],
            "env_key": guide["env_key"],
            "rate_limit": guide["rate_limit"],
            "notes": guide["notes"],
        })
    return {
        "base_url": base_url,
        "master_key": master_key,
        "master_key_masked": masked,
        "model_count": len(config.models),
        "provider_count": sum(1 for p in config.providers.values() if p.api_key),
        "top_models": available_top,
        "providers": provider_info,
    }


@router.get("/api/provider-guide/{provider_name}")
async def api_provider_guide(provider_name: str):
    guide = get_guide(provider_name)
    if not guide:
        raise HTTPException(404, f"No guide found for provider: {provider_name}")
    prov = config.providers.get(provider_name)
    has_key = bool(prov and prov.api_keys) if prov else False
    return {**guide, "has_key": has_key}


# ── Config exports ───────────────────────────────────────────────────────────


@router.get("/api/config/openclaw")
async def config_openclaw(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    base_url = _get_base_url()
    available_top = [m for m in TOP_RECOMMENDED_MODELS if m in config.models][:10]
    return {
        "api_key": config.master_key or "",
        "base_url": base_url,
        "default_model": available_top[0] if available_top else "",
        "models": available_top,
    }


@router.get("/api/config/hermes")
async def config_hermes(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    base_url = _get_base_url()
    available_top = [m for m in TOP_RECOMMENDED_MODELS if m in config.models][:10]
    return {
        "openai_api_key": config.master_key or "",
        "openai_base_url": base_url,
        "model": available_top[0] if available_top else "",
        "available_models": available_top,
    }


@router.get("/api/config/env")
async def config_env(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    base_url = _get_base_url()
    lines = [
        f"OPENAI_API_KEY={config.master_key or ''}",
        f"OPENAI_BASE_URL={base_url}",
        "DEFAULT_MODEL=nemotron-super-120b",
    ]
    return {"env_string": "\n".join(lines)}


@router.get("/api/config/export")
async def api_config_export(tool: str = "", authorization: str | None = Header(None)):
    verify_master_key(authorization)
    if not tool:
        raise HTTPException(400, "Missing 'tool' query parameter")

    base_url = _get_base_url()
    api_key = config.master_key or ""
    top_models = [m for m in TOP_RECOMMENDED_MODELS if m in config.models][:10]
    default_model = top_models[0] if top_models else "llama-3.3-70b"

    configs: dict[str, dict[str, Any]] = {
        "openclaw": {
            "name": "OpenClaw",
            "config": {
                "api_key": api_key, "base_url": base_url,
                "default_model": default_model, "models": top_models,
            },
        },
        "cursor": {
            "name": "Cursor",
            "config": {
                "apiKey": api_key, "apiBase": base_url, "model": default_model,
            },
        },
        "librechat": {
            "name": "LibreChat",
            "config": {
                "endpoints": {
                    "custom": [{
                        "name": "Free LLM Gateway",
                        "apiKey": api_key,
                        "baseURL": base_url,
                        "models": {"default": [default_model], "fetch": True},
                    }]
                }
            },
        },
        "open-webui": {
            "name": "Open WebUI",
            "config": {
                "OPENAI_API_BASE_URL": base_url,
                "OPENAI_API_KEY": api_key,
                "model": default_model,
            },
        },
        "continue-dev": {
            "name": "Continue.dev",
            "config": {
                "models": [{
                    "title": "Free LLM Gateway",
                    "provider": "openai",
                    "model": default_model,
                    "apiBase": base_url,
                    "apiKey": api_key,
                }],
            },
        },
        "jan": {
            "name": "Jan",
            "config": {"api_key": api_key, "base_url": base_url, "model": default_model},
        },
        "litellm": {
            "name": "LiteLLM",
            "config": {
                "model_list": [
                    {
                        "model_name": m,
                        "litellm_params": {
                            "model": f"openai/{m}",
                            "api_base": base_url,
                            "api_key": api_key,
                        },
                    }
                    for m in top_models[:5]
                ],
            },
        },
        "generic-openai": {
            "name": "Generic OpenAI SDK",
            "config": {
                "api_key": api_key, "base_url": base_url,
                "default_model": default_model, "models": top_models,
                "env_vars": {"OPENAI_API_KEY": api_key, "OPENAI_BASE_URL": base_url},
            },
        },
    }

    if tool not in configs:
        raise HTTPException(400, f"Unknown tool: {tool}. Supported: {', '.join(configs.keys())}")

    return {"tool": tool, **configs[tool]}


@router.get("/api/config/export/tools")
async def api_config_export_tools():
    """List all supported export tool names."""
    return {
        "tools": [
            {"id": "openclaw", "name": "OpenClaw"},
            {"id": "cursor", "name": "Cursor"},
            {"id": "librechat", "name": "LibreChat"},
            {"id": "open-webui", "name": "Open WebUI"},
            {"id": "continue-dev", "name": "Continue.dev"},
            {"id": "jan", "name": "Jan"},
            {"id": "litellm", "name": "LiteLLM"},
            {"id": "generic-openai", "name": "Generic OpenAI SDK"},
        ]
    }


# ── Smart Default API ────────────────────────────────────────────────────────


@router.get("/api/smart-default")
async def api_smart_default(task: str = "chat", authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return smart_default.get_default(task)


@router.get("/api/smart-default/all")
async def api_smart_default_all(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return smart_default.get_all_defaults()


# ── Benchmark API ────────────────────────────────────────────────────────────


@router.get("/api/benchmarks")
async def api_benchmarks():
    return benchmark_runner.get_results()


@router.get("/api/benchmarks/run")
async def api_benchmarks_run(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    if benchmark_runner.is_running():
        return {"status": "already_running", "message": "Benchmarks already in progress"}
    if not client:
        raise HTTPException(503, "Server not ready")
    asyncio.create_task(_async_benchmark_run())
    return {"status": "started", "message": "Benchmark run started"}


async def _async_benchmark_run() -> None:
    from routes import state
    try:
        result = await benchmark_runner.run_all(client)
        state.benchmark_results = result
        smart_default.update_benchmarks(result)
    except Exception as e:
        logger.warning("Benchmark run failed: %s", e)


# ── RTK Token Compression API ────────────────────────────────────────────────


@router.post("/api/compress")
async def api_compress(request: Request, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    body = await request.json()
    messages = body.get("messages", [])
    max_context = body.get("max_context", 20)
    if not messages:
        raise HTTPException(400, "Missing 'messages' field")
    result = compress_messages(messages, enabled=True, max_context=max_context)
    return {
        "original_tokens": result.original_count,
        "compressed_tokens": result.compressed_count,
        "saved_percent": result.saved_percent,
        "strategies_applied": result.strategies_applied,
        "messages": result.messages,
    }


# ── Format Translation API ───────────────────────────────────────────────────


@router.post("/api/translate")
async def api_translate(request: Request, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    body = await request.json()
    source = body.get("source_format") or detect_format(body.get("body", {}))
    target = body.get("target_format", "openai")
    request_body = body.get("body", {})
    if not request_body:
        raise HTTPException(400, "Missing 'body' field")
    from format_translator import translate_request
    translated = translate_request(request_body, target)
    return {
        "source_format": source,
        "target_format": target,
        "translated": translated,
    }


@router.get("/api/formats")
async def api_formats():
    return {
        "formats": [
            {"id": "openai", "name": "OpenAI", "description": "Standard OpenAI chat completions"},
            {"id": "claude", "name": "Anthropic Claude", "description": "Claude Messages API"},
            {"id": "gemini", "name": "Google Gemini", "description": "Gemini generateContent API"},
            {"id": "vertex", "name": "Google Vertex AI", "description": "Vertex AI prediction API"},
        ],
    }


# ── Playground API ───────────────────────────────────────────────────────────


@router.post("/api/playground")
async def api_playground(request: Request, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    body = await request.json()
    model = body.get("model", "")
    stream = body.get("stream", False)

    if not model:
        model = "llama-3.3-70b"
        body["model"] = model

    source_format = detect_format(body)
    if source_format != "openai":
        body = translate_to_openai(body, source_format)

    resolved = smart_router.resolve(model)
    model = resolved.resolved_name

    fallback_attempts: list[str] = []
    try:
        result, provider, provider_model = await router.route_request(model, body, client)
    except AllRateLimitedError:
        raise HTTPException(429, "All providers rate-limited. Try again later.")
    except Exception as e:
        raise HTTPException(502, f"Request failed: {e}")

    fallback_attempts.append(f"{provider}/{provider_model}")

    if stream and hasattr(result, "__aiter__"):
        from routes.chat import safe_stream
        return StreamingResponse(
            safe_stream(result, provider, provider_model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Routed-Via": f"{provider}/{provider_model}",
                "X-Fallback-Attempts": str(len(fallback_attempts)),
            },
        )

    response_data = result if isinstance(result, dict) else {"content": str(result)}
    if isinstance(response_data, dict):
        response_data.setdefault("model", provider_model)

    return {
        "response": response_data,
        "metadata": {
            "model_requested": body.get("model", ""),
            "model_resolved": model,
            "provider": provider,
            "provider_model": provider_model,
            "source_format": source_format,
            "routed_via": f"{provider}/{provider_model}",
            "fallback_attempts": fallback_attempts,
        },
    }


# ── Tool Calling Translation API ────────────────────────────────────────────


@router.post("/api/translate-tools")
async def api_translate_tools(request: Request, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    body = await request.json()
    tools = body.get("tools", [])
    target_provider = body.get("target_provider", "openai")
    if not tools:
        raise HTTPException(400, "Missing 'tools' field")
    translated = prepare_tools_for_provider(tools, target_provider)
    return {
        "source_format": "openai",
        "target_provider": target_provider,
        "translated": translated,
    }
