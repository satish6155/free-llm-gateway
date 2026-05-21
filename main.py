"""Free LLM Gateway — unified OpenAI-compatible API server."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cache import ResponseCache, response_cache
import sys
from pathlib import Path

from smart_router import SmartRouter
from config import load_config, AppConfig, discover_models, PROVIDER_DEFS
from health import HealthChecker
from key_manager import KeyManager
from providers import has_tool_calling
from request_queue import RequestQueue, request_queue, BLOCKING_QUEUE
from rate_limiter import RateLimiter
from router import Router, AllRateLimitedError
from smart_router import SmartRouter
from smart_default import SmartDefault
from tracking import UsageTracker, usage_tracker
from provider_guides import get_all_guides, get_guide
from benchmark import BenchmarkRunner
from token_compressor import compress_messages
from format_translator import detect_format, translate_to_openai, translate_response, FormatType
from custom_combos import combo_manager
from quota_tracker import quota_tracker
from request_db import request_db
from oauth_manager import oauth_manager
from rate_tracker import per_key_rate_tracker, PROVIDER_FREE_LIMITS
from sticky_sessions import sticky_sessions
from gateway_auth import gateway_auth, GatewayAuthManager
from tool_call_translator import prepare_tools_for_provider
from key_encryptor import EncryptedKeyStore
from pydantic import BaseModel, Field, field_validator, ValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Globals ──────────────────────────────────────────────────────────────────
config: AppConfig = load_config()
rate_limiter = RateLimiter()
health_checker = HealthChecker()
router = Router(config, rate_limiter, health_checker)
templates = Jinja2Templates(directory="templates")
key_manager = KeyManager(config.master_key or "default-key")
smart_router = SmartRouter(config.models)
benchmark_runner = BenchmarkRunner(config)
benchmark_results = benchmark_runner.get_results()
# Smart router for model aliases
smart_default = SmartDefault(config.models, benchmark_results)
# New feature modules
encrypted_key_store = EncryptedKeyStore(config.master_key or "default-key")
if config.master_key:
    import gateway_auth as _gw_mod
    _gw_mod.gateway_auth = GatewayAuthManager(encryption_key=config.master_key)
    gateway_auth = _gw_mod.gateway_auth
else:
    gateway_auth = None

# Shared httpx client (reused across requests for connection pooling)
_client: httpx.AsyncClient | None = None


def _sync_keys_to_config() -> None:
    """Push key_manager keys into config.providers so the router picks them up."""
    for name, prov in config.providers.items():
        km_keys = key_manager.get_keys(name)
        if km_keys:
            prov.api_keys = km_keys


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(http2=True, follow_redirects=True)

    # Auto-discover models from providers
    try:
        discovered = await discover_models(_client, config)
        if discovered:
            logger.info(
                "Discovered %d models: %s",
                len(discovered),
                ", ".join(list(discovered.keys())[:10]),
            )
    except Exception as e:
        logger.warning("Model discovery failed: %s", e)

    # Start background health checks
    health_checker.start(_client, config.providers)
    # Run an initial health check immediately
    await health_checker.check_all(_client, config.providers)

    # Sync key_manager keys into config providers
    _sync_keys_to_config()

    # Merge custom combos into model config
    combo_manager.merge_into_config(config.models)

    # Initialize per-key rate limits from known free tier limits
    for prov_name, model_limits in PROVIDER_FREE_LIMITS.items():
        for model_key, limits in model_limits.items():
            provider = config.providers.get(prov_name)
            if provider and provider.api_key:
                per_key_rate_tracker.set_limits(
                    prov_name, model_key, provider.api_key,
                    rpm=limits.rpm, rpd=limits.rpd,
                    tpm=limits.tpm, tpd=limits.tpd,
                )

    # Start sticky session cleanup loop
    sticky_sessions.start_cleanup_loop()

    # Initialize SQLite request log
    request_db.init()

    # Startup complete

    # Start request queue workers
    request_queue.set_router(router)
    await request_queue.start_workers(num_workers=3)

    logger.info(
        "Gateway started — %d models, %d providers configured",
        len(config.models),
        sum(1 for p in config.providers.values() if p.api_key),
    )
    yield

    # Shutdown
    await request_queue.stop_workers()
    await sticky_sessions.stop_cleanup()
    usage_tracker.flush()
    request_db.close()
    await health_checker.stop()
    if _client:
        await _client.aclose()


app = FastAPI(title="Free LLM Gateway", version="1.0.0", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Liveness probe ─────────────────────────────────────────────────────────────
@app.get("/api/ping")
async def ping():
    """Lightweight liveness probe for health checks and uptime monitoring."""
    return {"status": "ok", "timestamp": time.time()}


# ── Request validation (Pydantic models) ──────────────────────────────────────

class ChatCompletionRequest(BaseModel):
    """Validates incoming chat completion requests before routing to providers.

    Modeled after the OpenAI /v1/chat/completions spec. Invalid requests
    are rejected with 400 errors before they ever hit upstream providers.
    """
    messages: list[dict[str, Any]] = Field(..., min_length=1)
    model: str | None = None
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    stop: str | list[str] | None = None
    n: int | None = Field(default=None, gt=0)
    conversation_id: str | None = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid_roles = {"system", "user", "assistant", "tool"}
        for i, msg in enumerate(v):
            role = msg.get("role")
            if role not in valid_roles:
                raise ValueError(
                    f"messages[{i}].role must be one of {valid_roles}, got '{role}'"
                )
            # Assistant messages need content or tool_calls
            if role == "assistant":
                has_content = (
                    isinstance(msg.get("content"), str) and len(msg["content"]) > 0
                )
                has_tool_calls = bool(msg.get("tool_calls"))
                if not has_content and not has_tool_calls:
                    raise ValueError(
                        f"messages[{i}] (assistant) must have non-empty content or tool_calls"
                    )
            # Tool messages need tool_call_id
            if role == "tool" and not msg.get("tool_call_id"):
                raise ValueError(
                    f"messages[{i}] (tool) must include 'tool_call_id'"
                )
        return v


async def safe_stream(
    stream: Any, provider: str, provider_model: str,
) -> AsyncIterator[bytes]:
    """Wrap a provider stream to catch mid-stream errors.

    If the upstream connection breaks mid-stream, we emit an SSE error
    frame + [DONE] so the client sees a clean signal instead of a
    silently truncated stream.
    """
    import json as _json

    try:
        async for chunk in stream:
            yield chunk
    except Exception as exc:
        logger.error(
            "Mid-stream error from %s/%s: %s",
            provider, provider_model, exc,
        )
        # Emit an SSE error frame so the client knows what happened
        error_payload = _json.dumps({
            "error": {
                "message": f"Provider error ({provider}/{provider_model}): stream interrupted",
                "type": "stream_error",
            },
        })
        yield f"data: {error_payload}\n\n".encode()
        yield b"data: [DONE]\n\n"


# ── Auth middleware ───────────────────────────────────────────────────────────
def verify_master_key(authorization: str | None) -> None:
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

    # Fall back to master key — use constant-time comparison to prevent
    # timing attacks that could recover the key byte-by-byte.
    import hmac
    if config.master_key and hmac.compare_digest(token, config.master_key):
        return

    raise HTTPException(401, "Invalid API key")


# ── Chat completions ─────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    raw_body = await request.json()

    # ── Input validation: reject malformed requests early ──
    try:
        validated = ChatCompletionRequest(**raw_body)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Invalid request: {e.error_count()} error(s)",
                "type": "invalid_request_error",
                "errors": [
                    {"field": str(loc), "message": err["msg"]}
                    for loc, err in e.errors()
                ],
            },
        )

    body = raw_body  # keep full body for downstream compatibility
    model = validated.model or ""
    stream = validated.stream

    # ── Format translation: auto-detect and normalize to OpenAI ──
    source_format = detect_format(body)
    if source_format != "openai":
        logger.info("Format translation: %s -> openai", source_format)
        body = translate_to_openai(body, source_format)

    if not model:
        model = body.get("model", "")
        if not model:
            raise HTTPException(400, "Missing 'model' field")

    # ── RTK Token Compression ──
    messages = body.get("messages", [])
    if messages and len(messages) >= 6:
        rtk_enabled = os.environ.get("RTK_COMPRESSION", "true").lower() != "false"
        compression = compress_messages(messages, enabled=rtk_enabled)
        if compression.strategies_applied:
            body["messages"] = compression.messages
            logger.info(
                "RTK: %d -> %d tokens (%.1f%% saved)",
                compression.original_count, compression.compressed_count,
                compression.saved_percent,
            )

    # ── Smart routing: resolve model name via aliases and equivalence ──
    resolved = smart_router.resolve(model)
    if resolved.substitution:
        logger.info("Smart routing: %s", resolved.substitution)
    model = resolved.resolved_name

    # ── Tool calling auto-routing ──
    if has_tool_calling(body):
        model_cfg = config.models.get(model)
        if model_cfg and not model_cfg.capabilities.supports_tools:
            alt = smart_router.find_model_with_capability("supports_tools")
            if alt:
                logger.info(
                    "Auto-routing: '%s' doesn't support tools -> '%s'",
                    model, alt,
                )
                model = alt
            else:
                raise HTTPException(
                    400,
                    f"Model '{model}' does not support tool calling and no alternative found",
                )

    # ── Cache check (non-streaming only) ──
    cache_headers = {"X-Cache": "MISS"}
    if not stream:
        cache_key = ResponseCache.make_key(
            model, body.get("messages", []), body.get("temperature")
        )
        cached, hit = response_cache.get(cache_key)
        if hit:
            cache_headers["X-Cache"] = "HIT"
            cached.setdefault("model", model)
            return JSONResponse(content=cached, headers=cache_headers)

    # ── Sticky sessions: prefer same provider for conversation ──
    messages = body.get("messages", [])
    conversation_id = body.get("conversation_id") or sticky_sessions.extract_conversation_id(messages)
    preferred_provider, preferred_model = None, None
    if conversation_id:
        preferred_provider, preferred_model = sticky_sessions.get(conversation_id)

    # ── Tool calling translation for Gemini ──
    tools = body.get("tools")
    if tools:
        # Store original tools for response compatibility
        body["_original_tools"] = tools

    # ── Route request ──
    fallback_attempts: list[str] = []
    try:
        result, provider, provider_model = await router.route_request(model, body, _client)
    except AllRateLimitedError:
        # All providers rate-limited → queue the request
        return await _handle_rate_limited(model, body, stream)

    # Track fallback attempts for routing headers
    fallback_attempts.append(f"{provider}/{provider_model}")

    # ── Record sticky session ──
    if conversation_id:
        sticky_sessions.set(conversation_id, provider, provider_model)

    # ── Per-key rate tracking ──
    prov_cfg = config.providers.get(provider)
    if prov_cfg and prov_cfg.api_key:
        total_tokens = 0
        if isinstance(result, dict):
            usage = result.get("usage")
            if usage and isinstance(usage, dict):
                total_tokens = usage.get("total_tokens", 0) or 0
        per_key_rate_tracker.record_request(provider, provider_model, prov_cfg.api_key, tokens=total_tokens)

    # ── Record token usage ──
    if isinstance(result, dict):
        usage = result.get("usage")
        if usage and isinstance(usage, dict):
            usage_tracker.record(
                model=model,
                provider=provider,
                prompt_tokens=usage.get("prompt_tokens", 0) or 0,
                completion_tokens=usage.get("completion_tokens", 0) or 0,
                total_tokens=usage.get("total_tokens", 0) or 0,
            )
            # Quota tracking
            quota_tracker.record_request(provider, tokens=usage.get("total_tokens", 0) or 0)

        # ── Cache the response (non-streaming only) ──
        if not stream:
            response_cache.put(cache_key, result)

    # ── Return response ──
    routing_headers = {
        "X-Routed-Via": f"{provider}/{provider_model}",
        "X-Fallback-Attempts": str(len(fallback_attempts)),
        "X-Sticky-Session": "true" if conversation_id and preferred_provider == provider else "false",
    }

    if stream and hasattr(result, "__aiter__"):
        return StreamingResponse(
            safe_stream(result, provider, provider_model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Provider": provider,
                "X-Provider-Model": provider_model,
                "X-Cache": "BYPASS",
                "X-Source-Format": source_format,
                **routing_headers,
            },
        )

    # Non-streaming: ensure it's a dict
    if isinstance(result, dict):
        result.setdefault("model", provider_model)
    cache_headers["X-Provider"] = provider
    cache_headers["X-Provider-Model"] = provider_model
    cache_headers["X-Source-Format"] = source_format
    cache_headers.update(routing_headers)
    return JSONResponse(content=result, headers=cache_headers)


async def _handle_rate_limited(model: str, body: dict, stream: bool):
    """Handle the case where all providers are rate-limited."""
    if stream:
        raise HTTPException(429, "All providers rate-limited. Retry later.")

    try:
        req_id, wait_time, queued_req = await request_queue.enqueue(model, body)
    except RuntimeError:
        raise HTTPException(503, "Queue is full. All providers rate-limited. Retry later.")

    if BLOCKING_QUEUE and queued_req:
        # Block and wait for result
        try:
            result = await request_queue.get_result(req_id)
            return JSONResponse(
                content=result,
                headers={"X-Queued": "true", "X-Request-Id": req_id},
            )
        except Exception:
            raise HTTPException(504, "Queued request timed out")

    # Non-blocking: return 202 with polling info
    return JSONResponse(
        status_code=202,
        content={
            "id": req_id,
            "status": "queued",
            "estimated_wait_seconds": wait_time,
            "poll_url": f"/api/queue/{req_id}",
        },
        headers={"X-Request-Id": req_id},
    )


# ── Models list ──────────────────────────────────────────────────────────────
@app.get("/v1/models")
async def list_models(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    models = []
    for name, model_cfg in config.models.items():
        providers = [
            {"provider": fb.provider, "model": fb.model}
            for fb in model_cfg.fallbacks
        ]
        models.append({
            "id": name,
            "object": "model",
            "owned_by": "free-llm-gateway",
            "providers": providers,
            "context_window": model_cfg.context_window or None,
            "size_label": model_cfg.size_label or None,
            "capabilities": {
                "supports_tools": model_cfg.capabilities.supports_tools,
                "supports_vision": model_cfg.capabilities.supports_vision,
                "supports_streaming": model_cfg.capabilities.supports_streaming,
            },
        })
    return {"object": "list", "data": models}


# ── Embeddings (pass-through) ────────────────────────────────────────────────
@app.post("/v1/embeddings")
async def embeddings(request: Request, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    body = await request.json()
    model = body.get("model", "")

    fallbacks = router.get_fallbacks(model)
    if not fallbacks:
        raise HTTPException(400, f"Unknown model: {model}")

    for fb in fallbacks:
        provider = config.providers.get(fb.provider)
        if not provider or not provider.api_key:
            continue
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{provider.base_url}/embeddings"
        body["model"] = fb.model
        try:
            resp = await _client.post(url, headers=headers, json=body, timeout=60.0)
            if resp.status_code < 400:
                return resp.json()
        except Exception as e:
            logger.warning("Embedding provider %s failed: %s", fb.provider, e)
            continue

    raise HTTPException(502, "All embedding providers failed")


# ── Usage tracking API ───────────────────────────────────────────────────────
@app.get("/api/usage")
async def api_usage(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return usage_tracker.get_stats()


# ── Cache API ────────────────────────────────────────────────────────────────
@app.get("/api/cache")
async def api_cache_stats(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return response_cache.stats()


@app.delete("/api/cache")
async def api_cache_clear(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    cleared = response_cache.clear()
    return {"cleared": cleared}


@app.post("/api/cache/prune")
async def api_cache_prune(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    pruned = response_cache.prune_expired()
    return {"pruned": pruned}


# ── Queue API ────────────────────────────────────────────────────────────────
@app.get("/api/queue")
async def api_queue_stats(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return {
        **request_queue.stats(),
        "pending": request_queue.get_pending_list(),
    }


@app.get("/api/queue/{request_id}")
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


# ── Dashboard ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    from pathlib import Path
    html = Path("templates/dashboard.html").read_text()
    return HTMLResponse(content=html)


# ── Model resolution ─────────────────────────────────────────────────────────
@app.get("/api/models/resolve")
async def resolve_model(name: str, authorization: str | None = Header(None)):
    """Resolve a model name through aliases and equivalence mapping."""
    verify_master_key(authorization)
    if not name:
        raise HTTPException(400, "Missing 'name' query parameter")
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


@app.get("/api/status")
async def api_status():
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


# ── Health & Key Health API ──────────────────────────────────────────────────
@app.get("/api/health/keys")
async def api_health_keys(authorization: str | None = Header(None)):
    """Per-key health status for all providers."""
    verify_master_key(authorization)
    return health_checker.get_all_key_health()


@app.get("/api/health/keys/{provider}")
async def api_health_keys_provider(
    provider: str, authorization: str | None = Header(None),
):
    """Per-key health status for a specific provider."""
    verify_master_key(authorization)
    if provider not in config.providers:
        raise HTTPException(404, f"Provider '{provider}' not found")
    return health_checker.get_key_health(provider)


@app.post("/api/health/check")
async def api_health_check_now(authorization: str | None = Header(None)):
    """Trigger an immediate health check (providers + keys)."""
    verify_master_key(authorization)
    if not _client:
        raise HTTPException(503, "Server not ready")
    await health_checker.check_all(_client, config.providers)
    await health_checker.check_all_keys(_client, config.providers)
    return {
        "providers": health_checker.get_all_health(),
        "keys": health_checker.get_all_key_health(),
    }


# ── Model discovery ──────────────────────────────────────────────────────────
@app.post("/api/discover")
async def api_discover(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    if not _client:
        raise HTTPException(503, "Server not ready")
    discovered = await discover_models(_client, config)
    return {
        "discovered_count": len(discovered),
        "discovered_models": list(discovered.keys()),
        "total_models": len(config.models),
    }


# ── API Key Management ───────────────────────────────────────────────────────
@app.post("/api/keys")
async def add_key(request: Request):
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
        if _client:
            resp = await _client.get(test_url, headers=headers, timeout=15.0)
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


@app.delete("/api/keys/{provider}/{index}")
async def remove_key(provider: str, index: int):
    if not key_manager.remove_key(provider, index):
        raise HTTPException(404, "Key not found")
    _sync_keys_to_config()
    return {"ok": True}


@app.get("/api/keys")
async def list_keys():
    """List all keys: .env-sourced (read-only) + runtime-added (deletable)."""
    runtime_keys = key_manager.list_keys()
    # Build .env keys from config.providers
    env_keys: dict[str, list[dict[str, Any]]] = {}
    for name, prov in config.providers.items():
        if prov.api_keys:
            env_keys[name] = []
            for i, key in enumerate(prov.api_keys):
                masked = "****" + key[-4:] if len(key) > 4 else "****"
                # Check if this key also exists in runtime keys
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

    # Mark runtime keys with source
    all_keys: dict[str, list[dict[str, Any]]] = {}
    all_providers = set(list(runtime_keys.keys()) + list(env_keys.keys()))
    for pname in all_providers:
        entries = []
        # .env keys first
        for e in env_keys.get(pname, []):
            entries.append(e)
        # runtime keys
        for e in runtime_keys.get(pname, []):
            entry = {**e, "source": "runtime", "deletable": True}
            entries.append(entry)
        if entries:
            all_keys[pname] = entries
    return {"keys": all_keys}


@app.post("/api/keys/{provider}/validate")
async def validate_provider_keys(provider: str):
    """Validate all keys for a provider by testing each against the /models endpoint."""
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
            base_url = base_url.replace("{account_id}", os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""))
        test_url = f"{base_url}/models"
        headers = {"Authorization": f"Bearer " + api_key}
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/free-llm-gateway"
            headers["X-Title"] = "Free LLM Gateway"
        try:
            resp = await _client.get(test_url, headers=headers, timeout=15.0)
            valid = resp.status_code < 400
            key_manager.set_validated(provider, i, valid)
            results.append({"index": i, "valid": valid})
        except Exception as e:
            key_manager.set_validated(provider, i, False)
            results.append({"index": i, "valid": False, "error": str(e)[:200]})

    return {"provider": provider, "results": results}


@app.post("/api/keys/{provider}/{index}/validate")
async def validate_key(provider: str, index: int):
    keys = key_manager.get_keys(provider)
    if index < 0 or index >= len(keys):
        raise HTTPException(404, "Key not found")

    api_key = keys[index]
    prov_def = PROVIDER_DEFS.get(provider)
    if not prov_def:
        raise HTTPException(400, f"Unknown provider: {provider}")

    base_url = prov_def[1]
    if "{account_id}" in base_url:
        import os
        base_url = base_url.replace("{account_id}", os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""))

    # Try a lightweight models list request
    test_url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer " + api_key}
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/free-llm-gateway"
        headers["X-Title"] = "Free LLM Gateway"

    try:
        resp = await _client.get(test_url, headers=headers, timeout=15.0)
        valid = resp.status_code < 400
        key_manager.set_validated(provider, index, valid)
        if valid:
            return {"valid": True}
        else:
            return {"valid": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        key_manager.set_validated(provider, index, False)
        return {"valid": False, "error": str(e)[:200]}


@app.patch("/api/keys/{provider}/{index}/toggle")
async def toggle_key(provider: str, index: int, request: Request):
    """Enable or disable a specific API key by index.

    Disabled keys are added to the provider's disabled_keys list and
    skipped during routing. Re-enabling removes them from that list.
    """
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


# ── Connection info & Config export ──────────────────────────────────────────
TOP_RECOMMENDED_MODELS = [
    "nemotron-super-120b", "llama-3.3-70b", "deepseek-r1",
    "gemma-4-31b", "qwen3-coder", "mistral-large",
    "gpt-oss-120b", "hermes-3-405b", "minimax-m2.5", "qwen3-next-80b",
]


def _get_base_url() -> str:
    host = config.host if config.host != "0.0.0.0" else "localhost"
    return f"http://{host}:{config.port}/v1"


@app.post("/api/sync-providers")
async def api_sync_providers(authorization: str | None = Header(None)):
    """Sync providers from awesome-free-llm-apis upstream."""
    verify_master_key(authorization)
    import subprocess
    import json as _json
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "sync_providers.py")],
            capture_output=True, text=True, timeout=60,
        )
        # Reload config after sync
        global config
        config = load_config()
        # Count models
        total = len(config.models)
        return {
            "ok": True,
            "output": result.stdout[-500:] if result.stdout else "",
            "new_models": 0,  # sync_providers.py handles counting
            "providers": len(config.providers),
            "total_models": total,
        }
    except Exception as e:
        raise HTTPException(500, f"Sync failed: {e}")


@app.get("/api/auto-update")
async def api_auto_update(authorization: str | None = Header(None)):
    """Trigger auto-update: re-discover models from all providers."""
    verify_master_key(authorization)
    if not _client:
        raise HTTPException(503, "Server not ready")

    old_count = len(config.models)
    discovered = await discover_models(_client, config)
    new_count = len(config.models)
    added = new_count - old_count

    return {
        "ok": True,
        "previous_model_count": old_count,
        "current_model_count": new_count,
        "new_models_discovered": added,
        "discovered_models": list(discovered.keys())[:20] if discovered else [],
    }


@app.get("/api/connection-info")
async def api_connection_info():
    base_url = _get_base_url()
    master_key = config.master_key or ""
    masked = ""
    if master_key:
        masked = ("*" * max(0, len(master_key) - 4)) + master_key[-4:]
    else:
        masked = "(not set)"
    available_top = [m for m in TOP_RECOMMENDED_MODELS if m in config.models][:10]
    # Include provider guides with key status
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
        "provider_count": sum(1 for p in config.providers.values() if p.api_keys),
        "top_models": available_top,
        "providers": provider_info,
    }


@app.get("/api/provider-guide/{provider_name}")
async def api_provider_guide(provider_name: str):
    """Get detailed sign-up instructions for a provider."""
    guide = get_guide(provider_name)
    if not guide:
        raise HTTPException(404, f"No guide found for provider: {provider_name}")
    prov = config.providers.get(provider_name)
    has_key = bool(prov and prov.api_keys) if prov else False
    return {**guide, "has_key": has_key}


@app.get("/api/config/openclaw")
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


@app.get("/api/config/hermes")
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


@app.get("/api/config/env")
async def config_env(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    base_url = _get_base_url()
    lines = [
        f"OPENAI_API_KEY={config.master_key or ''}",
        f"OPENAI_BASE_URL={base_url}",
        "DEFAULT_MODEL=nemotron-super-120b",
    ]
    return {"env_string": "\n".join(lines)}


# ── Batch requests ────────────────────────────────────────────────────────────
BATCH_MAX_SIZE = int(os.environ.get("BATCH_MAX_SIZE", "10"))


# ── Smart Default API ────────────────────────────────────────────────────────
@app.get("/api/smart-default")
async def api_smart_default(task: str = "chat", authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return smart_default.get_default(task)


@app.get("/api/smart-default/all")
async def api_smart_default_all(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    return smart_default.get_all_defaults()


# ── Benchmark API ────────────────────────────────────────────────────────────
@app.get("/api/benchmarks")
async def api_benchmarks():
    return benchmark_runner.get_results()


@app.get("/api/benchmarks/run")
async def api_benchmarks_run(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    if benchmark_runner.is_running():
        return {"status": "already_running", "message": "Benchmarks already in progress"}
    if not _client:
        raise HTTPException(503, "Server not ready")
    asyncio.create_task(_async_benchmark_run())
    return {"status": "started", "message": "Benchmark run started"}


async def _async_benchmark_run() -> None:
    global benchmark_results
    try:
        # Update results progressively
        class LiveRunner:
            pass
        result = await benchmark_runner.run_all(_client)
        benchmark_results = result
        smart_default.update_benchmarks(result)
    except Exception as e:
        logger.warning("Benchmark run failed: %s", e)


# ── Analytics API ────────────────────────────────────────────────────────────
@app.get("/api/analytics")
async def api_analytics(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    stats = usage_tracker.get_stats()
    daily = stats.get("today", {})
    week = stats.get("week", {})
    all_time = stats.get("all_time", {})

    # Aggregate model and provider stats from all daily data
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

    # Success rates from router logs
    logs = router.get_logs(200)
    for log_entry in logs:
        p = log_entry.get("provider", "unknown")
        if p not in provider_success_map:
            provider_success_map[p] = {"success": 0, "total": 0}
        provider_success_map[p]["total"] += 1
        if log_entry.get("success"):
            provider_success_map[p]["success"] += 1

    # Average latency per model from logs
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

    # GPT-4 pricing for estimated savings
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


# ── Rich Analytics API (SQLite-backed) ───────────────────────────────────────

VALID_RANGES = {"24h", "7d", "30d"}


@app.get("/api/analytics/summary")
async def analytics_summary(
    range: str = "7d", authorization: str | None = Header(None),
):
    verify_master_key(authorization)
    r = range if range in VALID_RANGES else "7d"
    return request_db.get_summary(r)


@app.get("/api/analytics/by-model")
async def analytics_by_model(
    range: str = "7d", authorization: str | None = Header(None),
):
    verify_master_key(authorization)
    r = range if range in VALID_RANGES else "7d"
    return request_db.get_by_model(r)


@app.get("/api/analytics/by-provider")
async def analytics_by_provider(
    range: str = "7d", authorization: str | None = Header(None),
):
    verify_master_key(authorization)
    r = range if range in VALID_RANGES else "7d"
    return request_db.get_by_provider(r)


@app.get("/api/analytics/timeline")
async def analytics_timeline(
    range: str = "7d", interval: str = "day",
    authorization: str | None = Header(None),
):
    verify_master_key(authorization)
    r = range if range in VALID_RANGES else "7d"
    iv = interval if interval in ("hour", "day") else "day"
    return request_db.get_timeline(r, iv)


@app.get("/api/analytics/errors")
async def analytics_errors(
    range: str = "7d", authorization: str | None = Header(None),
):
    verify_master_key(authorization)
    r = range if range in VALID_RANGES else "7d"
    return request_db.get_errors(r)


# ── Key Health API ───────────────────────────────────────────────────────────
@app.post("/api/keys/validate-all")
async def api_validate_all_keys(authorization: str | None = Header(None)):
    verify_master_key(authorization)
    if not _client:
        raise HTTPException(503, "Server not ready")

    results = {}
    for name, prov_def in PROVIDER_DEFS.items():
        keys = key_manager.get_keys(name)
        # Also include .env keys
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
                resp = await _client.get(test_url, headers=headers, timeout=15.0)
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

    # Summary counts
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


# ── Fallback Chain API ───────────────────────────────────────────────────────

@app.get("/api/fallbacks")
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


@app.put("/api/fallbacks/{model}")
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

    # Build a lookup from the incoming order
    incoming = {
        (entry["provider"], entry["model"]): entry
        for entry in body
        if "provider" in entry and "model" in entry
    }

    # Update enabled state on existing fallbacks and reorder
    new_fallbacks: list = []
    for entry in body:
        provider = entry.get("provider", "")
        provider_model = entry.get("model", "")
        enabled = entry.get("enabled", True)
        # Find matching existing fallback or create new
        existing = next(
            (fb for fb in model_cfg.fallbacks
             if fb.provider == provider and fb.model == provider_model),
            None,
        )
        if existing:
            existing.enabled = enabled
            new_fallbacks.append(existing)
        else:
            from config import ModelFallback
            new_fallbacks.append(ModelFallback(
                provider=provider, model=provider_model, enabled=enabled,
            ))

    model_cfg.fallbacks = new_fallbacks
    return {"success": True, "model": model, "fallbacks": len(new_fallbacks)}


@app.post("/api/fallbacks/{model}/sort/{preset}")
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
        # Sort by ascending penalty (least penalized first)
        model_cfg.fallbacks.sort(
            key=lambda fb: router.penalty_tracker.get_penalty(fb.provider, fb.model),
        )
    elif preset == "health":
        # Sort: enabled first, then by health (healthy first)
        def _health_key(fb: Any) -> int:
            if not fb.enabled:
                return 1000
            if router.health_checker and not router.health_checker.is_available(fb.provider):
                return 500
            return router.penalty_tracker.get_penalty(fb.provider, fb.model)
        model_cfg.fallbacks.sort(key=_health_key)
    elif preset == "intelligence":
        # Sort by intelligence rank descending (smartest first)
        model_cfg.fallbacks.sort(
            key=lambda fb: -_get_fallback_rank(fb, "intelligence"),
        )
    elif preset == "speed":
        # Sort by speed rank descending (fastest first)
        model_cfg.fallbacks.sort(
            key=lambda fb: -_get_fallback_rank(fb, "speed"),
        )
    elif preset == "budget":
        # Sort by ascending token usage (least used first)
        usage = request_db.get_model_token_usage(30)
        def _budget_key(fb: Any) -> int:
            key = f"{fb.provider}:{fb.model}"
            return usage.get(key, {}).get("total_tokens", 0)
        model_cfg.fallbacks.sort(key=_budget_key)
    elif preset == "priority":
        # Reset to original order — no-op (already in original order)
        pass
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
    """Get intelligence or speed rank for a fallback's model.

    Looks up the rank from the ModelConfig that contains this fallback.
    Falls back to 0 (unknown) if not found.
    """
    for _model_name, model_cfg in config.models.items():
        for existing_fb in model_cfg.fallbacks:
            if existing_fb.provider == fb.provider and existing_fb.model == fb.model:
                if rank_type == "intelligence":
                    return model_cfg.intelligence_rank
                elif rank_type == "speed":
                    return model_cfg.speed_rank
    return 0


@app.get("/api/fallbacks/token-usage")
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


@app.get("/api/models/{model}/limits")
async def api_model_limits(model: str, authorization: str | None = Header(None)):
    """Get per-model rate limits for all fallbacks."""
    verify_master_key(authorization)
    model_cfg = config.models.get(model)
    if not model_cfg:
        raise HTTPException(404, f"Model '{model}' not found")

    limits = []
    for fb in model_cfg.fallbacks:
        provider = config.providers.get(fb.provider)
        active_key = provider.api_key if provider else ""
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


# ── Export Configs API ───────────────────────────────────────────────────────
@app.get("/api/config/export")
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
                "api_key": api_key,
                "base_url": base_url,
                "default_model": default_model,
                "models": top_models,
            },
        },
        "cursor": {
            "name": "Cursor",
            "config": {
                "apiKey": api_key,
                "apiBase": base_url,
                "model": default_model,
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
            "config": {
                "api_key": api_key,
                "base_url": base_url,
                "model": default_model,
            },
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
                "api_key": api_key,
                "base_url": base_url,
                "default_model": default_model,
                "models": top_models,
                "env_vars": {
                    "OPENAI_API_KEY": api_key,
                    "OPENAI_BASE_URL": base_url,
                },
            },
        },
    }

    if tool not in configs:
        raise HTTPException(400, f"Unknown tool: {tool}. Supported: {', '.join(configs.keys())}")

    return {"tool": tool, **configs[tool]}


@app.get("/api/config/export/tools")
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


# ── RTK Token Compression API ────────────────────────────────────────────────
@app.post("/api/compress")
async def api_compress(request: Request, authorization: str | None = Header(None)):
    """Compress a message list using RTK token compression."""
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
@app.post("/api/translate")
async def api_translate(request: Request, authorization: str | None = Header(None)):
    """Translate a request body between API formats."""
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


@app.get("/api/formats")
async def api_formats():
    """List supported API formats."""
    return {
        "formats": [
            {"id": "openai", "name": "OpenAI", "description": "Standard OpenAI chat completions"},
            {"id": "claude", "name": "Anthropic Claude", "description": "Claude Messages API"},
            {"id": "gemini", "name": "Google Gemini", "description": "Gemini generateContent API"},
            {"id": "vertex", "name": "Google Vertex AI", "description": "Vertex AI prediction API"},
        ],
    }


# ── Custom Combos API ────────────────────────────────────────────────────────
@app.get("/api/combos")
async def api_list_combos(authorization: str | None = Header(None)):
    """List all custom combos."""
    verify_master_key(authorization)
    return {"combos": combo_manager.list_combos()}


@app.post("/api/combos")
async def api_create_combo(request: Request, authorization: str | None = Header(None)):
    """Create a new custom combo."""
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
    # Merge into runtime config
    config.models[name] = combo.to_model_config()
    return {"ok": True, "combo": combo.name, "entries": len(combo.entries)}


@app.put("/api/combos/{name}")
async def api_update_combo(name: str, request: Request, authorization: str | None = Header(None)):
    """Update an existing combo."""
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


@app.delete("/api/combos/{name}")
async def api_delete_combo(name: str, authorization: str | None = Header(None)):
    """Delete a combo."""
    verify_master_key(authorization)
    if not combo_manager.delete_combo(name):
        raise HTTPException(404, f"Combo '{name}' not found")
    config.models.pop(name, None)
    return {"ok": True}


# ── Quota Tracking API ────────────────────────────────────────────────────────
@app.get("/api/quotas")
async def api_quotas(authorization: str | None = Header(None)):
    """Get quota status for all providers."""
    verify_master_key(authorization)
    return {
        "providers": quota_tracker.get_all_quotas(),
        "dashboard": quota_tracker.get_dashboard_summary(),
    }


@app.get("/api/quotas/{provider}")
async def api_quota_provider(provider: str, authorization: str | None = Header(None)):
    """Get quota status for a specific provider."""
    verify_master_key(authorization)
    quota = quota_tracker.get_quota(provider)
    if not quota:
        raise HTTPException(404, f"No quota data for provider: {provider}")
    return quota


@app.put("/api/quotas/{provider}")
async def api_set_quota_limits(
    provider: str,
    request: Request,
    authorization: str | None = Header(None),
):
    """Set custom quota limits for a provider."""
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
@app.get("/api/oauth/providers")
async def api_oauth_providers(authorization: str | None = Header(None)):
    """List available OAuth subscription providers and their connection status."""
    verify_master_key(authorization)
    return {"providers": oauth_manager.list_connections()}


@app.post("/api/oauth/authorize")
async def api_oauth_authorize(request: Request, authorization: str | None = Header(None)):
    """Start an OAuth flow for a subscription provider."""
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


@app.get("/api/oauth/callback")
async def api_oauth_callback(state: str, code: str):
    """Handle OAuth callback (redirected from provider)."""
    if not _client:
        raise HTTPException(503, "Server not ready")
    result = await oauth_manager.handle_callback(state, code, _client)
    if "error" in result:
        return HTMLResponse(content=f"<h3>OAuth Error</h3><p>{result['error']}</p>")
    return HTMLResponse(
        content=f"<h3>Connected!</h3><p>{result['provider']} is now connected.</p>"
        "<p>You can close this tab.</p>"
    )


@app.post("/api/oauth/refresh/{provider}")
async def api_oauth_refresh(provider: str, authorization: str | None = Header(None)):
    """Refresh an OAuth token."""
    verify_master_key(authorization)
    if not _client:
        raise HTTPException(503, "Server not ready")
    result = await oauth_manager.refresh_token(provider, _client)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.delete("/api/oauth/{provider}")
async def api_oauth_disconnect(provider: str, authorization: str | None = Header(None)):
    """Disconnect an OAuth provider."""
    verify_master_key(authorization)
    if not oauth_manager.remove_token(provider):
        raise HTTPException(404, f"No OAuth connection for: {provider}")
    return {"ok": True}




@app.post("/v1/batch")
async def batch_requests(request: Request, authorization: str | None = Header(None)):
    """Execute multiple chat completion requests in parallel.

    Body: {"requests": [{"model": "...", "messages": [...]}, ...]}
    Returns results as array in the same order. Each item is independent.
    """
    verify_master_key(authorization)
    body = await request.json()
    requests_list = body.get("requests", [])

    if not requests_list:
        return {"object": "batch", "results": []}

    if len(requests_list) > BATCH_MAX_SIZE:
        raise HTTPException(
            400,
            f"Batch size {len(requests_list)} exceeds maximum of {BATCH_MAX_SIZE}",
        )

    async def _process_batch_item(idx: int, req: dict) -> dict:
        model = req.get("model", "")
        if not model:
            return {
                "index": idx, "success": False,
                "error": "Missing 'model' field",
            }

        # Smart routing for each item
        resolved = smart_router.resolve(model)
        resolved_model = resolved.resolved_name

        # Tool auto-routing
        if has_tool_calling(req):
            model_cfg = config.models.get(resolved_model)
            if model_cfg and not model_cfg.capabilities.supports_tools:
                alt = smart_router.find_model_with_capability("supports_tools")
                if alt:
                    resolved_model = alt

        # Disable streaming in batch — not useful here
        batch_req = {**req, "stream": False, "model": resolved_model}

        try:
            result, provider, provider_model = await router.route_request(
                resolved_model, batch_req, _client,
            )
            if isinstance(result, dict):
                result.setdefault("model", provider_model)
            return {
                "index": idx, "success": True,
                "result": result, "provider": provider,
                "provider_model": provider_model,
                "substitution": resolved.substitution,
            }
        except Exception as e:
            return {"index": idx, "success": False, "error": str(e)[:500]}

    results = await asyncio.gather(
        *[_process_batch_item(i, req) for i, req in enumerate(requests_list)]
    )
    return {"object": "batch", "results": list(results)}


# ── Per-Key Rate Tracking API ────────────────────────────────────────────────
@app.get("/api/rate-tracking")
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


@app.get("/api/rate-tracking/{provider}")
async def api_rate_tracking_provider(provider: str, authorization: str | None = Header(None)):
    """Get per-key rate tracking for a specific provider."""
    verify_master_key(authorization)
    return {"provider": provider, "usage": per_key_rate_tracker.get_provider_usage(provider)}


@app.post("/api/rate-tracking/set-limits")
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


@app.post("/api/rate-tracking/cleanup")
async def api_rate_tracking_cleanup(authorization: str | None = Header(None)):
    """Clean up expired rate tracking entries."""
    verify_master_key(authorization)
    removed = per_key_rate_tracker.cleanup_expired()
    return {"removed": removed}


# ── Sticky Sessions API ──────────────────────────────────────────────────────
@app.get("/api/sessions")
async def api_sessions(authorization: str | None = Header(None)):
    """Get all active sticky sessions."""
    verify_master_key(authorization)
    return {
        "sessions": sticky_sessions.get_all(),
        "stats": sticky_sessions.get_stats(),
    }


@app.delete("/api/sessions/{conversation_id}")
async def api_session_remove(conversation_id: str, authorization: str | None = Header(None)):
    """Remove a sticky session."""
    verify_master_key(authorization)
    if not sticky_sessions.remove(conversation_id):
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@app.post("/api/sessions/cleanup")
async def api_sessions_cleanup(authorization: str | None = Header(None)):
    """Clean up expired sessions."""
    verify_master_key(authorization)
    removed = sticky_sessions.cleanup_expired()
    return {"removed": removed}


# ── Gateway Auth API ─────────────────────────────────────────────────────────
@app.get("/api/gateway-keys")
async def api_gateway_keys(authorization: str | None = Header(None)):
    """List all gateway API keys."""
    verify_master_key(authorization)
    if not gateway_auth:
        return {"keys": [], "stats": {"total_keys": 0}}
    return {"keys": gateway_auth.list_keys(), "stats": gateway_auth.get_stats()}


@app.post("/api/gateway-keys")
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


@app.delete("/api/gateway-keys/{name}")
async def api_revoke_gateway_key(name: str, authorization: str | None = Header(None)):
    """Revoke a gateway API key."""
    verify_master_key(authorization)
    if not gateway_auth:
        raise HTTPException(400, "Gateway auth not configured")
    if not gateway_auth.revoke_key(name):
        raise HTTPException(404, f"Key '{name}' not found")
    return {"ok": True}


@app.put("/api/gateway-keys/{name}/toggle")
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
@app.get("/api/encrypted-keys")
async def api_encrypted_keys(authorization: str | None = Header(None)):
    """List encrypted keys (masked)."""
    verify_master_key(authorization)
    return {"keys": encrypted_key_store.list_keys()}


@app.post("/api/encrypted-keys")
async def api_add_encrypted_key(request: Request, authorization: str | None = Header(None)):
    """Add a key to encrypted storage."""
    verify_master_key(authorization)
    body = await request.json()
    provider = body.get("provider", "").strip()
    key = body.get("key", "").strip()
    if not provider or not key:
        raise HTTPException(400, "Missing 'provider' or 'key'")
    index = encrypted_key_store.add_key(provider, key)
    # Also sync to config
    prov = config.providers.get(provider)
    if prov:
        all_keys = encrypted_key_store.get_keys(provider)
        prov.api_keys = all_keys
    return {"ok": True, "provider": provider, "index": index}


@app.delete("/api/encrypted-keys/{provider}/{index}")
async def api_remove_encrypted_key(provider: str, index: int, authorization: str | None = Header(None)):
    """Remove a key from encrypted storage."""
    verify_master_key(authorization)
    if not encrypted_key_store.remove_key(provider, index):
        raise HTTPException(404, "Key not found")
    # Sync to config
    prov = config.providers.get(provider)
    if prov:
        remaining = encrypted_key_store.get_keys(provider)
        prov.api_keys = remaining
    return {"ok": True}


# ── Playground API ───────────────────────────────────────────────────────────
@app.post("/api/playground")
async def api_playground(request: Request, authorization: str | None = Header(None)):
    """Interactive playground: send messages and get streaming responses.

    Accepts the same body as /v1/chat/completions but returns
    a JSON response with full metadata for the playground UI.
    """
    verify_master_key(authorization)
    body = await request.json()
    model = body.get("model", "")
    stream = body.get("stream", False)

    if not model:
        model = "llama-3.3-70b"
        body["model"] = model

    # Detect and translate format
    source_format = detect_format(body)
    if source_format != "openai":
        body = translate_to_openai(body, source_format)

    # Resolve model
    resolved = smart_router.resolve(model)
    model = resolved.resolved_name

    # Route with fallback tracking
    fallback_attempts: list[str] = []
    try:
        result, provider, provider_model = await router.route_request(model, body, _client)
    except AllRateLimitedError:
        raise HTTPException(429, "All providers rate-limited. Try again later.")
    except Exception as e:
        raise HTTPException(502, f"Request failed: {e}")

    fallback_attempts.append(f"{provider}/{provider_model}")

    # For streaming, return SSE
    if stream and hasattr(result, "__aiter__"):
        return StreamingResponse(
            safe_stream(result, provider, provider_model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Routed-Via": f"{provider}/{provider_model}",
                "X-Fallback-Attempts": str(len(fallback_attempts)),
            },
        )

    # For non-streaming, wrap with playground metadata
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
@app.post("/api/translate-tools")
async def api_translate_tools(request: Request, authorization: str | None = Header(None)):
    """Translate tool/function calling definitions between provider formats."""
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


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
    )
