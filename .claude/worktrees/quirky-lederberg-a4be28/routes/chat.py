"""Core LLM endpoints — chat completions, models, embeddings, batch."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
from typing import Any, AsyncIterator

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from cache import ResponseCache
from providers.base import has_tool_calling
from request_queue import BLOCKING_QUEUE
from router import AllRateLimitedError
from routes.state import (
    client, config, quota_tracker,
    request_queue, response_cache, router, smart_router,
    sticky_sessions, usage_tracker, verify_master_key,
)
from token_compressor import compress_messages
from format_translator import detect_format, translate_to_openai

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Request validation (Pydantic models) ──────────────────────────────────────


class ChatCompletionRequest(BaseModel):
    """Validates incoming chat completion requests before routing to providers."""
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
            if role == "assistant":
                has_content = (
                    isinstance(msg.get("content"), str) and len(msg["content"]) > 0
                )
                has_tool_calls = bool(msg.get("tool_calls"))
                if not has_content and not has_tool_calls:
                    raise ValueError(
                        f"messages[{i}] (assistant) must have non-empty content or tool_calls"
                    )
            if role == "tool" and not msg.get("tool_call_id"):
                raise ValueError(
                    f"messages[{i}] (tool) must include 'tool_call_id'"
                )
        return v


async def safe_stream(
    stream: Any, provider: str, provider_model: str,
) -> AsyncIterator[bytes]:
    """Wrap a provider stream to catch mid-stream errors."""
    try:
        async for chunk in stream:
            yield chunk
    except Exception as exc:
        logger.error(
            "Mid-stream error from %s/%s: %s",
            provider, provider_model, exc,
        )
        error_payload = _json.dumps({
            "error": {
                "message": f"Provider error ({provider}/{provider_model}): stream interrupted",
                "type": "stream_error",
            },
        })
        yield f"data: {error_payload}\n\n".encode()
        yield b"data: [DONE]\n\n"


# ── Chat completions ─────────────────────────────────────────────────────────


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(None)):
    verify_master_key(authorization)
    raw_body = await request.json()

    # Input validation
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

    body = raw_body
    model = validated.model or ""
    stream = validated.stream

    # Format translation
    source_format = detect_format(body)
    if source_format != "openai":
        logger.info("Format translation: %s -> openai", source_format)
        body = translate_to_openai(body, source_format)

    if not model:
        model = body.get("model", "")
        if not model:
            raise HTTPException(400, "Missing 'model' field")

    # RTK Token Compression
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

    # Smart routing
    resolved = smart_router.resolve(model)
    if resolved.substitution:
        logger.info("Smart routing: %s", resolved.substitution)
    model = resolved.resolved_name

    # Tool calling auto-routing
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

    # Cache check (non-streaming only)
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

    # Sticky sessions
    messages = body.get("messages", [])
    conversation_id = (
        body.get("conversation_id")
        or sticky_sessions.extract_conversation_id(messages)
    )
    preferred_provider, preferred_model = None, None
    if conversation_id:
        preferred_provider, preferred_model = sticky_sessions.get(conversation_id)

    # Tool calling translation for Gemini
    tools = body.get("tools")
    if tools:
        body["_original_tools"] = tools

    # Route request
    fallback_attempts: list[str] = []
    try:
        result, provider, provider_model = await router.route_request(
            model, body, client,
        )
    except AllRateLimitedError:
        return await _handle_rate_limited(model, body, stream)

    fallback_attempts.append(f"{provider}/{provider_model}")

    # Record sticky session
    if conversation_id:
        sticky_sessions.set(conversation_id, provider, provider_model)

    # Per-key rate tracking
    from rate_tracker import per_key_rate_tracker
    prov_cfg = config.providers.get(provider)
    if prov_cfg and prov_cfg.api_key:
        total_tokens = 0
        if isinstance(result, dict):
            usage = result.get("usage")
            if usage and isinstance(usage, dict):
                total_tokens = usage.get("total_tokens", 0) or 0
        per_key_rate_tracker.record_request(
            provider, provider_model, prov_cfg.api_key, tokens=total_tokens,
        )

    # Record token usage
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
            quota_tracker.record_request(
                provider, tokens=usage.get("total_tokens", 0) or 0,
            )

        # Cache response (non-streaming only)
        if not stream:
            response_cache.put(cache_key, result)

    # Return response
    routing_headers = {
        "X-Routed-Via": f"{provider}/{provider_model}",
        "X-Fallback-Attempts": str(len(fallback_attempts)),
        "X-Sticky-Session": (
            "true" if conversation_id and preferred_provider == provider else "false"
        ),
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
        try:
            result = await request_queue.get_result(req_id)
            return JSONResponse(
                content=result,
                headers={"X-Queued": "true", "X-Request-Id": req_id},
            )
        except Exception:
            raise HTTPException(504, "Queued request timed out")

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


@router.get("/v1/models")
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


@router.post("/v1/embeddings")
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
            resp = await client.post(url, headers=headers, json=body, timeout=60.0)
            if resp.status_code < 400:
                return resp.json()
        except Exception as e:
            logger.warning("Embedding provider %s failed: %s", fb.provider, e)
            continue

    raise HTTPException(502, "All embedding providers failed")


# ── Batch requests ────────────────────────────────────────────────────────────

BATCH_MAX_SIZE = int(os.environ.get("BATCH_MAX_SIZE", "10"))


@router.post("/v1/batch")
async def batch_requests(request: Request, authorization: str | None = Header(None)):
    """Execute multiple chat completion requests in parallel."""
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
            return {"index": idx, "success": False, "error": "Missing 'model' field"}

        resolved = smart_router.resolve(model)
        resolved_model = resolved.resolved_name

        if has_tool_calling(req):
            model_cfg = config.models.get(resolved_model)
            if model_cfg and not model_cfg.capabilities.supports_tools:
                alt = smart_router.find_model_with_capability("supports_tools")
                if alt:
                    resolved_model = alt

        batch_req = {**req, "stream": False, "model": resolved_model}

        try:
            result, provider, provider_model = await router.route_request(
                resolved_model, batch_req, client,
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
