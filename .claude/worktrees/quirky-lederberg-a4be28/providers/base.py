"""Abstract base provider and shared utilities for all provider adapters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 120.0  # seconds


# ── Exceptions ────────────────────────────────────────────────────────────────

class ProviderError(Exception):
    """Raised when a provider request fails."""

    def __init__(
        self,
        provider: str,
        status: int,
        message: str,
        retry_after: float | None = None,
    ):
        self.provider = provider
        self.status = status
        self.message = message
        self.retry_after = retry_after
        super().__init__(f"[{provider}] HTTP {status}: {message}")


# ── Utility functions ─────────────────────────────────────────────────────────

def has_tool_calling(payload: dict[str, Any]) -> bool:
    """Check if the request contains tool/function calling fields."""
    return bool(
        payload.get("tools") or payload.get("tool_choice")
        or payload.get("functions") or payload.get("function_call")
    )


def _is_rate_limited(status: int) -> bool:
    return status == 429


def _get_retry_after(resp: httpx.Response) -> float | None:
    """Extract Retry-After header value in seconds."""
    val = resp.headers.get("retry-after")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def build_openai_body(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize the request body for OpenAI-compatible providers."""
    body = {**payload}
    body["model"] = model
    return body


# ── Abstract base class ──────────────────────────────────────────────────────

class BaseProvider(ABC):
    """Base class for all LLM provider adapters.

    Each provider implements its own URL construction, header building,
    body formatting, and key validation logic.
    """

    platform: str = ""

    @abstractmethod
    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        provider_config: Any,
        model: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        """Send a chat completion request to this provider.

        Returns either a JSON dict (non-streaming) or an AsyncIterator[bytes]
        (streaming SSE chunks).
        """

    @abstractmethod
    def get_models_url(self, provider_config: Any) -> str:
        """URL for listing available models."""

    @abstractmethod
    def get_models_headers(
        self, provider_config: Any, api_key: str | None = None,
    ) -> dict[str, str]:
        """Headers for the models listing endpoint.

        If api_key is None, use provider_config.api_key.
        """

    def validate_key_url(self, provider_config: Any) -> str:
        """URL for validating an API key. Defaults to get_models_url()."""
        return self.get_models_url(provider_config)

    # ── Shared helpers ────────────────────────────────────────────────────

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        provider_name: str,
    ) -> AsyncIterator[bytes]:
        """Stream SSE chunks from a provider, with error handling."""
        try:
            async with client.stream(
                "POST", url, headers=headers, json=body, timeout=REQUEST_TIMEOUT,
            ) as resp:
                if _is_rate_limited(resp.status_code):
                    raise ProviderError(
                        provider_name, 429, "Rate limited",
                        retry_after=_get_retry_after(resp),
                    )
                if resp.status_code >= 400:
                    body_text = await resp.aread()
                    raise ProviderError(
                        provider_name, resp.status_code, body_text.decode()[:500],
                    )
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.TimeoutException:
            raise ProviderError(provider_name, 0, "Request timed out")

    def _check_response(
        self, resp: httpx.Response, provider_name: str,
    ) -> None:
        """Check an HTTP response and raise ProviderError on failure."""
        if _is_rate_limited(resp.status_code):
            raise ProviderError(
                provider_name, 429, "Rate limited",
                retry_after=_get_retry_after(resp),
            )
        if resp.status_code >= 400:
            raise ProviderError(provider_name, resp.status_code, resp.text[:500])
