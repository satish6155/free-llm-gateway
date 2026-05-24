"""OpenAI-compatible provider — used by 18 platforms that share the same API format."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

from providers.base import (
    BaseProvider,
    ProviderError,
    build_openai_body,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Platforms that use OpenAI-compatible /chat/completions
OPENAI_COMPAT_PLATFORMS = {
    "openrouter", "github", "groq", "cerebras", "nvidia",
    "siliconflow", "mistral", "llm7", "ollama",
    "deepseek", "together", "fireworks", "sambanova", "chutes",
    "openai", "perplexity", "xai", "novita",
}


class OpenAICompatibleProvider(BaseProvider):
    """Provider for platforms that follow the OpenAI chat/completions API."""

    def __init__(self, platform: str = "") -> None:
        self.platform = platform

    def _build_headers(self, provider_config: Any, api_key: str | None = None) -> dict[str, str]:
        key = api_key or provider_config.api_key or ""
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if self.platform == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/free-llm-gateway"
            headers["X-Title"] = "Free LLM Gateway"
        return headers

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        provider_config: Any,
        model: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        url = f"{provider_config.base_url}/chat/completions"
        headers = self._build_headers(provider_config)
        body = build_openai_body(model, payload)
        stream = payload.get("stream", False)

        if stream:
            return self._stream_response(client, url, headers, body, provider_config.name)

        try:
            resp = await client.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
        except httpx.TimeoutException:
            raise ProviderError(provider_config.name, 0, "Request timed out")

        self._check_response(resp, provider_config.name)
        return resp.json()

    def get_models_url(self, provider_config: Any) -> str:
        return f"{provider_config.base_url}/models"

    def get_models_headers(
        self, provider_config: Any, api_key: str | None = None,
    ) -> dict[str, str]:
        return self._build_headers(provider_config, api_key)
