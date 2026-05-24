"""Anthropic provider — placeholder for future Claude API support."""

from __future__ import annotations

from typing import Any

import httpx

from providers.base import BaseProvider, ProviderError, REQUEST_TIMEOUT


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API — placeholder for future native format support.

    Currently routes through OpenAI-compatible endpoints when available.
    """

    platform = "anthropic"

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        provider_config: Any,
        model: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        # Use OpenAI-compatible endpoint if Anthropic provides one
        url = f"{provider_config.base_url}/v1/chat/completions"
        headers = {
            "x-api-key": provider_config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {**payload, "model": model}

        try:
            resp = await client.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
        except httpx.TimeoutException:
            raise ProviderError(provider_config.name, 0, "Request timed out")

        self._check_response(resp, provider_config.name)
        return resp.json()

    def get_models_url(self, provider_config: Any) -> str:
        return f"{provider_config.base_url}/v1/models"

    def get_models_headers(
        self, provider_config: Any, api_key: str | None = None,
    ) -> dict[str, str]:
        key = api_key or provider_config.api_key or ""
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
