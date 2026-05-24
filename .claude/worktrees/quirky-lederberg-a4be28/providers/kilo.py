"""Kilo provider — OpenAI-compatible with different base URL path."""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from providers.base import (
    BaseProvider,
    ProviderError,
    build_openai_body,
    REQUEST_TIMEOUT,
)


class KiloProvider(BaseProvider):
    """Kilo AI — OpenAI-compatible with /v1/ path prefix."""

    platform = "kilo"

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        provider_config: Any,
        model: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        url = f"{provider_config.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {provider_config.api_key}",
            "Content-Type": "application/json",
        }
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
        return f"{provider_config.base_url}/v1/models"

    def get_models_headers(
        self, provider_config: Any, api_key: str | None = None,
    ) -> dict[str, str]:
        key = api_key or provider_config.api_key or ""
        return {"Authorization": f"Bearer {key}"}
