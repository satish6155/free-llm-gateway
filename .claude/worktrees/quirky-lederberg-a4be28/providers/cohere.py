"""Cohere provider — different endpoint and message format."""

from __future__ import annotations

from typing import Any

import httpx

from providers.base import BaseProvider, ProviderError, REQUEST_TIMEOUT


class CohereProvider(BaseProvider):
    """Cohere API — uses /chat endpoint with different message format."""

    platform = "cohere"

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        provider_config: Any,
        model: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{provider_config.base_url}/chat"
        headers = {
            "Authorization": f"Bearer {provider_config.api_key}",
            "Content-Type": "application/json",
        }
        messages = payload.get("messages", [])
        cohere_body: dict[str, Any] = {"model": model}
        if messages:
            last_msg = messages[-1] if messages else {}
            cohere_body["message"] = last_msg.get("content", "")
            if len(messages) > 1:
                cohere_body["chat_history"] = [
                    {"role": m["role"], "message": m["content"]}
                    for m in messages[:-1]
                    if m["role"] in ("user", "assistant")
                ]

        try:
            resp = await client.post(url, headers=headers, json=cohere_body, timeout=REQUEST_TIMEOUT)
        except httpx.TimeoutException:
            raise ProviderError(provider_config.name, 0, "Request timed out")

        self._check_response(resp, provider_config.name)
        return resp.json()

    def get_models_url(self, provider_config: Any) -> str:
        return f"{provider_config.base_url}/models"

    def get_models_headers(
        self, provider_config: Any, api_key: str | None = None,
    ) -> dict[str, str]:
        key = api_key or provider_config.api_key or ""
        return {"Authorization": f"Bearer {key}"}
