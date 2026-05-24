"""HuggingFace Inference API provider."""

from __future__ import annotations

from typing import Any

import httpx

from providers.base import BaseProvider, ProviderError, REQUEST_TIMEOUT


class HuggingFaceProvider(BaseProvider):
    """HuggingFace Inference API — different URL pattern and body format."""

    platform = "huggingface"

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        provider_config: Any,
        model: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{provider_config.base_url}/{model}"
        headers = {
            "Authorization": f"Bearer {provider_config.api_key}",
            "Content-Type": "application/json",
        }
        messages = payload.get("messages", [])
        hf_body: dict[str, Any] = {"model": model}
        if messages:
            hf_body["messages"] = messages
        hf_body["max_tokens"] = payload.get("max_tokens", 1024)
        hf_body["stream"] = payload.get("stream", False)

        try:
            resp = await client.post(url, headers=headers, json=hf_body, timeout=REQUEST_TIMEOUT)
        except httpx.TimeoutException:
            raise ProviderError(provider_config.name, 0, "Request timed out")

        self._check_response(resp, provider_config.name)
        return resp.json()

    def get_models_url(self, provider_config: Any) -> str:
        return provider_config.base_url

    def get_models_headers(
        self, provider_config: Any, api_key: str | None = None,
    ) -> dict[str, str]:
        key = api_key or provider_config.api_key or ""
        return {"Authorization": f"Bearer {key}"}
