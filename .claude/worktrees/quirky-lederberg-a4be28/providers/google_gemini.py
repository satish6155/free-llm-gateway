"""Google Gemini provider — different API format with contents/parts."""

from __future__ import annotations

from typing import Any

import httpx

from providers.base import BaseProvider, ProviderError, REQUEST_TIMEOUT


class GoogleGeminiProvider(BaseProvider):
    """Google Gemini API — uses generateContent endpoint with contents format."""

    platform = "google_gemini"

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        provider_config: Any,
        model: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = (
            f"{provider_config.base_url}/models/{model}"
            f":generateContent?key={provider_config.api_key}"
        )
        messages = payload.get("messages", [])
        contents = []
        for m in messages:
            role = "user" if m["role"] in ("user", "system") else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        gemini_body: dict[str, Any] = {"contents": contents}

        try:
            resp = await client.post(url, json=gemini_body, timeout=REQUEST_TIMEOUT)
        except httpx.TimeoutException:
            raise ProviderError(provider_config.name, 0, "Request timed out")

        self._check_response(resp, provider_config.name)
        return resp.json()

    def get_models_url(self, provider_config: Any) -> str:
        return f"{provider_config.base_url}/models?key={provider_config.api_key}"

    def get_models_headers(
        self, provider_config: Any, api_key: str | None = None,
    ) -> dict[str, str]:
        # Gemini uses query param auth, not Bearer headers
        return {}

    def validate_key_url(self, provider_config: Any) -> str:
        return f"{provider_config.base_url}/models?key={provider_config.api_key}"
