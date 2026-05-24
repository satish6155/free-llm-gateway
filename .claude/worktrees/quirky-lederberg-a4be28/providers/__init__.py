"""Provider registry — one adapter per LLM provider platform.

Usage:
    from providers import get_provider, send_to_provider, PROVIDER_REGISTRY
    p = get_provider("groq")
    result = await p.chat_completion(client, config, model, payload)
"""

from __future__ import annotations

from providers.base import BaseProvider, ProviderError, has_tool_calling
from providers.openai_compatible import OpenAICompatibleProvider
from providers.cloudflare import CloudflareProvider
from providers.huggingface import HuggingFaceProvider
from providers.cohere import CohereProvider
from providers.google_gemini import GoogleGeminiProvider
from providers.kilo import KiloProvider
from providers.anthropic import AnthropicProvider

# ── Provider registry ────────────────────────────────────────────────────────

PROVIDER_REGISTRY: dict[str, BaseProvider] = {}

# OpenAI-compatible providers (18 platforms share one implementation)
_OPENAI_COMPAT_PLATFORMS = (
    "openrouter", "github", "groq", "cerebras", "nvidia",
    "siliconflow", "mistral", "llm7", "ollama",
    "deepseek", "together", "fireworks", "sambanova", "chutes",
    "openai", "perplexity", "xai", "novita",
)

for _name in _OPENAI_COMPAT_PLATFORMS:
    PROVIDER_REGISTRY[_name] = OpenAICompatibleProvider(platform=_name)

# Providers with special API formats
PROVIDER_REGISTRY["cloudflare"] = CloudflareProvider()
PROVIDER_REGISTRY["huggingface"] = HuggingFaceProvider()
PROVIDER_REGISTRY["cohere"] = CohereProvider()
PROVIDER_REGISTRY["google_gemini"] = GoogleGeminiProvider()
PROVIDER_REGISTRY["kilo"] = KiloProvider()
PROVIDER_REGISTRY["anthropic"] = AnthropicProvider()


def get_provider(name: str) -> BaseProvider:
    """Look up a provider adapter by platform name."""
    p = PROVIDER_REGISTRY.get(name)
    if not p:
        raise ProviderError(name, 400, f"Unknown provider: {name}")
    return p


# Providers that don't support OpenAI tool/function calling format
SPECIAL_PROVIDERS = {"cloudflare", "huggingface", "cohere", "google_gemini", "kilo", "anthropic"}


async def send_to_provider(
    client, provider_config, model: str, payload: dict,
) -> dict | object:
    """Route a request to the correct provider adapter.

    This is the main entry point used by the router.
    Guards against tool-calling requests to providers that don't support it.
    """
    name = provider_config.name
    # Special providers don't support OpenAI tool calling format
    if name in SPECIAL_PROVIDERS and has_tool_calling(payload):
        raise ProviderError(
            name, 400,
            f"Provider {name} does not support tool calling for model {model}",
        )
    p = get_provider(name)
    return await p.chat_completion(client, provider_config, model, payload)


__all__ = [
    "BaseProvider",
    "ProviderError",
    "has_tool_calling",
    "get_provider",
    "send_to_provider",
    "PROVIDER_REGISTRY",
]
