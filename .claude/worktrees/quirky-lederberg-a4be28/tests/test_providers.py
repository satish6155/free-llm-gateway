"""Tests for all 24 provider adapters — registry, URL construction, headers, body formatting."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from providers import PROVIDER_REGISTRY, get_provider, send_to_provider
from providers.base import (
    BaseProvider,
    ProviderError,
    build_openai_body,
    has_tool_calling,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_provider_config(
    name: str = "test",
    base_url: str = "https://api.test.com",
    api_key: str = "sk-test-key-123",
) -> MagicMock:
    """Create a mock ProviderConfig for testing."""
    cfg = MagicMock()
    cfg.name = name
    cfg.base_url = base_url
    cfg.api_key = api_key
    return cfg


def _make_client(response_json: dict | None = None, status_code: int = 200) -> AsyncMock:
    """Create a mock httpx.AsyncClient that returns a canned response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = json.dumps(response_json or {})
    resp.json = MagicMock(return_value=response_json or {})
    resp.headers = httpx.Headers({})

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=resp)
    client.get = AsyncMock(return_value=resp)
    return client


def _run(coro):
    """Run an async coroutine synchronously in a fresh event loop."""
    return asyncio.run(coro)


# ── 1. Provider Registry ──────────────────────────────────────────────────────


class TestProviderRegistry:
    """Verify all 24 providers are registered and retrievable."""

    ALL_PROVIDERS = [
        "openrouter", "github", "groq", "cerebras", "nvidia",
        "siliconflow", "mistral", "llm7", "ollama",
        "deepseek", "together", "fireworks", "sambanova", "chutes",
        "openai", "perplexity", "xai", "novita",
        "cloudflare", "huggingface", "cohere", "google_gemini", "kilo",
        "anthropic",
    ]

    def test_registry_has_24_providers(self) -> None:
        assert len(PROVIDER_REGISTRY) == 24

    @pytest.mark.parametrize("name", ALL_PROVIDERS)
    def test_get_provider_returns_instance(self, name: str) -> None:
        p = get_provider(name)
        assert isinstance(p, BaseProvider)

    @pytest.mark.parametrize("name", ALL_PROVIDERS)
    def test_provider_has_platform(self, name: str) -> None:
        p = get_provider(name)
        assert p.platform, f"{name} has empty platform"

    def test_get_unknown_provider_raises(self) -> None:
        with pytest.raises(ProviderError, match="Unknown provider"):
            get_provider("nonexistent_provider_xyz")

    def test_all_providers_are_baseprovider(self) -> None:
        for name, provider in PROVIDER_REGISTRY.items():
            assert isinstance(provider, BaseProvider), f"{name} is not a BaseProvider"


# ── 2. OpenAI-Compatible Providers (18 platforms) ─────────────────────────────


class TestOpenAICompatibleProvider:
    """Verify the shared OpenAI-compatible provider works for all 18 platforms."""

    OPENAI_PLATFORMS = [
        "openrouter", "github", "groq", "cerebras", "nvidia",
        "siliconflow", "mistral", "llm7", "ollama",
        "deepseek", "together", "fireworks", "sambanova", "chutes",
        "openai", "perplexity", "xai", "novita",
    ]

    @pytest.mark.parametrize("name", OPENAI_PLATFORMS)
    def test_models_url(self, name: str) -> None:
        p = get_provider(name)
        cfg = _make_provider_config(name, "https://api.test.com/v1")
        assert p.get_models_url(cfg) == "https://api.test.com/v1/models"

    @pytest.mark.parametrize("name", OPENAI_PLATFORMS)
    def test_models_headers_with_bearer(self, name: str) -> None:
        p = get_provider(name)
        cfg = _make_provider_config(name, api_key="sk-mykey")
        headers = p.get_models_headers(cfg)
        assert headers["Authorization"] == "Bearer sk-mykey"

    @pytest.mark.parametrize("name", OPENAI_PLATFORMS)
    def test_models_headers_with_explicit_key(self, name: str) -> None:
        p = get_provider(name)
        cfg = _make_provider_config(name, api_key="sk-default")
        headers = p.get_models_headers(cfg, api_key="sk-explicit")
        assert headers["Authorization"] == "Bearer sk-explicit"

    @pytest.mark.parametrize("name", OPENAI_PLATFORMS)
    def test_chat_completion_url(self, name: str) -> None:
        """Chat completion should POST to {base_url}/chat/completions."""
        p = get_provider(name)
        cfg = _make_provider_config(name, "https://api.test.com/v1")
        client = _make_client({"id": "chatcmpl-1", "choices": []})

        _run(p.chat_completion(client, cfg, "model-x", {
            "messages": [{"role": "user", "content": "hi"}],
        }))

        client.post.assert_called_once()
        call_url = client.post.call_args[0][0]
        assert call_url == "https://api.test.com/v1/chat/completions"

    def test_openrouter_extra_headers(self) -> None:
        """OpenRouter should include HTTP-Referer and X-Title headers."""
        p = get_provider("openrouter")
        cfg = _make_provider_config("openrouter")
        client = _make_client({"id": "chatcmpl-1"})

        _run(p.chat_completion(client, cfg, "openai/gpt-oss-120b", {
            "messages": [{"role": "user", "content": "hi"}],
        }))

        headers = client.post.call_args[1]["headers"]
        assert headers["HTTP-Referer"] == "https://github.com/free-llm-gateway"
        assert headers["X-Title"] == "Free LLM Gateway"

    def test_non_openrouter_no_extra_headers(self) -> None:
        """Non-OpenRouter providers should NOT include Referer/Title headers."""
        p = get_provider("groq")
        cfg = _make_provider_config("groq")
        client = _make_client({"id": "chatcmpl-1"})

        _run(p.chat_completion(client, cfg, "llama3", {
            "messages": [{"role": "user", "content": "hi"}],
        }))

        headers = client.post.call_args[1]["headers"]
        assert "HTTP-Referer" not in headers
        assert "X-Title" not in headers

    def test_rate_limited_raises_provider_error(self) -> None:
        p = get_provider("groq")
        cfg = _make_provider_config("groq")
        client = _make_client(status_code=429)

        with pytest.raises(ProviderError) as exc_info:
            _run(p.chat_completion(client, cfg, "llama3", {
                "messages": [{"role": "user", "content": "hi"}],
            }))
        assert exc_info.value.status == 429

    def test_server_error_raises_provider_error(self) -> None:
        p = get_provider("groq")
        cfg = _make_provider_config("groq")
        client = _make_client(status_code=500)

        with pytest.raises(ProviderError) as exc_info:
            _run(p.chat_completion(client, cfg, "llama3", {
                "messages": [{"role": "user", "content": "hi"}],
            }))
        assert exc_info.value.status == 500

    def test_timeout_raises_provider_error(self) -> None:
        p = get_provider("groq")
        cfg = _make_provider_config("groq")
        client = _make_client()
        client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with pytest.raises(ProviderError) as exc_info:
            _run(p.chat_completion(client, cfg, "llama3", {
                "messages": [{"role": "user", "content": "hi"}],
            }))
        assert exc_info.value.status == 0

    def test_success_returns_json(self) -> None:
        p = get_provider("groq")
        cfg = _make_provider_config("groq")
        expected = {"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello!"}}]}
        client = _make_client(expected, status_code=200)

        result = _run(p.chat_completion(client, cfg, "llama3", {
            "messages": [{"role": "user", "content": "hi"}],
        }))
        assert result == expected

    def test_body_includes_model(self) -> None:
        p = get_provider("groq")
        cfg = _make_provider_config("groq")
        client = _make_client({"id": "1"})

        _run(p.chat_completion(client, cfg, "llama-3.3-70b", {
            "messages": [{"role": "user", "content": "hi"}],
        }))

        body = client.post.call_args[1]["json"]
        assert body["model"] == "llama-3.3-70b"


# ── 3. Cloudflare Provider ────────────────────────────────────────────────────


class TestCloudflareProvider:
    """Verify Cloudflare Workers AI provider."""

    def test_platform(self) -> None:
        assert get_provider("cloudflare").platform == "cloudflare"

    def test_models_url(self) -> None:
        cfg = _make_provider_config("cloudflare", "https://api.cloudflare.com/client/v4/accounts/acc123")
        url = get_provider("cloudflare").get_models_url(cfg)
        assert url == "https://api.cloudflare.com/client/v4/accounts/acc123/models"

    def test_chat_completion_url(self) -> None:
        p = get_provider("cloudflare")
        cfg = _make_provider_config("cloudflare", "https://api.cloudflare.com/client/v4/accounts/acc123")
        client = _make_client({"result": {}})

        _run(p.chat_completion(client, cfg, "@cf/meta/llama-3.1-8b-instruct", {
            "messages": [{"role": "user", "content": "hi"}],
        }))

        call_url = client.post.call_args[0][0]
        assert "/chat/completions" in call_url

    def test_bearer_auth(self) -> None:
        p = get_provider("cloudflare")
        cfg = _make_provider_config("cloudflare", api_key="cf-key-123")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "model", {
            "messages": [{"role": "user", "content": "hi"}],
        }))

        headers = client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer cf-key-123"


# ── 4. HuggingFace Provider ──────────────────────────────────────────────────


class TestHuggingFaceProvider:
    """Verify HuggingFace Inference API provider."""

    def test_platform(self) -> None:
        assert get_provider("huggingface").platform == "huggingface"

    def test_models_url_is_base_url(self) -> None:
        cfg = _make_provider_config("huggingface", "https://api-inference.huggingface.co/models")
        url = get_provider("huggingface").get_models_url(cfg)
        assert url == "https://api-inference.huggingface.co/models"

    def test_chat_url_includes_model(self) -> None:
        p = get_provider("huggingface")
        cfg = _make_provider_config("huggingface", "https://api-inference.huggingface.co/models")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "meta-llama/Llama-3.1-8B-Instruct", {
            "messages": [{"role": "user", "content": "hi"}],
        }))

        call_url = client.post.call_args[0][0]
        assert "meta-llama/Llama-3.1-8B-Instruct" in call_url

    def test_body_includes_messages(self) -> None:
        p = get_provider("huggingface")
        cfg = _make_provider_config("huggingface")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "model-x", {
            "messages": [{"role": "user", "content": "Hello HF"}],
        }))

        body = client.post.call_args[1]["json"]
        assert body["messages"] == [{"role": "user", "content": "Hello HF"}]


# ── 5. Cohere Provider ────────────────────────────────────────────────────────


class TestCohereProvider:
    """Verify Cohere API provider."""

    def test_platform(self) -> None:
        assert get_provider("cohere").platform == "cohere"

    def test_chat_url_uses_chat_endpoint(self) -> None:
        p = get_provider("cohere")
        cfg = _make_provider_config("cohere", "https://api.cohere.com/v2")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "command-r-plus", {
            "messages": [{"role": "user", "content": "hi"}],
        }))

        call_url = client.post.call_args[0][0]
        assert call_url.endswith("/chat")

    def test_body_converts_messages(self) -> None:
        """Cohere should convert messages: last→message, rest→chat_history."""
        p = get_provider("cohere")
        cfg = _make_provider_config("cohere")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "command-r-plus", {
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Second"},
            ],
        }))

        body = client.post.call_args[1]["json"]
        assert body["message"] == "Second"
        assert len(body.get("chat_history", [])) == 2

    def test_single_message_no_history(self) -> None:
        """Single message should not produce chat_history."""
        p = get_provider("cohere")
        cfg = _make_provider_config("cohere")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "command-r-plus", {
            "messages": [{"role": "user", "content": "Hello"}],
        }))

        body = client.post.call_args[1]["json"]
        assert body["message"] == "Hello"
        # With only 1 message, chat_history should not be set (len(messages) > 1 is False)
        assert "chat_history" not in body


# ── 6. Google Gemini Provider ─────────────────────────────────────────────────


class TestGoogleGeminiProvider:
    """Verify Google Gemini API provider."""

    def test_platform(self) -> None:
        assert get_provider("google_gemini").platform == "google_gemini"

    def test_url_contains_generate_content(self) -> None:
        p = get_provider("google_gemini")
        cfg = _make_provider_config("google_gemini", "https://generativelanguage.googleapis.com/v1beta")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "gemini-2.0-flash", {
            "messages": [{"role": "user", "content": "Hello"}],
        }))

        call_url = client.post.call_args[0][0]
        assert ":generateContent" in call_url
        assert "key=sk-test-key-123" in call_url

    def test_body_converts_to_contents(self) -> None:
        p = get_provider("google_gemini")
        cfg = _make_provider_config("google_gemini")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "gemini-2.0-flash", {
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ],
        }))

        body = client.post.call_args[1]["json"]
        assert "contents" in body
        # system → user, user → user (both mapped to "user" role in Gemini)
        assert body["contents"][0]["role"] == "user"
        assert body["contents"][0]["parts"][0]["text"] == "Be helpful"
        assert body["contents"][1]["role"] == "user"
        assert body["contents"][1]["parts"][0]["text"] == "Hello"

    def test_assistant_role_mapped_to_model(self) -> None:
        p = get_provider("google_gemini")
        cfg = _make_provider_config("google_gemini")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "gemini-2.0-flash", {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        }))

        body = client.post.call_args[1]["json"]
        roles = [c["role"] for c in body["contents"]]
        assert roles == ["user", "model", "user"]

    def test_models_url_uses_query_param(self) -> None:
        p = get_provider("google_gemini")
        cfg = _make_provider_config("google_gemini", "https://generativelanguage.googleapis.com/v1beta")
        url = p.get_models_url(cfg)
        assert "key=" in url
        assert "/models" in url

    def test_models_headers_empty(self) -> None:
        """Gemini uses query params, not Bearer headers."""
        p = get_provider("google_gemini")
        cfg = _make_provider_config("google_gemini")
        headers = p.get_models_headers(cfg)
        assert "Authorization" not in headers
        assert headers == {}

    def test_chat_does_not_use_bearer_header(self) -> None:
        """Gemini chat URL should use ?key= not Bearer header."""
        p = get_provider("google_gemini")
        cfg = _make_provider_config("google_gemini", "https://example.com")
        client = _make_client({})
        _run(p.chat_completion(client, cfg, "gemini-2.0-flash", {
            "messages": [{"role": "user", "content": "hi"}],
        }))
        # Gemini passes no headers dict (uses json= only)
        call_kwargs = client.post.call_args[1]
        assert "headers" not in call_kwargs or "Authorization" not in call_kwargs.get("headers", {})


# ── 7. Kilo Provider ──────────────────────────────────────────────────────────


class TestKiloProvider:
    """Verify Kilo provider (OpenAI-compatible with /v1/ prefix)."""

    def test_platform(self) -> None:
        assert get_provider("kilo").platform == "kilo"

    def test_url_has_v1_prefix(self) -> None:
        p = get_provider("kilo")
        cfg = _make_provider_config("kilo", "https://api.kilo.ai")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "model-x", {
            "messages": [{"role": "user", "content": "hi"}],
        }))

        call_url = client.post.call_args[0][0]
        assert call_url == "https://api.kilo.ai/v1/chat/completions"

    def test_models_url_has_v1_prefix(self) -> None:
        p = get_provider("kilo")
        cfg = _make_provider_config("kilo", "https://api.kilo.ai")
        url = p.get_models_url(cfg)
        assert url == "https://api.kilo.ai/v1/models"


# ── 8. Anthropic Provider ─────────────────────────────────────────────────────


class TestAnthropicProvider:
    """Verify Anthropic provider."""

    def test_platform(self) -> None:
        assert get_provider("anthropic").platform == "anthropic"

    def test_uses_x_api_key_header(self) -> None:
        p = get_provider("anthropic")
        cfg = _make_provider_config("anthropic", "https://api.anthropic.com")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "claude-3-5-sonnet-20241022", {
            "messages": [{"role": "user", "content": "hi"}],
        }))

        headers = client.post.call_args[1]["headers"]
        assert headers["x-api-key"] == "sk-test-key-123"
        assert "Authorization" not in headers

    def test_includes_anthropic_version(self) -> None:
        p = get_provider("anthropic")
        cfg = _make_provider_config("anthropic")
        client = _make_client({})

        _run(p.chat_completion(client, cfg, "claude-3-5-sonnet-20241022", {
            "messages": [{"role": "user", "content": "hi"}],
        }))

        headers = client.post.call_args[1]["headers"]
        assert "anthropic-version" in headers

    def test_models_headers_uses_x_api_key(self) -> None:
        p = get_provider("anthropic")
        cfg = _make_provider_config("anthropic")
        headers = p.get_models_headers(cfg)
        assert "x-api-key" in headers
        assert "Authorization" not in headers

    def test_models_url_has_v1_prefix(self) -> None:
        p = get_provider("anthropic")
        cfg = _make_provider_config("anthropic", "https://api.anthropic.com")
        url = p.get_models_url(cfg)
        assert url == "https://api.anthropic.com/v1/models"


# ── 9. Shared Utilities ──────────────────────────────────────────────────────


class TestSharedUtilities:
    """Verify shared provider utilities."""

    def test_has_tool_calling_with_tools(self) -> None:
        assert has_tool_calling({"tools": [{"type": "function"}]})

    def test_has_tool_calling_with_tool_choice(self) -> None:
        assert has_tool_calling({"tool_choice": "auto"})

    def test_has_tool_calling_with_functions(self) -> None:
        assert has_tool_calling({"functions": [{"name": "f"}]})

    def test_has_tool_calling_with_function_call(self) -> None:
        assert has_tool_calling({"function_call": "auto"})

    def test_has_tool_calling_empty(self) -> None:
        assert not has_tool_calling({"messages": []})

    def test_has_tool_calling_none_values(self) -> None:
        assert not has_tool_calling({"tools": None, "tool_choice": None})

    def test_build_openai_body_sets_model(self) -> None:
        body = build_openai_body("gpt-4", {"messages": [], "temperature": 0.7})
        assert body["model"] == "gpt-4"
        assert body["temperature"] == 0.7

    def test_build_openai_body_overwrites_model(self) -> None:
        body = build_openai_body("new-model", {"model": "old-model", "messages": []})
        assert body["model"] == "new-model"

    def test_provider_error_fields(self) -> None:
        err = ProviderError("groq", 429, "Rate limited", retry_after=30.0)
        assert err.provider == "groq"
        assert err.status == 429
        assert err.retry_after == 30.0
        assert "groq" in str(err)
        assert "429" in str(err)


# ── 10. send_to_provider gateway ──────────────────────────────────────────────


class TestSendToProvider:
    """Verify the send_to_provider routing function."""

    def test_routes_to_correct_provider(self) -> None:
        client = _make_client({"id": "1"})
        cfg = _make_provider_config("groq")

        result = _run(send_to_provider(client, cfg, "llama3", {
            "messages": [{"role": "user", "content": "hi"}],
        }))
        assert result == {"id": "1"}

    def test_rejects_tools_for_special_providers(self) -> None:
        """Special providers should reject tool-calling requests."""
        client = _make_client()
        for name in ("cloudflare", "huggingface", "cohere", "google_gemini", "kilo", "anthropic"):
            cfg = _make_provider_config(name)
            with pytest.raises(ProviderError, match="does not support tool calling"):
                _run(send_to_provider(client, cfg, "model", {
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "f"}}],
                }))

    def test_allows_tools_for_openai_compat(self) -> None:
        """OpenAI-compatible providers should accept tool-calling requests."""
        client = _make_client({"id": "1"})
        cfg = _make_provider_config("groq")

        result = _run(send_to_provider(client, cfg, "llama3", {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }))
        assert result == {"id": "1"}

    def test_unknown_provider_raises(self) -> None:
        client = _make_client()
        cfg = _make_provider_config("nonexistent_provider_xyz")

        with pytest.raises(ProviderError, match="Unknown provider"):
            _run(send_to_provider(client, cfg, "model", {
                "messages": [{"role": "user", "content": "hi"}],
            }))
