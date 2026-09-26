"""Tests for local LLM last-resort fallback."""

from config import AppConfig, ModelConfig, ModelFallback, ProviderConfig, local_llm_enabled, local_llm_model
from rate_limiter import RateLimiter
from router import Router


def _make_router(
    *,
    cloud_providers: dict[str, ProviderConfig] | None = None,
    local: ProviderConfig | None = None,
    models: dict[str, ModelConfig] | None = None,
) -> Router:
    providers = dict(cloud_providers or {})
    if local is not None:
        providers["local"] = local
    cfg = AppConfig(
        master_key="test",
        host="127.0.0.1",
        port=8080,
        default_rpm_limit=0,
        providers=providers,
        models=models or {
            "llama-test": ModelConfig(
                unified_name="llama-test",
                fallbacks=[
                    ModelFallback(provider="groq", model="llama-3.3-70b"),
                    ModelFallback(provider="cerebras", model="llama-3.3-70b"),
                ],
            )
        },
    )
    return Router(cfg, RateLimiter())


class TestLocalLlmConfigHelpers:
    def test_disabled_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
        monkeypatch.delenv("LOCAL_LLM_ENABLED", raising=False)
        assert local_llm_enabled() is False
        assert local_llm_model() == ""

    def test_enabled_when_model_set(self, monkeypatch) -> None:
        monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3.2")
        assert local_llm_enabled() is True
        assert local_llm_model() == "llama3.2"

    def test_enabled_via_flag(self, monkeypatch) -> None:
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
        monkeypatch.setenv("LOCAL_LLM_ENABLED", "true")
        assert local_llm_enabled() is True


class TestLocalFallbackChain:
    def test_appends_local_at_end(self, monkeypatch) -> None:
        monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3.2")
        r = _make_router(
            local=ProviderConfig(name="local", base_url="http://127.0.0.1:11434/v1", api_keys=["local"]),
        )
        fallbacks = r.get_fallbacks("llama-test")
        assert [fb.provider for fb in fallbacks] == ["groq", "cerebras", "local"]
        assert fallbacks[-1].model == "llama3.2"

    def test_does_not_duplicate_local(self, monkeypatch) -> None:
        monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3.2")
        r = _make_router(
            local=ProviderConfig(name="local", base_url="http://127.0.0.1:11434/v1", api_keys=["local"]),
            models={
                "llama-test": ModelConfig(
                    unified_name="llama-test",
                    fallbacks=[
                        ModelFallback(provider="groq", model="llama-3.3-70b"),
                        ModelFallback(provider="local", model="custom-local"),
                    ],
                )
            },
        )
        fallbacks = r.get_fallbacks("llama-test")
        assert sum(1 for fb in fallbacks if fb.provider == "local") == 1
        assert fallbacks[-1].model == "custom-local"

    def test_skips_when_disabled(self, monkeypatch) -> None:
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
        monkeypatch.delenv("LOCAL_LLM_ENABLED", raising=False)
        r = _make_router(
            local=ProviderConfig(name="local", base_url="http://127.0.0.1:11434/v1", api_keys=["local"]),
        )
        fallbacks = r.get_fallbacks("llama-test")
        assert all(fb.provider != "local" for fb in fallbacks)

    def test_local_stays_last_after_round_robin(self, monkeypatch) -> None:
        monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3.2")
        groq = ProviderConfig(name="groq", base_url="https://api.groq.com", api_keys=["g"])
        cerebras = ProviderConfig(name="cerebras", base_url="https://api.cerebras.ai", api_keys=["c"])
        local = ProviderConfig(name="local", base_url="http://127.0.0.1:11434/v1", api_keys=["local"])
        r = _make_router(
            cloud_providers={"groq": groq, "cerebras": cerebras},
            local=local,
        )
        fallbacks = r.get_fallbacks("llama-test")

        # Force round-robin to rotate cloud providers across calls
        first = r._select_provider("llama-test", fallbacks)
        second = r._select_provider("llama-test", fallbacks)
        assert first[-1][0].name == "local"
        assert second[-1][0].name == "local"
        assert {c[0].name for c in first[:-1]} == {"groq", "cerebras"}

    def test_load_providers_injects_placeholder_key(self, monkeypatch) -> None:
        monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3.2")
        monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
        monkeypatch.delenv("LOCAL_LLM_KEY", raising=False)
        from config import _load_providers

        providers = _load_providers()
        assert "local" in providers
        assert providers["local"].api_key == "local"
        assert providers["local"].base_url == "http://127.0.0.1:1234/v1"


class TestPreferredConnection:
    def _router(self, monkeypatch) -> Router:
        monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3.2")
        return _make_router(
            local=ProviderConfig(name="local", base_url="http://127.0.0.1:11434/v1", api_keys=["local"]),
        )

    def test_default_keeps_local_as_last_resort(self, monkeypatch) -> None:
        r = self._router(monkeypatch)
        chain = r.apply_preferred_connection(r.get_fallbacks("llama-test"), None)
        assert [fb.provider for fb in chain] == ["groq", "cerebras", "local"]
        assert chain[-1].model == "llama3.2"

    def test_preferred_local_goes_first_then_cloud(self, monkeypatch) -> None:
        r = self._router(monkeypatch)
        chain = r.apply_preferred_connection(r.get_fallbacks("llama-test"), "local")
        assert [fb.provider for fb in chain] == ["local", "groq", "cerebras"]
        assert chain[0].model == "llama3.2"

    def test_select_provider_honors_preferred_local(self, monkeypatch) -> None:
        monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3.2")
        groq = ProviderConfig(name="groq", base_url="https://api.groq.com", api_keys=["g"])
        local = ProviderConfig(name="local", base_url="http://127.0.0.1:11434/v1", api_keys=["local"])
        r = _make_router(cloud_providers={"groq": groq}, local=local)
        fallbacks = r.apply_preferred_connection(r.get_fallbacks("llama-test"), "local")
        selected = r._select_provider(
            "llama-test", fallbacks, preferred_connection="local",
        )
        assert selected[0][0].name == "local"
        assert selected[0][1] == "llama3.2"

    def test_preferred_cloud_provider_moves_front(self, monkeypatch) -> None:
        r = self._router(monkeypatch)
        chain = r.apply_preferred_connection(r.get_fallbacks("llama-test"), "cerebras")
        assert [fb.provider for fb in chain] == ["cerebras", "groq", "local"]

    def test_select_provider_pins_preferred_across_round_robin(self, monkeypatch) -> None:
        monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3.2")
        groq = ProviderConfig(name="groq", base_url="https://api.groq.com", api_keys=["g"])
        cerebras = ProviderConfig(name="cerebras", base_url="https://api.cerebras.ai", api_keys=["c"])
        local = ProviderConfig(name="local", base_url="http://127.0.0.1:11434/v1", api_keys=["local"])
        r = _make_router(cloud_providers={"groq": groq, "cerebras": cerebras}, local=local)
        fallbacks = r.apply_preferred_connection(r.get_fallbacks("llama-test"), "groq")
        first = r._select_provider("llama-test", fallbacks, preferred_connection="groq")
        second = r._select_provider("llama-test", fallbacks, preferred_connection="groq")
        assert first[0][0].name == "groq"
        assert second[0][0].name == "groq"

    def test_keeps_local_when_it_is_the_only_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("LOCAL_LLM_MODEL", "nomic-embed-text")
        r = _make_router(
            local=ProviderConfig(name="local", base_url="http://127.0.0.1:11434/v1", api_keys=["local"]),
            models={
                "nomic-embed-text": ModelConfig(
                    unified_name="nomic-embed-text",
                    fallbacks=[ModelFallback(provider="local", model="nomic-embed-text")],
                )
            },
        )
        chain = r.apply_preferred_connection(r.get_fallbacks("nomic-embed-text"), None)
        assert [fb.provider for fb in chain] == ["local"]

