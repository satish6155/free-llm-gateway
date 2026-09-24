"""Groq gpt-oss request shaping for openai/gpt-oss-* models."""

from __future__ import annotations

import os
from unittest.mock import patch

from providers import _build_openai_body, _is_groq_gpt_oss


class TestGroqGptOssBody:
    def test_detects_groq_gpt_oss_models(self) -> None:
        assert _is_groq_gpt_oss("groq", "openai/gpt-oss-20b")
        assert _is_groq_gpt_oss("groq", "openai/gpt-oss-120b")
        assert not _is_groq_gpt_oss("openrouter", "openai/gpt-oss-20b:free")
        assert not _is_groq_gpt_oss("groq", "llama-3.3-70b-versatile")

    def test_maps_max_tokens_and_sets_reasoning_effort(self) -> None:
        body = _build_openai_body(
            "openai/gpt-oss-20b",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "preferred_connection": "cloud",
            },
            provider_name="groq",
        )
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["max_completion_tokens"] == 100
        assert "max_tokens" not in body
        assert body["reasoning_effort"] == "low"
        assert "preferred_connection" not in body

    def test_preserves_explicit_reasoning_effort(self) -> None:
        body = _build_openai_body(
            "openai/gpt-oss-20b",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 50,
                "reasoning_effort": "medium",
            },
            provider_name="groq",
        )
        assert body["reasoning_effort"] == "medium"
        assert body["max_completion_tokens"] == 50

    def test_other_providers_unchanged(self) -> None:
        body = _build_openai_body(
            "openai/gpt-oss-20b:free",
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
            provider_name="openrouter",
        )
        assert body["max_tokens"] == 100
        assert "max_completion_tokens" not in body
        assert "reasoning_effort" not in body

    def test_reasoning_effort_env_override(self) -> None:
        with patch.dict(os.environ, {"GROQ_GPT_OSS_REASONING_EFFORT": "high"}):
            body = _build_openai_body(
                "openai/gpt-oss-20b",
                {"messages": [{"role": "user", "content": "hi"}]},
                provider_name="groq",
            )
        assert body["reasoning_effort"] == "high"
