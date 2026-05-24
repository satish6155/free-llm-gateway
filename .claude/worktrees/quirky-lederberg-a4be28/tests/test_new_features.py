"""Tests for the 4 features ported from FreeLLMAPI.

1. Timing-safe key comparison
2. Pydantic input validation
3. Dynamic penalty routing
4. Cooldown system
"""

import hmac
import time
from unittest.mock import MagicMock, patch

import pytest

from rate_tracker import PerKeyRateTracker
from router import PenaltyTracker, PENALTY_PER_429, MAX_PENALTY, DECAY_INTERVAL_S


# ── 1. Timing-safe key comparison ────────────────────────────────────────────


class TestTimingSafeKeyComparison:
    """Verify hmac.compare_digest is used for key validation."""

    def test_verify_master_key_uses_compare_digest(self) -> None:
        """verify_master_key should use hmac.compare_digest, not ==."""
        import inspect
        from routes.state import verify_master_key

        source = inspect.getsource(verify_master_key)
        assert "hmac.compare_digest" in source
        # Should NOT contain naive == comparison on the token
        assert "token ==" not in source
        assert "token==" not in source

    def test_gateway_auth_uses_compare_digest(self) -> None:
        """GatewayAuthManager.validate_key should use hmac.compare_digest."""
        import inspect
        from gateway_auth import GatewayAuthManager

        source = inspect.getsource(GatewayAuthManager.validate_key)
        assert "hmac.compare_digest" in source

    def test_compare_digest_rejects_wrong_key(self) -> None:
        """hmac.compare_digest correctly rejects mismatched keys."""
        assert not hmac.compare_digest("correct-key", "wrong-key")

    def test_compare_digest_accepts_matching_key(self) -> None:
        """hmac.compare_digest correctly accepts matching keys."""
        assert hmac.compare_digest("same-key", "same-key")


# ── 2. Pydantic input validation ─────────────────────────────────────────────


class TestPydanticInputValidation:
    """Verify ChatCompletionRequest validates input correctly."""

    def test_import_chat_completion_request(self) -> None:
        """ChatCompletionRequest should be importable from main."""
        from routes.chat import ChatCompletionRequest
        assert ChatCompletionRequest is not None

    def test_valid_minimal_request(self) -> None:
        """Minimal valid request should pass validation."""
        from routes.chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert len(req.messages) == 1
        assert req.temperature == 0.0
        assert req.stream is False

    def test_valid_full_request(self) -> None:
        """Full request with all fields should pass validation."""
        from routes.chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            stream=True,
        )
        assert req.model == "gpt-4"
        assert req.temperature == 0.7
        assert req.max_tokens == 100
        assert req.stream is True

    def test_empty_messages_rejected(self) -> None:
        """Empty messages list should be rejected."""
        from routes.chat import ChatCompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(messages=[])
        assert "at least 1" in str(exc_info.value).lower() or "min_length" in str(exc_info.value).lower()

    def test_temperature_out_of_range_rejected(self) -> None:
        """Temperature above 2 or below 0 should be rejected."""
        from routes.chat import ChatCompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=3.0,
            )

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=-0.5,
            )

    def test_max_tokens_must_be_positive(self) -> None:
        """max_tokens <= 0 should be rejected."""
        from routes.chat import ChatCompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=0,
            )

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=-10,
            )

    def test_top_p_out_of_range_rejected(self) -> None:
        """top_p above 1 or below 0 should be rejected."""
        from routes.chat import ChatCompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hi"}],
                top_p=1.5,
            )

    def test_invalid_role_rejected(self) -> None:
        """Messages with invalid roles should be rejected."""
        from routes.chat import ChatCompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[{"role": "invalid_role", "content": "Hi"}],
            )

    def test_stream_must_be_bool(self) -> None:
        """stream must be boolean — non-coerceable values rejected."""
        from routes.chat import ChatCompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hi"}],
                stream=[1, 2, 3],
            )


# ── 3. Dynamic penalty routing ───────────────────────────────────────────────


class TestPenaltyTracker:
    """Verify PenaltyTracker tracks 429 penalties with decay."""

    def setup_method(self) -> None:
        self.tracker = PenaltyTracker()

    def test_initial_penalty_is_zero(self) -> None:
        """No penalty for a model that hasn't been hit."""
        assert self.tracker.get_penalty("groq", "llama3") == 0

    def test_record_hit_increases_penalty(self) -> None:
        """Recording a 429 hit should increase penalty by PENALTY_PER_429."""
        self.tracker.record_hit("groq", "llama3")
        assert self.tracker.get_penalty("groq", "llama3") == PENALTY_PER_429

    def test_multiple_hits_stack(self) -> None:
        """Multiple hits should stack up to MAX_PENALTY."""
        for _ in range(5):
            self.tracker.record_hit("groq", "llama3")
        penalty = self.tracker.get_penalty("groq", "llama3")
        assert penalty <= MAX_PENALTY
        assert penalty > PENALTY_PER_429

    def test_penalty_caps_at_max(self) -> None:
        """Penalty should never exceed MAX_PENALTY."""
        for _ in range(20):
            self.tracker.record_hit("groq", "llama3")
        assert self.tracker.get_penalty("groq", "llama3") == MAX_PENALTY

    def test_record_success_reduces_penalty(self) -> None:
        """Recording success should reduce penalty by 1."""
        self.tracker.record_hit("groq", "llama3")
        assert self.tracker.get_penalty("groq", "llama3") == PENALTY_PER_429

        self.tracker.record_success("groq", "llama3")
        assert self.tracker.get_penalty("groq", "llama3") == PENALTY_PER_429 - 1

    def test_success_removes_zero_penalty(self) -> None:
        """When penalty reaches 0, the entry should be removed."""
        self.tracker.record_hit("groq", "llama3")
        for _ in range(PENALTY_PER_429):
            self.tracker.record_success("groq", "llama3")
        assert self.tracker.get_penalty("groq", "llama3") == 0

    def test_different_providers_independent(self) -> None:
        """Penalties for different providers should be independent."""
        self.tracker.record_hit("groq", "llama3")
        self.tracker.record_hit("cerebras", "llama3")
        self.tracker.record_hit("cerebras", "llama3")

        assert self.tracker.get_penalty("groq", "llama3") == PENALTY_PER_429
        assert self.tracker.get_penalty("cerebras", "llama3") == PENALTY_PER_429 * 2

    def test_get_all_penalties(self) -> None:
        """get_all_penalties should return sorted penalty list."""
        self.tracker.record_hit("groq", "llama3")
        self.tracker.record_hit("cerebras", "llama3")
        self.tracker.record_hit("cerebras", "llama3")

        penalties = self.tracker.get_all_penalties()
        assert len(penalties) == 2
        # Higher penalty first
        assert penalties[0]["penalty"] >= penalties[1]["penalty"]

    def test_time_decay_reduces_penalty(self) -> None:
        """Penalty should decay over time."""
        self.tracker.record_hit("groq", "llama3")
        assert self.tracker.get_penalty("groq", "llama3") == PENALTY_PER_429

        # Simulate time passing by manipulating last_hit
        key = "groq:llama3"
        entry = self.tracker._penalties[key]
        entry["last_hit"] = time.time() - DECAY_INTERVAL_S * 2  # 2 decay intervals

        penalty = self.tracker.get_penalty("groq", "llama3")
        assert penalty == PENALTY_PER_429 - 2  # -1 per decay interval


# ── 4. Cooldown system ───────────────────────────────────────────────────────


class TestCooldown:
    """Verify cooldown tracking in PerKeyRateTracker."""

    def setup_method(self) -> None:
        self.tracker = PerKeyRateTracker()

    def test_not_on_cooldown_by_default(self) -> None:
        """Keys should not be on cooldown initially."""
        assert not self.tracker.is_on_cooldown("groq", "llama3", "key1")

    def test_set_cooldown_activates(self) -> None:
        """Setting cooldown should make key show as on cooldown."""
        self.tracker.set_cooldown("groq", "llama3", "key1", duration_s=60)
        assert self.tracker.is_on_cooldown("groq", "llama3", "key1")

    def test_different_keys_independent(self) -> None:
        """Different keys should have independent cooldowns."""
        self.tracker.set_cooldown("groq", "llama3", "key1", duration_s=60)
        assert self.tracker.is_on_cooldown("groq", "llama3", "key1")
        assert not self.tracker.is_on_cooldown("groq", "llama3", "key2")

    def test_cooldown_expires(self) -> None:
        """Cooldown should expire after the specified duration."""
        self.tracker.set_cooldown("groq", "llama3", "key1", duration_s=60)

        # Simulate time passing by manipulating the stored expiry
        k = self.tracker._key("groq", "llama3", "key1")
        self.tracker._cooldowns[k] = time.time() - 1  # expired 1 second ago

        assert not self.tracker.is_on_cooldown("groq", "llama3", "key1")

    def test_default_cooldown_duration(self) -> None:
        """Default cooldown should be 30 seconds."""
        assert PerKeyRateTracker.DEFAULT_COOLDOWN_S == 30.0

    def test_cooldown_cleans_up_after_expiry(self) -> None:
        """Expired cooldown entries should be removed from the dict."""
        self.tracker.set_cooldown("groq", "llama3", "key1", duration_s=60)
        k = self.tracker._key("groq", "llama3", "key1")
        assert k in self.tracker._cooldowns

        # Expire it
        self.tracker._cooldowns[k] = time.time() - 1
        self.tracker.is_on_cooldown("groq", "llama3", "key1")

        # Should be cleaned up
        assert k not in self.tracker._cooldowns

    def test_cooldown_per_provider_model_key(self) -> None:
        """Cooldown should be specific to (provider, model, key) triple."""
        self.tracker.set_cooldown("groq", "llama3", "key1", duration_s=60)
        self.tracker.set_cooldown("groq", "mixtral", "key1", duration_s=60)

        assert self.tracker.is_on_cooldown("groq", "llama3", "key1")
        assert self.tracker.is_on_cooldown("groq", "mixtral", "key1")

    def test_max_retries_is_10(self) -> None:
        """MAX_RETRIES should be 10 for deep retry."""
        from router import MAX_RETRIES
        assert MAX_RETRIES == 10
