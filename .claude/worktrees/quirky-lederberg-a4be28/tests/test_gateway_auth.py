"""Tests for unified gateway API key management."""

import pytest
from gateway_auth import GatewayAuthManager


class TestGatewayAuth:
    def setup_method(self) -> None:
        import gateway_auth
        import tempfile
        from pathlib import Path
        self._original_file = gateway_auth.GATEWAY_KEYS_FILE
        self._tmpdir = tempfile.mkdtemp()
        gateway_auth.GATEWAY_KEYS_FILE = Path(self._tmpdir) / "test_gw_keys.json"
        self.auth = GatewayAuthManager(encryption_key="test-secret-key-for-testing")

    def teardown_method(self) -> None:
        import gateway_auth
        gateway_auth.GATEWAY_KEYS_FILE = self._original_file
        if hasattr(self, '_tmpdir'):
            import shutil
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_generate_key(self) -> None:
        """Should generate a key with correct prefix."""
        raw = GatewayAuthManager.generate_key()
        assert raw.startswith("fgk-")
        assert len(raw) > 20

    def test_create_and_validate_key(self) -> None:
        """Should create and validate a gateway key."""
        raw_key, gw_key = self.auth.create_key("test-user")
        assert gw_key.name == "test-user"
        assert gw_key.enabled

        validated = self.auth.validate_key(raw_key)
        assert validated is not None
        assert validated.name == "test-user"

    def test_validate_wrong_prefix(self) -> None:
        """Should reject keys without correct prefix."""
        result = self.auth.validate_key("sk-wrong-prefix")
        assert result is None

    def test_validate_unknown_key(self) -> None:
        """Should reject unknown keys."""
        result = self.auth.validate_key("fgk-nonexistentkey123456789")
        assert result is None

    def test_revoke_key(self) -> None:
        """Should revoke a key by name."""
        raw_key, _ = self.auth.create_key("revoke-me")
        assert self.auth.revoke_key("revoke-me")
        assert self.auth.validate_key(raw_key) is None

    def test_revoke_nonexistent(self) -> None:
        """Should return False for nonexistent key."""
        assert not self.auth.revoke_key("nonexistent")

    def test_toggle_key(self) -> None:
        """Should enable/disable a key."""
        raw_key, _ = self.auth.create_key("toggle-me")
        assert self.auth.toggle_key("toggle-me", False)
        assert self.auth.validate_key(raw_key) is None  # disabled

        assert self.auth.toggle_key("toggle-me", True)
        assert self.auth.validate_key(raw_key) is not None  # re-enabled

    def test_list_keys(self) -> None:
        """Should list keys without raw values."""
        self.auth.create_key("user-1")
        self.auth.create_key("user-2")
        keys = self.auth.list_keys()
        assert len(keys) == 2
        # Should not contain raw key or hash
        for k in keys:
            assert "key_hash" not in k
            assert "fgk-" not in str(k)

    def test_can_access_model_unrestricted(self) -> None:
        """Unrestricted keys should access all models."""
        _, gw_key = self.auth.create_key("open-user")
        assert self.auth.can_access_model(gw_key, "gpt-4")
        assert self.auth.can_access_model(gw_key, "claude-3")

    def test_can_access_model_restricted(self) -> None:
        """Restricted keys should only access allowed models."""
        _, gw_key = self.auth.create_key(
            "restricted-user",
            allowed_models=["gpt-4", "gpt-3.5-turbo"],
        )
        assert self.auth.can_access_model(gw_key, "gpt-4")
        assert not self.auth.can_access_model(gw_key, "claude-3")

    def test_can_access_provider_unrestricted(self) -> None:
        """Unrestricted keys should access all providers."""
        _, gw_key = self.auth.create_key("open-user")
        assert self.auth.can_access_provider(gw_key, "groq")
        assert self.auth.can_access_provider(gw_key, "cerebras")

    def test_can_access_provider_restricted(self) -> None:
        """Restricted keys should only access allowed providers."""
        _, gw_key = self.auth.create_key(
            "provider-restricted",
            allowed_providers=["groq", "cerebras"],
        )
        assert self.auth.can_access_provider(gw_key, "groq")
        assert not self.auth.can_access_provider(gw_key, "openai")

    def test_admin_bypasses_restrictions(self) -> None:
        """Admin keys should bypass all restrictions."""
        _, gw_key = self.auth.create_key(
            "admin-user",
            allowed_models=["gpt-4"],
            is_admin=True,
        )
        assert self.auth.can_access_model(gw_key, "claude-3")  # not in list but admin
        assert self.auth.can_access_provider(gw_key, "any-provider")

    def test_get_stats(self) -> None:
        """Should return auth statistics."""
        self.auth.create_key("user-a")
        self.auth.create_key("admin-a", is_admin=True)
        stats = self.auth.get_stats()
        assert stats["total_keys"] == 2
        assert stats["admin_keys"] == 1

    def test_request_count_increments(self) -> None:
        """Should increment request count on each validation."""
        raw_key, _ = self.auth.create_key("counter-user")
        self.auth.validate_key(raw_key)
        self.auth.validate_key(raw_key)
        self.auth.validate_key(raw_key)
        gw = self.auth.get_key_by_name("counter-user")
        assert gw is not None
        assert gw.request_count == 3
