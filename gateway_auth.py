"""Unified gateway API key management.

Provides a single API key that external clients use to authenticate
with the gateway, instead of needing provider-specific keys.
Supports multiple user keys with different permissions and rate limits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet

from key_encryptor import KeyEncryptor

logger = logging.getLogger(__name__)

GATEWAY_KEYS_FILE = Path(__file__).parent / "data" / "gateway_keys.json"


@dataclass
class GatewayKeyPermissions:
    """Permissions for a gateway API key."""

    allowed_models: list[str] = field(default_factory=list)  # empty = all
    allowed_providers: list[str] = field(default_factory=list)  # empty = all
    max_rpm: int = 0  # 0 = unlimited
    max_tpm: int = 0  # 0 = unlimited
    is_admin: bool = False


@dataclass
class GatewayKey:
    """A gateway API key with metadata and permissions."""

    key_hash: str  # SHA-256 hash of the actual key
    name: str
    created_at: float
    last_used_at: float = 0.0
    request_count: int = 0
    enabled: bool = True
    permissions: GatewayKeyPermissions = field(default_factory=GatewayKeyPermissions)

    def to_dict(self, include_hash: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "request_count": self.request_count,
            "enabled": self.enabled,
            "permissions": {
                "allowed_models": self.permissions.allowed_models,
                "allowed_providers": self.permissions.allowed_providers,
                "max_rpm": self.permissions.max_rpm,
                "max_tpm": self.permissions.max_tpm,
                "is_admin": self.permissions.is_admin,
            },
        }
        if include_hash:
            d["key_hash"] = self.key_hash
        return d


class GatewayAuthManager:
    """Manages unified gateway API keys.

    Features:
    - Generate/revoke gateway API keys
    - Validate incoming requests against gateway keys
    - Per-key rate limiting (RPM, TPM)
    - Per-key model/provider restrictions
    - Encrypted storage
    """

    GATEWAY_KEY_PREFIX = "fgk-"  # free-gateway-key

    def __init__(self, encryption_key: str = "") -> None:
        self._keys: dict[str, GatewayKey] = {}  # key_hash -> GatewayKey
        self._encryptor = KeyEncryptor(encryption_key) if encryption_key else None
        self._load()

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def generate_key() -> str:
        """Generate a new gateway API key."""
        raw = secrets.token_urlsafe(32)
        return f"{GatewayAuthManager.GATEWAY_KEY_PREFIX}{raw}"

    def create_key(
        self,
        name: str,
        allowed_models: list[str] | None = None,
        allowed_providers: list[str] | None = None,
        max_rpm: int = 0,
        max_tpm: int = 0,
        is_admin: bool = False,
    ) -> tuple[str, GatewayKey]:
        """Create a new gateway API key. Returns (raw_key, GatewayKey)."""
        raw_key = self.generate_key()
        key_hash = self._hash_key(raw_key)
        now = time.time()

        gw_key = GatewayKey(
            key_hash=key_hash,
            name=name,
            created_at=now,
            last_used_at=0.0,
            request_count=0,
            enabled=True,
            permissions=GatewayKeyPermissions(
                allowed_models=allowed_models or [],
                allowed_providers=allowed_providers or [],
                max_rpm=max_rpm,
                max_tpm=max_tpm,
                is_admin=is_admin,
            ),
        )

        self._keys[key_hash] = gw_key
        self._save()
        logger.info("Created gateway key: %s", name)
        return raw_key, gw_key

    def validate_key(self, raw_key: str) -> GatewayKey | None:
        """Validate a gateway API key. Returns GatewayKey if valid."""
        if not raw_key.startswith(self.GATEWAY_KEY_PREFIX):
            return None

        key_hash = self._hash_key(raw_key)
        gw_key = self._keys.get(key_hash)

        if not gw_key:
            return None
        if not gw_key.enabled:
            return None

        # Update usage stats
        gw_key.last_used_at = time.time()
        gw_key.request_count += 1
        self._save()

        return gw_key

    def revoke_key(self, name: str) -> bool:
        """Revoke a gateway key by name. Returns True if found."""
        for key_hash, gw_key in list(self._keys.items()):
            if gw_key.name == name:
                del self._keys[key_hash]
                self._save()
                logger.info("Revoked gateway key: %s", name)
                return True
        return False

    def toggle_key(self, name: str, enabled: bool) -> bool:
        """Enable or disable a gateway key. Returns True if found."""
        for gw_key in self._keys.values():
            if gw_key.name == name:
                gw_key.enabled = enabled
                self._save()
                return True
        return False

    def list_keys(self) -> list[dict[str, Any]]:
        """List all gateway keys (without raw key values)."""
        return [gw.to_dict(include_hash=False) for gw in self._keys.values()]

    def get_key_by_name(self, name: str) -> GatewayKey | None:
        """Get a gateway key by name."""
        for gw_key in self._keys.values():
            if gw_key.name == name:
                return gw_key
        return None

    def can_access_model(self, gw_key: GatewayKey, model: str) -> bool:
        """Check if a gateway key can access a specific model."""
        if gw_key.permissions.is_admin:
            return True
        if not gw_key.permissions.allowed_models:
            return True  # empty = all allowed
        return model in gw_key.permissions.allowed_models

    def can_access_provider(self, gw_key: GatewayKey, provider: str) -> bool:
        """Check if a gateway key can access a specific provider."""
        if gw_key.permissions.is_admin:
            return True
        if not gw_key.permissions.allowed_providers:
            return True
        return provider in gw_key.permissions.allowed_providers

    def _load(self) -> None:
        if not GATEWAY_KEYS_FILE.exists():
            self._keys = {}
            return
        try:
            raw = GATEWAY_KEYS_FILE.read_text()
            if self._encryptor:
                raw = self._encryptor.decrypt(raw)
            data = json.loads(raw)
            self._keys = {}
            for key_hash, key_data in data.items():
                perms = key_data.get("permissions", {})
                self._keys[key_hash] = GatewayKey(
                    key_hash=key_hash,
                    name=key_data.get("name", ""),
                    created_at=key_data.get("created_at", 0),
                    last_used_at=key_data.get("last_used_at", 0),
                    request_count=key_data.get("request_count", 0),
                    enabled=key_data.get("enabled", True),
                    permissions=GatewayKeyPermissions(
                        allowed_models=perms.get("allowed_models", []),
                        allowed_providers=perms.get("allowed_providers", []),
                        max_rpm=perms.get("max_rpm", 0),
                        max_tpm=perms.get("max_tpm", 0),
                        is_admin=perms.get("is_admin", False),
                    ),
                )
            logger.info("Loaded %d gateway keys", len(self._keys))
        except Exception as e:
            logger.warning("Failed to load gateway keys: %s", e)
            self._keys = {}

    def _save(self) -> None:
        GATEWAY_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {kh: gw.to_dict(include_hash=True) for kh, gw in self._keys.items()}
        raw = json.dumps(data, indent=2)
        if self._encryptor:
            raw = self._encryptor.encrypt(raw)
        GATEWAY_KEYS_FILE.write_text(raw)

    def get_stats(self) -> dict[str, Any]:
        """Get gateway auth statistics."""
        total = len(self._keys)
        active = sum(1 for k in self._keys.values() if k.enabled)
        admins = sum(1 for k in self._keys.values() if k.permissions.is_admin)
        total_requests = sum(k.request_count for k in self._keys.values())
        return {
            "total_keys": total,
            "active_keys": active,
            "admin_keys": admins,
            "total_requests": total_requests,
        }


# Global singleton — initialized in main.py after config load
gateway_auth: GatewayAuthManager | None = None
