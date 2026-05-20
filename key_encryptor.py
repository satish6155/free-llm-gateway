"""AES-256-GCM encrypted key storage.

Upgrades key management from Fernet (AES-128-CBC) to AES-256-GCM for
stronger encryption with authentication. Provides authenticated encryption
that protects both confidentiality and integrity of stored API keys.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

ENCRYPTED_KEYS_FILE = Path(__file__).parent / "data" / "encrypted_keys.json"
NONCE_SIZE = 12  # 96-bit nonce for AES-GCM


class KeyEncryptor:
    """AES-256-GCM encryptor for API key storage.

    Derives a 256-bit key from a master secret using SHA-256,
    then uses AES-256-GCM for authenticated encryption.
    """

    def __init__(self, master_secret: str) -> None:
        self._key = hashlib.sha256(master_secret.encode()).digest()
        self._aesgcm = AESGCM(self._key)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string. Returns base64-encoded nonce+ciphertext."""
        nonce = os.urandom(NONCE_SIZE)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode(), None)
        # Prepend nonce to ciphertext for storage
        combined = nonce + ciphertext
        return base64.b64encode(combined).decode()

    def decrypt(self, encrypted: str) -> str:
        """Decrypt a base64-encoded nonce+ciphertext string."""
        combined = base64.b64decode(encrypted)
        nonce = combined[:NONCE_SIZE]
        ciphertext = combined[NONCE_SIZE:]
        plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode()


class EncryptedKeyStore:
    """Manages encrypted API keys stored in data/encrypted_keys.json.

    Uses AES-256-GCM for encryption at rest. Each key is encrypted
    individually with a unique nonce.
    """

    def __init__(self, master_secret: str) -> None:
        self._encryptor = KeyEncryptor(master_secret)
        self._keys: dict[str, list[dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if not ENCRYPTED_KEYS_FILE.exists():
            self._keys = {}
            return
        try:
            raw = json.loads(ENCRYPTED_KEYS_FILE.read_text())
            for provider, entries in raw.items():
                self._keys[provider] = []
                for entry in entries:
                    decrypted = self._encryptor.decrypt(entry["key"])
                    self._keys[provider].append({
                        "key": decrypted,
                        "validated": entry.get("validated", False),
                        "added_at": entry.get("added_at", ""),
                    })
            logger.info(
                "Loaded %d encrypted keys for %d providers",
                sum(len(v) for v in self._keys.values()),
                len(self._keys),
            )
        except Exception as e:
            logger.warning("Failed to load encrypted keys: %s", e)
            self._keys = {}

    def _save(self) -> None:
        ENCRYPTED_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, list[dict[str, Any]]] = {}
        for provider, entries in self._keys.items():
            data[provider] = []
            for entry in entries:
                encrypted = self._encryptor.encrypt(entry["key"])
                data[provider].append({
                    "key": encrypted,
                    "validated": entry.get("validated", False),
                    "added_at": entry.get("added_at", ""),
                })
        ENCRYPTED_KEYS_FILE.write_text(json.dumps(data, indent=2))

    def add_key(self, provider: str, api_key: str) -> int:
        """Add a key for a provider. Returns index of new key."""
        if provider not in self._keys:
            self._keys[provider] = []
        from datetime import datetime, timezone
        self._keys[provider].append({
            "key": api_key,
            "validated": False,
            "added_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        return len(self._keys[provider]) - 1

    def remove_key(self, provider: str, index: int) -> bool:
        """Remove a key by provider and index."""
        entries = self._keys.get(provider, [])
        if index < 0 or index >= len(entries):
            return False
        entries.pop(index)
        if not entries:
            del self._keys[provider]
        self._save()
        return True

    def list_keys(self) -> dict[str, list[dict[str, Any]]]:
        """List all keys with masked values."""
        result: dict[str, list[dict[str, Any]]] = {}
        for provider, entries in self._keys.items():
            result[provider] = []
            for i, entry in enumerate(entries):
                key = entry["key"]
                masked = f"****{key[-4:]}" if len(key) > 4 else "****"
                result[provider].append({
                    "index": i,
                    "key_masked": masked,
                    "validated": entry.get("validated", False),
                    "added_at": entry.get("added_at", ""),
                })
        return result

    def get_keys(self, provider: str) -> list[str]:
        """Get all raw API keys for a provider."""
        return [e["key"] for e in self._keys.get(provider, [])]

    def get_first_key(self, provider: str) -> str | None:
        """Get the first available key for a provider."""
        keys = self.get_keys(provider)
        return keys[0] if keys else None

    def set_validated(self, provider: str, index: int, valid: bool) -> None:
        """Update validation status."""
        entries = self._keys.get(provider, [])
        if 0 <= index < len(entries):
            entries[index]["validated"] = valid
            self._save()

    def has_provider(self, provider: str) -> bool:
        return bool(self._keys.get(provider))

    def migrate_from_fernet(self, fernet_keys: dict[str, list[dict[str, Any]]]) -> int:
        """Migrate keys from Fernet-encrypted format to AES-256-GCM.

        Args:
            fernet_keys: Dict of provider -> list of key entries with decrypted "key" field

        Returns:
            Number of keys migrated
        """
        migrated = 0
        for provider, entries in fernet_keys.items():
            for entry in entries:
                if "key" in entry:
                    self.add_key(provider, entry["key"])
                    migrated += 1
        logger.info("Migrated %d keys from Fernet to AES-256-GCM", migrated)
        return migrated
