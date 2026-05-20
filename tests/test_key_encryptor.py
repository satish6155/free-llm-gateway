"""Tests for AES-256-GCM encrypted key storage."""

import json
import tempfile
from pathlib import Path
import pytest
from key_encryptor import KeyEncryptor, EncryptedKeyStore


class TestKeyEncryptor:
    def test_encrypt_decrypt_roundtrip(self) -> None:
        """Should encrypt and decrypt a string correctly."""
        enc = KeyEncryptor("test-secret")
        original = "sk-1234567890abcdef"
        encrypted = enc.encrypt(original)
        decrypted = enc.decrypt(encrypted)
        assert decrypted == original

    def test_different_secrets_fail(self) -> None:
        """Decryption with wrong secret should fail."""
        enc1 = KeyEncryptor("secret-1")
        enc2 = KeyEncryptor("secret-2")
        encrypted = enc1.encrypt("test-key")
        with pytest.raises(Exception):
            enc2.decrypt(encrypted)

    def test_same_input_different_ciphertext(self) -> None:
        """Same input should produce different ciphertext (random nonce)."""
        enc = KeyEncryptor("test-secret")
        original = "sk-same-key"
        enc1 = enc.encrypt(original)
        enc2 = enc.encrypt(original)
        assert enc1 != enc2  # Different nonces → different ciphertext
        assert enc.decrypt(enc1) == original
        assert enc.decrypt(enc2) == original

    def test_empty_string(self) -> None:
        """Should handle empty string."""
        enc = KeyEncryptor("test-secret")
        encrypted = enc.encrypt("")
        assert enc.decrypt(encrypted) == ""

    def test_long_string(self) -> None:
        """Should handle long strings."""
        enc = KeyEncryptor("test-secret")
        original = "x" * 10000
        encrypted = enc.encrypt(original)
        assert enc.decrypt(encrypted) == original

    def test_unicode(self) -> None:
        """Should handle unicode characters."""
        enc = KeyEncryptor("test-secret")
        original = "sk-密钥-🔑"
        encrypted = enc.encrypt(original)
        assert enc.decrypt(encrypted) == original


class TestEncryptedKeyStore:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.store_file = Path(self.tmpdir) / "test_keys.json"
        # Monkey-patch the file path
        import key_encryptor
        self._original_file = key_encryptor.ENCRYPTED_KEYS_FILE
        key_encryptor.ENCRYPTED_KEYS_FILE = self.store_file
        self.store = EncryptedKeyStore("master-secret")

    def teardown_method(self) -> None:
        import key_encryptor
        key_encryptor.ENCRYPTED_KEYS_FILE = self._original_file
        if self.store_file.exists():
            self.store_file.unlink()

    def test_add_and_get_key(self) -> None:
        """Should add and retrieve a key."""
        self.store.add_key("groq", "gsk-test-key-123")
        keys = self.store.get_keys("groq")
        assert len(keys) == 1
        assert keys[0] == "gsk-test-key-123"

    def test_multiple_keys(self) -> None:
        """Should handle multiple keys per provider."""
        self.store.add_key("groq", "key-1")
        self.store.add_key("groq", "key-2")
        self.store.add_key("groq", "key-3")
        keys = self.store.get_keys("groq")
        assert len(keys) == 3

    def test_remove_key(self) -> None:
        """Should remove a key by index."""
        self.store.add_key("groq", "key-1")
        self.store.add_key("groq", "key-2")
        assert self.store.remove_key("groq", 0)
        keys = self.store.get_keys("groq")
        assert len(keys) == 1
        assert keys[0] == "key-2"

    def test_list_keys_masked(self) -> None:
        """Should list keys with masked values."""
        self.store.add_key("groq", "gsk-abcdefghijklmnop")
        listed = self.store.list_keys()
        assert "groq" in listed
        assert listed["groq"][0]["key_masked"].endswith("mnop")
        assert "gsk-abcdefghijklmnop" not in str(listed)

    def test_persistence(self) -> None:
        """Should persist and reload keys."""
        self.store.add_key("groq", "persistent-key-123")

        # Create new store instance to test persistence
        store2 = EncryptedKeyStore("master-secret")
        keys = store2.get_keys("groq")
        assert len(keys) == 1
        assert keys[0] == "persistent-key-123"

    def test_has_provider(self) -> None:
        """Should check if provider has keys."""
        assert not self.store.has_provider("groq")
        self.store.add_key("groq", "test-key")
        assert self.store.has_provider("groq")

    def test_set_validated(self) -> None:
        """Should update validation status."""
        self.store.add_key("groq", "test-key")
        self.store.set_validated("groq", 0, True)
        listed = self.store.list_keys()
        assert listed["groq"][0]["validated"] is True

    def test_get_first_key(self) -> None:
        """Should return first key or None."""
        assert self.store.get_first_key("groq") is None
        self.store.add_key("groq", "first-key")
        assert self.store.get_first_key("groq") == "first-key"

    def test_file_is_encrypted(self) -> None:
        """Stored file should not contain plaintext keys."""
        self.store.add_key("groq", "plaintext-secret-key-12345")
        content = self.store_file.read_text()
        assert "plaintext-secret-key-12345" not in content
