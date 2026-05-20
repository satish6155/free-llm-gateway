"""Tests for sticky session management."""

import time
import pytest
from sticky_sessions import StickySessionManager


class TestStickySessions:
    def setup_method(self) -> None:
        self.manager = StickySessionManager(default_ttl=300)

    def test_create_and_get_session(self) -> None:
        """Should store and retrieve session."""
        self.manager.set("conv-123", "groq", "llama3")
        provider, model = self.manager.get("conv-123")
        assert provider == "groq"
        assert model == "llama3"

    def test_no_session_returns_none(self) -> None:
        """Should return None for unknown conversation."""
        provider, model = self.manager.get("unknown-conv")
        assert provider is None
        assert model is None

    def test_empty_conversation_id(self) -> None:
        """Should handle empty conversation ID gracefully."""
        self.manager.set("", "groq", "llama3")
        provider, model = self.manager.get("")
        assert provider is None

    def test_session_expiry(self) -> None:
        """Expired sessions should return None."""
        self.manager.set("conv-exp", "groq", "llama3", ttl=0.01)
        time.sleep(0.02)
        provider, model = self.manager.get("conv-exp")
        assert provider is None

    def test_session_touch_updates_last_used(self) -> None:
        """Getting a session should update last_used_at."""
        self.manager.set("conv-touch", "groq", "llama3")
        _, _ = self.manager.get("conv-touch")
        time.sleep(0.01)
        _, _ = self.manager.get("conv-touch")
        sessions = self.manager.get_all()
        assert len(sessions) >= 1
        assert sessions[0]["request_count"] == 2

    def test_remove_session(self) -> None:
        """Should remove a session."""
        self.manager.set("conv-rm", "groq", "llama3")
        assert self.manager.remove("conv-rm")
        provider, model = self.manager.get("conv-rm")
        assert provider is None

    def test_remove_nonexistent(self) -> None:
        """Removing nonexistent session should return False."""
        assert not self.manager.remove("nonexistent")

    def test_update_existing_session(self) -> None:
        """Should update provider on existing session."""
        self.manager.set("conv-upd", "groq", "llama3")
        self.manager.set("conv-upd", "cerebras", "llama3")
        provider, model = self.manager.get("conv-upd")
        assert provider == "cerebras"

    def test_get_all_active(self) -> None:
        """Should return all active sessions."""
        self.manager.set("conv-1", "groq", "llama3")
        self.manager.set("conv-2", "cerebras", "llama3")
        sessions = self.manager.get_all()
        assert len(sessions) == 2

    def test_get_stats(self) -> None:
        """Should return session statistics."""
        self.manager.set("conv-1", "groq", "llama3")
        self.manager.get("conv-1")
        stats = self.manager.get_stats()
        assert stats["total_sessions"] == 1
        assert stats["active_sessions"] == 1
        assert stats["total_requests"] == 1

    def test_cleanup_expired(self) -> None:
        """Should clean up expired sessions."""
        self.manager.set("conv-exp1", "groq", "llama3", ttl=0.01)
        self.manager.set("conv-active", "cerebras", "llama3", ttl=300)
        time.sleep(0.02)
        removed = self.manager.cleanup_expired()
        assert removed == 1
        assert self.manager.get_stats()["active_sessions"] == 1

    def test_extract_conversation_id(self) -> None:
        """Should extract conversation ID from messages."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello world"},
        ]
        conv_id = self.manager.extract_conversation_id(messages)
        assert conv_id.startswith("conv_")
        assert len(conv_id) > 5

    def test_extract_conversation_id_empty(self) -> None:
        """Should return empty for no messages."""
        conv_id = self.manager.extract_conversation_id([])
        assert conv_id == ""

    def test_extract_conversation_id_multimodal(self) -> None:
        """Should handle multimodal content."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            ]}
        ]
        conv_id = self.manager.extract_conversation_id(messages)
        assert conv_id.startswith("conv_")
