"""Sticky sessions: route consecutive requests in a conversation to the same provider.

Maintains a mapping of conversation_id -> (provider, model) with a configurable TTL.
Ensures conversation continuity by preferring the same provider that handled
previous requests in the same conversation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TTL = 1800  # 30 minutes
CLEANUP_INTERVAL = 300  # 5 minutes


@dataclass
class SessionEntry:
    """A sticky session binding a conversation to a provider."""

    conversation_id: str
    provider: str
    model: str
    created_at: float
    last_used_at: float
    request_count: int = 0
    ttl: float = DEFAULT_SESSION_TTL

    @property
    def is_expired(self) -> bool:
        return time.time() - self.last_used_at > self.ttl

    def touch(self) -> None:
        self.last_used_at = time.time()
        self.request_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "model": self.model,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "request_count": self.request_count,
            "ttl": self.ttl,
            "expires_in": max(0, self.ttl - (time.time() - self.last_used_at)),
        }


class StickySessionManager:
    """Manages sticky sessions for conversation continuity.

    When a conversation starts, the chosen provider is recorded.
    Subsequent requests with the same conversation_id will prefer
    the same provider, unless the session has expired.
    """

    def __init__(self, default_ttl: float = DEFAULT_SESSION_TTL) -> None:
        self._sessions: dict[str, SessionEntry] = {}
        self._lock = Lock()
        self._default_ttl = default_ttl
        self._cleanup_task: asyncio.Task | None = None

    def get(
        self, conversation_id: str,
    ) -> tuple[str | None, str | None]:
        """Get the (provider, model) for a conversation.

        Returns (None, None) if no active session exists.
        """
        if not conversation_id:
            return None, None

        with self._lock:
            entry = self._sessions.get(conversation_id)
            if not entry:
                return None, None

            if entry.is_expired:
                del self._sessions[conversation_id]
                logger.debug("Session expired: %s", conversation_id)
                return None, None

            entry.touch()
            return entry.provider, entry.model

    def set(
        self,
        conversation_id: str,
        provider: str,
        model: str,
        ttl: float | None = None,
    ) -> None:
        """Create or update a sticky session."""
        if not conversation_id:
            return

        now = time.time()
        with self._lock:
            existing = self._sessions.get(conversation_id)
            if existing:
                existing.provider = provider
                existing.model = model
                existing.touch()
            else:
                self._sessions[conversation_id] = SessionEntry(
                    conversation_id=conversation_id,
                    provider=provider,
                    model=model,
                    created_at=now,
                    last_used_at=now,
                    ttl=ttl or self._default_ttl,
                )

    def remove(self, conversation_id: str) -> bool:
        """Remove a sticky session. Returns True if session existed."""
        with self._lock:
            return self._sessions.pop(conversation_id, None) is not None

    def get_all(self) -> list[dict[str, Any]]:
        """Get all active sessions as dicts."""
        with self._lock:
            active = []
            expired = []
            for cid, entry in self._sessions.items():
                if entry.is_expired:
                    expired.append(cid)
                else:
                    active.append(entry.to_dict())
            for cid in expired:
                del self._sessions[cid]
            return active

    def get_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        with self._lock:
            total = len(self._sessions)
            active = sum(1 for e in self._sessions.values() if not e.is_expired)
            total_requests = sum(e.request_count for e in self._sessions.values())
        return {
            "total_sessions": total,
            "active_sessions": active,
            "expired_sessions": total - active,
            "total_requests": total_requests,
            "default_ttl": self._default_ttl,
        }

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        with self._lock:
            expired = [cid for cid, e in self._sessions.items() if e.is_expired]
            for cid in expired:
                del self._sessions[cid]
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)

    def start_cleanup_loop(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired sessions."""
        while True:
            try:
                self.cleanup_expired()
            except Exception as e:
                logger.error("Session cleanup error: %s", e)
            await asyncio.sleep(CLEANUP_INTERVAL)

    def extract_conversation_id(self, messages: list[dict[str, Any]]) -> str:
        """Extract or generate a conversation ID from messages.

        Strategy:
        1. Check for explicit conversation_id in the request
        2. Hash the first message content as a conversation identifier
        3. Return empty string if no messages
        """
        if not messages:
            return ""

        # Use hash of first user message as conversation ID
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    return f"conv_{hash(content) & 0xFFFFFFFFFFFFFFFF:x}"
                elif isinstance(content, list):
                    # Multimodal content
                    text_parts = [
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    combined = " ".join(text_parts)
                    if combined:
                        return f"conv_{hash(combined) & 0xFFFFFFFFFFFFFFFF:x}"
        return ""


# Global singleton
sticky_sessions = StickySessionManager()
