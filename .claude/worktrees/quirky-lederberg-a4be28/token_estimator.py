"""Token estimation from messages before routing to providers.

Provides a rough token count estimate so the router can pre-check
TPM/TPD limits before sending a request upstream. Uses a simple
heuristic of ~4 characters per token (consistent with OpenAI's
tiktoken for English text).

This is intentionally lightweight — no model-specific tokenizers,
just a fast approximation for rate limit gating.
"""

from __future__ import annotations


def estimate_tokens(messages: list[dict[str, object]]) -> int:
    """Estimate the total token count for a list of chat messages.

    Counts text content in all message roles, plus overhead per message
    (role markers, formatting). Multimodal content blocks are counted
    by their text parts only.

    Returns an estimated token count (always >= 1 if any content exists).
    """
    if not messages:
        return 0

    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            # Multimodal content blocks
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if isinstance(text, str):
                        total_chars += len(text)

    # ~4 characters per token (OpenAI tiktoken average for English)
    message_tokens = max(1, total_chars // 4) if total_chars > 0 else 0

    # Add per-message overhead: role markers, separators, formatting
    # OpenAI adds ~4 tokens overhead per message
    overhead = len(messages) * 4

    return message_tokens + overhead


def estimate_request_tokens(payload: dict[str, object]) -> int:
    """Estimate input tokens for a full request payload.

    Includes message tokens plus a small buffer for system parameters
    (model name, temperature, etc.).
    """
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return 0

    base = estimate_tokens(messages)

    # Add tokens for tools/functions if present
    tools = payload.get("tools")
    if isinstance(tools, list):
        import json
        tools_json = json.dumps(tools)
        base += max(1, len(tools_json) // 4)

    # Small buffer for system parameters
    return base + 10
