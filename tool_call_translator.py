"""Tool/function calling format translation between OpenAI and Gemini.

Translates OpenAI's tool/function calling format to Google Gemini's
functionDeclarations format and back, enabling cross-provider tool use.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── OpenAI → Gemini ──────────────────────────────────────────────────────────

def openai_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool definitions to Gemini function_declarations format.

    OpenAI format:
    [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
    ]

    Gemini format:
    [
        {
            "functionDeclarations": [{
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {...},
                    "required": [...]
                }
            }]
        }
    ]
    """
    gemini_tools: list[dict[str, Any]] = []

    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            declaration = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
            }

            params = func.get("parameters")
            if params:
                declaration["parameters"] = _convert_schema_to_gemini(params)

            gemini_declaration = {"functionDeclarations": [declaration]}

            # Merge into existing tool if same structure
            merged = False
            for existing in gemini_tools:
                if "functionDeclarations" in existing:
                    existing["functionDeclarations"].append(declaration)
                    merged = True
                    break
            if not merged:
                gemini_tools.append(gemini_declaration)

    return gemini_tools


def _convert_schema_to_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON Schema type format to Gemini's schema format."""
    gemini_schema: dict[str, Any] = {}

    # Convert type
    json_type = schema.get("type", "object")
    gemini_schema["type"] = _json_type_to_gemini(json_type)

    if "description" in schema:
        gemini_schema["description"] = schema["description"]

    if "properties" in schema:
        gemini_schema["properties"] = {}
        for prop_name, prop_def in schema["properties"].items():
            gemini_schema["properties"][prop_name] = _convert_schema_to_gemini(prop_def)

    if "required" in schema:
        gemini_schema["required"] = schema["required"]

    if "enum" in schema:
        gemini_schema["enum"] = schema["enum"]

    if "items" in schema:
        gemini_schema["items"] = _convert_schema_to_gemini(schema["items"])

    return gemini_schema


def _json_type_to_gemini(json_type: str) -> str:
    """Convert JSON Schema type to Gemini type string."""
    mapping = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "object": "OBJECT",
        "array": "ARRAY",
    }
    return mapping.get(json_type, "STRING")


# ── Gemini → OpenAI ──────────────────────────────────────────────────────────

def gemini_tools_to_openai(gemini_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Gemini function_declarations back to OpenAI tool format."""
    openai_tools: list[dict[str, Any]] = []

    for tool in gemini_tools:
        declarations = tool.get("functionDeclarations", [])
        for decl in declarations:
            params = decl.get("parameters")
            openai_func: dict[str, Any] = {
                "name": decl.get("name", ""),
                "description": decl.get("description", ""),
            }
            if params:
                openai_func["parameters"] = _convert_schema_to_openai(params)

            openai_tools.append({
                "type": "function",
                "function": openai_func,
            })

    return openai_tools


def _convert_schema_to_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert Gemini schema format back to JSON Schema."""
    openai_schema: dict[str, Any] = {}

    gemini_type = schema.get("type", "OBJECT")
    openai_schema["type"] = _gemini_type_to_json(gemini_type)

    if "description" in schema:
        openai_schema["description"] = schema["description"]

    if "properties" in schema:
        openai_schema["properties"] = {}
        for prop_name, prop_def in schema["properties"].items():
            openai_schema["properties"][prop_name] = _convert_schema_to_openai(prop_def)

    if "required" in schema:
        openai_schema["required"] = schema["required"]

    if "enum" in schema:
        openai_schema["enum"] = schema["enum"]

    if "items" in schema:
        openai_schema["items"] = _convert_schema_to_openai(schema["items"])

    return openai_schema


def _gemini_type_to_json(gemini_type: str) -> str:
    """Convert Gemini type string to JSON Schema type."""
    mapping = {
        "STRING": "string",
        "NUMBER": "number",
        "INTEGER": "integer",
        "BOOLEAN": "boolean",
        "OBJECT": "object",
        "ARRAY": "array",
    }
    return mapping.get(gemini_type, "string")


# ── Tool call response translation ──────────────────────────────────────────

def translate_tool_calls_to_gemini(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate OpenAI tool_calls in assistant message to Gemini functionCall format.

    OpenAI: {"tool_calls": [{"id": "call_xxx", "function": {"name": "f", "arguments": "{}"}}]}
    Gemini: {"functionCall": {"name": "f", "args": {...}}}
    """
    gemini_calls: list[dict[str, Any]] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {"raw": args_str}

        gemini_calls.append({
            "functionCall": {
                "name": name,
                "args": args,
            }
        })
    return gemini_calls


def translate_gemini_function_call_to_openai(
    function_call: dict[str, Any], call_id: str | None = None,
) -> dict[str, Any]:
    """Translate Gemini functionCall to OpenAI tool_calls format.

    Accepts both wrapped and unwrapped formats:
    - Wrapped: {"functionCall": {"name": "f", "args": {...}}}
    - Unwrapped: {"name": "f", "args": {...}}

    Returns OpenAI format:
    {"tool_calls": [{"id": "call_xxx", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    """
    import secrets as _secrets

    # Handle both wrapped and unwrapped formats
    inner = function_call.get("functionCall", function_call)
    name = inner.get("name", "")
    args = inner.get("args", {})
    tool_call_id = call_id or f"call_{_secrets.token_hex(12)}"

    return {
        "tool_calls": [{
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args),
            },
        }],
        "role": "assistant",
    }


def translate_gemini_function_response_to_openai(
    function_response: dict[str, Any],
) -> dict[str, Any]:
    """Translate Gemini functionResponse to OpenAI tool message format.

    Accepts both wrapped and unwrapped formats:
    - Wrapped: {"functionResponse": {"name": "f", "response": {"content": "..."}}}
    - Unwrapped: {"name": "f", "response": {"content": "..."}}

    Returns OpenAI format:
    {"role": "tool", "tool_call_id": "call_xxx", "content": "..."}
    """
    inner = function_response.get("functionResponse", function_response)
    name = inner.get("name", "")
    response = inner.get("response", {})
    content = response.get("content", "")
    if isinstance(content, dict):
        content = json.dumps(content)

    return {
        "role": "tool",
        "tool_call_id": f"call_{hash(name) & 0xFFFFFFFF:x}",
        "content": str(content),
    }


def prepare_tools_for_provider(
    tools: list[dict[str, Any]] | None,
    target_provider: str,
) -> list[dict[str, Any]] | None:
    """Translate tools to the target provider's format.

    Args:
        tools: OpenAI-format tool definitions
        target_provider: Provider name to translate for

    Returns:
        Tools in the target provider's format, or None if no tools
    """
    if not tools:
        return None

    if target_provider == "google_gemini":
        result = openai_tools_to_gemini(tools)
        return result if result else None

    # Most providers use OpenAI format directly
    return tools
