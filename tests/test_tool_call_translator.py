"""Tests for tool/function calling translation between OpenAI and Gemini."""

import json
import pytest
from tool_call_translator import (
    openai_tools_to_gemini,
    gemini_tools_to_openai,
    translate_tool_calls_to_gemini,
    translate_gemini_function_call_to_openai,
    translate_gemini_function_response_to_openai,
    prepare_tools_for_provider,
)


SAMPLE_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]


class TestOpenAIToGemini:
    def test_basic_conversion(self) -> None:
        """Should convert OpenAI tools to Gemini format."""
        result = openai_tools_to_gemini(SAMPLE_OPENAI_TOOLS)
        assert len(result) >= 1
        decls = result[0].get("functionDeclarations", [])
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"
        assert decls[0]["description"] == "Get current weather for a location"

    def test_type_conversion(self) -> None:
        """Should convert JSON Schema types to Gemini types."""
        result = openai_tools_to_gemini(SAMPLE_OPENAI_TOOLS)
        params = result[0]["functionDeclarations"][0]["parameters"]
        assert params["type"] == "OBJECT"
        assert params["properties"]["location"]["type"] == "STRING"

    def test_enum_preserved(self) -> None:
        """Should preserve enum values."""
        result = openai_tools_to_gemini(SAMPLE_OPENAI_TOOLS)
        unit_prop = result[0]["functionDeclarations"][0]["parameters"]["properties"]["unit"]
        assert "enum" in unit_prop
        assert "celsius" in unit_prop["enum"]

    def test_required_preserved(self) -> None:
        """Should preserve required fields."""
        result = openai_tools_to_gemini(SAMPLE_OPENAI_TOOLS)
        params = result[0]["functionDeclarations"][0]["parameters"]
        assert "required" in params
        assert "location" in params["required"]

    def test_empty_tools(self) -> None:
        """Should handle empty tool list."""
        result = openai_tools_to_gemini([])
        assert result == []

    def test_multiple_tools(self) -> None:
        """Should handle multiple tools."""
        tools = SAMPLE_OPENAI_TOOLS + [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {"type": "string"},
                        },
                    },
                },
            }
        ]
        result = openai_tools_to_gemini(tools)
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 2


class TestGeminiToOpenAI:
    def test_basic_conversion(self) -> None:
        """Should convert Gemini tools back to OpenAI format."""
        gemini_tools = [
            {
                "functionDeclarations": [{
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "location": {"type": "STRING", "description": "City"},
                        },
                        "required": ["location"],
                    },
                }]
            }
        ]
        result = gemini_tools_to_openai(gemini_tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"

    def test_roundtrip(self) -> None:
        """Should roundtrip OpenAI → Gemini → OpenAI."""
        gemini = openai_tools_to_gemini(SAMPLE_OPENAI_TOOLS)
        back = gemini_tools_to_openai(gemini)
        assert back[0]["function"]["name"] == "get_weather"
        assert back[0]["function"]["parameters"]["properties"]["location"]["type"] == "string"


class TestToolCallTranslation:
    def test_translate_tool_calls_to_gemini(self) -> None:
        """Should translate OpenAI tool_calls to Gemini functionCall."""
        tool_calls = [
            {
                "id": "call_abc123",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ]
        result = translate_tool_calls_to_gemini(tool_calls)
        assert len(result) == 1
        assert "functionCall" in result[0]
        assert result[0]["functionCall"]["name"] == "get_weather"
        assert result[0]["functionCall"]["args"]["location"] == "Paris"

    def test_translate_gemini_function_call(self) -> None:
        """Should translate Gemini functionCall to OpenAI format."""
        fc = {"functionCall": {"name": "get_weather", "args": {"location": "Paris"}}}
        result = translate_gemini_function_call_to_openai(fc)
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])
        assert args["location"] == "Paris"

    def test_translate_gemini_function_response(self) -> None:
        """Should translate Gemini functionResponse to OpenAI tool message."""
        fr = {"functionResponse": {"name": "get_weather", "response": {"content": "22°C"}}}
        result = translate_gemini_function_response_to_openai(fr)
        assert result["role"] == "tool"
        assert "22°C" in result["content"]

    def test_translate_function_response_dict_content(self) -> None:
        """Should handle dict content in function response."""
        fr = {
            "functionResponse": {
                "name": "get_weather",
                "response": {"content": {"temp": 22, "unit": "celsius"}},
            }
        }
        result = translate_gemini_function_response_to_openai(fr)
        assert result["role"] == "tool"
        parsed = json.loads(result["content"])
        assert parsed["temp"] == 22


class TestPrepareToolsForProvider:
    def test_gemini_provider(self) -> None:
        """Should translate tools for Gemini provider."""
        result = prepare_tools_for_provider(SAMPLE_OPENAI_TOOLS, "google_gemini")
        assert result is not None
        assert "functionDeclarations" in result[0]

    def test_openai_compatible_provider(self) -> None:
        """Should pass through tools for OpenAI-compatible providers."""
        result = prepare_tools_for_provider(SAMPLE_OPENAI_TOOLS, "groq")
        assert result == SAMPLE_OPENAI_TOOLS

    def test_no_tools(self) -> None:
        """Should return None when no tools provided."""
        assert prepare_tools_for_provider(None, "groq") is None
        assert prepare_tools_for_provider([], "groq") is None
