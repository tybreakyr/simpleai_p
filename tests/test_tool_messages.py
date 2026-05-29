"""
Unit tests for tool-call / tool-result message serialization across providers.

These exercise the neutral ``Message`` tool fields (``tool_calls`` on assistant
turns, ``role="tool"`` results) and verify each provider translates them into
its native wire format. SDKs are mocked; no network access required.
"""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch

from llm_provider.models import (
    ChatRequest,
    Message,
    ProviderConfig,
    ToolCall,
)


def _assistant_call(name="get_thing", args=None, call_id="call_1", content=""):
    return Message(
        role="assistant",
        content=content,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args or {"x": 1})],
    )


def _tool_result(call_id="call_1", name="get_thing", content="the result"):
    return Message(role="tool", content=content, tool_call_id=call_id, name=name)


def _tool_exchange_messages():
    """user → assistant(tool_call) → tool(result)."""
    return [
        Message(role="user", content="please act"),
        _assistant_call(),
        _tool_result(),
    ]


# ---------------------------------------------------------------------------
# Model-level validation
# ---------------------------------------------------------------------------

class TestMessageToolFields(unittest.TestCase):
    def test_tool_result_requires_call_id(self):
        with self.assertRaises(ValueError):
            Message(role="tool", content="x")

    def test_assistant_tool_call_allows_empty_content(self):
        msg = _assistant_call(content="")
        self.assertEqual(msg.content, "")
        self.assertEqual(len(msg.tool_calls), 1)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class TestOpenAIToolMessages(unittest.TestCase):
    def _provider(self):
        config = ProviderConfig(
            host="https://api.openai.com", default_model="gpt-4o",
            api_key="k", timeout=30.0, retry_attempts=1,
        )
        with patch.dict(sys.modules, {"openai": MagicMock()}):
            from llm_provider.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider.__new__(OpenAIProvider)
            p._config = config
            return p

    def test_serializes_tool_exchange(self):
        p = self._provider()
        req = ChatRequest(messages=_tool_exchange_messages())
        msgs = p._build_kwargs(req)["messages"]
        # user, assistant(tool_calls), tool
        self.assertEqual(msgs[0], {"role": "user", "content": "please act"})
        asst = msgs[1]
        self.assertEqual(asst["role"], "assistant")
        self.assertIsNone(asst["content"])
        tc = asst["tool_calls"][0]
        self.assertEqual(tc["id"], "call_1")
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "get_thing")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"x": 1})
        self.assertEqual(
            msgs[2],
            {"role": "tool", "tool_call_id": "call_1", "content": "the result"},
        )

    def test_supports_tool_role(self):
        self.assertIn("tool", self._provider().supported_features().supported_roles)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class TestOllamaToolMessages(unittest.TestCase):
    def _provider(self):
        config = ProviderConfig(
            host="http://localhost:11434", default_model="llama3.1",
            api_key=None, timeout=30.0, retry_attempts=1,
        )
        from llm_provider.providers.ollama_provider import OllamaProvider
        p = OllamaProvider.__new__(OllamaProvider)
        p._config = config
        return p

    def test_serializes_tool_exchange(self):
        p = self._provider()
        req = ChatRequest(messages=_tool_exchange_messages())
        msgs = p._build_payload(req)["messages"]
        asst = msgs[1]
        self.assertEqual(asst["role"], "assistant")
        # Ollama takes arguments as a dict, not a JSON string.
        self.assertEqual(asst["tool_calls"][0]["function"]["arguments"], {"x": 1})
        self.assertEqual(
            msgs[2],
            {"role": "tool", "tool_call_id": "call_1", "content": "the result"},
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class TestAnthropicToolMessages(unittest.TestCase):
    def _provider(self):
        config = ProviderConfig(
            host="https://api.anthropic.com", default_model="claude-sonnet-4-6",
            api_key="k", timeout=30.0, retry_attempts=1,
        )
        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            from llm_provider.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider.__new__(AnthropicProvider)
            p._config = config
            p._max_tokens = 1024
            return p

    def test_serializes_tool_exchange(self):
        p = self._provider()
        req = ChatRequest(messages=_tool_exchange_messages())
        msgs = p._build_kwargs(req)["messages"]
        asst = msgs[1]
        self.assertEqual(asst["role"], "assistant")
        # Empty content → no text block, only tool_use.
        self.assertEqual(asst["content"], [
            {"type": "tool_use", "id": "call_1", "name": "get_thing", "input": {"x": 1}},
        ])
        # tool result rides in a user turn as a tool_result block.
        self.assertEqual(msgs[2], {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "the result"},
            ],
        })

    def test_merges_consecutive_tool_results(self):
        p = self._provider()
        req = ChatRequest(messages=[
            Message(role="user", content="go"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="a", name="t", arguments={}),
                ToolCall(id="b", name="t", arguments={}),
            ]),
            _tool_result(call_id="a", content="ra"),
            _tool_result(call_id="b", content="rb"),
        ])
        msgs = p._build_kwargs(req)["messages"]
        # Two tool results collapse into a single user turn with two blocks.
        self.assertEqual(len(msgs), 3)
        self.assertEqual(len(msgs[2]["content"]), 2)
        self.assertEqual(msgs[2]["content"][0]["tool_use_id"], "a")
        self.assertEqual(msgs[2]["content"][1]["tool_use_id"], "b")


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class TestGeminiToolMessages(unittest.TestCase):
    def _provider_and_types(self):
        config = ProviderConfig(
            host="https://generativelanguage.googleapis.com",
            default_model="gemini-1.5-flash", api_key="k",
            timeout=30.0, retry_attempts=1,
        )
        mock_genai = MagicMock()
        ctx = patch.dict(sys.modules, {
            "google": MagicMock(),
            "google.genai": mock_genai,
            "google.genai.types": mock_genai.types,
        })
        ctx.start()
        from llm_provider.providers.gemini_provider import GeminiProvider
        p = GeminiProvider.__new__(GeminiProvider)
        p._config = config
        return p, mock_genai.types, ctx

    def test_serializes_function_call_and_response(self):
        p, types, ctx = self._provider_and_types()
        try:
            req = ChatRequest(messages=_tool_exchange_messages())
            model_name, contents, _ = p._build_kwargs(req)
            # user + model(function_call) + user(function_response)
            self.assertEqual(len(contents), 3)
            types.FunctionCall.assert_called_once_with(name="get_thing", args={"x": 1})
            types.FunctionResponse.assert_called_once_with(
                name="get_thing", response={"result": "the result"}
            )
        finally:
            ctx.stop()


if __name__ == "__main__":
    unittest.main()
