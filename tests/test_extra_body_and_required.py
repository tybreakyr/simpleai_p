"""
Tests for the namespaced ``extra_body`` passthrough on ChatRequest and the
``tool_choice="required"`` synonym for ``"any"``.

Each provider should:
  * read its own bucket from ``request.extra_body`` and merge it into the
    outgoing payload,
  * leave other providers' buckets untouched (drop them silently),
  * accept ``tool_choice="required"`` (normalised to ``"any"`` at request
    construction time).

Construction-time validation:
  * unknown top-level keys raise ``ValueError``,
  * non-dict bucket values raise ``ValueError``.
"""

import importlib.util
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from llm_provider.models import (
    ChatRequest,
    Message,
    ProviderConfig,
    ToolSchema,
)

_HAS_OPENAI_SDK = importlib.util.find_spec("openai") is not None
_HAS_ANTHROPIC_SDK = importlib.util.find_spec("anthropic") is not None


_TOOL = ToolSchema(
    name="echo",
    description="Echo input.",
    input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
)


# ---------------------------------------------------------------------------
# ChatRequest validation
# ---------------------------------------------------------------------------


class TestChatRequestValidation(unittest.TestCase):
    def test_required_normalises_to_any(self):
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[_TOOL],
            tool_choice="required",
        )
        self.assertEqual(req.tool_choice, "any")

    def test_any_passthrough(self):
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[_TOOL],
            tool_choice="any",
        )
        self.assertEqual(req.tool_choice, "any")

    def test_extra_body_unknown_provider_rejected(self):
        with self.assertRaises(ValueError):
            ChatRequest(
                messages=[Message(role="user", content="hi")],
                extra_body={"openi": {"foo": 1}},  # typo
            )

    def test_extra_body_non_dict_bucket_rejected(self):
        with self.assertRaises(ValueError):
            ChatRequest(
                messages=[Message(role="user", content="hi")],
                extra_body={"openai": "not-a-dict"},
            )

    def test_extra_body_valid_buckets_accepted(self):
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            extra_body={
                "openai": {"repetition_penalty": 1.1},
                "anthropic": {"top_k": 40},
                "gemini": {},
                "ollama": {"keep_alive": "10m"},
            },
        )
        self.assertEqual(req.extra_body["openai"]["repetition_penalty"], 1.1)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


def _openai_provider():
    """Return an OpenAIProvider with the openai SDK stubbed."""
    from llm_provider.providers.openai_provider import OpenAIProvider

    cfg = ProviderConfig(
        host="https://api.openai.com",
        default_model="gpt-4o",
        api_key="test",
        timeout=30.0,
        retry_attempts=1,
    )
    with patch("openai.OpenAI", MagicMock()):
        return OpenAIProvider(cfg)


@unittest.skipUnless(_HAS_OPENAI_SDK, "openai SDK not installed")
class TestOpenAIExtraBody(unittest.TestCase):
    def test_own_bucket_merged_into_kwargs(self):
        provider = _openai_provider()
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            extra_body={"openai": {"repetition_penalty": 1.1, "seed": 42}},
        )
        kwargs = provider._build_kwargs(req)
        self.assertEqual(kwargs["repetition_penalty"], 1.1)
        self.assertEqual(kwargs["seed"], 42)

    def test_other_provider_bucket_not_leaked(self):
        provider = _openai_provider()
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            extra_body={"anthropic": {"top_k": 40}},
        )
        kwargs = provider._build_kwargs(req)
        self.assertNotIn("top_k", kwargs)

    def test_required_maps_to_openai_required(self):
        provider = _openai_provider()
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[_TOOL],
            tool_choice="required",
        )
        kwargs = provider._build_kwargs(req)
        self.assertEqual(kwargs["tool_choice"], "required")


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------


def _ollama_provider():
    from llm_provider.providers.ollama_provider import OllamaProvider

    cfg = ProviderConfig(
        host="http://localhost:11434",
        default_model="llama3",
        timeout=30.0,
        retry_attempts=1,
    )
    return OllamaProvider(cfg)


class TestOllamaExtraBodyAndRequired(unittest.TestCase):
    def test_own_bucket_merged_into_payload(self):
        provider = _ollama_provider()
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            extra_body={"ollama": {"keep_alive": "10m"}},
        )
        payload = provider._build_payload(req)
        self.assertEqual(payload["keep_alive"], "10m")

    def test_other_provider_bucket_not_leaked(self):
        provider = _ollama_provider()
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            extra_body={"openai": {"repetition_penalty": 1.1}},
        )
        payload = provider._build_payload(req)
        self.assertNotIn("repetition_penalty", payload)

    def test_required_emulation_retries_with_nudge(self):
        from llm_provider.models import ChatResponse, ToolCall

        provider = _ollama_provider()
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[_TOOL],
            tool_choice="required",  # normalised to "any"
        )

        empty = ChatResponse(message="prose, no tool", tool_calls=None)
        good = ChatResponse(
            message="",
            tool_calls=[ToolCall(id="call_1", name="echo", arguments={"x": "ok"})],
        )

        with patch.object(provider, "_execute_with_retry", side_effect=[empty, good]) as m:
            response = provider.chat(req)
        self.assertEqual(m.call_count, 2)
        self.assertIsNotNone(response.tool_calls)
        self.assertEqual(response.tool_calls[0].name, "echo")

    def test_required_emulation_raises_when_still_no_tool_call(self):
        from llm_provider.errors import InvalidResponseError
        from llm_provider.models import ChatResponse

        provider = _ollama_provider()
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[_TOOL],
            tool_choice="required",
        )

        empty = ChatResponse(message="prose only", tool_calls=None)
        with patch.object(provider, "_execute_with_retry", side_effect=[empty, empty]):
            with self.assertRaises(InvalidResponseError):
                provider.chat(req)


class TestOllamaRequiredEmulationAsync(unittest.IsolatedAsyncioTestCase):
    async def test_required_emulation_retries_with_nudge_async(self):
        from llm_provider.models import ChatResponse, ToolCall

        provider = _ollama_provider()
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[_TOOL],
            tool_choice="required",
        )

        empty = ChatResponse(message="prose, no tool", tool_calls=None)
        good = ChatResponse(
            message="",
            tool_calls=[ToolCall(id="call_1", name="echo", arguments={"x": "ok"})],
        )

        fake_retry = AsyncMock(side_effect=[empty, good])
        with patch("llm_provider.retry._async_retry_with_backoff", fake_retry):
            response = await provider.achat(req)
        self.assertEqual(fake_retry.await_count, 2)
        self.assertIsNotNone(response.tool_calls)
        self.assertEqual(response.tool_calls[0].name, "echo")

    async def test_required_emulation_raises_when_still_no_tool_call_async(self):
        from llm_provider.errors import InvalidResponseError
        from llm_provider.models import ChatResponse

        provider = _ollama_provider()
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[_TOOL],
            tool_choice="required",
        )

        empty = ChatResponse(message="prose only", tool_calls=None)
        fake_retry = AsyncMock(side_effect=[empty, empty])
        with patch("llm_provider.retry._async_retry_with_backoff", fake_retry):
            with self.assertRaises(InvalidResponseError):
                await provider.achat(req)


# ---------------------------------------------------------------------------
# Anthropic & Gemini extra_body merging (unit-level, no SDK calls)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_ANTHROPIC_SDK, "anthropic SDK not installed")
class TestAnthropicExtraBody(unittest.TestCase):
    def test_own_bucket_merged(self):
        with patch("anthropic.Anthropic", MagicMock()):
            from llm_provider.providers.anthropic_provider import AnthropicProvider

            cfg = ProviderConfig(
                host="https://api.anthropic.com",
                default_model="claude-sonnet-4-6",
                api_key="test",
                timeout=30.0,
                retry_attempts=1,
            )
            provider = AnthropicProvider(cfg)
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            extra_body={"anthropic": {"top_k": 40}},
        )
        kwargs = provider._build_kwargs(req)
        self.assertEqual(kwargs["top_k"], 40)

    def test_other_provider_bucket_not_leaked(self):
        with patch("anthropic.Anthropic", MagicMock()):
            from llm_provider.providers.anthropic_provider import AnthropicProvider

            cfg = ProviderConfig(
                host="https://api.anthropic.com",
                default_model="claude-sonnet-4-6",
                api_key="test",
                timeout=30.0,
                retry_attempts=1,
            )
            provider = AnthropicProvider(cfg)
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            extra_body={"openai": {"repetition_penalty": 1.1}},
        )
        kwargs = provider._build_kwargs(req)
        self.assertNotIn("repetition_penalty", kwargs)


if __name__ == "__main__":
    unittest.main()
