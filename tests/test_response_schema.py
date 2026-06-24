"""
Tests for schema-driven structured output (``ChatRequest.response_schema``),
the Qwen3 ``/no_think`` prefix, the concurrency semaphore, and the mlx provider.

Each provider maps ``response_schema`` to its native enforcement mechanism:
  * OpenAI / mlx → ``response_format`` (json_schema, or json_object for mlx)
  * Gemini       → ``response_mime_type`` + ``response_schema``
  * Ollama       → ``format`` (JSON Schema)
  * Anthropic    → a forced single synthetic tool, read back as a dict

and decodes the result into ``ChatResponse.structured_data`` as a plain dict,
raising a retryable ``JSONParseFailedError`` when decoding fails.
"""

import asyncio
import importlib.util
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

from llm_provider.errors import JSONParseFailedError
from llm_provider.models import (
    ChatRequest,
    Message,
    ProviderConfig,
    SystemPrompt,
    ToolSchema,
)
from llm_provider.retry import RetryConfig

_HAS_GEMINI_SDK = importlib.util.find_spec("google.genai") is not None

_SCHEMA = {
    "type": "object",
    "title": "Person Record",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


def _retry_cfg(attempts=1):
    return RetryConfig(max_retries=attempts, base_delay=0.01, max_delay=0.01, backoff_factor=1.0)


# ─── Provider factory helpers (no live SDK calls) ────────────────────────────


def _openai_provider(extra_settings=None):
    with patch.dict(sys.modules, {"openai": MagicMock()}):
        from llm_provider.providers.openai_provider import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
    p._config = ProviderConfig(
        host="https://api.openai.com/v1",
        default_model="gpt-4o",
        api_key="test-key",
        extra_settings=extra_settings or {},
    )
    p._client = MagicMock()
    p._retry_config = _retry_cfg()
    return p


def _anthropic_provider():
    with patch.dict(sys.modules, {"anthropic": MagicMock()}):
        from llm_provider.providers.anthropic_provider import AnthropicProvider

        p = AnthropicProvider.__new__(AnthropicProvider)
    p._config = ProviderConfig(
        host="https://api.anthropic.com",
        default_model="claude-3-5-sonnet",
        api_key="test-key",
    )
    p._client = MagicMock()
    p._max_tokens = 8192
    p._retry_config = _retry_cfg()
    return p


def _ollama_provider(extra_settings=None):
    with patch.dict(sys.modules, {"requests": MagicMock(), "httpx": MagicMock()}):
        from llm_provider.providers.ollama_provider import OllamaProvider

        p = OllamaProvider.__new__(OllamaProvider)
    p._config = ProviderConfig(
        host="http://localhost:11434",
        default_model="llama3",
        api_key="",
        extra_settings=extra_settings or {},
    )
    p._base_url = "http://localhost:11434/api"
    p._session = MagicMock()
    p._retry_config = _retry_cfg()
    return p


# ─── ChatRequest validation ──────────────────────────────────────────────────


class TestResponseSchemaValidation(unittest.TestCase):
    def test_non_dict_rejected(self):
        with self.assertRaises(ValueError):
            ChatRequest(messages=[Message(role="user", content="hi")], response_schema="nope")

    def test_combined_with_tools_rejected(self):
        tool = ToolSchema(name="t", description="d", input_schema={"type": "object"})
        with self.assertRaises(ValueError):
            ChatRequest(
                messages=[Message(role="user", content="hi")],
                response_schema=_SCHEMA,
                tools=[tool],
            )

    def test_accepts_dict(self):
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        self.assertEqual(req.response_schema, _SCHEMA)


# ─── OpenAI ──────────────────────────────────────────────────────────────────


class TestOpenAIResponseSchema(unittest.TestCase):
    def test_json_schema_response_format(self):
        provider = _openai_provider()
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        kwargs = provider._build_kwargs(req)
        rf = kwargs["response_format"]
        self.assertEqual(rf["type"], "json_schema")
        self.assertEqual(rf["json_schema"]["schema"], _SCHEMA)
        self.assertEqual(rf["json_schema"]["name"], "person_record")  # from title

    def test_json_object_mode_for_mlx(self):
        provider = _openai_provider(extra_settings={"structured_output_format": "json_object"})
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        kwargs = provider._build_kwargs(req)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_json_object_mode_injects_schema_into_prompt(self):
        # json_object only forces valid JSON, not the shape, so the schema must
        # be described in the prompt (mlx-lm also ignores json_schema entirely).
        provider = _openai_provider(extra_settings={"structured_output_format": "json_object"})
        req = ChatRequest(
            messages=[Message(role="user", content="Extract")], response_schema=_SCHEMA
        )
        kwargs = provider._build_kwargs(req)
        last = kwargs["messages"][-1]["content"]
        self.assertIn("JSON Schema", last)
        self.assertIn('"properties"', last)

    def test_json_schema_mode_does_not_inject_prompt(self):
        provider = _openai_provider()  # default json_schema (real enforcement)
        req = ChatRequest(
            messages=[Message(role="user", content="Extract")], response_schema=_SCHEMA
        )
        kwargs = provider._build_kwargs(req)
        self.assertEqual(kwargs["messages"][-1]["content"], "Extract")

    def test_parse_decodes_to_dict(self):
        provider = _openai_provider()
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        resp = MagicMock()
        msg = MagicMock(content='{"name": "Ada", "age": 36}', tool_calls=None)
        resp.choices = [MagicMock(message=msg, finish_reason="stop")]
        out = provider._parse_response(resp, req)
        self.assertEqual(out.structured_data, {"name": "Ada", "age": 36})

    def test_parse_failure_raises_retryable(self):
        provider = _openai_provider()
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        resp = MagicMock()
        msg = MagicMock(content="totally not json", tool_calls=None)
        resp.choices = [MagicMock(message=msg, finish_reason="stop")]
        with self.assertRaises(JSONParseFailedError) as ctx:
            provider._parse_response(resp, req)
        self.assertTrue(ctx.exception.retryable)


# ─── Ollama ────────────────────────────────────────────────────────────────


class TestOllamaResponseSchema(unittest.TestCase):
    def test_format_field_set(self):
        provider = _ollama_provider()
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        payload = provider._build_payload(req)
        self.assertEqual(payload["format"], _SCHEMA)

    def test_control_keys_not_leaked_into_payload(self):
        provider = _ollama_provider(
            extra_settings={
                "disable_thinking": True,
                "structured_output_format": "json_object",
                "keep_alive": "10m",
            }
        )
        req = ChatRequest(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(req)
        self.assertNotIn("disable_thinking", payload)
        self.assertNotIn("structured_output_format", payload)
        self.assertEqual(payload["keep_alive"], "10m")  # genuine ollama option still forwarded

    def test_parse_decodes_to_dict(self):
        provider = _ollama_provider()
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        data = {"message": {"content": '{"name": "Bo", "age": 7}'}}
        out = provider._parse_response(data, req)
        self.assertEqual(out.structured_data, {"name": "Bo", "age": 7})


# ─── Anthropic (forced synthetic tool) ───────────────────────────────────────


class TestAnthropicResponseSchema(unittest.TestCase):
    def test_forces_synthetic_tool(self):
        provider = _anthropic_provider()
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        kwargs = provider._build_kwargs(req)
        self.assertEqual(len(kwargs["tools"]), 1)
        self.assertEqual(kwargs["tools"][0]["name"], "emit_structured_output")
        self.assertEqual(kwargs["tools"][0]["input_schema"], _SCHEMA)
        self.assertEqual(kwargs["tool_choice"], {"type": "tool", "name": "emit_structured_output"})

    def test_parse_reads_tool_arguments(self):
        provider = _anthropic_provider()
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        block = MagicMock()
        block.type = "tool_use"
        block.id = "call_1"
        block.name = "emit_structured_output"
        block.input = {"name": "Ada", "age": 36}
        resp = MagicMock(content=[block], stop_reason="tool_use")
        out = provider._parse_response(resp, req)
        self.assertEqual(out.structured_data, {"name": "Ada", "age": 36})
        self.assertIsNone(out.tool_calls)  # synthetic tool is hidden from the caller

    def test_parse_missing_tool_raises(self):
        provider = _anthropic_provider()
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I won't comply"
        resp = MagicMock(content=[text_block], stop_reason="end_turn")
        with self.assertRaises(JSONParseFailedError):
            provider._parse_response(resp, req)


# ─── Gemini ──────────────────────────────────────────────────────────────────


@unittest.skipUnless(_HAS_GEMINI_SDK, "google-genai SDK not installed")
class TestGeminiResponseSchema(unittest.TestCase):
    def _provider(self):
        from llm_provider.providers.gemini_provider import GeminiProvider

        p = GeminiProvider.__new__(GeminiProvider)
        p._config = ProviderConfig(
            host="https://generativelanguage.googleapis.com",
            default_model="gemini-1.5-flash",
            api_key="test-key",
        )
        p._client = MagicMock()
        p._rate_limit_rpm = 0
        p._last_request_at = 0.0
        p._rate_lock = threading.Lock()
        p._retry_config = _retry_cfg()
        return p

    def test_response_schema_and_mime_set(self):
        provider = self._provider()
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        captured = {}
        from google.genai import types

        with patch.object(
            types, "GenerateContentConfig", side_effect=lambda **kw: captured.update(kw)
        ):
            provider._build_kwargs(req)
        self.assertEqual(captured["response_mime_type"], "application/json")
        # additionalProperties-free schema survives the gemini sanitizer
        self.assertEqual(captured["response_schema"]["properties"], _SCHEMA["properties"])

    def test_parse_decodes_to_dict(self):
        provider = self._provider()
        req = ChatRequest(messages=[Message(role="user", content="hi")], response_schema=_SCHEMA)
        resp = MagicMock()
        resp.text = '{"name": "Cy", "age": 5}'
        resp.candidates = []
        resp.function_calls = None
        out = provider._parse_response(resp, req)
        self.assertEqual(out.structured_data, {"name": "Cy", "age": 5})


# ─── /no_think prefix ─────────────────────────────────────────────────────────


class TestNoThinkPrefix(unittest.TestCase):
    def test_applied_for_qwen3_when_enabled(self):
        p = _openai_provider(extra_settings={"disable_thinking": True})
        p._config.default_model = "mlx-community/Qwen3.5-9B-MLX-4bit"
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            system_prompt=SystemPrompt(content="Be terse."),
        )
        kwargs = p._build_kwargs(req)
        self.assertEqual(kwargs["messages"][0]["role"], "system")
        self.assertTrue(kwargs["messages"][0]["content"].startswith("/no_think\n"))

    def test_injected_even_without_system_prompt(self):
        p = _openai_provider(extra_settings={"disable_thinking": True})
        p._config.default_model = "Qwen3-8B"
        req = ChatRequest(messages=[Message(role="user", content="hi")])
        kwargs = p._build_kwargs(req)
        self.assertTrue(kwargs["messages"][0]["content"].startswith("/no_think"))

    def test_noop_for_non_qwen3(self):
        p = _openai_provider(extra_settings={"disable_thinking": True})
        p._config.default_model = "gpt-4o"
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            system_prompt=SystemPrompt(content="Be terse."),
        )
        kwargs = p._build_kwargs(req)
        self.assertEqual(kwargs["messages"][0]["content"], "Be terse.")

    def test_noop_when_disabled(self):
        p = _ollama_provider()  # disable_thinking not set
        p._config.default_model = "qwen3:8b"
        req = ChatRequest(
            messages=[Message(role="user", content="hi")],
            system_prompt=SystemPrompt(content="Hello."),
        )
        payload = p._build_payload(req)
        self.assertEqual(payload["messages"][0]["content"], "Hello.")


# ─── Concurrency semaphore + mlx provider ────────────────────────────────────


class TestConcurrencyAndMlx(unittest.IsolatedAsyncioTestCase):
    def test_provider_config_carries_max_concurrent(self):
        cfg = ProviderConfig(host="h", default_model="m", max_concurrent=3)
        self.assertEqual(cfg.max_concurrent, 3)

    async def test_semaphore_limits_concurrency(self):
        from llm_provider.providers.base_provider import BaseProvider

        class _Dummy(BaseProvider):
            def chat(self, request):  # pragma: no cover - not used
                raise NotImplementedError

            async def achat(self, request):
                async def _op():
                    return await self._track()

                return await self._arun_with_limit(_op, "achat")

            def list_models(self):  # pragma: no cover
                return []

            def name(self):  # pragma: no cover
                return "dummy"

            def is_available(self):  # pragma: no cover
                return True

            def supported_features(self):  # pragma: no cover
                return None

        provider = _Dummy(ProviderConfig(host="h", default_model="m", max_concurrent=2))
        state = {"active": 0, "peak": 0}

        async def _track():
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.02)
            state["active"] -= 1
            return "ok"

        provider._track = _track
        await asyncio.gather(*[provider.achat(None) for _ in range(6)])
        self.assertLessEqual(state["peak"], 2)

    def test_create_mlx_provider_defaults(self):
        from llm_provider.providers import create_mlx_provider

        with patch.dict(sys.modules, {"openai": MagicMock()}):
            provider = create_mlx_provider({"default_model": "mlx-community/Qwen3.5-9B"})
        es = provider._config.extra_settings
        self.assertEqual(es["base_url"], "http://localhost:8000/v1")
        self.assertEqual(es["structured_output_format"], "json_object")
        self.assertEqual(provider._config.api_key, "mlx-lm")


if __name__ == "__main__":
    unittest.main()
