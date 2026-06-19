import asyncio
import json
import sys
import threading
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from llm_provider.models import ChatRequest, Message, ToolSchema, ToolCall, ProviderConfig
from llm_provider.retry import RetryConfig


def _retry_cfg(attempts=1):
    return RetryConfig(max_retries=attempts, base_delay=0.01, max_delay=0.01, backoff_factor=1.0)


# ─── Provider factory helpers ────────────────────────────────────────────────

def _anthropic_provider(async_client=None):
    with patch.dict(sys.modules, {"anthropic": MagicMock()}):
        from llm_provider.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider.__new__(AnthropicProvider)
    p._config = ProviderConfig(
        host="https://api.anthropic.com", default_model="claude-3-5-sonnet", api_key="test-key"
    )
    p._client = MagicMock()
    p._max_tokens = 8192
    p._retry_config = _retry_cfg()
    if async_client is not None:
        p._async_client = async_client
    return p


def _openai_provider(async_client=None):
    with patch.dict(sys.modules, {"openai": MagicMock()}):
        from llm_provider.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider.__new__(OpenAIProvider)
    p._config = ProviderConfig(
        host="https://api.openai.com/v1", default_model="gpt-4o", api_key="test-key"
    )
    p._client = MagicMock()
    p._retry_config = _retry_cfg()
    if async_client is not None:
        p._async_client = async_client
    return p


def _gemini_provider():
    mock_genai = MagicMock()
    with patch.dict(sys.modules, {
        "google": MagicMock(),
        "google.genai": mock_genai,
        "google.genai.types": mock_genai.types,
    }):
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


def _ollama_provider(async_client=None):
    with patch.dict(sys.modules, {"requests": MagicMock(), "httpx": MagicMock()}):
        from llm_provider.providers.ollama_provider import OllamaProvider
        p = OllamaProvider.__new__(OllamaProvider)
    p._config = ProviderConfig(
        host="http://localhost:11434", default_model="llama3", api_key=""
    )
    p._base_url = "http://localhost:11434/api"
    p._session = MagicMock()
    p._retry_config = _retry_cfg()
    if async_client is not None:
        p._async_client = async_client
    return p


# ─── Anthropic sync ──────────────────────────────────────────────────────────

class TestAnthropicToolCalling(unittest.TestCase):
    def setUp(self):
        self.provider = _anthropic_provider()

    def _text_response(self, text="ok"):
        r = MagicMock()
        r.content = [MagicMock(type="text", text=text)]
        r.stop_reason = "end_turn"
        return r

    def _tool_response(self, *tool_specs):
        r = MagicMock()
        blocks = []
        for call_id, name, args in tool_specs:
            b = MagicMock(type="tool_use", id=call_id, input=args)
            b.name = name
            blocks.append(b)
        r.content = blocks
        r.stop_reason = "tool_use"
        return r

    def test_chat_with_tools_passes_schema_to_sdk(self):
        self.provider._client.messages.create.return_value = self._text_response()
        tools = [ToolSchema(name="weather", description="get weather", input_schema={"type": "object"})]
        self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")], tools=tools))

        kwargs = self.provider._client.messages.create.call_args.kwargs
        self.assertIn("tools", kwargs)
        self.assertEqual(kwargs["tools"], [
            {"name": "weather", "description": "get weather", "input_schema": {"type": "object"}}
        ])

    def test_response_parses_tool_calls(self):
        self.provider._client.messages.create.return_value = self._tool_response(
            ("call_123", "weather", {"loc": "NY"})
        )
        resp = self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")]))
        self.assertIsNotNone(resp.tool_calls)
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].id, "call_123")
        self.assertEqual(resp.tool_calls[0].name, "weather")
        self.assertEqual(resp.tool_calls[0].arguments, {"loc": "NY"})
        self.assertEqual(resp.stop_reason, "tool_use")

    def test_multiple_tool_calls_in_one_response(self):
        self.provider._client.messages.create.return_value = self._tool_response(
            ("t1", "weather", {"loc": "NY"}),
            ("t2", "weather", {"loc": "LA"}),
        )
        resp = self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")]))
        self.assertEqual(len(resp.tool_calls), 2)

    def test_response_without_tool_calls_has_none(self):
        self.provider._client.messages.create.return_value = self._text_response("hello")
        resp = self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")]))
        self.assertIsNone(resp.tool_calls)
        self.assertEqual(resp.stop_reason, "end_turn")

    def test_tool_choice_forwarded(self):
        self.provider._client.messages.create.return_value = self._text_response()
        self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")], tool_choice="any"))
        kwargs = self.provider._client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], {"type": "any"})

    def test_stop_reason_populated_for_tool_use(self):
        self.provider._client.messages.create.return_value = self._tool_response(
            ("id1", "fn", {})
        )
        resp = self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")]))
        self.assertEqual(resp.stop_reason, "tool_use")


# ─── Anthropic async ─────────────────────────────────────────────────────────

class TestAnthropicToolCallingAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.async_client = MagicMock()
        self.provider = _anthropic_provider(async_client=self.async_client)

    def _tool_response(self, call_id, name, args):
        r = MagicMock()
        b = MagicMock(type="tool_use", id=call_id, input=args)
        b.name = name
        r.content = [b]
        r.stop_reason = "tool_use"
        return r

    async def test_achat_with_tools_returns_tool_calls(self):
        self.async_client.messages.create = AsyncMock(
            return_value=self._tool_response("call_abc", "weather", {"loc": "SF"})
        )
        tools = [ToolSchema(name="weather", description="get weather", input_schema={"type": "object"})]
        resp = await self.provider.achat(
            ChatRequest(messages=[Message(role="user", content="hi")], tools=tools)
        )
        self.assertIsNotNone(resp.tool_calls)
        self.assertEqual(resp.tool_calls[0].id, "call_abc")
        self.assertEqual(resp.tool_calls[0].name, "weather")
        self.assertEqual(resp.tool_calls[0].arguments, {"loc": "SF"})
        self.assertEqual(resp.stop_reason, "tool_use")

    async def test_achat_tools_schema_forwarded_to_sdk(self):
        r = MagicMock()
        r.content = [MagicMock(type="text", text="ok")]
        r.stop_reason = "end_turn"
        self.async_client.messages.create = AsyncMock(return_value=r)
        tools = [ToolSchema(name="search", description="web search", input_schema={"type": "object"})]
        await self.provider.achat(
            ChatRequest(messages=[Message(role="user", content="hi")], tools=tools)
        )
        kwargs = self.async_client.messages.create.call_args.kwargs
        self.assertIn("tools", kwargs)
        self.assertEqual(kwargs["tools"][0]["name"], "search")


# ─── OpenAI sync ─────────────────────────────────────────────────────────────

class TestOpenAIToolCalling(unittest.TestCase):
    def setUp(self):
        self.provider = _openai_provider()

    def _text_completion(self, text="ok"):
        choice = MagicMock()
        choice.message.content = text
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        r = MagicMock()
        r.choices = [choice]
        return r

    def _tool_completion(self, *tool_specs):
        tcs = []
        for call_id, name, args in tool_specs:
            fn = MagicMock()
            fn.name = name
            fn.arguments = json.dumps(args)
            tcs.append(MagicMock(id=call_id, type="function", function=fn))
        choice = MagicMock()
        choice.message.content = ""
        choice.message.tool_calls = tcs
        choice.finish_reason = "tool_calls"
        r = MagicMock()
        r.choices = [choice]
        return r

    def test_chat_with_tools_passes_schema_to_sdk(self):
        self.provider._client.chat.completions.create.return_value = self._text_completion()
        tools = [ToolSchema(name="weather", description="get weather", input_schema={"type": "object"})]
        self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")], tools=tools))
        kwargs = self.provider._client.chat.completions.create.call_args.kwargs
        self.assertIn("tools", kwargs)
        self.assertEqual(kwargs["tools"], [{
            "type": "function",
            "function": {"name": "weather", "description": "get weather", "parameters": {"type": "object"}}
        }])

    def test_response_parses_tool_calls(self):
        self.provider._client.chat.completions.create.return_value = self._tool_completion(
            ("call_123", "weather", {"loc": "NY"})
        )
        resp = self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")]))
        self.assertIsNotNone(resp.tool_calls)
        self.assertEqual(resp.tool_calls[0].id, "call_123")
        self.assertEqual(resp.tool_calls[0].name, "weather")
        self.assertEqual(resp.tool_calls[0].arguments, {"loc": "NY"})
        self.assertEqual(resp.stop_reason, "tool_calls")

    def test_tool_choice_forwarded(self):
        self.provider._client.chat.completions.create.return_value = self._text_completion()
        self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")], tool_choice="weather"))
        kwargs = self.provider._client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], {"type": "function", "function": {"name": "weather"}})

    def test_tool_choice_auto(self):
        self.provider._client.chat.completions.create.return_value = self._text_completion()
        self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")], tool_choice="auto"))
        kwargs = self.provider._client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], "auto")

    def test_tool_choice_any_maps_to_required(self):
        self.provider._client.chat.completions.create.return_value = self._text_completion()
        self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")], tool_choice="any"))
        kwargs = self.provider._client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], "required")


# ─── OpenAI async ────────────────────────────────────────────────────────────

class TestOpenAIToolCallingAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.async_client = MagicMock()
        self.provider = _openai_provider(async_client=self.async_client)

    async def test_achat_with_tools_returns_tool_calls(self):
        fn = MagicMock()
        fn.name = "weather"
        fn.arguments = json.dumps({"loc": "NY"})
        tc = MagicMock(id="call_xyz", type="function", function=fn)
        choice = MagicMock()
        choice.message.content = "Checking."
        choice.message.tool_calls = [tc]
        choice.finish_reason = "tool_calls"
        r = MagicMock()
        r.choices = [choice]
        self.async_client.chat.completions.create = AsyncMock(return_value=r)

        tools = [ToolSchema(name="weather", description="get weather", input_schema={"type": "object"})]
        resp = await self.provider.achat(
            ChatRequest(messages=[Message(role="user", content="hi")], tools=tools)
        )
        self.assertIsNotNone(resp.tool_calls)
        self.assertEqual(resp.tool_calls[0].id, "call_xyz")
        self.assertEqual(resp.tool_calls[0].name, "weather")
        self.assertEqual(resp.tool_calls[0].arguments, {"loc": "NY"})

    async def test_achat_tool_choice_forwarded(self):
        choice = MagicMock()
        choice.message.content = "ok"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        r = MagicMock()
        r.choices = [choice]
        self.async_client.chat.completions.create = AsyncMock(return_value=r)
        await self.provider.achat(
            ChatRequest(messages=[Message(role="user", content="hi")], tool_choice="any")
        )
        kwargs = self.async_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], "required")


# ─── Gemini sync ─────────────────────────────────────────────────────────────

class TestGeminiToolCalling(unittest.TestCase):
    def setUp(self):
        mock_genai = MagicMock()
        self.patcher = patch.dict(sys.modules, {
            "google": MagicMock(),
            "google.genai": mock_genai,
            "google.genai.types": mock_genai.types,
        })
        self.patcher.start()
        self.provider = _gemini_provider()

    def tearDown(self):
        self.patcher.stop()

    def test_chat_with_tools_passes_schema_to_sdk(self):
        self.provider._client.models.generate_content.return_value = MagicMock(
            text="hello", function_calls=[]
        )
        tools = [ToolSchema(name="weather", description="get weather", input_schema={"type": "object"})]
        self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")], tools=tools))
        self.provider._client.models.generate_content.assert_called_once()

    def test_response_parses_tool_calls(self):
        fc = MagicMock()
        fc.name = "weather"
        fc.args = {"loc": "NY"}
        fc.id = "call_123"
        self.provider._client.models.generate_content.return_value = MagicMock(
            text="ok", function_calls=[fc]
        )
        resp = self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")]))
        self.assertIsNotNone(resp.tool_calls)
        self.assertEqual(resp.tool_calls[0].name, "weather")
        self.assertEqual(resp.tool_calls[0].arguments, {"loc": "NY"})
        self.assertEqual(resp.tool_calls[0].id, "call_123")

    def test_multiple_tool_calls_get_unique_fallback_ids(self):
        fc1 = MagicMock()
        fc1.name = "weather"
        fc1.args = {"loc": "NY"}
        fc1.id = None
        fc2 = MagicMock()
        fc2.name = "weather"
        fc2.args = {"loc": "LA"}
        fc2.id = None
        self.provider._client.models.generate_content.return_value = MagicMock(
            text="", function_calls=[fc1, fc2]
        )
        resp = self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")]))
        ids = [tc.id for tc in resp.tool_calls]
        self.assertEqual(len(set(ids)), 2, "Duplicate tool calls must receive unique IDs")


# ─── Gemini async ────────────────────────────────────────────────────────────

class TestGeminiToolCallingAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        mock_genai = MagicMock()
        self.patcher = patch.dict(sys.modules, {
            "google": MagicMock(),
            "google.genai": mock_genai,
            "google.genai.types": mock_genai.types,
        })
        self.patcher.start()
        self.provider = _gemini_provider()

    def tearDown(self):
        self.patcher.stop()

    async def test_achat_with_tools_returns_tool_calls(self):
        fc = MagicMock()
        fc.name = "weather"
        fc.args = {"loc": "LA"}
        fc.id = "call_gemini_1"
        self.provider._client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(text="On it.", function_calls=[fc])
        )
        tools = [ToolSchema(name="weather", description="get weather", input_schema={"type": "object"})]
        resp = await self.provider.achat(
            ChatRequest(messages=[Message(role="user", content="hi")], tools=tools)
        )
        self.assertIsNotNone(resp.tool_calls)
        self.assertEqual(resp.tool_calls[0].id, "call_gemini_1")
        self.assertEqual(resp.tool_calls[0].name, "weather")
        self.assertEqual(resp.tool_calls[0].arguments, {"loc": "LA"})

    async def test_achat_duplicate_tool_calls_get_unique_ids(self):
        fc1 = MagicMock()
        fc1.name = "weather"
        fc1.args = {"loc": "NY"}
        fc1.id = None
        fc2 = MagicMock()
        fc2.name = "weather"
        fc2.args = {"loc": "LA"}
        fc2.id = None
        self.provider._client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(text="", function_calls=[fc1, fc2])
        )
        resp = await self.provider.achat(ChatRequest(messages=[Message(role="user", content="hi")]))
        ids = [tc.id for tc in resp.tool_calls]
        self.assertEqual(len(set(ids)), 2, "Duplicate tool calls must get unique IDs")


# ─── Ollama sync ─────────────────────────────────────────────────────────────

class TestOllamaToolCalling(unittest.TestCase):
    def setUp(self):
        self.provider = _ollama_provider()

    def test_chat_with_tools_passes_schema_to_sdk(self):
        self.provider._session.post.return_value = MagicMock(
            json=MagicMock(return_value={"message": {"content": "hello"}})
        )
        tools = [ToolSchema(name="weather", description="get weather", input_schema={"type": "object"})]
        self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")], tools=tools))
        kwargs = self.provider._session.post.call_args.kwargs
        payload = kwargs["json"]
        self.assertIn("tools", payload)
        self.assertEqual(payload["tools"], [{
            "type": "function",
            "function": {"name": "weather", "description": "get weather", "parameters": {"type": "object"}}
        }])

    def test_response_parses_tool_calls(self):
        self.provider._session.post.return_value = MagicMock(json=MagicMock(return_value={
            "message": {
                "content": "Checking.",
                "tool_calls": [{"function": {"name": "weather", "arguments": {"loc": "NY"}}}],
            }
        }))
        resp = self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")]))
        self.assertIsNotNone(resp.tool_calls)
        self.assertEqual(resp.tool_calls[0].name, "weather")
        self.assertEqual(resp.tool_calls[0].arguments, {"loc": "NY"})
        self.assertTrue(resp.tool_calls[0].id.startswith("call_"))

    def test_tool_choice_forwarded(self):
        self.provider._session.post.return_value = MagicMock(
            json=MagicMock(return_value={"message": {"content": "ok"}})
        )
        tools = [ToolSchema(name="weather", description="get weather", input_schema={"type": "object"})]
        self.provider.chat(ChatRequest(
            messages=[Message(role="user", content="hi")], tools=tools, tool_choice="auto"
        ))
        payload = self.provider._session.post.call_args.kwargs["json"]
        self.assertIn("tool_choice", payload)
        self.assertEqual(payload["tool_choice"], "auto")

    def test_multiple_tool_calls_get_unique_ids(self):
        self.provider._session.post.return_value = MagicMock(json=MagicMock(return_value={
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "weather", "arguments": {"loc": "NY"}}},
                    {"function": {"name": "weather", "arguments": {"loc": "LA"}}},
                ],
            }
        }))
        resp = self.provider.chat(ChatRequest(messages=[Message(role="user", content="hi")]))
        ids = [tc.id for tc in resp.tool_calls]
        self.assertEqual(len(set(ids)), 2, "Multiple tool calls must have unique IDs")


# ─── Ollama async ────────────────────────────────────────────────────────────

class TestOllamaToolCallingAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.async_client = MagicMock()
        self.mock_httpx = MagicMock()
        self.patcher = patch.dict(sys.modules, {"requests": MagicMock(), "httpx": self.mock_httpx})
        self.patcher.start()
        self.provider = _ollama_provider(async_client=self.async_client)

    def tearDown(self):
        self.patcher.stop()

    async def test_achat_with_tools_returns_tool_calls(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": "Checking.",
                "tool_calls": [{"function": {"name": "weather", "arguments": {"loc": "Boston"}}}],
            }
        }
        self.async_client.post = AsyncMock(return_value=mock_response)
        tools = [ToolSchema(name="weather", description="get weather", input_schema={"type": "object"})]
        resp = await self.provider.achat(
            ChatRequest(messages=[Message(role="user", content="hi")], tools=tools)
        )
        self.assertIsNotNone(resp.tool_calls)
        self.assertEqual(resp.tool_calls[0].name, "weather")
        self.assertEqual(resp.tool_calls[0].arguments, {"loc": "Boston"})
        self.assertTrue(resp.tool_calls[0].id.startswith("call_"))

    async def test_achat_tool_choice_forwarded(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        self.async_client.post = AsyncMock(return_value=mock_response)
        tools = [ToolSchema(name="weather", description="get weather", input_schema={"type": "object"})]
        await self.provider.achat(ChatRequest(
            messages=[Message(role="user", content="hi")], tools=tools, tool_choice="auto"
        ))
        payload = self.async_client.post.call_args.kwargs.get("json", {})
        self.assertIn("tool_choice", payload)
        self.assertEqual(payload["tool_choice"], "auto")

    async def test_achat_multiple_tool_calls_get_unique_ids(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "weather", "arguments": {"loc": "NY"}}},
                    {"function": {"name": "weather", "arguments": {"loc": "LA"}}},
                ],
            }
        }
        self.async_client.post = AsyncMock(return_value=mock_response)
        resp = await self.provider.achat(ChatRequest(messages=[Message(role="user", content="hi")]))
        ids = [tc.id for tc in resp.tool_calls]
        self.assertEqual(len(set(ids)), 2, "Multiple tool calls must get unique IDs")


# ─── Nested→flat tool-param adaptation (per-model capability) ─────────────────

def _openai_provider_for_model(default_model, extra_settings=None):
    with patch.dict(sys.modules, {"openai": MagicMock()}):
        from llm_provider.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider.__new__(OpenAIProvider)
    p._config = ProviderConfig(
        host="https://api.openai.com/v1",
        default_model=default_model,
        api_key="test-key",
        extra_settings=extra_settings or {},
    )
    p._client = MagicMock()
    p._retry_config = _retry_cfg()
    return p


def _nested_tool():
    return ToolSchema(
        name="submit",
        description="submit a nested payload",
        input_schema={
            "type": "object",
            "properties": {
                "say": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "determination": {"type": "integer"}},
                    "required": ["text", "determination"],
                },
            },
            "required": ["say"],
        },
    )


def _openai_tool_completion(name, args):
    fn = MagicMock()
    fn.name = name
    fn.arguments = json.dumps(args)
    tc = MagicMock(id="call_1", type="function", function=fn)
    choice = MagicMock()
    choice.message.content = ""
    choice.message.tool_calls = [tc]
    choice.finish_reason = "tool_calls"
    r = MagicMock()
    r.choices = [choice]
    return r


class TestNestedToolParamFlattening(unittest.TestCase):
    def test_weak_model_gets_flat_schema_and_renested_args(self):
        provider = _openai_provider_for_model("mlx-community/Qwen3.5-9B-MLX-4bit")
        # The model "sees" flat args and answers in the flat shape.
        provider._client.chat.completions.create.return_value = _openai_tool_completion(
            "submit", {"say__text": "hi", "say__determination": 80}
        )

        resp = provider.chat(ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[_nested_tool()],
            tool_choice="submit",
        ))

        # Wire schema sent to the SDK is flattened (no nested object property).
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        params = kwargs["tools"][0]["function"]["parameters"]
        self.assertEqual(set(params["properties"]), {"say__text", "say__determination"})
        # Forced-tool name is preserved.
        self.assertEqual(kwargs["tool_choice"], {"type": "function", "function": {"name": "submit"}})
        # Returned args are re-nested for the caller.
        self.assertEqual(resp.tool_calls[0].arguments, {"say": {"text": "hi", "determination": 80}})

    def test_capable_model_passes_nested_schema_through(self):
        provider = _openai_provider_for_model("gpt-4o")
        provider._client.chat.completions.create.return_value = _openai_tool_completion(
            "submit", {"say": {"text": "hi", "determination": 80}}
        )
        resp = provider.chat(ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[_nested_tool()],
        ))
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        params = kwargs["tools"][0]["function"]["parameters"]
        self.assertIn("say", params["properties"])
        self.assertEqual(params["properties"]["say"]["type"], "object")
        # Args unchanged (already nested).
        self.assertEqual(resp.tool_calls[0].arguments, {"say": {"text": "hi", "determination": 80}})

    def test_extra_settings_override_forces_flatten_on_capable_model(self):
        provider = _openai_provider_for_model("gpt-4o", extra_settings={"flatten_tool_params": True})
        provider._client.chat.completions.create.return_value = _openai_tool_completion(
            "submit", {"say__text": "hi", "say__determination": 80}
        )
        resp = provider.chat(ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[_nested_tool()],
        ))
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        params = kwargs["tools"][0]["function"]["parameters"]
        self.assertEqual(set(params["properties"]), {"say__text", "say__determination"})
        self.assertEqual(resp.tool_calls[0].arguments, {"say": {"text": "hi", "determination": 80}})

    def test_override_can_disable_flatten_on_weak_model(self):
        provider = _openai_provider_for_model("qwen3", extra_settings={"flatten_tool_params": False})
        provider._client.chat.completions.create.return_value = _openai_tool_completion(
            "submit", {"say": {"text": "hi", "determination": 80}}
        )
        provider.chat(ChatRequest(
            messages=[Message(role="user", content="hi")],
            tools=[_nested_tool()],
        ))
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        params = kwargs["tools"][0]["function"]["parameters"]
        self.assertIn("say", params["properties"])  # left nested


if __name__ == '__main__':
    unittest.main()
