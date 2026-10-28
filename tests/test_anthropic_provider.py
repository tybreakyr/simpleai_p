"""
Unit tests for the Anthropic provider implementation.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from llm_provider.errors import (
    ConnectionFailedError,
    LLMError,
    ModelNotAvailableError,
    RateLimitExceededError,
    TimeoutError,
)
from llm_provider.models import ChatRequest, Message, ProviderConfig, SystemPrompt, ToolCall
from llm_provider.retry import RetryConfig


def _make_config(**kwargs):
    defaults = dict(
        host="https://api.anthropic.com",
        default_model="claude-sonnet-4-6",
        api_key="test-key",
        timeout=30.0,
        retry_attempts=1,
    )
    defaults.update(kwargs)
    return ProviderConfig(**defaults)


def _make_provider(config=None):
    """Return an AnthropicProvider with the anthropic SDK mocked out."""
    if config is None:
        config = _make_config()

    with patch.dict(sys.modules, {"anthropic": MagicMock()}):
        from llm_provider.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._config = config
        provider._client = MagicMock()
        provider._max_tokens = 8192
        provider._retry_config = RetryConfig(
            max_retries=config.retry_attempts,
            base_delay=0.01,
            max_delay=0.01,
            backoff_factor=1.0,
        )
        return provider


def _make_message_response(text="Hello from Anthropic"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _make_fake_anthropic_exc():
    """Build a fake anthropic module with typed exception classes."""
    fake = MagicMock()
    fake.RateLimitError = type("RateLimitError", (Exception,), {})
    fake.APITimeoutError = type("APITimeoutError", (Exception,), {})
    fake.APIConnectionError = type("APIConnectionError", (Exception,), {})
    fake.NotFoundError = type("NotFoundError", (Exception,), {})
    fake.AuthenticationError = type("AuthenticationError", (Exception,), {})
    return fake


class TestAnthropicProviderInit(unittest.TestCase):
    def test_requires_api_key(self):
        config = _make_config(api_key=None)
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            from llm_provider.providers.anthropic_provider import AnthropicProvider

            with self.assertRaises(ValueError):
                AnthropicProvider(config)

    def test_name(self):
        self.assertEqual(_make_provider().name(), "anthropic")

    def test_supported_features(self):
        features = _make_provider().supported_features()
        self.assertTrue(features.structured_output)
        self.assertTrue(features.streaming)
        self.assertTrue(features.vision)
        self.assertTrue(features.function_calling)
        self.assertTrue(features.temperature)
        self.assertTrue(features.top_p)
        self.assertEqual(features.context_window, 200_000)

    def test_default_max_tokens(self):
        provider = _make_provider()
        self.assertEqual(provider._max_tokens, 8192)

    def test_custom_max_tokens_from_extra_settings(self):
        config = _make_config(extra_settings={"max_tokens": 4096})
        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            from llm_provider.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider.__new__(AnthropicProvider)
            provider._config = config
            provider._client = MagicMock()
            provider._max_tokens = int(config.extra_settings.get("max_tokens", 8192))
            provider._retry_config = RetryConfig(
                max_retries=1, base_delay=0.01, max_delay=0.01, backoff_factor=1.0
            )
        self.assertEqual(provider._max_tokens, 4096)


class TestAnthropicProviderChat(unittest.TestCase):
    def setUp(self):
        self.provider = _make_provider()

    def test_chat_success(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        response = self.provider.chat(request)
        self.assertEqual(response.message, "Hello from Anthropic")

    def test_chat_concatenates_multiple_blocks(self):
        b1, b2 = MagicMock(), MagicMock()
        b1.type = "text"
        b1.text = "Hello"
        b2.type = "text"
        b2.text = " world"
        resp = MagicMock()
        resp.content = [b1, b2]
        self.provider._client.messages.create.return_value = resp
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        response = self.provider.chat(request)
        self.assertEqual(response.message, "Hello world")

    def test_chat_with_system_prompt(self):
        self.provider._client.messages.create.return_value = _make_message_response("OK")
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            system_prompt=SystemPrompt(content="Be concise"),
        )
        self.provider.chat(request)
        call_kwargs = self.provider._client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["system"], "Be concise")

    def test_chat_without_system_prompt_omits_system_key(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        self.provider.chat(request)
        call_kwargs = self.provider._client.messages.create.call_args[1]
        self.assertNotIn("system", call_kwargs)

    def test_chat_uses_request_model(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model="claude-haiku-4-5",
        )
        self.provider.chat(request)
        call_kwargs = self.provider._client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "claude-haiku-4-5")

    def test_chat_falls_back_to_default_model(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        self.provider.chat(request)
        call_kwargs = self.provider._client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "claude-sonnet-4-6")

    def test_chat_always_sends_max_tokens(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        self.provider.chat(request)
        call_kwargs = self.provider._client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["max_tokens"], 8192)

    def test_chat_passes_temperature(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        request = ChatRequest(messages=[Message(role="user", content="Hi")], temperature=0.5)
        self.provider.chat(request)
        call_kwargs = self.provider._client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.5)

    def test_chat_passes_top_p(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        request = ChatRequest(messages=[Message(role="user", content="Hi")], top_p=0.95)
        self.provider.chat(request)
        call_kwargs = self.provider._client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["top_p"], 0.95)

    def test_chat_omits_temperature_when_not_set(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        self.provider.chat(request)
        call_kwargs = self.provider._client.messages.create.call_args[1]
        self.assertNotIn("temperature", call_kwargs)

    def test_chat_messages_mapped_correctly(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        request = ChatRequest(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
                Message(role="user", content="How are you?"),
            ]
        )
        self.provider.chat(request)
        call_kwargs = self.provider._client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0], {"role": "user", "content": "Hello"})
        self.assertEqual(messages[1], {"role": "assistant", "content": "Hi there"})
        self.assertEqual(messages[2], {"role": "user", "content": "How are you?"})

    def test_chat_skips_blocks_without_text_attr(self):
        block_with_text = MagicMock(spec=["text", "type"])
        block_with_text.type = "text"
        block_with_text.text = "Hello"
        block_without_text = MagicMock(spec=["type", "id", "name", "input"])  # no 'text' attribute
        block_without_text.type = "tool_use"
        block_without_text.id = "call_1"
        block_without_text.name = "func"
        block_without_text.input = {}
        resp = MagicMock()
        resp.content = [block_with_text, block_without_text]
        self.provider._client.messages.create.return_value = resp
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        response = self.provider.chat(request)
        self.assertEqual(response.message, "Hello")


def _block(type_, **fields):
    block = MagicMock()
    block.type = type_
    for key, value in fields.items():
        setattr(block, key, value)
    return block


class TestAnthropicSamplingParams(unittest.TestCase):
    """Current models 400 on non-default sampling, so it is only sent to older ones."""

    def setUp(self):
        self.provider = _make_provider()
        self.provider._client.messages.create.return_value = _make_message_response()

    def _sent(self, model):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model=model,
            temperature=0.7,
            top_p=0.9,
        )
        self.provider.chat(request)
        return self.provider._client.messages.create.call_args[1]

    def test_kept_for_models_that_accept_it(self):
        for model in (
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-3-5-haiku-20241022",
        ):
            with self.subTest(model=model):
                kwargs = self._sent(model)
                self.assertEqual(kwargs["temperature"], 0.7)
                self.assertEqual(kwargs["top_p"], 0.9)

    def test_dropped_for_current_models(self):
        for model in (
            "claude-haiku-5-5",
            "claude-sonnet-5-5",
            "claude-sonnet-5",
            "claude-opus-5-5",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-fable-5-1",
        ):
            with self.subTest(model=model):
                kwargs = self._sent(model)
                self.assertNotIn("temperature", kwargs)
                self.assertNotIn("top_p", kwargs)

    def test_judged_on_the_default_model_when_request_has_none(self):
        provider = _make_provider(_make_config(default_model="claude-haiku-5-5"))
        provider._client.messages.create.return_value = _make_message_response()
        provider.chat(ChatRequest(messages=[Message(role="user", content="Hi")], temperature=0.0))
        self.assertNotIn("temperature", provider._client.messages.create.call_args[1])


class TestAnthropicThinkingRoundTrip(unittest.TestCase):
    """A tool-use turn's thinking blocks must go back to the API unmodified."""

    THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    REDACTED = {"type": "redacted_thinking", "data": "opaque"}

    def setUp(self):
        self.provider = _make_provider()

    def test_thinking_rides_on_every_tool_call(self):
        resp = MagicMock()
        resp.content = [
            _block("thinking", thinking="", signature="sig-abc"),
            _block("redacted_thinking", data="opaque"),
            _block("text", text="Looking that up."),
            _block("tool_use", id="tu_1", name="lookup", input={"q": "a"}),
            _block("tool_use", id="tu_2", name="lookup", input={"q": "b"}),
        ]
        self.provider._client.messages.create.return_value = resp
        response = self.provider.chat(ChatRequest(messages=[Message(role="user", content="Hi")]))

        self.assertEqual(response.message, "Looking that up.")
        for call in response.tool_calls:
            self.assertEqual(call.thought_signature, [self.THINKING, self.REDACTED])

    def test_no_thinking_leaves_calls_bare(self):
        resp = MagicMock()
        resp.content = [_block("tool_use", id="tu_1", name="lookup", input={})]
        self.provider._client.messages.create.return_value = resp
        response = self.provider.chat(ChatRequest(messages=[Message(role="user", content="Hi")]))
        self.assertIsNone(response.tool_calls[0].thought_signature)

    def _history(self, calls):
        return [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Looking that up.", tool_calls=calls),
            Message(role="tool", content="result", tool_call_id=calls[0].id),
        ]

    def test_thinking_leads_the_echoed_assistant_turn(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        thinking = [self.THINKING, self.REDACTED]
        calls = [
            ToolCall(id="tu_1", name="lookup", arguments={"q": "a"}, thought_signature=thinking),
            ToolCall(id="tu_2", name="lookup", arguments={"q": "b"}, thought_signature=thinking),
        ]
        self.provider.chat(ChatRequest(messages=self._history(calls)))

        assistant = self.provider._client.messages.create.call_args[1]["messages"][1]
        types = [block["type"] for block in assistant["content"]]
        self.assertEqual(types, ["thinking", "redacted_thinking", "text", "tool_use", "tool_use"])
        self.assertEqual(assistant["content"][0], self.THINKING)

    def test_survives_echoing_a_subset_of_calls(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        calls = [
            ToolCall(id="tu_2", name="lookup", arguments={}, thought_signature=[self.THINKING])
        ]
        self.provider.chat(ChatRequest(messages=self._history(calls)))

        assistant = self.provider._client.messages.create.call_args[1]["messages"][1]
        self.assertEqual(assistant["content"][0], self.THINKING)

    def test_calls_without_thinking_send_no_thinking_block(self):
        self.provider._client.messages.create.return_value = _make_message_response()
        calls = [ToolCall(id="tu_1", name="lookup", arguments={})]
        self.provider.chat(ChatRequest(messages=self._history(calls)))

        assistant = self.provider._client.messages.create.call_args[1]["messages"][1]
        self.assertEqual([b["type"] for b in assistant["content"]], ["text", "tool_use"])


class TestAnthropicProviderErrorClassification(unittest.TestCase):
    def setUp(self):
        self.provider = _make_provider()

    def test_rate_limit_error(self):
        fake = _make_fake_anthropic_exc()
        exc = fake.RateLimitError("rate limited")
        exc.response = None
        with patch.dict(sys.modules, {"anthropic": fake}):
            with self.assertRaises(RateLimitExceededError):
                self.provider._classify_anthropic_error(exc)

    def test_rate_limit_error_with_retry_after_header(self):
        fake = _make_fake_anthropic_exc()
        exc = fake.RateLimitError("rate limited")
        exc.response = MagicMock()
        exc.response.headers = {"retry-after": "30"}
        with patch.dict(sys.modules, {"anthropic": fake}):
            with self.assertRaises(RateLimitExceededError) as ctx:
                self.provider._classify_anthropic_error(exc)
        self.assertEqual(ctx.exception.retry_after, 30.0)

    def test_timeout_error(self):
        fake = _make_fake_anthropic_exc()
        exc = fake.APITimeoutError("timed out")
        with patch.dict(sys.modules, {"anthropic": fake}):
            with self.assertRaises(TimeoutError):
                self.provider._classify_anthropic_error(exc)

    def test_connection_error(self):
        fake = _make_fake_anthropic_exc()
        exc = fake.APIConnectionError("connection refused")
        with patch.dict(sys.modules, {"anthropic": fake}):
            with self.assertRaises(ConnectionFailedError):
                self.provider._classify_anthropic_error(exc)

    def test_not_found_error(self):
        fake = _make_fake_anthropic_exc()
        exc = fake.NotFoundError("model not found")
        with patch.dict(sys.modules, {"anthropic": fake}):
            with self.assertRaises(ModelNotAvailableError):
                self.provider._classify_anthropic_error(exc)

    def test_auth_error(self):
        fake = _make_fake_anthropic_exc()
        exc = fake.AuthenticationError("bad api key")
        with patch.dict(sys.modules, {"anthropic": fake}):
            with self.assertRaises(LLMError) as ctx:
                self.provider._classify_anthropic_error(exc)
        self.assertFalse(ctx.exception.retryable)

    def test_llm_error_passthrough(self):
        already_classified = RateLimitExceededError(
            message="already wrapped", operation="chat", cause=None
        )
        self.provider._client.messages.create.side_effect = already_classified
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        with self.assertRaises(RateLimitExceededError):
            self.provider.chat(request)


class TestAnthropicProviderListModels(unittest.TestCase):
    def test_list_models(self):
        provider = _make_provider()
        m1, m2 = MagicMock(), MagicMock()
        m1.id = "claude-opus-4-7"
        m2.id = "claude-sonnet-4-6"
        provider._client.models.list.return_value = [m1, m2]
        models = provider.list_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, "claude-opus-4-7")
        self.assertEqual(models[1].name, "claude-sonnet-4-6")

    def test_is_available_true(self):
        provider = _make_provider()
        provider._client.models.list.return_value = iter([MagicMock()])
        self.assertTrue(provider.is_available())

    def test_is_available_false_on_exception(self):
        provider = _make_provider()
        provider._client.models.list.side_effect = Exception("network error")
        self.assertFalse(provider.is_available())


class TestCreateAnthropicProvider(unittest.TestCase):
    def test_factory_creates_provider(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            from llm_provider.providers.anthropic_provider import (
                AnthropicProvider,
                create_anthropic_provider,
            )

            provider = create_anthropic_provider(
                {
                    "api_key": "test-key",
                    "default_model": "claude-sonnet-4-6",
                }
            )
        self.assertIsInstance(provider, AnthropicProvider)
        self.assertEqual(provider.name(), "anthropic")

    def test_factory_requires_api_key(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            from llm_provider.providers.anthropic_provider import create_anthropic_provider

            with self.assertRaises(ValueError):
                create_anthropic_provider({"default_model": "claude-sonnet-4-6"})

    def test_factory_defaults_model(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            from llm_provider.providers.anthropic_provider import create_anthropic_provider

            provider = create_anthropic_provider({"api_key": "test-key"})
        self.assertEqual(provider._config.default_model, "claude-sonnet-4-6")

    def test_factory_respects_max_tokens_extra_setting(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            from llm_provider.providers.anthropic_provider import create_anthropic_provider

            provider = create_anthropic_provider(
                {
                    "api_key": "test-key",
                    "extra_settings": {"max_tokens": 4096},
                }
            )
        self.assertEqual(provider._max_tokens, 4096)

    def test_factory_sets_correct_host(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            from llm_provider.providers.anthropic_provider import create_anthropic_provider

            provider = create_anthropic_provider({"api_key": "test-key"})
        self.assertEqual(provider._config.host, "https://api.anthropic.com")


if __name__ == "__main__":
    unittest.main()


class TestUsage(unittest.TestCase):
    """Token counts ride the response so callers can measure what a call cost."""

    def _parse(self, usage):

        from llm_provider.models import ChatRequest, Message

        provider = _make_provider()
        response = _make_message_response("hi")
        response.usage = usage
        req = ChatRequest(messages=[Message(role="user", content="x")])
        return provider._parse_response(response, req)

    def test_reported_token_counts_are_kept(self):
        from types import SimpleNamespace

        out = self._parse(
            SimpleNamespace(
                input_tokens=1200,
                output_tokens=340,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=None,
            )
        )
        self.assertEqual(
            out.usage, {"input_tokens": 1200, "output_tokens": 340, "cache_read_input_tokens": 0}
        )

    def test_no_usage_reported_is_none(self):
        self.assertIsNone(self._parse(None).usage)
