"""
Unit tests for the OpenAI provider implementation.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from llm_provider.models import ProviderConfig, ChatRequest, Message, SystemPrompt
from llm_provider.errors import (
    RateLimitExceededError, TimeoutError, ConnectionFailedError,
    ModelNotAvailableError, LLMError, InvalidResponseError,
)
from llm_provider.retry import RetryConfig


def _make_config(**kwargs):
    defaults = dict(
        host="https://api.openai.com",
        default_model="gpt-4o",
        api_key="test-key",
        timeout=30.0,
        retry_attempts=1,
    )
    defaults.update(kwargs)
    return ProviderConfig(**defaults)


def _make_provider(config=None):
    """Return an OpenAIProvider with the openai SDK mocked out."""
    if config is None:
        config = _make_config()

    with patch.dict(sys.modules, {"openai": MagicMock()}):
        from llm_provider.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._config = config
        provider._client = MagicMock()
        provider._retry_config = RetryConfig(
            max_retries=config.retry_attempts,
            base_delay=0.01,
            max_delay=0.01,
            backoff_factor=1.0,
        )
        return provider


def _make_completion(content="Hello from OpenAI"):
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


class TestOpenAIProviderInit(unittest.TestCase):
    def test_requires_api_key(self):
        config = _make_config(api_key=None)
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from llm_provider.providers.openai_provider import OpenAIProvider
            with self.assertRaises(ValueError):
                OpenAIProvider(config)

    def test_name(self):
        provider = _make_provider()
        self.assertEqual(provider.name(), "openai")

    def test_supported_features(self):
        provider = _make_provider()
        features = provider.supported_features()
        self.assertTrue(features.structured_output)
        self.assertTrue(features.streaming)
        self.assertTrue(features.vision)
        self.assertTrue(features.function_calling)
        self.assertTrue(features.temperature)
        self.assertTrue(features.top_p)
        self.assertEqual(features.context_window, 128_000)


class TestOpenAIProviderChat(unittest.TestCase):
    def setUp(self):
        self.provider = _make_provider()

    def test_chat_success(self):
        self.provider._client.chat.completions.create.return_value = _make_completion()
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        response = self.provider.chat(request)
        self.assertEqual(response.message, "Hello from OpenAI")

    def test_chat_with_system_prompt(self):
        self.provider._client.chat.completions.create.return_value = _make_completion("OK")
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            system_prompt=SystemPrompt(content="Be helpful"),
        )
        response = self.provider.chat(request)
        self.assertEqual(response.message, "OK")

        call_kwargs = self.provider._client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "Be helpful"})
        self.assertEqual(messages[1], {"role": "user", "content": "Hi"})

    def test_chat_uses_request_model(self):
        self.provider._client.chat.completions.create.return_value = _make_completion()
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model="gpt-4o-mini",
        )
        self.provider.chat(request)
        call_kwargs = self.provider._client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4o-mini")

    def test_chat_falls_back_to_default_model(self):
        self.provider._client.chat.completions.create.return_value = _make_completion()
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        self.provider.chat(request)
        call_kwargs = self.provider._client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4o")

    def test_chat_passes_temperature(self):
        self.provider._client.chat.completions.create.return_value = _make_completion()
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            temperature=0.7,
        )
        self.provider.chat(request)
        call_kwargs = self.provider._client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)

    def test_chat_passes_top_p(self):
        self.provider._client.chat.completions.create.return_value = _make_completion()
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            top_p=0.9,
        )
        self.provider.chat(request)
        call_kwargs = self.provider._client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["top_p"], 0.9)

    def test_chat_no_choices_raises(self):
        resp = MagicMock()
        resp.choices = []
        self.provider._client.chat.completions.create.return_value = resp
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        with self.assertRaises(InvalidResponseError):
            self.provider.chat(request)

    def test_chat_none_content_raises(self):
        choice = MagicMock()
        choice.message.content = None
        resp = MagicMock()
        resp.choices = [choice]
        self.provider._client.chat.completions.create.return_value = resp
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        with self.assertRaises(InvalidResponseError):
            self.provider.chat(request)


class TestOpenAIProviderErrorClassification(unittest.TestCase):
    def setUp(self):
        self.provider = _make_provider()

    def _raise_via_chat(self, exc):
        self.provider._client.chat.completions.create.side_effect = exc
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        self.provider.chat(request)

    def _patch_openai_exc(self, cls_name, base=Exception):
        """Dynamically build a fake openai exception class."""
        return type(cls_name, (base,), {})

    def test_rate_limit_error(self):
        import sys
        fake_openai = MagicMock()
        RateLimitErr = type("RateLimitError", (Exception,), {})
        fake_openai.RateLimitError = RateLimitErr
        fake_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
        fake_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
        fake_openai.NotFoundError = type("NotFoundError", (Exception,), {})
        fake_openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
        fake_openai.APIStatusError = type("APIStatusError", (Exception,), {})

        exc = RateLimitErr("rate limited")
        exc.response = None

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with self.assertRaises(RateLimitExceededError):
                self.provider._classify_openai_error(exc)

    def test_timeout_error(self):
        import sys
        fake_openai = MagicMock()
        fake_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        TimeoutErr = type("APITimeoutError", (Exception,), {})
        fake_openai.APITimeoutError = TimeoutErr
        fake_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
        fake_openai.NotFoundError = type("NotFoundError", (Exception,), {})
        fake_openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
        fake_openai.APIStatusError = type("APIStatusError", (Exception,), {})

        exc = TimeoutErr("timed out")
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with self.assertRaises(TimeoutError):
                self.provider._classify_openai_error(exc)

    def test_connection_error(self):
        import sys
        fake_openai = MagicMock()
        fake_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        fake_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
        ConnErr = type("APIConnectionError", (Exception,), {})
        fake_openai.APIConnectionError = ConnErr
        fake_openai.NotFoundError = type("NotFoundError", (Exception,), {})
        fake_openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
        fake_openai.APIStatusError = type("APIStatusError", (Exception,), {})

        exc = ConnErr("connection refused")
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with self.assertRaises(ConnectionFailedError):
                self.provider._classify_openai_error(exc)

    def test_not_found_error(self):
        import sys
        fake_openai = MagicMock()
        fake_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        fake_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
        fake_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
        NotFoundErr = type("NotFoundError", (Exception,), {})
        fake_openai.NotFoundError = NotFoundErr
        fake_openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
        fake_openai.APIStatusError = type("APIStatusError", (Exception,), {})

        exc = NotFoundErr("model not found")
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with self.assertRaises(ModelNotAvailableError):
                self.provider._classify_openai_error(exc)

    def test_auth_error(self):
        import sys
        fake_openai = MagicMock()
        fake_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        fake_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
        fake_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
        fake_openai.NotFoundError = type("NotFoundError", (Exception,), {})
        AuthErr = type("AuthenticationError", (Exception,), {})
        fake_openai.AuthenticationError = AuthErr
        fake_openai.APIStatusError = type("APIStatusError", (Exception,), {})

        exc = AuthErr("bad api key")
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with self.assertRaises(LLMError):
                self.provider._classify_openai_error(exc)


class TestOpenAIProviderListModels(unittest.TestCase):
    def test_list_models(self):
        provider = _make_provider()
        m1, m2 = MagicMock(), MagicMock()
        m1.id = "gpt-4o"
        m2.id = "gpt-4o-mini"
        provider._client.models.list.return_value = [m1, m2]

        models = provider.list_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, "gpt-4o")
        self.assertEqual(models[1].name, "gpt-4o-mini")

    def test_is_available_true(self):
        provider = _make_provider()
        provider._client.models.list.return_value = iter([MagicMock()])
        self.assertTrue(provider.is_available())

    def test_is_available_false_on_exception(self):
        provider = _make_provider()
        provider._client.models.list.side_effect = Exception("network error")
        self.assertFalse(provider.is_available())


class TestCreateOpenAIProvider(unittest.TestCase):
    def test_factory_creates_provider(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from llm_provider.providers.openai_provider import create_openai_provider, OpenAIProvider
            provider = create_openai_provider({
                "api_key": "test-key",
                "default_model": "gpt-4o",
            })
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.name(), "openai")

    def test_factory_requires_api_key(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from llm_provider.providers.openai_provider import create_openai_provider
            with self.assertRaises(ValueError):
                create_openai_provider({"default_model": "gpt-4o"})

    def test_factory_defaults_model(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from llm_provider.providers.openai_provider import create_openai_provider
            provider = create_openai_provider({"api_key": "test-key"})
        self.assertEqual(provider._config.default_model, "gpt-4o")

    def test_factory_respects_base_url(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from llm_provider.providers.openai_provider import create_openai_provider
            provider = create_openai_provider({
                "api_key": "test-key",
                "extra_settings": {"base_url": "https://my-proxy.example.com"},
            })
        self.assertEqual(provider._config.extra_settings["base_url"], "https://my-proxy.example.com")


if __name__ == "__main__":
    unittest.main()
