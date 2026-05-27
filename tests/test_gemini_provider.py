"""
Unit tests for the Gemini provider implementation.
"""

import sys
import threading
import unittest
from unittest.mock import MagicMock, patch
from llm_provider.models import ProviderConfig, ChatRequest, Message, SystemPrompt
from llm_provider.errors import RateLimitExceededError
from llm_provider.retry import RetryConfig


def _make_config(**kwargs):
    defaults = dict(
        host="https://generativelanguage.googleapis.com",
        default_model="gemini-1.5-flash",
        api_key="test-key",
        timeout=30.0,
        retry_attempts=1,
    )
    defaults.update(kwargs)
    return ProviderConfig(**defaults)


def _make_provider(config=None):
    """Return a GeminiProvider with the google.genai SDK mocked out."""
    if config is None:
        config = _make_config()

    mock_genai = MagicMock()
    with patch.dict(sys.modules, {
        "google": MagicMock(),
        "google.genai": mock_genai,
        "google.genai.types": mock_genai.types,
    }):
        from llm_provider.providers.gemini_provider import GeminiProvider
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._config = config
        provider._client = MagicMock()
        provider._rate_limit_rpm = config.rate_limit
        provider._last_request_at = 0.0
        provider._rate_lock = threading.Lock()
        provider._retry_config = RetryConfig(
            max_retries=config.retry_attempts,
            base_delay=0.01,
            max_delay=0.01,
            backoff_factor=1.0,
        )
        return provider


class TestGeminiProviderInit(unittest.TestCase):
    def test_requires_api_key(self):
        config = _make_config(api_key=None)
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": MagicMock(),
        }):
            from llm_provider.providers.gemini_provider import GeminiProvider
            with self.assertRaises(ValueError):
                GeminiProvider(config)

    def test_name(self):
        provider = _make_provider()
        self.assertEqual(provider.name(), "gemini")

    def test_supported_features(self):
        provider = _make_provider()
        features = provider.supported_features()
        self.assertTrue(features.structured_output)
        self.assertTrue(features.streaming)
        self.assertTrue(features.vision)
        self.assertTrue(features.temperature)
        self.assertTrue(features.top_p)


class TestGeminiProviderChat(unittest.TestCase):
    def setUp(self):
        self.mock_genai = MagicMock()
        self.patcher = patch.dict(sys.modules, {
            "google": MagicMock(),
            "google.genai": self.mock_genai,
            "google.genai.types": self.mock_genai.types,
        })
        self.patcher.start()
        self.provider = _make_provider()

    def tearDown(self):
        self.patcher.stop()

    def _mock_response(self, text="Hello from Gemini"):
        resp = MagicMock()
        resp.text = text
        return resp

    def test_chat_success(self):
        self.provider._client.models.generate_content.return_value = self._mock_response()
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        response = self.provider.chat(request)
        self.assertEqual(response.message, "Hello from Gemini")

    def test_chat_with_system_prompt(self):
        self.provider._client.models.generate_content.return_value = self._mock_response("OK")
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            system_prompt=SystemPrompt(content="Be concise"),
        )
        response = self.provider.chat(request)
        self.assertEqual(response.message, "OK")
        call_kwargs = self.provider._client.models.generate_content.call_args[1]
        self.assertIsNotNone(call_kwargs.get("config"))

    def test_chat_rate_limit_429_raises(self):
        self.provider._client.models.generate_content.side_effect = Exception(
            "429 RESOURCE_EXHAUSTED retryDelay: '5s'"
        )
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        with self.assertRaises(RateLimitExceededError):
            self.provider.chat(request)

    def test_chat_daily_quota_not_retryable(self):
        self.provider._client.models.generate_content.side_effect = Exception(
            "429 RESOURCE_EXHAUSTED PerDay quota exceeded"
        )
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        with self.assertRaises(RateLimitExceededError) as ctx:
            self.provider.chat(request)
        self.assertFalse(ctx.exception.retryable)

    def test_chat_uses_request_model(self):
        self.provider._client.models.generate_content.return_value = self._mock_response()
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model="gemini-2.0-flash",
        )
        self.provider.chat(request)
        call_kwargs = self.provider._client.models.generate_content.call_args[1]
        self.assertEqual(call_kwargs["model"], "gemini-2.0-flash")


class TestGeminiProviderListModels(unittest.TestCase):
    def test_list_models_filters_gemini(self):
        provider = _make_provider()
        m1, m2, m3 = MagicMock(), MagicMock(), MagicMock()
        m1.name = "models/gemini-1.5-pro"
        m2.name = "models/gemini-1.5-flash"
        m3.name = "models/text-bison-001"
        provider._client.models.list.return_value = [m1, m2, m3]

        models = provider.list_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, "models/gemini-1.5-pro")

    def test_is_available_true(self):
        provider = _make_provider()
        provider._client.models.list.return_value = iter([MagicMock()])
        self.assertTrue(provider.is_available())

    def test_is_available_false_on_exception(self):
        provider = _make_provider()
        provider._client.models.list.side_effect = Exception("network error")
        self.assertFalse(provider.is_available())


class TestParseGeminiRetryDelay(unittest.TestCase):
    def test_json_style_delay(self):
        from llm_provider.providers.gemini_provider import _parse_gemini_retry_delay
        result = _parse_gemini_retry_delay("'retryDelay': '9s'")
        self.assertAlmostEqual(result, 10.0)

    def test_prose_style_delay(self):
        from llm_provider.providers.gemini_provider import _parse_gemini_retry_delay
        result = _parse_gemini_retry_delay("Please retry in 9.3758776s")
        self.assertAlmostEqual(result, 10.375, places=2)

    def test_no_delay_returns_none(self):
        from llm_provider.providers.gemini_provider import _parse_gemini_retry_delay
        self.assertIsNone(_parse_gemini_retry_delay("some other error"))


class TestIsDailyQuota(unittest.TestCase):
    def test_per_day_detected(self):
        from llm_provider.providers.gemini_provider import _is_daily_quota
        self.assertTrue(_is_daily_quota("PerDay quota exceeded"))
        self.assertTrue(_is_daily_quota("per_day limit hit"))
        self.assertFalse(_is_daily_quota("per-minute rate limit"))


class TestCreateGeminiProvider(unittest.TestCase):
    def test_factory_creates_provider(self):
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": MagicMock(),
        }):
            from llm_provider.providers.gemini_provider import create_gemini_provider, GeminiProvider
            provider = create_gemini_provider({
                "api_key": "test-key",
                "default_model": "gemini-1.5-flash",
            })
        self.assertIsInstance(provider, GeminiProvider)
        self.assertEqual(provider.name(), "gemini")

    def test_factory_requires_api_key(self):
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": MagicMock(),
        }):
            from llm_provider.providers.gemini_provider import create_gemini_provider
            with self.assertRaises(ValueError):
                create_gemini_provider({"default_model": "gemini-1.5-flash"})


class TestSanitizeSchemaForGemini(unittest.TestCase):
    """Gemini's tool schema rejects certain JSON Schema fields. The provider
    sanitizes them at conversion time so other consumers don't have to."""

    def setUp(self):
        from llm_provider.providers.gemini_provider import _sanitize_schema_for_gemini
        self.fn = _sanitize_schema_for_gemini

    def test_strips_top_level_additional_properties(self):
        schema = {"type": "object", "additionalProperties": {"type": "string"}}
        self.assertEqual(self.fn(schema), {"type": "object"})

    def test_strips_nested_additional_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "object", "additionalProperties": {"type": "string"}},
                "counts": {"type": "object", "additionalProperties": {"type": "integer"}},
            },
            "required": ["tags"],
        }
        result = self.fn(schema)
        self.assertNotIn("additionalProperties", result["properties"]["tags"])
        self.assertNotIn("additionalProperties", result["properties"]["counts"])
        # Other fields preserved
        self.assertEqual(result["required"], ["tags"])
        self.assertEqual(result["properties"]["tags"]["type"], "object")

    def test_strips_inside_lists(self):
        schema = {
            "type": "object",
            "allOf": [
                {"type": "object", "additionalProperties": {"type": "string"}},
                {"type": "object"},
            ],
        }
        result = self.fn(schema)
        self.assertNotIn("additionalProperties", result["allOf"][0])

    def test_preserves_schemas_without_offending_keys(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        self.assertEqual(self.fn(schema), schema)

    def test_non_dict_passthrough(self):
        self.assertEqual(self.fn("string"), "string")
        self.assertEqual(self.fn(42), 42)
        self.assertIsNone(self.fn(None))

    def test_applied_during_tool_conversion(self):
        """End-to-end: a ChatRequest with additionalProperties in a tool schema
        should have it stripped by the time it reaches the Gemini SDK call."""
        from llm_provider.models import ToolSchema
        mock_genai = MagicMock()
        patcher = patch.dict(sys.modules, {
            "google": MagicMock(),
            "google.genai": mock_genai,
            "google.genai.types": mock_genai.types,
        })
        patcher.start()
        try:
            provider = _make_provider()
            # Stub out the SDK call so we can inspect what gets sent
            mock_resp = MagicMock()
            mock_resp.text = "ok"
            mock_resp.candidates = []
            provider._client.models.generate_content = MagicMock(return_value=mock_resp)

            tool = ToolSchema(
                name="t",
                description="d",
                input_schema={
                    "type": "object",
                    "properties": {
                        "m": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                    "required": ["m"],
                },
            )
            req = ChatRequest(
                messages=[Message(role="user", content="hi")],
                model="gemini-1.5-flash",
                tools=[tool],
            )
            provider.chat(req)

            # GenerateContentConfig was instantiated with our (sanitized) tools
            call_kwargs = mock_genai.types.GenerateContentConfig.call_args.kwargs
            tools_arg = call_kwargs["tools"]
            params = tools_arg[0]["function_declarations"][0]["parameters"]
            self.assertNotIn("additionalProperties", params["properties"]["m"])
            self.assertEqual(params["properties"]["m"]["type"], "object")
        finally:
            patcher.stop()


if __name__ == "__main__":
    unittest.main()
