import asyncio
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from llm_provider.models import ChatRequest, Message, ProviderConfig
from llm_provider.retry import RetryConfig


def _retry_cfg(attempts=2):
    return RetryConfig(max_retries=attempts, base_delay=0.01, max_delay=0.01, backoff_factor=1.0)


# ─── Anthropic ───────────────────────────────────────────────────────────────

class TestAsyncAnthropic(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.async_client = MagicMock()
        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            from llm_provider.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider.__new__(AnthropicProvider)
        p._config = ProviderConfig(
            host="https://api.anthropic.com",
            default_model="claude-3-5-sonnet",
            api_key="test-key",
            retry_attempts=2,
        )
        p._client = MagicMock()
        p._max_tokens = 8192
        p._retry_config = _retry_cfg(2)
        p._async_client = self.async_client
        self.provider = p

    def _ok_response(self, text="Hello from async"):
        mock = MagicMock()
        mock.content = [MagicMock(type="text", text=text)]
        mock.stop_reason = "end_turn"
        return mock

    async def test_achat_returns_response(self):
        self.async_client.messages.create = AsyncMock(return_value=self._ok_response())
        req = ChatRequest(messages=[Message(role="user", content="hi")])
        resp = await self.provider.achat(req)
        self.assertEqual(resp.message, "Hello from async")

    async def test_achat_retries_on_transient_error(self):
        from llm_provider.errors import ConnectionFailedError
        ok = self._ok_response("Success")
        err = ConnectionFailedError("transient", "achat")
        self.async_client.messages.create = AsyncMock(side_effect=[err, err, ok])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            resp = await self.provider.achat(ChatRequest(messages=[Message(role="user", content="hi")]))
        self.assertEqual(resp.message, "Success")
        self.assertEqual(self.async_client.messages.create.call_count, 3)

    async def test_concurrent_achat_with_gather(self):
        async def slow_create(*a, **kw):
            await asyncio.sleep(0.05)
            return self._ok_response("Done")
        self.async_client.messages.create = AsyncMock(side_effect=slow_create)
        req = ChatRequest(messages=[Message(role="user", content="hi")])
        start = time.monotonic()
        resps = await asyncio.gather(*[self.provider.achat(req) for _ in range(5)])
        self.assertLess(time.monotonic() - start, 0.3)
        self.assertEqual(len(resps), 5)

    async def test_alist_models_default_uses_thread(self):
        self.provider._client.models.list.return_value = [MagicMock(id="m1"), MagicMock(id="m2")]
        models = await self.provider.alist_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, "m1")


# ─── OpenAI ──────────────────────────────────────────────────────────────────

class TestAsyncOpenAI(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.async_client = MagicMock()
        with patch.dict(sys.modules, {"openai": MagicMock()}):
            from llm_provider.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider.__new__(OpenAIProvider)
        p._config = ProviderConfig(
            host="https://api.openai.com/v1",
            default_model="gpt-4o",
            api_key="test-key",
            retry_attempts=2,
        )
        p._client = MagicMock()
        p._retry_config = _retry_cfg(2)
        p._async_client = self.async_client
        self.provider = p

    def _ok_completion(self, text="Hello from async"):
        choice = MagicMock()
        choice.message.content = text
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    async def test_achat_returns_response(self):
        self.async_client.chat.completions.create = AsyncMock(return_value=self._ok_completion())
        req = ChatRequest(messages=[Message(role="user", content="hi")])
        resp = await self.provider.achat(req)
        self.assertEqual(resp.message, "Hello from async")

    async def test_concurrent_achat_with_gather(self):
        async def slow_create(*a, **kw):
            await asyncio.sleep(0.05)
            return self._ok_completion("Done")
        self.async_client.chat.completions.create = AsyncMock(side_effect=slow_create)
        req = ChatRequest(messages=[Message(role="user", content="hi")])
        start = time.monotonic()
        resps = await asyncio.gather(*[self.provider.achat(req) for _ in range(5)])
        self.assertLess(time.monotonic() - start, 0.3)
        self.assertEqual(len(resps), 5)

    async def test_alist_models_default_uses_thread(self):
        self.provider._client.models.list.return_value = [MagicMock(id="m1")]
        models = await self.provider.alist_models()
        self.assertEqual(len(models), 1)


# ─── Gemini ──────────────────────────────────────────────────────────────────

class TestAsyncGemini(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_genai = MagicMock()
        self.patcher = patch.dict(sys.modules, {
            "google": MagicMock(),
            "google.genai": self.mock_genai,
            "google.genai.types": self.mock_genai.types,
        })
        self.patcher.start()
        from llm_provider.providers.gemini_provider import GeminiProvider
        p = GeminiProvider.__new__(GeminiProvider)
        p._config = ProviderConfig(
            host="https://generativelanguage.googleapis.com",
            default_model="gemini-1.5-flash",
            api_key="test-key",
            retry_attempts=2,
        )
        p._client = MagicMock()
        p._rate_limit_rpm = 0
        p._last_request_at = 0.0
        p._rate_lock = threading.Lock()
        p._retry_config = _retry_cfg(2)
        self.provider = p

    def tearDown(self):
        self.patcher.stop()

    async def test_achat_returns_response(self):
        self.provider._client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(text="Hello from async", function_calls=[])
        )
        req = ChatRequest(messages=[Message(role="user", content="hi")])
        resp = await self.provider.achat(req)
        self.assertEqual(resp.message, "Hello from async")

    async def test_concurrent_achat_with_gather(self):
        async def slow_create(*a, **kw):
            await asyncio.sleep(0.05)
            return MagicMock(text="Done", function_calls=[])
        self.provider._client.aio.models.generate_content = AsyncMock(side_effect=slow_create)
        req = ChatRequest(messages=[Message(role="user", content="hi")])
        start = time.monotonic()
        resps = await asyncio.gather(*[self.provider.achat(req) for _ in range(5)])
        self.assertLess(time.monotonic() - start, 0.3)
        self.assertEqual(len(resps), 5)


# ─── Ollama ──────────────────────────────────────────────────────────────────

class TestAsyncOllama(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.async_client = MagicMock()
        self.mock_httpx = MagicMock()
        self.patcher = patch.dict(sys.modules, {"requests": MagicMock(), "httpx": self.mock_httpx})
        self.patcher.start()
        from llm_provider.providers.ollama_provider import OllamaProvider
        p = OllamaProvider.__new__(OllamaProvider)
        p._config = ProviderConfig(
            host="http://localhost:11434",
            default_model="llama3",
            api_key="",
            retry_attempts=2,
        )
        p._base_url = "http://localhost:11434/api"
        p._session = MagicMock()
        p._retry_config = _retry_cfg(2)
        p._async_client = self.async_client
        self.provider = p

    def tearDown(self):
        self.patcher.stop()

    def _ok_response(self, text="Hello from async"):
        mock = MagicMock()
        mock.json.return_value = {"message": {"content": text}}
        return mock

    async def test_achat_returns_response(self):
        self.async_client.post = AsyncMock(return_value=self._ok_response())
        req = ChatRequest(messages=[Message(role="user", content="hi")])
        resp = await self.provider.achat(req)
        self.assertEqual(resp.message, "Hello from async")

    async def test_concurrent_achat_with_gather(self):
        async def slow_post(*a, **kw):
            await asyncio.sleep(0.05)
            return self._ok_response("Done")
        self.async_client.post = AsyncMock(side_effect=slow_post)
        req = ChatRequest(messages=[Message(role="user", content="hi")])
        start = time.monotonic()
        resps = await asyncio.gather(*[self.provider.achat(req) for _ in range(5)])
        self.assertLess(time.monotonic() - start, 0.3)
        self.assertEqual(len(resps), 5)


if __name__ == '__main__':
    unittest.main()
