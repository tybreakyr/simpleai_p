"""
Tests for thinking/CoT enable and disable on Ollama thinking models (e.g. Qwen3.5).

Ollama thinking models (Qwen3 family) support a top-level ``think`` field in
the /api/chat payload:
  - ``think: true``  — enable chain-of-thought (model emits <think>…</think>)
  - ``think: false`` — disable chain-of-thought (fast non-thinking mode)

Two control surfaces exist:
  1. ``ProviderConfig.extra_settings`` — applied to every request from that provider
  2. ``ChatRequest.extra_body["ollama"]`` — per-request override (applied after
     extra_settings, so it wins on conflict)

Unit tests validate payload construction without any network calls.
Integration tests run against a live Ollama server and are skipped when the
model is not available.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_provider.models import ChatRequest, Message, ProviderConfig
from llm_provider.providers.ollama_provider import OllamaProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THINKING_MODEL = "qwen3.5:latest"


def _provider(extra_settings=None):
    cfg = ProviderConfig(
        host="http://localhost:11434",
        default_model=THINKING_MODEL,
        timeout=120.0,
        retry_attempts=1,
        extra_settings=extra_settings or {},
    )
    return OllamaProvider(cfg)


def _req(extra_body=None):
    return ChatRequest(
        messages=[Message(role="user", content="What is 2+2?")],
        extra_body=extra_body,
    )


# ---------------------------------------------------------------------------
# Unit tests — payload construction (no network)
# ---------------------------------------------------------------------------


class TestThinkingPayloadConstruction(unittest.TestCase):
    """Verify that think: true/false land in the outgoing payload correctly."""

    def test_extra_settings_think_false_disables(self):
        provider = _provider(extra_settings={"think": False})
        payload = provider._build_payload(_req())
        self.assertIn("think", payload)
        self.assertIs(payload["think"], False)

    def test_extra_settings_think_true_enables(self):
        provider = _provider(extra_settings={"think": True})
        payload = provider._build_payload(_req())
        self.assertIn("think", payload)
        self.assertIs(payload["think"], True)

    def test_no_think_setting_omits_field(self):
        provider = _provider()
        payload = provider._build_payload(_req())
        self.assertNotIn("think", payload)

    def test_extra_body_think_false_disables(self):
        provider = _provider()
        payload = provider._build_payload(_req(extra_body={"ollama": {"think": False}}))
        self.assertIn("think", payload)
        self.assertIs(payload["think"], False)

    def test_extra_body_think_true_enables(self):
        provider = _provider()
        payload = provider._build_payload(_req(extra_body={"ollama": {"think": True}}))
        self.assertIn("think", payload)
        self.assertIs(payload["think"], True)

    def test_extra_body_overrides_extra_settings_disabled_to_enabled(self):
        """Per-request extra_body must win over provider-level extra_settings."""
        provider = _provider(extra_settings={"think": False})
        payload = provider._build_payload(_req(extra_body={"ollama": {"think": True}}))
        self.assertIs(payload["think"], True)

    def test_extra_body_overrides_extra_settings_enabled_to_disabled(self):
        provider = _provider(extra_settings={"think": True})
        payload = provider._build_payload(_req(extra_body={"ollama": {"think": False}}))
        self.assertIs(payload["think"], False)

    def test_other_provider_bucket_does_not_set_think(self):
        provider = _provider()
        payload = provider._build_payload(_req(extra_body={"openai": {"think": True}}))
        self.assertNotIn("think", payload)

    def test_model_is_forwarded_correctly(self):
        provider = _provider()
        payload = provider._build_payload(_req())
        self.assertEqual(payload["model"], THINKING_MODEL)


# ---------------------------------------------------------------------------
# Integration tests — live Ollama (skipped when model not available)
# ---------------------------------------------------------------------------


def _ollama_has_model(model_name: str) -> bool:
    try:
        import requests

        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(m == model_name or m.startswith(model_name.split(":")[0]) for m in models)
    except Exception:
        return False


_MODEL_AVAILABLE = _ollama_has_model(THINKING_MODEL)
_SKIP_REASON = f"Ollama not running or model '{THINKING_MODEL}' not installed"


@unittest.skipUnless(_MODEL_AVAILABLE, _SKIP_REASON)
class TestThinkingIntegration(unittest.TestCase):
    """Live integration tests against Ollama with the Qwen3.5 thinking model."""

    def _chat(self, think: bool) -> str:
        provider = _provider(extra_settings={"think": think})
        req = ChatRequest(
            messages=[Message(role="user", content="What is 2+2? Answer in one word.")],
        )
        response = provider.chat(req)
        return response.message

    def test_thinking_disabled_returns_response(self):
        """think=False: model responds without emitting <think> blocks."""
        reply = self._chat(think=False)
        self.assertTrue(len(reply.strip()) > 0, "Expected a non-empty response")
        self.assertNotIn("<think>", reply, "think=False should suppress reasoning tags")

    def test_thinking_enabled_returns_response(self):
        """think=True: model still returns a usable response (may include <think> block)."""
        reply = self._chat(think=True)
        self.assertTrue(len(reply.strip()) > 0, "Expected a non-empty response")

    def test_per_request_think_false_via_extra_body(self):
        """Per-request extra_body disables thinking even without provider-level setting."""
        provider = _provider()
        req = ChatRequest(
            messages=[Message(role="user", content="What is 2+2? Answer in one word.")],
            extra_body={"ollama": {"think": False}},
        )
        response = provider.chat(req)
        self.assertTrue(len(response.message.strip()) > 0)
        self.assertNotIn("<think>", response.message)

    def test_per_request_think_true_via_extra_body(self):
        """Per-request extra_body enables thinking even without provider-level setting."""
        provider = _provider()
        req = ChatRequest(
            messages=[Message(role="user", content="What is 2+2? Answer in one word.")],
            extra_body={"ollama": {"think": True}},
        )
        response = provider.chat(req)
        self.assertTrue(len(response.message.strip()) > 0)

    def test_extra_body_overrides_provider_setting(self):
        """Provider has think=False; per-request extra_body sets think=True — model should think."""
        provider = _provider(extra_settings={"think": False})
        req = ChatRequest(
            messages=[Message(role="user", content="What is 2+2? Answer in one word.")],
            extra_body={"ollama": {"think": True}},
        )
        response = provider.chat(req)
        self.assertTrue(len(response.message.strip()) > 0)


if __name__ == "__main__":
    unittest.main()
