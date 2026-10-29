"""Live model-behaviour check: current Claude models through the Anthropic provider.

Skipped unless ``ANTHROPIC_API_KEY`` is set. From the ``stories`` repo root::

    ANTHROPIC_API_KEY=$(security find-generic-password -s stories \\
        -a STORYTELLER_ANTHROPIC_API_KEY -w) python -m pytest tests/test_anthropic_current_models_live.py

Model defaults to ``claude-haiku-5-5``; override with ``ANTHROPIC_LIVE_MODEL``.
Each call is a few hundred tokens — well under a cent on Haiku 5.5.

Arms:
  * **sampling** — subject: a request carrying temperature/top_p succeeds through
    the provider. Control: the same values sent straight to the SDK are rejected,
    which is what makes the provider's gate load-bearing.
  * **forced tool call** — ``tool_choice="required"`` (the storyteller wizards'
    shape) returns a real tool call.
  * **thinking round trip** — subject: a tool turn that thought is replayed with
    its thinking blocks and the follow-up succeeds. Control: the same history with
    the thinking blocks stripped, recorded rather than asserted — on 2026-10-27
    the API accepted it, so replay is guidance-compliance, not a 400 guard.
  * **prompt caching** — subject: the same long system prompt sent twice through
    the provider is written then read from cache, and so is a continued
    conversation's history. Control: the same prompt sent straight to the SDK
    without markers never caches, which is what makes the markers load-bearing.
    Each run salts its prompts so an earlier run's cache entries can't satisfy it.
"""

import os
import unittest
import uuid

from llm_provider.models import ChatRequest, Message, ToolCall, ToolSchema
from llm_provider.providers.anthropic_provider import create_anthropic_provider

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = os.environ.get("ANTHROPIC_LIVE_MODEL", "claude-haiku-5-5")

SHIP_TOOL = ToolSchema(
    name="book_shipment",
    description="Book a shipment of a crate to a warehouse. Call once you have decided.",
    input_schema={
        "type": "object",
        "properties": {
            "warehouse": {"type": "string", "description": "Warehouse code, e.g. W3."},
            "crates": {"type": "integer", "description": "Number of crates to ship."},
        },
        "required": ["warehouse", "crates"],
    },
)

# A small allocation puzzle: the model has to reason before it can call the tool,
# which is what makes it emit thinking blocks ahead of the tool_use block.
PUZZLE = (
    "Warehouses W1, W2 and W3 have capacities 7, 11 and 13 crates and currently hold "
    "5, 4 and 12 crates. Exactly 6 crates must go to a single warehouse that ends up "
    "with an even number of crates and does not exceed capacity. Work out which "
    "warehouse qualifies, then book the shipment. Book exactly one shipment."
)


def _provider():
    return create_anthropic_provider(
        {
            "host": "https://api.anthropic.com",
            "default_model": MODEL,
            "api_key": API_KEY,
            "extra_settings": {"max_tokens": 8000},
        }
    )


@unittest.skipUnless(API_KEY, "ANTHROPIC_API_KEY not set")
class TestSamplingParams(unittest.TestCase):
    def test_subject_provider_accepts_request_with_sampling(self):
        response = _provider().chat(
            ChatRequest(
                messages=[Message(role="user", content="Reply with the single word: ready")],
                temperature=0.0,
                top_p=0.9,
            )
        )
        self.assertIn("ready", response.message.lower())

    def test_control_sdk_rejects_the_same_values(self):
        import anthropic

        client = anthropic.Anthropic(api_key=API_KEY)
        with self.assertRaises(anthropic.BadRequestError):
            client.messages.create(
                model=MODEL,
                max_tokens=200,
                temperature=0.0,
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            )


@unittest.skipUnless(API_KEY, "ANTHROPIC_API_KEY not set")
class TestForcedToolCall(unittest.TestCase):
    def test_required_tool_choice_returns_a_tool_call(self):
        response = _provider().chat(
            ChatRequest(
                messages=[Message(role="user", content=PUZZLE)],
                tools=[SHIP_TOOL],
                tool_choice="required",
            )
        )
        self.assertTrue(response.tool_calls)
        call = response.tool_calls[0]
        self.assertEqual(call.name, "book_shipment")
        self.assertIsInstance(call.arguments, dict)


@unittest.skipUnless(API_KEY, "ANTHROPIC_API_KEY not set")
class TestThinkingRoundTrip(unittest.TestCase):
    def _first_turn(self):
        provider = _provider()
        response = provider.chat(
            ChatRequest(
                messages=[Message(role="user", content=PUZZLE)],
                tools=[SHIP_TOOL],
                tool_choice="auto",
                extra_body={"anthropic": {"output_config": {"effort": "high"}}},
            )
        )
        self.assertTrue(response.tool_calls, "model answered without calling the tool")
        if not response.tool_calls[0].thought_signature:
            self.skipTest("model did not think on this sample; round trip not exercised")
        return provider, response

    def _follow_up(self, provider, response, calls):
        history = [
            Message(role="user", content=PUZZLE),
            Message(role="assistant", content=response.message, tool_calls=calls),
            *[
                Message(role="tool", content="booked: confirmation #A17", tool_call_id=c.id)
                for c in calls
            ],
        ]
        return provider.chat(
            ChatRequest(
                messages=history,
                tools=[SHIP_TOOL],
                tool_choice="auto",
                extra_body={"anthropic": {"output_config": {"effort": "high"}}},
            )
        )

    def test_subject_replayed_thinking_is_accepted(self):
        provider, response = self._first_turn()
        final = self._follow_up(provider, response, response.tool_calls)
        self.assertTrue(final.message or final.tool_calls)

    def test_control_stripped_thinking(self):
        provider, response = self._first_turn()
        stripped = [
            ToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in response.tool_calls
        ]
        try:
            self._follow_up(provider, response, stripped)
        except Exception as exc:  # noqa: BLE001 — recording what the API does
            print(f"\n[control] stripped thinking rejected: {type(exc).__name__}: {exc}")
        else:
            print("\n[control] stripped thinking accepted by the API")


if __name__ == "__main__":
    unittest.main()


# ~6K tokens: comfortably past every current model's minimum cacheable prefix.
_RULES = "\n".join(
    f"Rule {i}: crates bound for warehouse W{i % 3 + 1} are logged in ledger {i} "
    "and checked against the manifest before loading."
    for i in range(150)
)


def _usage(response, field):
    return (response.usage or {}).get(field, 0)


@unittest.skipUnless(API_KEY, "ANTHROPIC_API_KEY not set")
class TestPromptCaching(unittest.TestCase):
    def _system(self):
        return f"Run {uuid.uuid4()}. You are a shipping clerk.\n{_RULES}"

    def test_subject_system_prompt_is_written_then_read(self):
        from llm_provider.models import SystemPrompt

        provider = _provider()
        request = ChatRequest(
            messages=[Message(role="user", content="Reply with the single word: ready")],
            system_prompt=SystemPrompt(content=self._system()),
        )
        first = provider.chat(request)
        second = provider.chat(request)
        print(f"\nsystem prompt: first={first.usage} second={second.usage}")
        self.assertGreater(_usage(first, "cache_creation_input_tokens"), 0)
        self.assertGreater(_usage(second, "cache_read_input_tokens"), 0)

    def test_subject_conversation_history_is_written_then_read(self):
        provider = _provider()
        request = ChatRequest(
            messages=[
                Message(role="user", content=self._system()),
                Message(role="assistant", content="Understood. I will follow every rule."),
                Message(role="user", content="Reply with the single word: ready"),
            ]
        )
        first = provider.chat(request)
        second = provider.chat(request)
        print(f"\nhistory: first={first.usage} second={second.usage}")
        self.assertGreater(_usage(first, "cache_creation_input_tokens"), 0)
        self.assertGreater(_usage(second, "cache_read_input_tokens"), 0)

    def test_control_unmarked_sdk_request_never_caches(self):
        import anthropic

        client = anthropic.Anthropic(api_key=API_KEY)
        kwargs = dict(
            model=MODEL,
            max_tokens=200,
            system=self._system(),
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        )
        first = client.messages.create(**kwargs)
        second = client.messages.create(**kwargs)
        print(f"\ncontrol: first={first.usage} second={second.usage}")
        self.assertFalse(first.usage.cache_creation_input_tokens)
        self.assertFalse(second.usage.cache_read_input_tokens)
