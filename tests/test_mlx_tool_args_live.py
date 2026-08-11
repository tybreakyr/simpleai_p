"""Live model-behaviour check: double-encoded tool arguments from mlx-lm.

Skipped unless a local OpenAI-compatible server is reachable. Bring one up from
the ``stories`` repo root with::

    ./scripts/start.sh --story qwen3-mlx-4bit --memory gemini-flash-2.5

which serves ``mlx-community/Qwen3.5-9B-MLX-4bit`` at ``http://localhost:8000/v1``.
Override with ``MLX_LM_BASE_URL`` / ``MLX_LM_MODEL``.

Arms:
  * **subject** — a tool with an array-of-objects parameter and a realistic,
    long prompt. This is the shape mlx-lm double-encodes. Post-fix the caller
    must see a real ``list``.
  * **control** — a scalar-only tool on the same server, which has never shown
    the quirk and must be unaffected by the decoder.
"""

import os
import unittest

import requests

from llm_provider.models import ChatRequest, Message, ProviderConfig, SystemPrompt, ToolSchema

BASE_URL = os.environ.get("MLX_LM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MLX_LM_MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")


def _server_up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/models", timeout=3).status_code < 500
    except Exception:
        return False


QUESTIONNAIRE_TOOL = ToolSchema(
    name="generate_questionnaire",
    description="Produce the character-creation questionnaire for the player.",
    input_schema={
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "description": "The questions to ask, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The question text."},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Two to four answer options.",
                        },
                    },
                    "required": ["text", "options"],
                },
            }
        },
        "required": ["questions"],
    },
)

SCALAR_TOOL = ToolSchema(
    name="name_the_protagonist",
    description="Choose a name and a one-line description for the protagonist.",
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["name", "description"],
    },
)

# Long/realistic prompt — a minimal prompt did not reproduce the quirk.
LONG_PROMPT = (
    "You are running character creation for an interactive story. The setting is a "
    "coastal town in late autumn where the ferry has stopped running and the only "
    "hotel is half-shuttered for the season. The player will play a returning "
    "resident who left fifteen years ago after a fire nobody talks about. Build a "
    "questionnaire of exactly five questions that establishes who they were before "
    "they left, why they came back now, what they are avoiding, who in town still "
    "owes them something, and what they are carrying in their bag. Each question "
    "must offer three to four concrete, mutually exclusive options written in the "
    "voice of the setting — no abstractions, no 'other'. Call the tool."
)


@unittest.skipUnless(_server_up(), f"No OpenAI-compatible server at {BASE_URL}")
class TestMlxDoubleEncodedToolArgsLive(unittest.TestCase):
    """Marked integration: requires a live local model server."""

    integration = True

    @classmethod
    def setUpClass(cls):
        from llm_provider.providers.openai_provider import OpenAIProvider

        cls.provider = OpenAIProvider(
            ProviderConfig(
                host=BASE_URL,
                default_model=MODEL,
                api_key="not-needed",
                timeout=300.0,
                retry_attempts=1,
                extra_settings={"base_url": BASE_URL, "max_tokens": 8192},
            )
        )

    def test_array_of_objects_arrives_as_a_list(self):
        response = self.provider.chat(
            ChatRequest(
                messages=[Message(role="user", content=LONG_PROMPT)],
                system_prompt=SystemPrompt(content="You always answer via the provided tool."),
                tools=[QUESTIONNAIRE_TOOL],
                tool_choice="required",
            )
        )
        self.assertTrue(response.tool_calls, "model returned no tool call")
        questions = response.tool_calls[0].arguments["questions"]
        self.assertIsInstance(questions, list, f"questions came back as {type(questions)}")
        self.assertTrue(questions)
        for q in questions:
            self.assertIsInstance(q, dict)
            self.assertIsInstance(q.get("options"), list)

    def test_control_scalar_only_tool_is_unaffected(self):
        response = self.provider.chat(
            ChatRequest(
                messages=[Message(role="user", content=LONG_PROMPT)],
                system_prompt=SystemPrompt(content="You always answer via the provided tool."),
                tools=[SCALAR_TOOL],
                tool_choice="required",
            )
        )
        self.assertTrue(response.tool_calls, "model returned no tool call")
        args = response.tool_calls[0].arguments
        self.assertIsInstance(args["name"], str)
        self.assertIsInstance(args["description"], str)


if __name__ == "__main__":
    unittest.main()
