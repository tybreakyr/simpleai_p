---
name: simpleai
description: >
  Guide Claude to write correct, idiomatic Python code using the `llm_provider`
  package from the `simpleai_p` library. Use this skill whenever the user wants
  to write code that imports from `llm_provider`, uses the `simpleai_p` library,
  wants to build multi-provider LLM support, wants to switch between Anthropic /
  OpenAI / Gemini / Ollama, or asks how to do chat, async chat, tool calling,
  structured output, or error handling with this library. Also trigger when the
  user asks "how do I use llm_provider", "write me a provider", "set up a
  factory", "call Claude/GPT/Gemini/Ollama with this library", or any variant.
  Do NOT trigger for code that directly uses the `anthropic`, `openai`, or
  `google.genai` SDKs without going through `llm_provider`.
---

# simpleai / llm_provider Skill

The `llm_provider` package (`simpleai_p` repo) is a unified Python abstraction
over Anthropic, OpenAI, Google Gemini, and Ollama. All four providers expose the
same `Provider` interface — swap providers by changing configuration, not code.

## Installation

```bash
# Install from the repo root (editable)
pip install -e .

# With a specific provider's SDK
pip install -e ".[anthropic]"   # anthropic>=0.97.0
pip install -e ".[openai]"      # openai>=2.0.0
pip install -e ".[gemini]"      # google-genai>=1.73.0
pip install -e ".[all]"         # all three cloud SDKs

# Ollama only needs `requests` (already in base dependencies)
```

Python 3.13+ is required.

## Core Imports

```python
from llm_provider import (
    # Provider constructors
    create_anthropic_provider,
    create_openai_provider,
    create_gemini_provider,
    create_ollama_provider,

    # Factory
    ProviderFactory, FactoryConfig,

    # Data models
    ProviderConfig, ChatRequest, ChatResponse,
    Message, SystemPrompt,
    ToolSchema, ToolCall,

    # Error types
    LLMError, ConnectionFailedError, TimeoutError,
    RateLimitExceededError, ModelNotAvailableError,
    JSONParseFailedError, InvalidConfigurationError,
    is_retryable,
)
```

## Pattern 1 — Direct Provider (no factory)

Use when you only need one provider or want minimal setup.

```python
from llm_provider import create_anthropic_provider, ChatRequest, Message

provider = create_anthropic_provider({
    "api_key": "sk-ant-...",
    "default_model": "claude-sonnet-4-6",
    "host": "https://api.anthropic.com",
    "timeout": 60.0,
    "retry_attempts": 3,
})

request = ChatRequest(
    messages=[Message(role="user", content="Explain async/await in Python.")],
)
response = provider.chat(request)
print(response.message)  # str
print(response.stop_reason)  # "end_turn" | "tool_use" | "max_tokens" | None
```

`create_*_provider` takes a plain `dict`, not a `ProviderConfig` instance.

## Pattern 2 — Factory Pattern (multi-provider apps)

Use when managing multiple providers or swapping at runtime.

```python
from llm_provider import (
    ProviderFactory, FactoryConfig, ProviderConfig,
    create_anthropic_provider, create_openai_provider,
    ChatRequest, Message,
)

factory = ProviderFactory()
factory.register_provider("anthropic", create_anthropic_provider)
factory.register_provider("openai", create_openai_provider)

factory.load_config(FactoryConfig(
    default_provider="anthropic",
    provider_configs={
        "anthropic": ProviderConfig(
            host="https://api.anthropic.com",
            default_model="claude-sonnet-4-6",
            api_key="sk-ant-...",
            timeout=60.0,
        ),
        "openai": ProviderConfig(
            host="https://api.openai.com",
            default_model="gpt-4o",
            api_key="sk-...",
            timeout=60.0,
        ),
    },
))

provider = factory.get_default_provider()
openai_provider = factory.create_provider_by_name("openai")

if factory.is_provider_available("anthropic"):
    provider = factory.create_provider_by_name("anthropic")
```

## Pattern 3 — System Prompts

```python
from llm_provider import ChatRequest, Message, SystemPrompt

request = ChatRequest(
    messages=[Message(role="user", content="Summarize this document: ...")],
    system_prompt=SystemPrompt("You are a concise technical writer."),
)
```

Anthropic passes `system_prompt` as a top-level `system` field. OpenAI and Ollama
inject it as a prepended `{"role": "system"}` message. The caller doesn't need to
handle this difference.

## Pattern 4 — Multi-turn Conversations

```python
history = [
    Message(role="user", content="What is the capital of France?"),
    Message(role="assistant", content="Paris."),
    Message(role="user", content="What is its population?"),
]
response = provider.chat(ChatRequest(messages=history))
```

Valid roles: `"user"` and `"assistant"`. Prefer `SystemPrompt` over a
`Message(role="system", ...)` for system instructions.

## Pattern 5 — Async Chat

```python
import asyncio
from llm_provider import create_anthropic_provider, ChatRequest, Message

provider = create_anthropic_provider({
    "api_key": "sk-ant-...",
    "default_model": "claude-sonnet-4-6",
    "host": "https://api.anthropic.com",
})

async def main():
    request = ChatRequest(messages=[Message(role="user", content="Hello!")])
    response = await provider.achat(request)
    print(response.message)

asyncio.run(main())
```

Concurrent requests:

```python
async def batch(provider, prompts):
    tasks = [
        provider.achat(ChatRequest(messages=[Message(role="user", content=p)]))
        for p in prompts
    ]
    responses = await asyncio.gather(*tasks)
    return [r.message for r in responses]
```

Ollama's `achat` requires `pip install httpx` (not included in base dependencies).

## Pattern 6 — Structured Output

```python
from dataclasses import dataclass
from typing import Optional
from llm_provider import ChatRequest, Message

@dataclass
class ExtractedPerson:
    name: str
    age: int
    email: Optional[str] = None

request = ChatRequest(
    messages=[Message(
        role="user",
        content="Extract into JSON: John Doe, 34, john@example.com",
    )],
    structured_output_type=ExtractedPerson,
)
response = provider.chat(request)

if response.structured_data:
    person: ExtractedPerson = response.structured_data
else:
    print("Parse failed, raw response:", response.message)
```

The library tries multiple JSON extraction strategies (direct parse → repair →
markdown block → brace counting). Use `structured_output_type=dict` to get the
raw parsed dict. Dataclass field names must match JSON keys exactly.

If `tools` is also set and the model returns tool calls, `structured_data` is
`None` — tool calls take priority.

## Pattern 7 — Tool Calling

```python
from llm_provider import ChatRequest, Message, ToolSchema

weather_tool = ToolSchema(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
)

request = ChatRequest(
    messages=[Message(role="user", content="What's the weather in Tokyo?")],
    tools=[weather_tool],
    tool_choice="auto",  # "auto" | "any" | "none" | specific tool name
)
response = provider.chat(request)

if response.tool_calls:
    for call in response.tool_calls:
        print(call.name)       # "get_weather"
        print(call.arguments)  # {"city": "Tokyo"}
        print(call.id)         # provider-generated call ID
```

`tool_choice` across providers:

| Value | Anthropic | OpenAI | Gemini | Ollama |
|---|---|---|---|---|
| `"auto"` | model decides | model decides | AUTO | forwarded |
| `"any"` | must use a tool | mapped to `"required"` | ANY | forwarded |
| `"none"` | no tools | no tools | NONE | forwarded |
| tool name | forces that tool | forces that function | ANY + allowed | forwarded |

Continuing after tool calls — send the result back as a `"user"` message:

```python
tool_result = {"temperature": 22, "condition": "sunny"}

followup = ChatRequest(
    messages=[
        Message(role="user", content="What's the weather in Tokyo?"),
        Message(role="assistant", content=response.message or ""),
        Message(
            role="user",
            content=f"Tool result for {response.tool_calls[0].name}: {tool_result}",
        ),
    ],
    tools=[weather_tool],
)
final = provider.chat(followup)
print(final.message)
```

## Pattern 8 — Temperature and Top-p

```python
request = ChatRequest(
    messages=[Message(role="user", content="Write a haiku about autumn.")],
    temperature=0.9,   # 0.0 = deterministic, higher = more random
    top_p=0.95,
)
```

Both are `Optional[float]`; omit to use provider defaults.

## Provider Configuration Reference

### Anthropic

```python
ProviderConfig(
    host="https://api.anthropic.com",
    default_model="claude-sonnet-4-6",   # or "claude-opus-4-7"
    api_key="sk-ant-...",
    timeout=60.0,
    retry_attempts=3,
    extra_settings={"max_tokens": 8192},  # Anthropic API requires this; default 8192
)
```

Override `max_tokens` for long responses: `extra_settings={"max_tokens": 16384}`.

### OpenAI

```python
ProviderConfig(
    host="https://api.openai.com",
    default_model="gpt-4o",
    api_key="sk-...",
    timeout=60.0,
    retry_attempts=3,
    extra_settings={
        "max_tokens": 4096,
        "base_url": "https://...",  # optional: Azure, Groq, or other compatible endpoint
    },
)
```

### Google Gemini

```python
ProviderConfig(
    host="https://generativelanguage.googleapis.com",
    default_model="gemini-2.0-flash",   # or "gemini-1.5-pro"
    api_key="AIza...",                  # from aistudio.google.com
    timeout=60.0,
    retry_attempts=3,
    rate_limit=15,  # requests-per-minute; set this on the free tier to avoid 429s
)
```

### Ollama (local)

```python
ProviderConfig(
    host="http://localhost:11434",
    default_model="llama3.1:latest",  # must be pulled: `ollama pull llama3.1`
    timeout=120.0,                    # first load can be slow
    retry_attempts=3,
    extra_settings={
        "think": False,       # disable CoT on thinking models (e.g., Qwen3)
        "keep_alive": "10m",  # keep model in VRAM between requests
    },
)
```

`extra_settings` for Ollama are forwarded as top-level fields in `/api/chat`.

## Error Handling

```python
from llm_provider import (
    LLMError, ConnectionFailedError, TimeoutError,
    RateLimitExceededError, ModelNotAvailableError,
    JSONParseFailedError, InvalidConfigurationError,
)

try:
    response = provider.chat(request)
except ConnectionFailedError as e:
    print(f"Network error: {e.message}")
except TimeoutError:
    print("Request timed out")
except RateLimitExceededError as e:
    wait = getattr(e, "retry_after", None)
    print(f"Rate limited. Retry after: {wait}s" if wait else "Rate limited.")
except ModelNotAvailableError as e:
    print(f"Model not found: {e.message}")
except JSONParseFailedError as e:
    print(f"Structured output parse failed: {e.message}")
except LLMError as e:
    print(f"[{e.error_type.value}] {e.message} (retryable={e.retryable})")
```

All `LLMError` subclasses carry: `e.message`, `e.error_type`, `e.retryable`,
`e.retry_count`, `e.cause` (the underlying exception).

The library applies exponential backoff automatically (`base_delay=2s`,
`max_delay=30s`, `backoff_factor=2.0`) for retryable errors up to `retry_attempts`
times. Non-retryable errors (`ModelNotAvailableError`, `InvalidConfigurationError`,
auth failures) surface immediately.

## Checking Features and Listing Models

```python
features = provider.supported_features()
if features.function_calling:
    # safe to pass tools=...
    ...
if features.async_supported:
    # safe to call achat
    ...

# List available models (result is cached)
models = factory.list_models("anthropic")
for m in models:
    print(m.name)

# Direct on provider
models = provider.list_models()

# Lightweight availability ping
if provider.is_available():
    ...
```

## Common Mistakes

1. **Missing `host`** — `ProviderConfig` requires `host` even though cloud
   providers ignore it. Use the canonical URL strings from the config reference.

2. **Anthropic `max_tokens`** — the Anthropic API rejects requests without it.
   The library defaults to 8192; pass `extra_settings={"max_tokens": 16384}` for
   long completions.

3. **Gemini role mapping** — Gemini uses `"model"` internally (not `"assistant"`).
   The provider handles the mapping; always use `"assistant"` in `Message` objects.

4. **Structured output + tools conflict** — if `tools` is set and the model
   returns tool calls, `response.structured_data` is `None`. Check
   `response.tool_calls` first when both are in play.

5. **Ollama async needs httpx** — `pip install httpx` is required for `achat`
   with Ollama; it is not included in the base dependencies.

6. **`FactoryConfig` default_provider must exist in provider_configs** — the
   string in `default_provider` must be a key in `provider_configs`, or
   `FactoryConfig.__post_init__` raises `ValueError`.
