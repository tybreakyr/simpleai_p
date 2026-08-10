# LLM Provider Abstraction Library

A lightweight Python library that provides a unified abstraction layer for interacting with multiple Large Language Model (LLM) providers. This library allows you to work with different LLM providers through a single interface, regardless of the underlying provider implementation.

## Features

- **Unified Interface**: Work with multiple LLM providers through a single, consistent API
- **Provider Abstraction**: Switch between providers without changing your code
- **Error Handling**: Comprehensive error handling with retry logic
- **Structured Output**: Robust JSON extraction and native tool-calling capabilities.
- **Multimodal Input**: Send images alongside text via `ImagePart` (base64) or forward-only `ImageUrl`, with fail-fast vision gating per provider.
- **Image Generation & Editing**: Text-to-image plus img2img (editing, inpainting, variations) via `generate_image()` — supply a source `image` (+`mask`) to route to the provider's edit endpoint; results returned as `ImagePart` for symmetry with image input.
- **Model-Aware Tool Schemas**: Automatically flatten nested tool parameters for models that can't handle them (e.g. small local Qwen models) and re-nest the response — callers always declare the natural nested schema.
- **Async Support**: Native `asyncio` support across all providers (`achat`).
- **Retry Mechanism**: Configurable exponential backoff retry logic (sync and async).
- **Thread-Safe**: Factory pattern with thread-safe provider and model caching.
- **Type Safety**: Full type hints and support for structured output parsing.

## Installation

```bash
pip install -r requirements.txt
```

Or install the package:

```bash
pip install -e .
```

For provider-specific dependencies:

```bash
pip install -e ".[gemini]"     # Google Gemini
pip install -e ".[openai]"     # OpenAI
pip install -e ".[anthropic]"  # Anthropic Claude
pip install -e ".[all]"        # All providers
```

## Quick Start

### Basic Usage

```python
from llm_provider import (
    ProviderFactory,
    FactoryConfig,
    ProviderConfig,
    ChatRequest,
    Message,
    create_ollama_provider,
)

# Create factory
factory = ProviderFactory()

# Register Ollama provider
factory.register_provider("ollama", create_ollama_provider)

# Create provider configuration
ollama_config = ProviderConfig(host="http://localhost:11434", default_model="llama2", timeout=30.0)

# Create factory configuration
factory_config = FactoryConfig(
    default_provider="ollama", provider_configs={"ollama": ollama_config}
)

# Load configuration
factory.load_config(factory_config)

# Get default provider
provider = factory.get_default_provider()

# Create chat request
request = ChatRequest(messages=[Message(role="user", content="Hello, how are you?")])

# Send request
response = provider.chat(request)
print(response.message)
```

### Async Usage

All providers support an asynchronous interface (`achat`) for concurrent applications.

```python
import asyncio
from llm_provider import ChatRequest, Message


async def fetch_chat():
    request = ChatRequest(messages=[Message(role="user", content="Hello!")])
    response = await provider.achat(request)
    print(response.message)


asyncio.run(fetch_chat())
```

### Structured Output

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class Person:
    name: str
    age: int
    email: Optional[str] = None


# Create request with structured output type
request = ChatRequest(
    messages=[
        Message(role="user", content="Extract person info from: John Doe, 30, john@example.com")
    ],
    structured_output_type=Person,
)

# Send request
response = provider.chat(request)

# Access structured data
if response.structured_data:
    person = response.structured_data
    print(f"Name: {person.name}, Age: {person.age}")
```

`structured_output_type` relies on prompting + robust JSON extraction. For
**provider-enforced** structured output, pass a raw JSON Schema via
`response_schema` instead — each provider maps it to its native mechanism and
returns the decoded object as a plain `dict` in `response.structured_data`:

| Provider | Native enforcement |
|----------|--------------------|
| OpenAI / mlx | `response_format` (`json_schema`, or `json_object` when `extra_settings["structured_output_format"]="json_object"`) |
| Gemini | `response_mime_type="application/json"` + `response_schema` |
| Ollama | `format=<schema>` |
| Anthropic | a forced single synthetic tool (no native JSON-schema mode) |

```python
schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}
request = ChatRequest(
    messages=[Message(role="user", content="Extract: John Doe, 30")],
    response_schema=schema,  # mutually exclusive with `tools`
    temperature=0.0,
)
response = provider.chat(request)
person: dict = response.structured_data  # {"name": "John Doe", "age": 30}
```

If the model returns text that can't be decoded to JSON, a retryable
`JSONParseFailedError` is raised so the retry layer re-rolls the call.

### Model & concurrency controls

Two `ProviderConfig` knobs cover common local-model needs:

- `max_concurrent`: cap concurrent in-flight async requests (an `asyncio.Semaphore`),
  useful for single-model servers like mlx-lm / Ollama that serialise requests.
- `extra_settings["disable_thinking"]`: when `True` and the model name contains
  `qwen3`, prepend `/no_think` to the system prompt (a no-op for other models).

The `create_mlx_provider` factory wraps the OpenAI provider with mlx-lm-friendly
defaults (`base_url=http://localhost:8000/v1`, `structured_output_format="json_object"`,
placeholder `api_key="mlx-lm"`, `vision=False`). Vision is opt-in for mlx-lm because
model support varies; set `extra_settings={"vision": True}` when using a vision-capable
mlx-vlm model.

### Tool Calling

Providers support native tool calling (function calling) where available.

```python
from llm_provider import ChatRequest, Message, ToolSchema

tools = [
    ToolSchema(
        name="get_weather",
        description="Get current weather",
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
    )
]

request = ChatRequest(
    messages=[Message(role="user", content="What's the weather in London?")],
    tools=tools,
    tool_choice="auto",
)

response = provider.chat(request)
if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool called: {call.name} with args {call.arguments}")
```

**`tool_choice` values:**

| Value | Behaviour |
|-------|-----------|
| `"auto"` | Model decides whether to call a tool |
| `"none"` | Tool calling disabled |
| `"any"` / `"required"` | Model must call a tool (`"required"` is accepted as a synonym and normalised to `"any"` internally) |
| `"<tool_name>"` | Model must call the named tool |

> **Ollama note:** Ollama has no native required-tool mode. When `tool_choice="required"` (or `"any"`) is used with Ollama and the first response contains no tool call, the provider automatically retries with a stricter system-prompt nudge. If the retry also returns no tool call, `InvalidResponseError` is raised.

### Nested tool parameters & per-model capabilities

Some models — notably small local ones such as Qwen3-family — mishandle **nested
objects** in tool-call parameters (they error or drop fields under load). Rather
than forcing every caller to hand-flatten such schemas, the library does it
automatically based on a small **per-model capability registry**:

- A tool whose `input_schema` contains a fixed-key nested object is **flattened
  on the wire** (`say: {text, determination}` → `say__text`, `say__determination`)
  before the request is sent, and the returned arguments are **re-nested** before
  they reach you. You declare — and receive — the natural nested shape.
- Only fixed-key objects flatten. Scalars, enums, arrays (including
  arrays-of-objects), and `additionalProperties` maps pass through untouched, so
  batch/map tool schemas are never altered. Tool names and `tool_choice` are
  preserved, so forced-tool behaviour is unaffected.
- The decision is per **model name** (`ChatRequest.model` or
  `ProviderConfig.default_model`), so the same provider can serve a weak model and
  a capable one. The registry lives in `llm_provider.model_capabilities`
  (`supports_nested_tool_params(name)`); unknown models are assumed capable.

```python
from llm_provider import supports_nested_tool_params

supports_nested_tool_params("gpt-4o")  # True  → schema sent as-is
supports_nested_tool_params("mlx-community/Qwen3.5-9B-MLX-4bit")  # False → auto-flattened
```

Override the decision per provider with
`extra_settings["flatten_tool_params"] = True | False` (beats the registry in
either direction). The pure codec is also exposed directly:
`flatten_tool_schema(schema) -> (flat_schema, mapping)` and
`renest_arguments(flat_args, mapping)`.

### Multi-turn Tool Loops (tool results)

To run an agentic loop — the model calls a tool, you execute it, feed the result
back, and let the model continue — replay the exchange using two extra `Message`
shapes. Roles map to each provider's native wire format automatically.

- An **assistant** turn that invoked tools carries `tool_calls` (echo back the
  `ToolCall` objects from `response.tool_calls`; `content` may be empty).
- A **tool-result** turn uses `role="tool"` with `tool_call_id` (the id of the
  call it answers), optional `name`, and the stringified result in `content`.

```python
from llm_provider import ChatRequest, Message

messages = [Message(role="user", content="What's the weather in London?")]

response = provider.chat(ChatRequest(messages=messages, tools=tools, tool_choice="auto"))

if response.tool_calls:
    call = response.tool_calls[0]
    # 1. Record the model's tool-call turn verbatim.
    messages.append(
        Message(role="assistant", content=response.message, tool_calls=response.tool_calls)
    )
    # 2. Execute the tool and feed the result back.
    result = get_weather(**call.arguments)  # your function
    messages.append(Message(role="tool", content=str(result), tool_call_id=call.id, name=call.name))
    # 3. Continue — the model now narrates/acts on the result.
    final = provider.chat(ChatRequest(messages=messages, tools=tools, tool_choice="auto"))
    print(final.message)
```

> **Gemini note:** Gemini 3.x requires the opaque `thought_signature` on a
> replayed `functionCall` part. It is captured into `ToolCall.thought_signature`
> on parse and replayed automatically when you echo the call back — no caller
> action needed. (Other providers leave the field `None`.)

### Per-request Provider Passthrough (`extra_body`)

`ChatRequest` accepts an `extra_body` dict that lets you pass provider-specific parameters on a per-request basis without touching `ProviderConfig`. Keys must be one of the known provider names (`"anthropic"`, `"openai"`, `"gemini"`, `"ollama"`); unknown keys raise `ValueError` at construction time to catch typos early. Each provider merges only its own bucket — other buckets are silently dropped.

```python
request = ChatRequest(
    messages=[Message(role="user", content="Tell me a story.")],
    extra_body={
        "anthropic": {"top_k": 40},
        "openai": {"seed": 42},
        "gemini": {},
        "ollama": {"keep_alive": "10m"},
    },
)
```

This is useful when you need a one-off parameter (e.g. `top_k` for a single creative call) without changing the provider's default configuration.

### Multimodal / image input

A `Message`'s `content` is normally a plain `str`. To send images, pass a list of
content parts instead. Text-only callers are unaffected — a bare string keeps working
exactly as before.

```python
from llm_provider import Message, TextPart, ImagePart, ImageUrl

# Inline base64 image (the universal path — works on every vision-capable provider)
with open("photo.png", "rb") as f:
    image = ImagePart.from_bytes(f.read(), "image/png")

request = ChatRequest(
    messages=[
        Message(role="user", content=[TextPart("What's in this image?"), image]),
    ],
)

# Or reference an image by URL (OpenAI and Anthropic only — see note below)
Message(role="user", content=[TextPart("Describe this"), ImageUrl("https://example.com/cat.jpg")])
```

Two image source types:

- **`ImagePart`** — base64 bytes + MIME type. The universal path; every vision-capable
  provider accepts it. Use `ImagePart.from_bytes(raw, "image/png")` to build one.
- **`ImageUrl`** — a **forward-only** URL. The library never fetches it; the URL is passed
  straight to the provider, which fetches it itself. Supported by **OpenAI** and
  **Anthropic** only. **Gemini** and **Ollama** require inline data and raise
  `ValidationError` asking for an `ImagePart`.

**Vision gating.** If a message carries an image but the target provider/model does not
support vision, a `ValidationError` is raised before any network call. Anthropic, OpenAI
and Gemini report vision support by default. Local providers are model-dependent, so they
require opt-in: **Ollama** (`llava`, `llama3.2-vision`, ...) and **mlx-lm**
(`create_mlx_provider`, e.g. an mlx-vlm model) both default vision off — enable it with
`extra_settings={"vision": True}`:

```python
config = ProviderConfig(
    host="http://localhost:11434",
    default_model="llava",
    extra_settings={"vision": True},  # required for image input on Ollama
)
```

### Image generation

Providers that support text-to-image expose `generate_image()` (and async
`agenerate_image()`). Generated images come back as `ImagePart`s — the same type used for
image *input* — so a result can be fed straight back into a `Message` for a describe/edit
loop.

```python
from llm_provider import ImageGenerationRequest

response = provider.generate_image(
    ImageGenerationRequest(prompt="a red bicycle on a beach", size="1024x1024", n=1)
)
image = response.images[0]  # an ImagePart (base64; MIME in image.media_type)

# persist it
import base64

with open("bike.png", "wb") as f:
    f.write(base64.b64decode(image.data))

# ...or feed it back into a chat turn
from llm_provider import Message, TextPart

Message(role="user", content=[TextPart("Describe this"), image])
```

#### Editing, inpainting, and variations (img2img)

Supplying a source `image` on the request switches `generate_image()` to img2img and
routes to the provider's edit endpoint — same method, no new call:

```python
from llm_provider import ImageGenerationRequest, ImagePart

src = ImagePart.from_bytes(open("photo.png", "rb").read(), "image/png")

# Edit — source image + prompt
provider.generate_image(ImageGenerationRequest(prompt="make it a snowy night", image=src))

# Inpainting — restrict the edit to a masked region (OpenAI only)
mask = ImagePart.from_bytes(open("mask.png", "rb").read(), "image/png")
provider.generate_image(ImageGenerationRequest(prompt="replace the sky", image=src, mask=mask))

# Variation — source image, no prompt (OpenAI only)
provider.generate_image(ImageGenerationRequest(image=src))
```

**Provider support:**

| Provider | Text→image | Edit / inpaint / variation | Notes |
|----------|-----------|----------------------------|-------|
| OpenAI | `images.generate` | `images.edit` (+`mask` for inpainting), `images.create_variation` (no prompt) | Default model `gpt-image-1`; set `extra_settings["image_model"]` to override (e.g. `"dall-e-3"`). Variations default to `dall-e-2` (`extra_settings["variation_model"]`). `size`/`quality` supported (`quality` rejected with `ValidationError` for `dall-e-2`). |
| Gemini | Imagen (`generate_images`) | flash image (`generate_content`, image+text→image) | Generation default `imagen-3.0-generate-002`; edit default `gemini-3.1-flash-image` (override via `extra_settings["image_edit_model"]`). **No mask input** — a `mask` raises `ValidationError`; variations (no prompt) are unsupported. Imagen sizes via `extra_body={"gemini": {"aspect_ratio": "16:9"}}`. |
| Anthropic / Ollama | — | — | No image API; raise `ValidationError`. |
| mlx-lm | — | — | No images endpoint; `image_generation` defaults off, raises `ValidationError`. |

Calling `generate_image()` on a provider/model without support for the requested mode
raises `ValidationError` before any network call.

### Error Handling

```python
from llm_provider.errors import LLMError, ConnectionFailedError, TimeoutError, is_retryable

try:
    response = provider.chat(request)
except ConnectionFailedError as e:
    print(f"Connection failed: {e}")
    if is_retryable(e):
        print("This error can be retried")
except TimeoutError as e:
    print(f"Request timed out: {e}")
except LLMError as e:
    print(f"LLM error: {e}")
```

### Listing Models

```python
# List available models for a provider
models = factory.list_models("ollama")
for model in models:
    print(f"Model: {model.name}")
```

### Checking Provider Availability

```python
# Check if provider is available
if factory.is_provider_available("ollama"):
    print("Provider is available")
else:
    print("Provider is not available")
```

## Architecture

### Core Components

1. **Provider Interface**: Abstract base class that all providers must implement
2. **Factory Pattern**: Manages provider creation, caching, and lifecycle
3. **Error Handling**: Comprehensive error types with retryability classification
4. **Retry Mechanism**: Exponential backoff retry logic
5. **JSON Extraction**: Robust extraction of JSON from LLM responses
6. **Configuration System**: Structured configuration with validation

### Data Models

- `Message`: Represents a single message in a conversation; `content` is `str` or `list[ContentPart]`
- `TextPart`: Plain-text segment of a multimodal message
- `ImagePart`: Inline base64 image + MIME type (universal; every vision provider accepts it)
- `ImageUrl`: Forward-only http(s) URL; passed straight to providers that fetch it (OpenAI, Anthropic)
- `ContentPart`: Type alias for `TextPart | ImagePart | ImageUrl`
- `SystemPrompt`: System-level instructions
- `ChatRequest`: Input structure for chat operations (supports `tools`, `tool_choice`, and `extra_body`)
- `ChatResponse`: Output structure from chat operations (supports `tool_calls`)
- `ImageGenerationRequest`: Input for image generation/editing (`prompt`, `model`, `n`, `size`, `quality`, `extra_body`, plus img2img `image` and `mask`). `prompt` is required for text→image; when `image` is set it's optional (omit it for an OpenAI variation). Exposes `is_edit` / `is_variation` mode helpers.
- `ImageGenerationResponse`: Output from image generation or editing (`list[ImagePart]` + optional `revised_prompt`)
- `ToolSchema`: Definition of an available tool (function)
- `ToolCall`: A tool invocation requested by the model
- `Model`: Represents an available LLM model
- `ProviderFeatures`: Describes provider capabilities (per provider)
- `ModelCapabilities`: Per-model quirks (e.g. `supports_nested_tool_params`); drives automatic tool-schema flattening
- `ProviderConfig`: Configuration for a provider instance
- `FactoryConfig`: Configuration for the provider factory

### Error Types

- `ConnectionFailedError`: Network/service unavailable
- `TimeoutError`: Request exceeded time limit
- `InvalidResponseError`: Malformed API response
- `JSONParseFailedError`: Structured output parsing failure
- `ModelNotAvailableError`: Requested model doesn't exist
- `RateLimitExceededError`: API rate limiting
- `InvalidConfigurationError`: Configuration errors
- `ValidationError`: Field-level validation errors

## Provider Implementations

### Ollama Provider

The Ollama provider supports local and remote Ollama instances via its REST API.

```python
from llm_provider.providers import create_ollama_provider

# Register Ollama provider
factory.register_provider("ollama", create_ollama_provider)

# Configuration
ollama_config = ProviderConfig(
    host="http://localhost:11434",  # Default Ollama URL
    default_model="llama3",
    timeout=30.0,
)
```

**Ollama-specific `extra_settings`:**

The `extra_settings` dict is forwarded as top-level fields in the Ollama `/api/chat` payload, enabling model-specific options:

| Setting | Type | Description |
|---------|------|-------------|
| `think` | bool | Disable chain-of-thought on thinking models (e.g. `false` for Qwen3) |
| `keep_alive` | str | Keep model in memory between requests (e.g. `"10m"`) |
| `vision` | bool | Enable image input for vision-capable models (e.g. `llava`, `llama3.2-vision`); defaults to `False` |

```python
ollama_config = ProviderConfig(
    host="http://localhost:11434",
    default_model="qwen3:8b",
    extra_settings={"think": False, "keep_alive": "10m"},
)
```

### Gemini Provider

The Gemini provider uses the `google-genai` SDK to connect to Google's Gemini API. Requires an API key from [Google AI Studio](https://aistudio.google.com/).

**Install the extra dependency first:**

```bash
pip install google-genai>=1.0.0
```

```python
from llm_provider.providers import create_gemini_provider

# Register Gemini provider
factory.register_provider("gemini", create_gemini_provider)

# Configuration
gemini_config = ProviderConfig(
    host="https://generativelanguage.googleapis.com",  # for interface consistency
    default_model="gemini-2.0-flash",
    api_key="YOUR_GEMINI_API_KEY",
    timeout=60.0,
)
```

**Supported Gemini features:**
- Structured output (via JSON extraction)
- Vision / multimodal input
- System prompts (`system_instruction`)
- Temperature and top_p sampling
- Up to ~1M token context window (model-dependent)

### OpenAI Provider

The OpenAI provider uses the `openai` SDK (v1.x+). Requires an API key from [OpenAI](https://platform.openai.com/).

**Install the extra dependency first:**

```bash
pip install openai>=1.0.0
```

```python
from llm_provider.providers import create_openai_provider

factory.register_provider("openai", create_openai_provider)

openai_config = ProviderConfig(
    host="https://api.openai.com",
    default_model="gpt-4o",
    api_key="YOUR_OPENAI_API_KEY",
    timeout=60.0,
)
```

**Supported OpenAI features:**
- Structured output (via JSON extraction)
- Vision / multimodal input (model-dependent, e.g. gpt-4o)
- System prompts
- Temperature and top_p sampling
- Function calling
- Up to 128k token context window (model-dependent)

**OpenAI-compatible endpoints** — set `base_url` in `extra_settings` to point at any
OpenAI-compatible API (Azure OpenAI, local proxy, Groq, etc.):

```python
openai_config = ProviderConfig(
    host="https://api.openai.com",
    default_model="gpt-4o",
    api_key="YOUR_KEY",
    extra_settings={"base_url": "https://your-azure-endpoint.openai.azure.com/"},
)
```

### Anthropic Provider

The Anthropic provider uses the `anthropic` SDK. Requires an API key from [Anthropic](https://console.anthropic.com/).

**Install the extra dependency first:**

```bash
pip install anthropic
```

```python
from llm_provider.providers import create_anthropic_provider

factory.register_provider("anthropic", create_anthropic_provider)

anthropic_config = ProviderConfig(
    host="https://api.anthropic.com",
    default_model="claude-sonnet-4-6",
    api_key="YOUR_ANTHROPIC_API_KEY",
    timeout=60.0,
)
```

**Supported Anthropic features:**
- Structured output (via JSON extraction)
- Vision / multimodal input (model-dependent)
- System prompts (passed as top-level `system` parameter)
- Temperature and top_p sampling
- Function calling (tool use)
- Up to 200k token context window (model-dependent)

**Anthropic-specific `extra_settings`:**

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `max_tokens` | int | `8192` | Maximum tokens to generate (required by the API) |

```python
anthropic_config = ProviderConfig(
    host="https://api.anthropic.com",
    default_model="claude-opus-4-7",
    api_key="YOUR_ANTHROPIC_API_KEY",
    extra_settings={"max_tokens": 16384},
)
```

**Using multiple providers together:**

```python
from llm_provider import ProviderFactory
from llm_provider.providers import (
    create_ollama_provider,
    create_openai_provider,
    create_anthropic_provider,
)

factory = ProviderFactory()
factory.register_provider("ollama", create_ollama_provider)
factory.register_provider("openai", create_openai_provider)
factory.register_provider("anthropic", create_anthropic_provider)

# Use fast local model for cheap tasks
local = factory.create_provider("ollama", ollama_config)
filter_result = local.chat(filter_request)

# Use cloud model for high-quality reasoning
cloud = factory.create_provider("anthropic", anthropic_config)
recommendation = cloud.chat(recommend_request)
```

## Creating Custom Providers

To create a custom provider, implement the `Provider` interface:

```python
from llm_provider import Provider, ChatRequest, ChatResponse, Model, ProviderFeatures
from llm_provider.models import ProviderConfig


class MyCustomProvider(Provider):
    def __init__(self, config: ProviderConfig):
        self._config = config

    def chat(self, request: ChatRequest) -> ChatResponse:
        # Implement chat logic
        pass

    def list_models(self) -> List[Model]:
        # Implement model listing
        pass

    def name(self) -> str:
        return "my_custom_provider"

    def is_available(self) -> bool:
        # Implement availability check
        pass

    def supported_features(self) -> ProviderFeatures:
        # Return supported features
        pass


# Factory function
def create_my_provider(config_dict: dict) -> Provider:
    config = ProviderConfig(**config_dict)
    return MyCustomProvider(config)


# Register provider
factory.register_provider("my_provider", create_my_provider)
```

## Configuration

### Factory Configuration

```python
factory_config = FactoryConfig(
    default_provider="ollama",
    provider_configs={
        "ollama": ProviderConfig(
            host="http://localhost:11434", default_model="llama2", timeout=30.0, retry_attempts=3
        )
    },
    model_preferences={"summarization": "llama2", "code": "codellama"},
    fallback_providers=["ollama", "backup_ollama"],
)
```

### Provider Configuration

```python
provider_config = ProviderConfig(
    host="http://localhost:11434",
    default_model="llama2",
    timeout=30.0,
    retry_attempts=3,
    api_key="optional-api-key",
    rate_limit=60,  # requests per minute
    extra_settings={"custom_setting": "value"},
)
```

## Testing

Run tests with pytest:

```bash
pytest
```

Or with unittest:

```bash
python -m unittest discover tests
```

## Design Document

This library was created based on the comprehensive design specification in `CREATEME.md`. The CREATEME.md file contains detailed architecture documentation, implementation guidelines, and design principles that were used to build this Python implementation.

The design document is language-agnostic and can be used as a blueprint to recreate this library in any programming language. It covers:

- Core architecture and component design
- Provider interface specifications
- Error handling patterns
- Retry mechanisms
- JSON extraction strategies
- Configuration systems
- Testing considerations

Whether you're porting this library to another language or building a similar abstraction layer, the CREATEME.md file serves as a complete reference for the design decisions and implementation patterns used in this codebase.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting (configured in `pyproject.toml`). Before opening a PR, run:

```bash
pip install -e ".[dev]"
ruff check .
ruff format .
```

Ruff is pinned to `0.16.2` in the `dev` extra of `pyproject.toml` and in CI. Install the `dev` extra rather than a bare `pip install ruff`: an unpinned ruff can pull in formatting changes (such as the Markdown code-block formatting added in 0.16) that CI doesn't apply, causing local and CI results to diverge. Bump both places together.

The CI workflow (`.github/workflows/ci.yml`) runs `ruff check` and `ruff format --check` automatically; PRs that fail the lint or format step will not be merged.

### Automated checks

Pull requests run an LLM **doc-drift check** (`.github/workflows/docs-check.yml`,
`anthropics/claude-code-action`): it inspects the PR diff and, if this README or
other docs lag the code, **pushes verified doc-fix commits straight to the PR
branch**. Because the bot commits to the branch, always `git fetch` + rebase
before pushing more work to an open PR — **never force-push over its commits.**
The check needs the `CLAUDE_CODE_OAUTH_TOKEN` repo secret (already configured).

