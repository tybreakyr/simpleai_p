# LLM Provider Abstraction Library

A lightweight Python library that provides a unified abstraction layer for interacting with multiple Large Language Model (LLM) providers. This library allows you to work with different LLM providers through a single interface, regardless of the underlying provider implementation.

## Features

- **Unified Interface**: Work with multiple LLM providers through a single, consistent API
- **Provider Abstraction**: Switch between providers without changing your code
- **Error Handling**: Comprehensive error handling with retry logic
- **Structured Output**: Robust JSON extraction and native tool-calling capabilities.
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
    ProviderFactory, FactoryConfig, ProviderConfig,
    ChatRequest, Message, create_ollama_provider
)

# Create factory
factory = ProviderFactory()

# Register Ollama provider
factory.register_provider("ollama", create_ollama_provider)

# Create provider configuration
ollama_config = ProviderConfig(
    host="http://localhost:11434",
    default_model="llama2",
    timeout=30.0
)

# Create factory configuration
factory_config = FactoryConfig(
    default_provider="ollama",
    provider_configs={
        "ollama": ollama_config
    }
)

# Load configuration
factory.load_config(factory_config)

# Get default provider
provider = factory.get_default_provider()

# Create chat request
request = ChatRequest(
    messages=[
        Message(role="user", content="Hello, how are you?")
    ]
)

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
    structured_output_type=Person
)

# Send request
response = provider.chat(request)

# Access structured data
if response.structured_data:
    person = response.structured_data
    print(f"Name: {person.name}, Age: {person.age}")
```

### Tool Calling

Providers support native tool calling (function calling) where available.

```python
from llm_provider import ChatRequest, Message, ToolSchema

tools = [
    ToolSchema(
        name="get_weather",
        description="Get current weather",
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}}
    )
]

request = ChatRequest(
    messages=[Message(role="user", content="What's the weather in London?")],
    tools=tools,
    tool_choice="auto"
)

response = provider.chat(request)
if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool called: {call.name} with args {call.arguments}")
```

### Error Handling

```python
from llm_provider.errors import (
    LLMError, ConnectionFailedError, TimeoutError,
    is_retryable
)

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

- `Message`: Represents a single message in a conversation
- `SystemPrompt`: System-level instructions
- `ChatRequest`: Input structure for chat operations (supports `tools` and `tool_choice`)
- `ChatResponse`: Output structure from chat operations (supports `tool_calls`)
- `ToolSchema`: Definition of an available tool (function)
- `ToolCall`: A tool invocation requested by the model
- `Model`: Represents an available LLM model
- `ProviderFeatures`: Describes provider capabilities
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
    timeout=30.0
)
```

**Ollama-specific `extra_settings`:**

The `extra_settings` dict is forwarded as top-level fields in the Ollama `/api/chat` payload, enabling model-specific options:

| Setting | Type | Description |
|---------|------|-------------|
| `think` | bool | Disable chain-of-thought on thinking models (e.g. `false` for Qwen3) |
| `keep_alive` | str | Keep model in memory between requests (e.g. `"10m"`) |

```python
ollama_config = ProviderConfig(
    host="http://localhost:11434",
    default_model="qwen3:8b",
    extra_settings={"think": False, "keep_alive": "10m"}
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
    timeout=60.0
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
    timeout=60.0
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
    extra_settings={"base_url": "https://your-azure-endpoint.openai.azure.com/"}
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
    timeout=60.0
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
    extra_settings={"max_tokens": 16384}
)
```

**Using multiple providers together:**

```python
from llm_provider import ProviderFactory
from llm_provider.providers import (
    create_ollama_provider, create_openai_provider, create_anthropic_provider
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
            host="http://localhost:11434",
            default_model="llama2",
            timeout=30.0,
            retry_attempts=3
        )
    },
    model_preferences={
        "summarization": "llama2",
        "code": "codellama"
    },
    fallback_providers=["ollama", "backup_ollama"]
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
    extra_settings={
        "custom_setting": "value"
    }
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

