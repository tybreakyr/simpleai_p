# LLM Provider Abstraction Library

A lightweight Python library that provides a unified abstraction layer for interacting with multiple Large Language Model (LLM) providers. This library allows you to work with different LLM providers through a single interface, regardless of the underlying provider implementation.

## Features

- **Unified Interface**: Work with multiple LLM providers through a single, consistent API
- **Provider Abstraction**: Switch between providers without changing your code
- **Error Handling**: Comprehensive error handling with retry logic
- **Structured Output**: Robust JSON extraction and parsing from LLM responses
- **Retry Mechanism**: Configurable exponential backoff retry logic
- **Thread-Safe**: Factory pattern with thread-safe provider and model caching
- **Type Safety**: Full type hints and support for structured output parsing

## Installation

```bash
pip install -r requirements.txt
```

Or install the package:

```bash
pip install -e .
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
- `ChatRequest`: Input structure for chat operations
- `ChatResponse`: Output structure from chat operations
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

The Ollama provider supports local and remote Ollama instances.

```python
from llm_provider.providers import create_ollama_provider

# Register Ollama provider
factory.register_provider("ollama", create_ollama_provider)

# Configuration
ollama_config = ProviderConfig(
    host="http://localhost:11434",  # Default Ollama URL
    default_model="llama2",
    timeout=30.0
)
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

