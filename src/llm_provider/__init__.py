"""
LLM Provider Abstraction Library

A unified abstraction layer for interacting with multiple LLM providers.
"""

__version__ = "0.1.0"

# Core interfaces
# Configuration
from .config import (
    load_factory_config_from_dict,
    load_factory_config_from_json,
    validate_factory_config,
    validate_provider_config,
)

# Error handling
from .errors import (
    ConnectionFailedError,
    ErrorType,
    InvalidConfigurationError,
    InvalidResponseError,
    JSONParseFailedError,
    LLMError,
    ModelNotAvailableError,
    OperationFailedError,
    RateLimitExceededError,
    TimeoutError,
    ValidationError,
    classify_error,
    is_retryable,
)
from .factory import ProviderFactory

# Utilities
from .json_extractor import extract_json, parse_structured_output

# Model capabilities + tool-schema flattening codec
from .model_capabilities import (
    ModelCapabilities,
    get_model_capabilities,
    supports_nested_tool_params,
)

# Data models
from .models import (
    ChatRequest,
    ChatResponse,
    FactoryConfig,
    Message,
    Model,
    ProviderConfig,
    ProviderFeatures,
    SystemPrompt,
    ToolCall,
    ToolSchema,
)
from .provider import Provider

# Provider implementations
from .providers import (
    create_anthropic_provider,
    create_gemini_provider,
    create_mlx_provider,
    create_ollama_provider,
    create_openai_provider,
)
from .retry import RetryConfig, retry_with_backoff, retryable
from .schema_transform import (
    flatten_tool_schema,
    renest_arguments,
    schema_has_flattenable_nesting,
)

__all__ = [
    # Version
    "__version__",
    # Core interfaces
    "Provider",
    "ProviderFactory",
    # Data models
    "Message",
    "SystemPrompt",
    "ChatRequest",
    "ChatResponse",
    "Model",
    "ProviderFeatures",
    "ProviderConfig",
    "FactoryConfig",
    "ToolSchema",
    "ToolCall",
    # Error handling
    "LLMError",
    "ValidationError",
    "ConnectionFailedError",
    "TimeoutError",
    "InvalidResponseError",
    "JSONParseFailedError",
    "ModelNotAvailableError",
    "RateLimitExceededError",
    "InvalidConfigurationError",
    "OperationFailedError",
    "ErrorType",
    "is_retryable",
    "classify_error",
    # Utilities
    "extract_json",
    "parse_structured_output",
    "RetryConfig",
    "retry_with_backoff",
    "retryable",
    # Model capabilities + schema codec
    "ModelCapabilities",
    "get_model_capabilities",
    "supports_nested_tool_params",
    "flatten_tool_schema",
    "renest_arguments",
    "schema_has_flattenable_nesting",
    # Configuration
    "validate_factory_config",
    "validate_provider_config",
    "load_factory_config_from_dict",
    "load_factory_config_from_json",
    # Provider implementations
    "create_ollama_provider",
    "create_gemini_provider",
    "create_openai_provider",
    "create_mlx_provider",
    "create_anthropic_provider",
]
