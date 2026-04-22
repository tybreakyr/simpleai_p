"""
LLM Provider Abstraction Library

A unified abstraction layer for interacting with multiple LLM providers.
"""

__version__ = "0.1.0"

# Core interfaces
from .provider import Provider
from .factory import ProviderFactory

# Data models
from .models import (
    Message,
    SystemPrompt,
    ChatRequest,
    ChatResponse,
    Model,
    ProviderFeatures,
    ProviderConfig,
    FactoryConfig,
)

# Error handling
from .errors import (
    LLMError,
    ValidationError,
    ConnectionFailedError,
    TimeoutError,
    InvalidResponseError,
    JSONParseFailedError,
    ModelNotAvailableError,
    RateLimitExceededError,
    InvalidConfigurationError,
    OperationFailedError,
    ErrorType,
    is_retryable,
    classify_error,
)

# Utilities
from .json_extractor import extract_json, parse_structured_output
from .retry import RetryConfig, retry_with_backoff, retryable

# Configuration
from .config import (
    validate_factory_config,
    validate_provider_config,
    load_factory_config_from_dict,
    load_factory_config_from_json,
)

# Provider implementations
from .providers import create_ollama_provider, create_gemini_provider

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
    
    # Configuration
    "validate_factory_config",
    "validate_provider_config",
    "load_factory_config_from_dict",
    "load_factory_config_from_json",
    
    # Provider implementations
    "create_ollama_provider",
    "create_gemini_provider",
]
