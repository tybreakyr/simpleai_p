"""
Core data structures for the LLM provider abstraction library.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, TypeVar, Generic
from enum import Enum


T = TypeVar('T')


class MessageRole(str, Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str
    content: str

    def __post_init__(self):
        """Validate message after initialization."""
        if not self.role:
            raise ValueError("Message role cannot be empty")
        if not isinstance(self.content, str):
            raise ValueError("Message content must be a string")


@dataclass
class SystemPrompt:
    """Represents system-level instructions."""
    content: str

    def __post_init__(self):
        """Validate system prompt after initialization."""
        if not isinstance(self.content, str):
            raise ValueError("System prompt content must be a string")


@dataclass
class ChatRequest(Generic[T]):
    """Input structure for chat operations."""
    messages: List[Message]
    system_prompt: Optional[SystemPrompt] = None
    structured_output_type: Optional[type[T]] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    def __post_init__(self):
        """Validate chat request after initialization."""
        if not self.messages:
            raise ValueError("Chat request must contain at least one message")


@dataclass
class ChatResponse(Generic[T]):
    """Output structure from chat operations."""
    message: str
    structured_data: Optional[T] = None

    def __post_init__(self):
        """Validate chat response after initialization."""
        if not isinstance(self.message, str):
            raise ValueError("Chat response message must be a string")


@dataclass
class Model:
    """Represents an available LLM model."""
    name: str

    def __post_init__(self):
        """Validate model after initialization."""
        if not self.name:
            raise ValueError("Model name cannot be empty")


@dataclass
class ProviderFeatures:
    """Describes provider capabilities."""
    structured_output: bool = False
    streaming: bool = False
    vision: bool = False
    context_window: int = 0
    supported_roles: List[str] = field(default_factory=lambda: ["user", "assistant"])
    function_calling: bool = False
    temperature: bool = True
    top_p: bool = True

    def __post_init__(self):
        """Validate provider features after initialization."""
        if self.context_window < 0:
            raise ValueError("Context window must be non-negative")
        if not self.supported_roles:
            raise ValueError("Provider must support at least one role")


@dataclass
class ProviderConfig:
    """Configuration for a provider instance."""
    host: str
    default_model: str
    timeout: float = 30.0
    retry_attempts: int = 3
    api_key: Optional[str] = None
    rate_limit: Optional[int] = None  # requests per minute
    extra_settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate provider configuration after initialization."""
        if not self.host:
            raise ValueError("Provider host cannot be empty")
        if not self.default_model:
            raise ValueError("Provider default model cannot be empty")
        if self.timeout <= 0:
            raise ValueError("Provider timeout must be positive")
        if self.retry_attempts < 0:
            raise ValueError("Provider retry attempts must be non-negative")
        if self.rate_limit is not None and self.rate_limit <= 0:
            raise ValueError("Provider rate limit must be positive if specified")


@dataclass
class FactoryConfig:
    """Configuration for the provider factory."""
    default_provider: str
    provider_configs: Dict[str, ProviderConfig]
    model_preferences: Dict[str, str] = field(default_factory=dict)  # task -> model mapping
    fallback_providers: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate factory configuration after initialization."""
        if not self.default_provider:
            raise ValueError("Factory default provider cannot be empty")
        if not self.provider_configs:
            raise ValueError("Factory must have at least one provider configuration")
        if self.default_provider not in self.provider_configs:
            raise ValueError(
                f"Factory default provider '{self.default_provider}' must exist in provider_configs"
            )

