"""
Core data structures for the LLM provider abstraction library.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, TypeVar, Generic
from enum import Enum
from uuid import uuid4


T = TypeVar('T')


class MessageRole(str, Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Represents a single message in a conversation.

    Beyond plain text turns, a ``Message`` can also represent the two halves
    of a tool-calling exchange so multi-turn tool loops can be replayed back
    to a provider:

    - An *assistant* turn that invoked tools carries ``tool_calls`` (the calls
      the model made). ``content`` may be empty for a tool-only turn.
    - A *tool-result* turn (``role="tool"``) carries ``tool_call_id`` (the id
      of the call it answers) and optionally ``name``; ``content`` holds the
      stringified tool result.

    Providers translate these neutral fields into their native wire format.
    """
    role: str
    content: str
    tool_calls: Optional[List["ToolCall"]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self):
        """Validate message after initialization."""
        if not self.role:
            raise ValueError("Message role cannot be empty")
        if not isinstance(self.content, str):
            raise ValueError("Message content must be a string")
        if self.role == MessageRole.TOOL.value and not self.tool_call_id:
            raise ValueError("Tool-result message requires a tool_call_id")


@dataclass
class SystemPrompt:
    """Represents system-level instructions."""
    content: str

    def __post_init__(self):
        """Validate system prompt after initialization."""
        if not isinstance(self.content, str):
            raise ValueError("System prompt content must be a string")


@dataclass
class ToolSchema:
    """Describes a tool/function the model may call.

    Use JSON Schema in ``input_schema`` to constrain the arguments the model
    must supply.  This neutral format is translated to each provider's native
    wire format inside the provider implementation.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]

    def __post_init__(self):
        if not self.name:
            raise ValueError("ToolSchema name cannot be empty")
        if not isinstance(self.input_schema, dict):
            raise ValueError("ToolSchema input_schema must be a dict")


@dataclass
class ToolCall:
    """A single tool invocation returned by the model.

    ``id`` is provider-generated for Anthropic/OpenAI; Ollama does not return
    one so the Ollama provider synthesises ``call_<12-hex-chars>`` instead.
    """
    id: str
    name: str
    arguments: Dict[str, Any]
    # Opaque provider-specific data that must be replayed verbatim when this
    # call is sent back in conversation history. Gemini 3.x uses it to carry a
    # ``thought_signature`` on the functionCall part (required, or the API 400s).
    # Other providers leave it None.
    thought_signature: Optional[Any] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("ToolCall name cannot be empty")
        if not isinstance(self.arguments, dict):
            raise ValueError("ToolCall arguments must be a dict")

    @staticmethod
    def make_id() -> str:
        return f"call_{uuid4().hex[:12]}"


_KNOWN_PROVIDERS = frozenset({"openai", "anthropic", "gemini", "ollama"})


@dataclass
class ChatRequest(Generic[T]):
    """Input structure for chat operations."""
    messages: List[Message]
    system_prompt: Optional[SystemPrompt] = None
    structured_output_type: Optional[type[T]] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[ToolSchema]] = None
    tool_choice: Optional[str] = None  # "auto" | "any" | "required" | "none" | tool name
    # Raw JSON Schema (dict) for schema-driven structured output. Each provider
    # maps this to its native enforcement mechanism (OpenAI response_format,
    # Gemini response_schema, Ollama format, Anthropic forced tool) and returns
    # the decoded object as a plain dict in ``ChatResponse.structured_data``.
    # Mutually exclusive with caller-supplied ``tools`` (the Anthropic path
    # needs the tool slot to enforce the schema).
    response_schema: Optional[Dict[str, Any]] = None
    # Per-provider passthrough into the outgoing request body. Top-level keys
    # must be one of the known provider names; unknown keys raise ValueError
    # so typos don't silently no-op. Each provider merges its own bucket into
    # its outgoing payload — other buckets are dropped.
    extra_body: Optional[Dict[str, Dict[str, Any]]] = None

    def __post_init__(self):
        """Validate chat request after initialization."""
        if not self.messages:
            raise ValueError("Chat request must contain at least one message")
        if self.tool_choice == "required":
            self.tool_choice = "any"
        if self.response_schema is not None:
            if not isinstance(self.response_schema, dict):
                raise ValueError("response_schema must be a dict (JSON Schema)")
            if self.tools:
                raise ValueError(
                    "response_schema cannot be combined with caller-supplied tools"
                )
        if self.extra_body is not None:
            if not isinstance(self.extra_body, dict):
                raise ValueError("extra_body must be a dict of provider-name -> dict")
            unknown = set(self.extra_body) - _KNOWN_PROVIDERS
            if unknown:
                raise ValueError(
                    f"extra_body has unknown provider key(s) {sorted(unknown)}; "
                    f"allowed: {sorted(_KNOWN_PROVIDERS)}"
                )
            for k, v in self.extra_body.items():
                if not isinstance(v, dict):
                    raise ValueError(
                        f"extra_body[{k!r}] must be a dict; got {type(v).__name__}"
                    )


@dataclass
class ChatResponse(Generic[T]):
    """Output structure from chat operations."""
    message: str
    structured_data: Optional[T] = None
    tool_calls: Optional[List[ToolCall]] = None
    stop_reason: Optional[str] = None  # "end_turn" | "tool_use" | "max_tokens"

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
    async_supported: bool = False

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
    # Cap on concurrent in-flight async requests (asyncio.Semaphore). Useful for
    # single-model local servers (mlx-lm, ollama) that serialise requests.
    # None / <=0 means unlimited.
    max_concurrent: Optional[int] = None
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

