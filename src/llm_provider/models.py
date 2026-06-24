"""
Core data structures for the LLM provider abstraction library.
"""

import base64
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar
from uuid import uuid4

T = TypeVar("T")


class MessageRole(StrEnum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class TextPart:
    """A plain-text segment of a multimodal message."""

    text: str

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise ValueError("TextPart text must be a string")


@dataclass
class ImagePart:
    """An inline image carried as base64-encoded bytes plus its MIME type.

    This is the universal image path: every vision-capable provider accepts
    inline base64 data. Use :meth:`from_bytes` to build one from raw bytes.
    """

    data: str  # base64-encoded image bytes
    media_type: str  # e.g. "image/jpeg", "image/png", "image/gif", "image/webp"

    def __post_init__(self):
        if not isinstance(self.data, str) or not self.data:
            raise ValueError("ImagePart data must be a non-empty base64 string")
        if not isinstance(self.media_type, str) or not self.media_type.startswith("image/"):
            raise ValueError("ImagePart media_type must be an image/* MIME type")

    @classmethod
    def from_bytes(cls, raw: bytes, media_type: str) -> "ImagePart":
        """Build an ImagePart from raw image bytes (pure base64 — no I/O)."""
        return cls(base64.b64encode(raw).decode("ascii"), media_type)


@dataclass
class ImageUrl:
    """A forward-only image URL.

    The library NEVER fetches this URL; it is passed straight through to
    providers that fetch it themselves (OpenAI, Anthropic), so there is no
    server-side request-forgery surface on our side. Providers that do not
    accept arbitrary URLs (Gemini, Ollama) raise a clear error asking for an
    :class:`ImagePart` instead.
    """

    url: str

    def __post_init__(self):
        # Reject file:/data:/etc. so a forwarded URL can't surprise a provider.
        if not isinstance(self.url, str) or not (
            self.url.startswith("https://") or self.url.startswith("http://")
        ):
            raise ValueError("ImageUrl must be an http(s) URL")


# A single piece of message content. A plain ``str`` content is equivalent to
# ``[TextPart(str)]`` and remains the fully back-compatible text-only path.
ContentPart = TextPart | ImagePart | ImageUrl


def as_content_parts(content: "str | list[ContentPart]") -> list[ContentPart]:
    """Normalize message content to a list of parts (str -> single TextPart)."""
    return [TextPart(content)] if isinstance(content, str) else content


def message_has_images(messages: list["Message"]) -> bool:
    """Whether any message carries image content (ImagePart or ImageUrl)."""
    return any(
        isinstance(m.content, list) and any(isinstance(p, (ImagePart, ImageUrl)) for p in m.content)
        for m in messages
    )


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

    ``content`` is either a plain ``str`` (text-only, the back-compatible path)
    or a list of :data:`ContentPart` (``TextPart``/``ImagePart``/``ImageUrl``)
    for multimodal turns. List content is only valid on ``user``/``assistant``
    turns; tool-call and tool-result turns remain text-only.
    """

    role: str
    content: "str | list[ContentPart]"
    tool_calls: list["ToolCall"] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self):
        """Validate message after initialization."""
        if not self.role:
            raise ValueError("Message role cannot be empty")
        if isinstance(self.content, str):
            pass
        elif isinstance(self.content, list):
            if not self.content:
                raise ValueError("Message content list cannot be empty")
            if not all(isinstance(p, (TextPart, ImagePart, ImageUrl)) for p in self.content):
                raise ValueError(
                    "Message content list items must be TextPart, ImagePart or ImageUrl"
                )
            if self.role == MessageRole.TOOL.value or self.tool_calls:
                raise ValueError("Tool-call and tool-result messages must use string content")
        else:
            raise ValueError("Message content must be a string or a list of content parts")
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
    input_schema: dict[str, Any]

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
    arguments: dict[str, Any]
    # Opaque provider-specific data that must be replayed verbatim when this
    # call is sent back in conversation history. Gemini 3.x uses it to carry a
    # ``thought_signature`` on the functionCall part (required, or the API 400s).
    # Other providers leave it None.
    thought_signature: Any | None = None

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

    messages: list[Message]
    system_prompt: SystemPrompt | None = None
    structured_output_type: type[T] | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[ToolSchema] | None = None
    tool_choice: str | None = None  # "auto" | "any" | "required" | "none" | tool name
    # Raw JSON Schema (dict) for schema-driven structured output. Each provider
    # maps this to its native enforcement mechanism (OpenAI response_format,
    # Gemini response_schema, Ollama format, Anthropic forced tool) and returns
    # the decoded object as a plain dict in ``ChatResponse.structured_data``.
    # Mutually exclusive with caller-supplied ``tools`` (the Anthropic path
    # needs the tool slot to enforce the schema).
    response_schema: dict[str, Any] | None = None
    # Per-provider passthrough into the outgoing request body. Top-level keys
    # must be one of the known provider names; unknown keys raise ValueError
    # so typos don't silently no-op. Each provider merges its own bucket into
    # its outgoing payload — other buckets are dropped.
    extra_body: dict[str, dict[str, Any]] | None = None

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
                raise ValueError("response_schema cannot be combined with caller-supplied tools")
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
                    raise ValueError(f"extra_body[{k!r}] must be a dict; got {type(v).__name__}")


@dataclass
class ChatResponse(Generic[T]):
    """Output structure from chat operations."""

    message: str
    structured_data: T | None = None
    tool_calls: list[ToolCall] | None = None
    stop_reason: str | None = None  # "end_turn" | "tool_use" | "max_tokens"

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
    supported_roles: list[str] = field(default_factory=lambda: ["user", "assistant"])
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
    api_key: str | None = None
    rate_limit: int | None = None  # requests per minute
    # Cap on concurrent in-flight async requests (asyncio.Semaphore). Useful for
    # single-model local servers (mlx-lm, ollama) that serialise requests.
    # None / <=0 means unlimited.
    max_concurrent: int | None = None
    extra_settings: dict[str, Any] = field(default_factory=dict)

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
    provider_configs: dict[str, ProviderConfig]
    model_preferences: dict[str, str] = field(default_factory=dict)  # task -> model mapping
    fallback_providers: list[str] = field(default_factory=list)

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
