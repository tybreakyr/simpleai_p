"""
Anthropic provider implementation via the anthropic SDK.

Uses the current anthropic SDK client interface (Anthropic().messages.create).

Install:
    pip install anthropic
    # or
    pip install "llm-provider[anthropic]"

Supported features:
    - Chat with system prompts (top-level system parameter)
    - Structured output (JSON extraction)
    - Temperature and top_p sampling
    - Up to 200k token context window (model-dependent)

Notes:
    - Anthropic's messages API requires max_tokens. Set via extra_settings
      {"max_tokens": N} or it defaults to 8192.
    - Only "user" and "assistant" roles are valid in the messages list;
      system prompts are passed as a separate top-level parameter.
    - The SDK has built-in retry logic (2 retries by default). This provider
      layer adds its own retry on top via BaseProvider for consistency.
"""

from __future__ import annotations

from typing import Any, TypeVar

from ..errors import (
    ConnectionFailedError,
    ErrorType,
    JSONParseFailedError,
    LLMError,
    ModelNotAvailableError,
    RateLimitExceededError,
    TimeoutError,
)
from ..json_extractor import parse_structured_output
from ..models import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    ImageUrl,
    Model,
    ProviderConfig,
    ProviderFeatures,
    TextPart,
)
from ..provider import Provider
from .base_provider import BaseProvider

T = TypeVar("T")

_DEFAULT_MAX_TOKENS = 8192

# Anthropic has no native JSON-schema response_format, so schema-driven
# structured output is enforced by forcing a single synthetic tool whose
# input_schema is the requested schema and reading its arguments back.
_STRUCTURED_OUTPUT_TOOL = "emit_structured_output"


class AnthropicProvider(BaseProvider[T]):
    """
    Anthropic provider via the anthropic SDK.

    api_key is required. max_tokens (required by the API) defaults to 8192
    and can be overridden via extra_settings: {"max_tokens": N}.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("AnthropicProvider requires an 'api_key' in ProviderConfig")

        from anthropic import Anthropic

        self._client = Anthropic(
            api_key=config.api_key,
            timeout=config.timeout,
            # Disable the SDK's built-in retries; BaseProvider handles retries.
            max_retries=0,
        )
        self._max_tokens: int = int(config.extra_settings.get("max_tokens", _DEFAULT_MAX_TOKENS))

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    @staticmethod
    def _content_to_blocks(content: list[Any]) -> list[dict[str, Any]]:
        """Translate multimodal content parts into Anthropic content blocks."""
        blocks: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, TextPart):
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": part.data,
                        },
                    }
                )
            elif isinstance(part, ImageUrl):
                blocks.append({"type": "image", "source": {"type": "url", "url": part.url}})
        return blocks

    def _build_kwargs(self, request: ChatRequest[T]) -> dict[str, Any]:
        self._assert_image_support(request)
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "assistant" and msg.tool_calls:
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                messages.append({"role": "assistant", "content": blocks})
            elif msg.role == "tool":
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
                # Anthropic requires tool_result blocks to ride in a user turn;
                # merge consecutive tool results into the same user message.
                if (
                    messages
                    and messages[-1]["role"] == "user"
                    and isinstance(messages[-1]["content"], list)
                ):
                    messages[-1]["content"].append(result_block)
                else:
                    messages.append({"role": "user", "content": [result_block]})
            elif isinstance(msg.content, list):
                messages.append({"role": msg.role, "content": self._content_to_blocks(msg.content)})
            else:
                messages.append({"role": msg.role, "content": msg.content})

        kwargs: dict[str, Any] = {
            "model": request.model or self._config.default_model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt.content
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.response_schema is not None:
            # Force a single synthetic tool to enforce the schema.
            kwargs["tools"] = [
                {
                    "name": _STRUCTURED_OUTPUT_TOOL,
                    "description": "Return the result as a structured object matching the schema.",
                    "input_schema": request.response_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": _STRUCTURED_OUTPUT_TOOL}
        elif request.tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]
        if request.tool_choice and request.response_schema is None:
            if request.tool_choice in ("auto", "any"):
                kwargs["tool_choice"] = {"type": request.tool_choice}
            else:
                kwargs["tool_choice"] = {"type": "tool", "name": request.tool_choice}

        if request.extra_body:
            kwargs.update(request.extra_body.get("anthropic", {}))

        return kwargs

    def _parse_response(self, response: Any, request: ChatRequest[T]) -> ChatResponse[T]:
        text_blocks = [
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ]
        message_content = "".join(text_blocks)

        from ..models import ToolCall

        tool_calls = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        structured_data: T | None = None
        if request.response_schema is not None:
            # The forced synthetic tool carries the structured result in its
            # arguments. Surface it as structured_data and hide the tool call.
            emitted = next(
                (tc for tc in tool_calls if tc.name == _STRUCTURED_OUTPUT_TOOL),
                None,
            )
            if emitted is None:
                raise JSONParseFailedError(
                    "Anthropic did not emit the structured-output tool call",
                    operation="chat",
                )
            structured_data = emitted.arguments
            tool_calls = []
        elif request.structured_output_type and not tool_calls:
            try:
                structured_data = parse_structured_output(
                    message_content, request.structured_output_type
                )
            except Exception:
                pass

        return ChatResponse(
            message=message_content,
            structured_data=structured_data,
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=getattr(response, "stop_reason", None),
        )

    def chat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        request, _flat_map = self._maybe_flatten_tools(request)

        def _chat() -> ChatResponse[T]:
            kwargs = self._build_kwargs(request)
            try:
                response = self._client.messages.create(**kwargs)
                return self._parse_response(response, request)
            except LLMError:
                raise
            except Exception as e:
                self._classify_anthropic_error(e)

        return self._maybe_renest_tool_calls(self._execute_with_retry(_chat, "chat"), _flat_map)

    async def achat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        if not hasattr(self, "_async_client"):
            from anthropic import AsyncAnthropic

            self._async_client = AsyncAnthropic(
                api_key=self._config.api_key,
                timeout=self._config.timeout,
                max_retries=0,
            )

        request, _flat_map = self._maybe_flatten_tools(request)

        async def _achat() -> ChatResponse[T]:
            kwargs = self._build_kwargs(request)
            try:
                response = await self._async_client.messages.create(**kwargs)
                return self._parse_response(response, request)
            except LLMError:
                raise
            except Exception as e:
                self._classify_anthropic_error(e)

        return self._maybe_renest_tool_calls(
            await self._arun_with_limit(_achat, "achat"), _flat_map
        )

    def list_models(self) -> list[Model]:
        def _list_models() -> list[Model]:
            try:
                return [Model(name=m.id) for m in self._client.models.list()]
            except Exception as e:
                self._classify_anthropic_error(e)

        return self._execute_with_retry(_list_models, "list_models")

    def name(self) -> str:
        return "anthropic"

    def is_available(self) -> bool:
        try:
            next(iter(self._client.models.list()))
            return True
        except Exception:
            return False

    def supported_features(self) -> ProviderFeatures:
        return ProviderFeatures(
            structured_output=True,
            streaming=True,
            vision=True,
            context_window=200_000,
            supported_roles=["user", "assistant"],
            function_calling=True,
            temperature=True,
            top_p=True,
            async_supported=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_anthropic_error(self, e: Exception) -> None:
        """Map anthropic SDK exceptions to LLMError subclasses."""
        try:
            from anthropic import (
                APIConnectionError,
                APITimeoutError,
                AuthenticationError,
                NotFoundError,
                RateLimitError,
            )
        except ImportError:
            self._classify_and_raise_error(e, "anthropic")
            return

        if isinstance(e, RateLimitError):
            retry_after: float | None = None
            if hasattr(e, "response") and e.response is not None:
                raw = e.response.headers.get("retry-after")
                if raw:
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        pass
            raise RateLimitExceededError(
                message=f"Anthropic rate limit exceeded: {e}",
                operation="anthropic",
                cause=e,
                retry_after=retry_after,
            ) from e

        if isinstance(e, APITimeoutError):
            raise TimeoutError(
                f"Anthropic request timed out: {e}", operation="anthropic", cause=e
            ) from e

        if isinstance(e, APIConnectionError):
            raise ConnectionFailedError(
                f"Failed to connect to Anthropic: {e}", operation="anthropic", cause=e
            ) from e

        if isinstance(e, NotFoundError):
            raise ModelNotAvailableError(
                f"Anthropic model not found: {e}", operation="anthropic", cause=e
            ) from e

        if isinstance(e, AuthenticationError):
            raise LLMError(
                error_type=ErrorType.OPERATION_FAILED,
                message=f"Anthropic authentication failed: {e}",
                retryable=False,
                operation="anthropic",
                cause=e,
            ) from e

        self._classify_and_raise_error(e, "anthropic")


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------


def create_anthropic_provider(config_dict: dict[str, Any]) -> Provider:
    """
    Factory function to create an AnthropicProvider.

    Args:
        config_dict: keys:
            - api_key:        Anthropic API key (required)
            - default_model:  e.g. "claude-sonnet-4-6" (required)
            - timeout:        seconds (default 60.0)
            - retry_attempts: (default 3)
            - extra_settings: optional dict; supports "max_tokens" (default 8192)
    """
    api_key = config_dict.get("api_key")
    default_model = config_dict.get("default_model", "claude-sonnet-4-6")
    timeout = float(config_dict.get("timeout", 60.0))
    retry_attempts = int(config_dict.get("retry_attempts", 3))
    rate_limit = config_dict.get("rate_limit")
    max_concurrent = config_dict.get("max_concurrent")
    extra_settings = config_dict.get("extra_settings", {})

    if not api_key:
        raise ValueError("Anthropic provider requires 'api_key' in configuration")

    provider_config = ProviderConfig(
        host="https://api.anthropic.com",
        default_model=default_model,
        timeout=timeout,
        retry_attempts=retry_attempts,
        api_key=api_key,
        rate_limit=rate_limit,
        max_concurrent=max_concurrent,
        extra_settings=extra_settings,
    )

    return AnthropicProvider(provider_config)
