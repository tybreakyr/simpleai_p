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

from typing import Dict, Any, List, Optional, TypeVar

from ..models import (
    ChatRequest, ChatResponse, Model, ProviderConfig,
    ProviderFeatures,
)
from ..provider import Provider
from ..errors import (
    ConnectionFailedError, TimeoutError,
    RateLimitExceededError, ModelNotAvailableError,
    LLMError, ErrorType,
)
from ..json_extractor import parse_structured_output
from .base_provider import BaseProvider


T = TypeVar('T')

_DEFAULT_MAX_TOKENS = 8192


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
        self._max_tokens: int = int(
            config.extra_settings.get("max_tokens", _DEFAULT_MAX_TOKENS)
        )

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def _build_kwargs(self, request: ChatRequest[T]) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        kwargs: Dict[str, Any] = {
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

        if request.tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                }
                for tool in request.tools
            ]
        if request.tool_choice:
            if request.tool_choice in ("auto", "any"):
                kwargs["tool_choice"] = {"type": request.tool_choice}
            else:
                kwargs["tool_choice"] = {"type": "tool", "name": request.tool_choice}
                
        return kwargs

    def _parse_response(self, response: Any, request: ChatRequest[T]) -> ChatResponse[T]:
        text_blocks = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        message_content = "".join(text_blocks)

        from ..models import ToolCall
        tool_calls = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        structured_data: Optional[T] = None
        if request.structured_output_type and not tool_calls:
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
            stop_reason=getattr(response, "stop_reason", None)
        )

    def chat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        def _chat() -> ChatResponse[T]:
            kwargs = self._build_kwargs(request)
            try:
                response = self._client.messages.create(**kwargs)
                return self._parse_response(response, request)
            except LLMError:
                raise
            except Exception as e:
                self._classify_anthropic_error(e)

        return self._execute_with_retry(_chat, "chat")

    async def achat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        from ..retry import _async_retry_with_backoff
        if not hasattr(self, "_async_client"):
            from anthropic import AsyncAnthropic
            self._async_client = AsyncAnthropic(
                api_key=self._config.api_key,
                timeout=self._config.timeout,
                max_retries=0,
            )
            
        async def _achat() -> ChatResponse[T]:
            kwargs = self._build_kwargs(request)
            try:
                response = await self._async_client.messages.create(**kwargs)
                return self._parse_response(response, request)
            except LLMError:
                raise
            except Exception as e:
                self._classify_anthropic_error(e)

        return await _async_retry_with_backoff(_achat, self._retry_config, "achat")

    def list_models(self) -> List[Model]:
        def _list_models() -> List[Model]:
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
                RateLimitError, APITimeoutError, APIConnectionError,
                NotFoundError, AuthenticationError,
            )
        except ImportError:
            self._classify_and_raise_error(e, "anthropic")
            return

        if isinstance(e, RateLimitError):
            retry_after: Optional[float] = None
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

def create_anthropic_provider(config_dict: Dict[str, Any]) -> Provider:
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
        extra_settings=extra_settings,
    )

    return AnthropicProvider(provider_config)
