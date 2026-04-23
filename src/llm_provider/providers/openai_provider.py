"""
OpenAI provider implementation via the openai SDK (v1.x+).

Uses the current openai>=1.0.0 client interface. The legacy module-level
functions (openai.ChatCompletion.create, etc.) are not used.

Install:
    pip install openai>=1.0.0
    # or
    pip install "llm-provider[openai]"

Supported features:
    - Chat with system prompts
    - Structured output (JSON extraction)
    - Vision / multimodal (model-dependent, e.g. gpt-4o)
    - Temperature and top_p sampling
    - Function calling
    - Up to 128k token context window (model-dependent)
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
    InvalidResponseError, RateLimitExceededError,
    ModelNotAvailableError, LLMError, ErrorType,
)
from ..json_extractor import parse_structured_output
from .base_provider import BaseProvider


T = TypeVar('T')


class OpenAIProvider(BaseProvider[T]):
    """
    OpenAI provider via openai SDK v1.x.

    api_key is required. Optionally set base_url via extra_settings to point
    at a compatible endpoint (e.g. Azure OpenAI, a local proxy).
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("OpenAIProvider requires an 'api_key' in ProviderConfig")

        from openai import OpenAI

        kwargs: Dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout,
        }
        base_url = config.extra_settings.get("base_url")
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def chat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        def _chat() -> ChatResponse[T]:
            messages: List[Dict[str, str]] = []

            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt.content})

            for msg in request.messages:
                messages.append({"role": msg.role, "content": msg.content})

            kwargs: Dict[str, Any] = {
                "model": request.model or self._config.default_model,
                "messages": messages,
            }
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p

            try:
                response = self._client.chat.completions.create(**kwargs)

                choice = response.choices[0] if response.choices else None
                if not choice or choice.message.content is None:
                    raise InvalidResponseError(
                        "OpenAI response contained no content", operation="chat"
                    )

                message_content: str = choice.message.content

                structured_data: Optional[T] = None
                if request.structured_output_type:
                    try:
                        structured_data = parse_structured_output(
                            message_content, request.structured_output_type
                        )
                    except Exception:
                        pass

                return ChatResponse(
                    message=message_content,
                    structured_data=structured_data,
                )

            except LLMError:
                raise
            except Exception as e:
                self._classify_openai_error(e)

        return self._execute_with_retry(_chat, "chat")

    def list_models(self) -> List[Model]:
        def _list_models() -> List[Model]:
            try:
                return [Model(name=m.id) for m in self._client.models.list()]
            except Exception as e:
                self._classify_openai_error(e)

        return self._execute_with_retry(_list_models, "list_models")

    def name(self) -> str:
        return "openai"

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
            context_window=128_000,
            supported_roles=["system", "user", "assistant"],
            function_calling=True,
            temperature=True,
            top_p=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_openai_error(self, e: Exception) -> None:
        """Map openai SDK exceptions to LLMError subclasses."""
        try:
            from openai import (
                RateLimitError, APITimeoutError, APIConnectionError,
                NotFoundError, AuthenticationError, APIStatusError,
            )
        except ImportError:
            self._classify_and_raise_error(e, "openai")
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
                message=f"OpenAI rate limit exceeded: {e}",
                operation="openai",
                cause=e,
                retry_after=retry_after,
            ) from e

        if isinstance(e, APITimeoutError):
            raise TimeoutError(
                f"OpenAI request timed out: {e}", operation="openai", cause=e
            ) from e

        if isinstance(e, APIConnectionError):
            raise ConnectionFailedError(
                f"Failed to connect to OpenAI: {e}", operation="openai", cause=e
            ) from e

        if isinstance(e, NotFoundError):
            raise ModelNotAvailableError(
                f"OpenAI model not found: {e}", operation="openai", cause=e
            ) from e

        if isinstance(e, AuthenticationError):
            raise LLMError(
                error_type=ErrorType.OPERATION_FAILED,
                message=f"OpenAI authentication failed: {e}",
                retryable=False,
                operation="openai",
                cause=e,
            ) from e

        self._classify_and_raise_error(e, "openai")


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------

def create_openai_provider(config_dict: Dict[str, Any]) -> Provider:
    """
    Factory function to create an OpenAIProvider.

    Args:
        config_dict: keys:
            - api_key:        OpenAI API key (required)
            - default_model:  e.g. "gpt-4o" or "gpt-4o-mini" (required)
            - timeout:        seconds (default 60.0)
            - retry_attempts: (default 3)
            - rate_limit:     optional requests-per-minute cap (informational only)
            - extra_settings: optional dict; supports "base_url" for custom endpoints
    """
    api_key = config_dict.get("api_key")
    default_model = config_dict.get("default_model", "gpt-4o")
    timeout = float(config_dict.get("timeout", 60.0))
    retry_attempts = int(config_dict.get("retry_attempts", 3))
    rate_limit = config_dict.get("rate_limit")
    extra_settings = config_dict.get("extra_settings", {})

    if not api_key:
        raise ValueError("OpenAI provider requires 'api_key' in configuration")

    provider_config = ProviderConfig(
        host="https://api.openai.com",
        default_model=default_model,
        timeout=timeout,
        retry_attempts=retry_attempts,
        api_key=api_key,
        rate_limit=rate_limit,
        extra_settings=extra_settings,
    )

    return OpenAIProvider(provider_config)
