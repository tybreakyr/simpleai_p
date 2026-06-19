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

import json
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

    def _build_kwargs(self, request: ChatRequest[T]) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = []
        system_content = self._maybe_no_think(
            request.system_prompt.content if request.system_prompt else None
        )
        if system_content:
            messages.append({
                "role": "system",
                "content": system_content
            })

        for msg in request.messages:
            if msg.role == "assistant" and msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            elif msg.role == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            else:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        kwargs: Dict[str, Any] = {
            "model": request.model or self._config.default_model,
            "messages": messages,
        }
        
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
            
        if "max_tokens" in self._config.extra_settings:
            kwargs["max_tokens"] = int(self._config.extra_settings["max_tokens"])

        if request.response_schema is not None:
            rf = self._build_response_format(request.response_schema)
            kwargs["response_format"] = rf
            if rf.get("type") == "json_object":
                # json_object only guarantees *valid* JSON, not the schema shape
                # (and some OpenAI-compatible servers like mlx-lm ignore the
                # json_schema mode entirely), so describe the schema in the
                # prompt to steer the model to the right structure.
                instruction = (
                    "\n\nYou MUST respond with a single JSON object that conforms "
                    "to this JSON Schema, with no extra wrapping keys:\n"
                    + json.dumps(request.response_schema)
                )
                messages[-1]["content"] = (messages[-1].get("content") or "") + instruction

        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                    }
                }
                for tool in request.tools
            ]
        if request.tool_choice:
            if request.tool_choice in ("auto", "none"):
                kwargs["tool_choice"] = request.tool_choice
            elif request.tool_choice == "any":
                kwargs["tool_choice"] = "required"
            else:
                kwargs["tool_choice"] = {"type": "function", "function": {"name": request.tool_choice}}

        if request.extra_body:
            kwargs.update(request.extra_body.get("openai", {}))

        return kwargs

    def _build_response_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Map a JSON Schema to an OpenAI ``response_format``.

        Honours ``extra_settings["structured_output_format"]``:
        ``"json_object"`` (used by mlx-lm, which doesn't support json_schema)
        falls back to plain JSON mode; otherwise the schema is enforced via
        ``json_schema``.
        """
        fmt = self._config.extra_settings.get("structured_output_format", "json_schema")
        if fmt == "json_object":
            return {"type": "json_object"}
        name = schema.get("title", "structured_response")
        name = name.replace(" ", "_").lower() if isinstance(name, str) else "structured_response"
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
                "strict": False,  # strict support is model-dependent
            },
        }

    def _parse_response(self, response: Any, request: ChatRequest[T]) -> ChatResponse[T]:
        choice = response.choices[0] if response.choices else None
        if not choice or (choice.message.content is None and not getattr(choice.message, "tool_calls", None)):
            raise InvalidResponseError(
                "OpenAI response contained no content", operation="chat"
            )

        message_content = choice.message.content or ""
        
        from ..models import ToolCall
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = tc.function.arguments if isinstance(tc.function.arguments, dict) else {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        structured_data: Optional[T] = None
        if request.response_schema is not None and not tool_calls:
            structured_data = self._decode_structured_dict(message_content)
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
            stop_reason=choice.finish_reason
        )

    def chat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        request, _flat_map = self._maybe_flatten_tools(request)

        def _chat() -> ChatResponse[T]:
            kwargs = self._build_kwargs(request)
            try:
                response = self._client.chat.completions.create(**kwargs)
                return self._parse_response(response, request)
            except LLMError:
                raise
            except Exception as e:
                self._classify_openai_error(e)

        return self._maybe_renest_tool_calls(
            self._execute_with_retry(_chat, "chat"), _flat_map
        )

    async def achat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        if not hasattr(self, "_async_client"):
            from openai import AsyncOpenAI
            _async_base_url = self._config.extra_settings.get("base_url") or None
            self._async_client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=_async_base_url,
                timeout=self._config.timeout,
                max_retries=0,
            )

        request, _flat_map = self._maybe_flatten_tools(request)

        async def _achat() -> ChatResponse[T]:
            kwargs = self._build_kwargs(request)
            try:
                response = await self._async_client.chat.completions.create(**kwargs)
                return self._parse_response(response, request)
            except LLMError:
                raise
            except Exception as e:
                self._classify_openai_error(e)

        return self._maybe_renest_tool_calls(
            await self._arun_with_limit(_achat, "achat"), _flat_map
        )

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
            supported_roles=["system", "user", "assistant", "tool"],
            function_calling=True,
            temperature=True,
            top_p=True,
            async_supported=True,
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
    max_concurrent = config_dict.get("max_concurrent")
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
        max_concurrent=max_concurrent,
        extra_settings=extra_settings,
    )

    return OpenAIProvider(provider_config)


def create_mlx_provider(config_dict: Dict[str, Any]) -> Provider:
    """
    Factory for an mlx-lm backend, which exposes an OpenAI-compatible API.

    Thin wrapper over :func:`create_openai_provider` with mlx-friendly defaults
    so consumers don't need to know mlx specifics:
      - ``extra_settings["base_url"]`` defaults to ``http://localhost:8000/v1``
      - ``extra_settings["structured_output_format"]`` defaults to
        ``"json_object"`` (mlx-lm doesn't support OpenAI's ``json_schema`` mode)
      - ``api_key`` defaults to ``"mlx-lm"`` (the OpenAI client requires some
        key; mlx-lm ignores it)
    """
    cfg = dict(config_dict)
    extra = dict(cfg.get("extra_settings", {}))
    extra.setdefault("base_url", "http://localhost:8000/v1")
    # mlx-lm doesn't support OpenAI's json_schema response_format, so downgrade
    # to plain json_object. An explicit json_object passes through unchanged.
    if extra.get("structured_output_format") in (None, "json_schema"):
        extra["structured_output_format"] = "json_object"
    cfg["extra_settings"] = extra
    if not cfg.get("api_key"):
        cfg["api_key"] = "mlx-lm"
    return create_openai_provider(cfg)
