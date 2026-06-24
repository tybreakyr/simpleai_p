"""
Base provider helper class for common provider implementation patterns.
"""

import asyncio
import dataclasses
import json
from abc import ABC
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from ..errors import (
    ConnectionFailedError,
    JSONParseFailedError,
    LLMError,
    TimeoutError,
    classify_error,
)
from ..json_extractor import extract_json, parse_structured_output
from ..model_capabilities import supports_nested_tool_params
from ..models import ChatRequest, ChatResponse, ProviderConfig, ToolCall, ToolSchema
from ..provider import Provider
from ..retry import RetryConfig, retry_with_backoff
from ..schema_transform import FlattenMapping, flatten_tool_schema, renest_arguments

T = TypeVar("T")


class BaseProvider(Provider[T], ABC, Generic[T]):
    """
    Base class for provider implementations with common functionality.

    Provides:
    - Retry logic integration
    - Error classification helpers
    - Structured output parsing integration
    - Configuration management
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize base provider.

        Args:
            config: Provider configuration
        """
        self._config = config
        self._retry_config = RetryConfig(
            max_retries=config.retry_attempts, base_delay=2.0, max_delay=30.0, backoff_factor=2.0
        )
        mc = getattr(config, "max_concurrent", None)
        # asyncio.Semaphore (3.10+) binds to the running loop lazily, so it is
        # safe to construct here outside an event loop.
        self._semaphore: asyncio.Semaphore | None = asyncio.Semaphore(mc) if mc and mc > 0 else None

    @property
    def config(self) -> ProviderConfig:
        """Get provider configuration."""
        return self._config

    @property
    def retry_config(self) -> RetryConfig:
        """Get retry configuration."""
        return self._retry_config

    def _execute_with_retry(self, operation: Callable[[], Any], operation_name: str = "") -> Any:
        """
        Execute an operation with retry logic.

        Args:
            operation: Function to execute (no arguments)
            operation_name: Name of the operation for error reporting

        Returns:
            Result of the operation
        """
        return retry_with_backoff(operation, self._retry_config, operation_name)

    async def _arun_with_limit(self, operation: Callable[[], Any], operation_name: str = "") -> Any:
        """Run an async operation under retry + the optional concurrency cap.

        Wraps ``_async_retry_with_backoff`` and, when ``max_concurrent`` is set,
        holds the provider's semaphore so no more than N requests are in flight
        — matching single-model local servers that serialise requests.
        """
        # Imported lazily so tests that patch ``llm_provider.retry`` are honoured.
        from ..retry import _async_retry_with_backoff

        semaphore = getattr(self, "_semaphore", None)
        if semaphore is None:
            return await _async_retry_with_backoff(operation, self._retry_config, operation_name)
        async with semaphore:
            return await _async_retry_with_backoff(operation, self._retry_config, operation_name)

    def _maybe_no_think(self, system_content: str | None) -> str | None:
        """Prepend ``/no_think`` for Qwen3 models when disable_thinking is set.

        ``/no_think`` is a Qwen3-specific soft switch that suppresses the
        reasoning phase. The model-name guard makes this a no-op for every other
        model, so it is safe to call unconditionally during message building.
        Controlled by ``extra_settings["disable_thinking"]``.
        """
        if not self._config.extra_settings.get("disable_thinking"):
            return system_content
        if "qwen3" not in (self._config.default_model or "").lower():
            return system_content
        return "/no_think\n" + (system_content or "")

    # ------------------------------------------------------------------
    # Nested→flat tool-param adaptation (for models that can't handle
    # nested tool-call parameters). Request-local: the mapping is computed
    # and consumed within a single chat/achat call, so concurrent calls
    # never share state. See ``schema_transform`` / ``model_capabilities``.
    # ------------------------------------------------------------------

    def _should_flatten_tool_params(self, model_name: str | None) -> bool:
        """Whether to flatten nested tool params for this model.

        An explicit ``extra_settings["flatten_tool_params"]`` (bool) overrides the
        per-model registry in either direction; otherwise fall back to the model's
        registered capability.
        """
        override = self._config.extra_settings.get("flatten_tool_params")
        if override is not None:
            return bool(override)
        return not supports_nested_tool_params(model_name)

    def _maybe_flatten_tools(
        self, request: ChatRequest[T]
    ) -> tuple[ChatRequest[T], dict[str, FlattenMapping]]:
        """Flatten nested tool schemas when the target model needs it.

        Returns a (possibly rewritten) request plus ``{tool_name: mapping}`` for
        re-nesting the response. Tool names and ``tool_choice`` are never altered,
        so forced-tool behavior is preserved. A no-op (returns the request as-is
        with an empty mapping) when there are no tools, the model is capable, or no
        tool actually has flattenable nesting.
        """
        if not request.tools:
            return request, {}
        model = request.model or self._config.default_model
        if not self._should_flatten_tool_params(model):
            return request, {}

        new_tools: list[ToolSchema] = []
        mappings: dict[str, FlattenMapping] = {}
        changed = False
        for tool in request.tools:
            flat_schema, mapping = flatten_tool_schema(tool.input_schema)
            if mapping:
                changed = True
                mappings[tool.name] = mapping
                new_tools.append(
                    ToolSchema(
                        name=tool.name,
                        description=tool.description,
                        input_schema=flat_schema,
                    )
                )
            else:
                new_tools.append(tool)

        if not changed:
            return request, {}
        return dataclasses.replace(request, tools=new_tools), mappings

    def _maybe_renest_tool_calls(
        self, response: ChatResponse[T], mappings: dict[str, FlattenMapping]
    ) -> ChatResponse[T]:
        """Re-nest flattened tool-call arguments using the per-tool mappings."""
        if not mappings or not response.tool_calls:
            return response
        restored = [
            ToolCall(
                id=tc.id,
                name=tc.name,
                arguments=renest_arguments(tc.arguments, mappings[tc.name]),
                thought_signature=tc.thought_signature,
            )
            if tc.name in mappings
            else tc
            for tc in response.tool_calls
        ]
        return dataclasses.replace(response, tool_calls=restored)

    def _decode_structured_dict(self, text: str) -> dict[str, Any]:
        """Decode a JSON object from response text for ``response_schema`` calls.

        Uses the robust ``extract_json`` heuristics, then ``json.loads``. On
        failure raises a retryable ``JSONParseFailedError`` so the retry layer
        re-rolls the call (mirrors the old collective "retry on invalid JSON").
        """
        try:
            return json.loads(extract_json(text))
        except Exception as e:
            raise JSONParseFailedError(
                f"Failed to decode JSON for response_schema: {e}", cause=e
            ) from e

    def _handle_structured_output(self, text: str, output_type: type[T] | None) -> T | None:
        """
        Handle structured output parsing.

        Args:
            text: Response text
            output_type: Optional type to parse into

        Returns:
            Parsed structured data or None
        """
        if output_type is None:
            return None

        try:
            return parse_structured_output(text, output_type)
        except Exception:
            # If parsing fails, return None (raw message will still be available)
            return None

    def _classify_and_raise_error(
        self, error: Exception, operation_name: str, status_code: int | None = None
    ) -> None:
        """
        Classify an error and raise appropriate LLMError.

        Args:
            error: Original exception
            operation_name: Name of the operation that failed
            status_code: Optional HTTP status code
        """
        error_type, retryable = classify_error(str(error), status_code, error)

        # Create appropriate error based on type
        error_message = str(error)

        if error_type.value == "connection_failed":
            raise ConnectionFailedError(error_message, operation_name, error) from error
        elif error_type.value == "timeout":
            raise TimeoutError(error_message, operation_name, error) from error
        else:
            # Use generic LLMError
            raise LLMError(
                error_type=error_type,
                message=error_message,
                retryable=retryable,
                operation=operation_name,
                cause=error,
            ) from error

    def _get_timeout(self) -> float:
        """Get configured timeout."""
        return self._config.timeout
