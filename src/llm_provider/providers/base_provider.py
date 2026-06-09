"""
Base provider helper class for common provider implementation patterns.
"""

import asyncio
from typing import Dict, Any, Optional, TypeVar, Generic, Callable, Type
from abc import ABC

from ..models import ChatRequest, ChatResponse, ProviderConfig, ProviderFeatures
from ..provider import Provider
from ..retry import RetryConfig, retry_with_backoff
from ..json_extractor import parse_structured_output, extract_json
from ..errors import (
    LLMError, classify_error, ConnectionFailedError, TimeoutError,
    JSONParseFailedError,
)
import json


T = TypeVar('T')


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
            max_retries=config.retry_attempts,
            base_delay=2.0,
            max_delay=30.0,
            backoff_factor=2.0
        )
        mc = getattr(config, "max_concurrent", None)
        # asyncio.Semaphore (3.10+) binds to the running loop lazily, so it is
        # safe to construct here outside an event loop.
        self._semaphore: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(mc) if mc and mc > 0 else None
        )
    
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

    async def _arun_with_limit(
        self, operation: Callable[[], Any], operation_name: str = ""
    ) -> Any:
        """Run an async operation under retry + the optional concurrency cap.

        Wraps ``_async_retry_with_backoff`` and, when ``max_concurrent`` is set,
        holds the provider's semaphore so no more than N requests are in flight
        — matching single-model local servers that serialise requests.
        """
        # Imported lazily so tests that patch ``llm_provider.retry`` are honoured.
        from ..retry import _async_retry_with_backoff

        semaphore = getattr(self, "_semaphore", None)
        if semaphore is None:
            return await _async_retry_with_backoff(
                operation, self._retry_config, operation_name
            )
        async with semaphore:
            return await _async_retry_with_backoff(
                operation, self._retry_config, operation_name
            )

    def _maybe_no_think(self, system_content: Optional[str]) -> Optional[str]:
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

    def _decode_structured_dict(self, text: str) -> Dict[str, Any]:
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
    
    def _handle_structured_output(
        self, 
        text: str, 
        output_type: Optional[Type[T]]
    ) -> Optional[T]:
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
        self,
        error: Exception,
        operation_name: str,
        status_code: Optional[int] = None
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
                cause=error
            ) from error
    
    def _get_timeout(self) -> float:
        """Get configured timeout."""
        return self._config.timeout

