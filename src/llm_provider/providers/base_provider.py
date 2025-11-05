"""
Base provider helper class for common provider implementation patterns.
"""

from typing import Dict, Any, Optional, TypeVar, Generic, Callable, Type
from abc import ABC

from ..models import ChatRequest, ChatResponse, ProviderConfig, ProviderFeatures
from ..provider import Provider
from ..retry import RetryConfig, retry_with_backoff
from ..json_extractor import parse_structured_output
from ..errors import LLMError, classify_error, ConnectionFailedError, TimeoutError


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

