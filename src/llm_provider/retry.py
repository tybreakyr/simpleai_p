"""
Retry mechanism with exponential backoff for LLM provider operations.
"""

import time
from dataclasses import dataclass
from typing import Callable, TypeVar, Optional
from functools import wraps

from .errors import LLMError, is_retryable, OperationFailedError


T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0

    def __post_init__(self):
        """Validate retry configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")
        if self.backoff_factor <= 0:
            raise ValueError("backoff_factor must be positive")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")


def calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate exponential backoff delay for a given attempt.
    
    Args:
        attempt: The attempt number (0-indexed)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    delay = config.base_delay * (config.backoff_factor ** attempt)
    return min(delay, config.max_delay)


def retry_with_backoff(
    func: Callable[[], T],
    config: RetryConfig,
    operation_name: str = ""
) -> T:
    """
    Execute a function with retry logic and exponential backoff.
    
    Args:
        func: Function to execute (no arguments)
        config: Retry configuration
        operation_name: Name of the operation for error reporting
        
    Returns:
        Result of the function
        
    Raises:
        LLMError: If all retries fail or a non-retryable error occurs
    """
    last_error: Optional[Exception] = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_error = e
            
            # Check if error is retryable
            if not is_retryable(e):
                # Wrap non-LLM errors
                if not isinstance(e, LLMError):
                    raise OperationFailedError(
                        message=f"Operation failed: {str(e)}",
                        operation=operation_name,
                        retryable=False,
                        cause=e
                    ) from e
                # Re-raise LLM errors that are not retryable
                raise
            
            # If this was the last attempt, raise the error
            if attempt >= config.max_retries:
                # Update error with retry count
                if isinstance(e, LLMError):
                    e.retry_count = attempt + 1
                    e.operation = operation_name
                    raise
                else:
                    raise OperationFailedError(
                        message=f"Operation failed after {attempt + 1} attempts: {str(e)}",
                        operation=operation_name,
                        retryable=True,
                        retry_count=attempt + 1,
                        cause=e
                    ) from e
            
            # Calculate delay and wait before retrying
            delay = calculate_backoff_delay(attempt, config)
            time.sleep(delay)
    
    # Should never reach here, but just in case
    if last_error:
        raise OperationFailedError(
            message=f"Operation failed: {str(last_error)}",
            operation=operation_name,
            retryable=True,
            retry_count=config.max_retries,
            cause=last_error
        ) from last_error
    
    raise OperationFailedError(
        message="Operation failed with unknown error",
        operation=operation_name,
        retryable=True
    )


def retryable(operation_name: str = "", retry_config: Optional[RetryConfig] = None):
    """
    Decorator for automatically retrying functions with exponential backoff.
    
    Args:
        operation_name: Name of the operation for error reporting
        retry_config: Retry configuration (uses default if not provided)
    
    Usage:
        @retryable(operation_name="chat")
        def chat(self, request):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            config = retry_config or RetryConfig()
            op_name = operation_name or func.__name__
            
            def _call():
                return func(*args, **kwargs)
            
            return retry_with_backoff(_call, config, op_name)
        
        return wrapper
    return decorator

