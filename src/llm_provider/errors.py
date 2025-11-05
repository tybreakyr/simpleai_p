"""
Error handling system for the LLM provider abstraction library.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ErrorType(str, Enum):
    """Types of errors that can occur."""
    CONNECTION_FAILED = "connection_failed"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    JSON_PARSE_FAILED = "json_parse_failed"
    MODEL_NOT_AVAILABLE = "model_not_available"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_CONFIGURATION = "invalid_configuration"
    OPERATION_FAILED = "operation_failed"


@dataclass
class LLMError(Exception):
    """Base exception for all LLM provider errors."""
    error_type: ErrorType
    message: str
    retryable: bool = False
    operation: str = ""
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    cause: Optional[Exception] = None

    def __str__(self) -> str:
        """Return string representation of the error."""
        parts = [f"[{self.error_type.value}] {self.message}"]
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        if self.retry_count > 0:
            parts.append(f"Retry count: {self.retry_count}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " | ".join(parts)


@dataclass
class ValidationError(LLMError):
    """Error for configuration/input validation issues."""
    field_errors: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize validation error."""
        self.error_type = ErrorType.INVALID_CONFIGURATION
        self.retryable = False
        if not self.message and self.field_errors:
            self.message = f"Validation failed for fields: {', '.join(self.field_errors.keys())}"

    def __str__(self) -> str:
        """Return string representation with field errors."""
        base = super().__str__()
        if self.field_errors:
            errors = ", ".join(f"{k}: {v}" for k, v in self.field_errors.items())
            return f"{base} | Field errors: {errors}"
        return base


class ConnectionFailedError(LLMError):
    """Error when connection to provider fails."""

    def __init__(self, message: str, operation: str = "", cause: Optional[Exception] = None):
        super().__init__(
            error_type=ErrorType.CONNECTION_FAILED,
            message=message,
            retryable=True,
            operation=operation,
            cause=cause
        )


class TimeoutError(LLMError):
    """Error when request exceeds time limit."""

    def __init__(self, message: str, operation: str = "", cause: Optional[Exception] = None):
        super().__init__(
            error_type=ErrorType.TIMEOUT,
            message=message,
            retryable=True,
            operation=operation,
            cause=cause
        )


class InvalidResponseError(LLMError):
    """Error when API response is malformed."""

    def __init__(self, message: str, operation: str = "", cause: Optional[Exception] = None):
        super().__init__(
            error_type=ErrorType.INVALID_RESPONSE,
            message=message,
            retryable=True,
            operation=operation,
            cause=cause
        )


class JSONParseFailedError(LLMError):
    """Error when structured output parsing fails."""

    def __init__(self, message: str, operation: str = "", cause: Optional[Exception] = None):
        super().__init__(
            error_type=ErrorType.JSON_PARSE_FAILED,
            message=message,
            retryable=True,
            operation=operation,
            cause=cause
        )


class ModelNotAvailableError(LLMError):
    """Error when requested model doesn't exist."""

    def __init__(self, message: str, operation: str = "", cause: Optional[Exception] = None):
        super().__init__(
            error_type=ErrorType.MODEL_NOT_AVAILABLE,
            message=message,
            retryable=False,
            operation=operation,
            cause=cause
        )


class RateLimitExceededError(LLMError):
    """Error when API rate limit is exceeded."""

    def __init__(self, message: str, operation: str = "", cause: Optional[Exception] = None):
        super().__init__(
            error_type=ErrorType.RATE_LIMIT_EXCEEDED,
            message=message,
            retryable=True,
            operation=operation,
            cause=cause
        )


class InvalidConfigurationError(LLMError):
    """Error when configuration is invalid."""

    def __init__(self, message: str, operation: str = "", cause: Optional[Exception] = None):
        super().__init__(
            error_type=ErrorType.INVALID_CONFIGURATION,
            message=message,
            retryable=False,
            operation=operation,
            cause=cause
        )


class OperationFailedError(LLMError):
    """General operation failure error."""

    def __init__(self, message: str, operation: str = "", retryable: bool = False, cause: Optional[Exception] = None):
        super().__init__(
            error_type=ErrorType.OPERATION_FAILED,
            message=message,
            retryable=retryable,
            operation=operation,
            cause=cause
        )


def is_retryable(error: Exception) -> bool:
    """Check if an error is retryable."""
    if isinstance(error, LLMError):
        return error.retryable
    return False


def classify_error(error_message: str, status_code: Optional[int] = None, 
                   exception: Optional[Exception] = None) -> tuple[ErrorType, bool]:
    """
    Classify an error based on message and status code.
    
    Returns:
        Tuple of (error_type, retryable)
    """
    error_message_lower = error_message.lower()
    
    # Network/connection errors - retryable
    if any(term in error_message_lower for term in ["connection", "network", "unreachable", "refused"]):
        return (ErrorType.CONNECTION_FAILED, True)
    
    # Timeout errors - retryable
    if any(term in error_message_lower for term in ["timeout", "timed out", "time out"]):
        return (ErrorType.TIMEOUT, True)
    
    # Rate limit errors - retryable
    if status_code == 429 or any(term in error_message_lower for term in ["rate limit", "too many requests"]):
        return (ErrorType.RATE_LIMIT_EXCEEDED, True)
    
    # Authentication/authorization errors - not retryable
    if status_code in (401, 403) or any(term in error_message_lower for term in ["unauthorized", "forbidden", "auth", "api key"]):
        return (ErrorType.OPERATION_FAILED, False)
    
    # Not found errors - not retryable
    if status_code == 404 or any(term in error_message_lower for term in ["not found", "does not exist"]):
        return (ErrorType.MODEL_NOT_AVAILABLE, False)
    
    # Bad request errors - generally not retryable
    if status_code == 400:
        return (ErrorType.INVALID_RESPONSE, False)
    
    # Server errors - retryable
    if status_code and 500 <= status_code < 600:
        return (ErrorType.OPERATION_FAILED, True)
    
    # Default to operation failed, retryable
    return (ErrorType.OPERATION_FAILED, True)

