"""
Unit tests for error handling system.
"""

import unittest
from llm_provider.errors import (
    LLMError,
    ValidationError,
    ConnectionFailedError,
    TimeoutError,
    is_retryable,
    classify_error,
    ErrorType
)


class TestErrors(unittest.TestCase):
    """Test cases for error handling."""
    
    def test_llm_error_basic(self):
        """Test basic LLMError creation."""
        error = LLMError(
            error_type=ErrorType.OPERATION_FAILED,
            message="Test error",
            retryable=True
        )
        self.assertEqual(error.error_type, ErrorType.OPERATION_FAILED)
        self.assertTrue(error.retryable)
        self.assertIn("Test error", str(error))
    
    def test_validation_error(self):
        """Test ValidationError with field errors."""
        error = ValidationError(
            error_type=ErrorType.INVALID_CONFIGURATION,
            message="Validation failed",
            field_errors={"field1": "Error 1", "field2": "Error 2"}
        )
        self.assertEqual(error.error_type, ErrorType.INVALID_CONFIGURATION)
        self.assertFalse(error.retryable)
        self.assertEqual(len(error.field_errors), 2)
        self.assertIn("field1", str(error))
    
    def test_connection_failed_error(self):
        """Test ConnectionFailedError."""
        error = ConnectionFailedError("Connection failed", "test_operation")
        self.assertEqual(error.error_type, ErrorType.CONNECTION_FAILED)
        self.assertTrue(error.retryable)
        self.assertEqual(error.operation, "test_operation")
    
    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError("Request timed out", "test_operation")
        self.assertEqual(error.error_type, ErrorType.TIMEOUT)
        self.assertTrue(error.retryable)
    
    def test_is_retryable(self):
        """Test is_retryable helper function."""
        retryable_error = ConnectionFailedError("Connection failed")
        non_retryable_error = ValidationError(
            error_type=ErrorType.INVALID_CONFIGURATION,
            message="Validation failed"
        )
        
        self.assertTrue(is_retryable(retryable_error))
        self.assertFalse(is_retryable(non_retryable_error))
        
        # Non-LLM errors should return False
        regular_error = ValueError("Regular error")
        self.assertFalse(is_retryable(regular_error))
    
    def test_classify_error_connection(self):
        """Test error classification for connection errors."""
        error_type, retryable = classify_error("Connection refused", None)
        self.assertEqual(error_type, ErrorType.CONNECTION_FAILED)
        self.assertTrue(retryable)
    
    def test_classify_error_timeout(self):
        """Test error classification for timeout errors."""
        error_type, retryable = classify_error("Request timed out", None)
        self.assertEqual(error_type, ErrorType.TIMEOUT)
        self.assertTrue(retryable)
    
    def test_classify_error_rate_limit(self):
        """Test error classification for rate limit errors."""
        error_type, retryable = classify_error("Rate limit exceeded", 429)
        self.assertEqual(error_type, ErrorType.RATE_LIMIT_EXCEEDED)
        self.assertTrue(retryable)
    
    def test_classify_error_auth(self):
        """Test error classification for auth errors."""
        error_type, retryable = classify_error("Unauthorized", 401)
        self.assertEqual(error_type, ErrorType.OPERATION_FAILED)
        self.assertFalse(retryable)
    
    def test_classify_error_not_found(self):
        """Test error classification for not found errors."""
        error_type, retryable = classify_error("Model not found", 404)
        self.assertEqual(error_type, ErrorType.MODEL_NOT_AVAILABLE)
        self.assertFalse(retryable)
    
    def test_classify_error_server_error(self):
        """Test error classification for server errors."""
        error_type, retryable = classify_error("Server error", 500)
        self.assertEqual(error_type, ErrorType.OPERATION_FAILED)
        self.assertTrue(retryable)


if __name__ == '__main__':
    unittest.main()

