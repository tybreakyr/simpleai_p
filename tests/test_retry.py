"""
Unit tests for retry mechanism.
"""

import unittest
import time
from unittest.mock import Mock, patch
from llm_provider.retry import (
    RetryConfig,
    calculate_backoff_delay,
    retry_with_backoff
)
from llm_provider.errors import ConnectionFailedError, ValidationError


class TestRetry(unittest.TestCase):
    """Test cases for retry mechanism."""
    
    def test_retry_config_validation(self):
        """Test retry configuration validation."""
        # Valid config
        config = RetryConfig(max_retries=3, base_delay=2.0, max_delay=30.0, backoff_factor=2.0)
        self.assertEqual(config.max_retries, 3)
        
        # Invalid max_retries
        with self.assertRaises(ValueError):
            RetryConfig(max_retries=-1)
        
        # Invalid base_delay
        with self.assertRaises(ValueError):
            RetryConfig(base_delay=-1)
    
    def test_calculate_backoff_delay(self):
        """Test backoff delay calculation."""
        config = RetryConfig(base_delay=2.0, max_delay=30.0, backoff_factor=2.0)
        
        # First attempt (attempt 0)
        delay = calculate_backoff_delay(0, config)
        self.assertEqual(delay, 2.0)
        
        # Second attempt (attempt 1)
        delay = calculate_backoff_delay(1, config)
        self.assertEqual(delay, 4.0)
        
        # Third attempt (attempt 2)
        delay = calculate_backoff_delay(2, config)
        self.assertEqual(delay, 8.0)
        
        # Should cap at max_delay
        delay = calculate_backoff_delay(10, config)
        self.assertEqual(delay, 30.0)
    
    def test_retry_with_backoff_success(self):
        """Test retry with successful operation."""
        config = RetryConfig(max_retries=3)
        
        func = Mock(return_value="success")
        result = retry_with_backoff(func, config, "test")
        
        self.assertEqual(result, "success")
        func.assert_called_once()
    
    def test_retry_with_backoff_retryable_error(self):
        """Test retry with retryable error that eventually succeeds."""
        config = RetryConfig(max_retries=3, base_delay=0.01)
        
        func = Mock(side_effect=[
            ConnectionFailedError("Connection failed", "test"),
            ConnectionFailedError("Connection failed", "test"),
            "success"
        ])
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = retry_with_backoff(func, config, "test")
        
        self.assertEqual(result, "success")
        self.assertEqual(func.call_count, 3)
    
    def test_retry_with_backoff_non_retryable_error(self):
        """Test retry with non-retryable error."""
        config = RetryConfig(max_retries=3)
        
        from llm_provider.errors import ErrorType
        func = Mock(side_effect=ValidationError(
            error_type=ErrorType.INVALID_CONFIGURATION,
            message="Validation failed"
        ))
        
        with self.assertRaises(ValidationError):
            retry_with_backoff(func, config, "test")
        
        # Should only be called once (no retries for non-retryable)
        func.assert_called_once()
    
    def test_retry_with_backoff_max_retries(self):
        """Test retry exhausts max retries."""
        config = RetryConfig(max_retries=2, base_delay=0.01)
        
        func = Mock(side_effect=ConnectionFailedError("Connection failed", "test"))
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with self.assertRaises(ConnectionFailedError) as context:
                retry_with_backoff(func, config, "test")
        
        # Should have retried max_retries + 1 times (initial + retries)
        self.assertEqual(func.call_count, 3)
        self.assertEqual(context.exception.retry_count, 3)


if __name__ == '__main__':
    unittest.main()

