"""
Unit tests for provider implementations.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from llm_provider.providers.ollama_provider import OllamaProvider, create_ollama_provider
from llm_provider.models import (
    ProviderConfig,
    ChatRequest,
    Message,
    SystemPrompt
)


class TestOllamaProvider(unittest.TestCase):
    """Test cases for Ollama provider."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            host="http://localhost:11434",
            default_model="llama2",
            timeout=30.0,
            retry_attempts=3
        )
    
    @patch('llm_provider.providers.ollama_provider.requests')
    def test_chat_success(self, mock_requests):
        """Test successful chat request."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": "Hello, how can I help?"
            }
        }
        mock_response.raise_for_status = Mock()
        mock_requests.Session.return_value.post.return_value = mock_response
        
        provider = OllamaProvider(self.config)
        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        
        response = provider.chat(request)
        
        self.assertEqual(response.message, "Hello, how can I help?")
    
    @patch('llm_provider.providers.ollama_provider.requests')
    def test_chat_with_system_prompt(self, mock_requests):
        """Test chat with system prompt."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": "Response"
            }
        }
        mock_response.raise_for_status = Mock()
        mock_requests.Session.return_value.post.return_value = mock_response
        
        provider = OllamaProvider(self.config)
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            system_prompt=SystemPrompt(content="You are helpful")
        )
        
        response = provider.chat(request)
        self.assertEqual(response.message, "Response")
    
    @patch('llm_provider.providers.ollama_provider.requests')
    def test_list_models(self, mock_requests):
        """Test listing models."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "mistral"}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_requests.Session.return_value.get.return_value = mock_response
        
        provider = OllamaProvider(self.config)
        models = provider.list_models()
        
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, "llama2")
        self.assertEqual(models[1].name, "mistral")
    
    def test_name(self):
        """Test provider name."""
        provider = OllamaProvider(self.config)
        self.assertEqual(provider.name(), "ollama")
    
    @patch('llm_provider.providers.ollama_provider.requests')
    def test_is_available(self, mock_requests):
        """Test availability check."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_requests.Session.return_value.get.return_value = mock_response
        
        provider = OllamaProvider(self.config)
        self.assertTrue(provider.is_available())
    
    def test_supported_features(self):
        """Test supported features."""
        provider = OllamaProvider(self.config)
        features = provider.supported_features()
        
        self.assertTrue(features.structured_output)
        self.assertTrue(features.streaming)
        self.assertTrue(features.temperature)
        self.assertTrue(features.top_p)
    
    def test_create_ollama_provider(self):
        """Test factory function."""
        config_dict = {
            "host": "http://localhost:11434",
            "default_model": "llama2",
            "timeout": 30.0
        }
        provider = create_ollama_provider(config_dict)
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.name(), "ollama")
    
    def test_create_ollama_provider_missing_model(self):
        """Test factory function with missing model."""
        config_dict = {
            "host": "http://localhost:11434"
        }
        with self.assertRaises(ValueError):
            create_ollama_provider(config_dict)


if __name__ == '__main__':
    unittest.main()

