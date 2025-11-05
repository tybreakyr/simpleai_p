"""
Unit tests for data models.
"""

import unittest
from llm_provider.models import (
    Message,
    SystemPrompt,
    ChatRequest,
    ChatResponse,
    Model,
    ProviderFeatures,
    ProviderConfig,
    FactoryConfig
)


class TestModels(unittest.TestCase):
    """Test cases for data models."""
    
    def test_message(self):
        """Test Message model."""
        msg = Message(role="user", content="Hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")
        
        # Empty role should raise error
        with self.assertRaises(ValueError):
            Message(role="", content="Hello")
    
    def test_system_prompt(self):
        """Test SystemPrompt model."""
        prompt = SystemPrompt(content="You are a helpful assistant")
        self.assertEqual(prompt.content, "You are a helpful assistant")
    
    def test_chat_request(self):
        """Test ChatRequest model."""
        messages = [Message(role="user", content="Hello")]
        request = ChatRequest(messages=messages)
        self.assertEqual(len(request.messages), 1)
        
        # Empty messages should raise error
        with self.assertRaises(ValueError):
            ChatRequest(messages=[])
    
    def test_chat_response(self):
        """Test ChatResponse model."""
        response = ChatResponse(message="Hello, how can I help?")
        self.assertEqual(response.message, "Hello, how can I help?")
        self.assertIsNone(response.structured_data)
    
    def test_model(self):
        """Test Model model."""
        model = Model(name="gpt-4")
        self.assertEqual(model.name, "gpt-4")
        
        # Empty name should raise error
        with self.assertRaises(ValueError):
            Model(name="")
    
    def test_provider_features(self):
        """Test ProviderFeatures model."""
        features = ProviderFeatures(
            structured_output=True,
            streaming=True,
            context_window=8192
        )
        self.assertTrue(features.structured_output)
        self.assertTrue(features.streaming)
        self.assertEqual(features.context_window, 8192)
        
        # Negative context window should raise error
        with self.assertRaises(ValueError):
            ProviderFeatures(context_window=-1)
    
    def test_provider_config(self):
        """Test ProviderConfig model."""
        config = ProviderConfig(
            host="http://localhost:8000",
            default_model="gpt-4",
            timeout=30.0,
            retry_attempts=3
        )
        self.assertEqual(config.host, "http://localhost:8000")
        self.assertEqual(config.default_model, "gpt-4")
        self.assertEqual(config.timeout, 30.0)
        
        # Invalid timeout should raise error
        with self.assertRaises(ValueError):
            ProviderConfig(host="http://localhost", default_model="model", timeout=-1)
    
    def test_factory_config(self):
        """Test FactoryConfig model."""
        provider_config = ProviderConfig(
            host="http://localhost",
            default_model="model"
        )
        factory_config = FactoryConfig(
            default_provider="provider1",
            provider_configs={"provider1": provider_config}
        )
        self.assertEqual(factory_config.default_provider, "provider1")
        self.assertEqual(len(factory_config.provider_configs), 1)
        
        # Default provider must exist in configs
        with self.assertRaises(ValueError):
            FactoryConfig(
                default_provider="nonexistent",
                provider_configs={"provider1": provider_config}
            )


if __name__ == '__main__':
    unittest.main()

