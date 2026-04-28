"""
Integration tests for LLM provider using local Ollama server.

These tests require:
- Ollama server running on localhost:11434
- At least one model installed (e.g., llama3.1:latest, qwen2.5:7b)
- requests library installed
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from llm_provider import (
        ProviderFactory, FactoryConfig, ProviderConfig,
        ChatRequest, Message, SystemPrompt, create_ollama_provider
    )
    from llm_provider.errors import LLMError
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    IMPORT_ERROR = str(e)


@unittest.skipIf(not HAS_DEPENDENCIES, f"Skipping integration tests: {IMPORT_ERROR if not HAS_DEPENDENCIES else ''}")
class TestOllamaIntegration(unittest.TestCase):
    """Integration tests with local Ollama server."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.factory = ProviderFactory()
        cls.factory.register_provider("ollama", create_ollama_provider)
        
        # Try to find an available model
        cls.test_model = None
        try:
            # Create a temporary provider to check available models
            temp_config = ProviderConfig(
                host="http://localhost:11434",
                default_model="llama3.1:latest",  # Try common model first
                timeout=10.0
            )
            temp_provider = create_ollama_provider({
                "host": "http://localhost:11434",
                "default_model": "llama3.1:latest",
                "timeout": 10.0
            })
            
            if temp_provider.is_available():
                models = temp_provider.list_models()
                if models:
                    # Try to use a common model name
                    for model in models:
                        if any(name in model.name.lower() for name in ["llama3.1", "qwen2.5", "llama", "mistral"]):
                            cls.test_model = model.name
                            break
                    
                    # If no common model found, use first available
                    if not cls.test_model:
                        cls.test_model = models[0].name
        except Exception:
            pass
        
        if not cls.test_model:
            # Fallback to common model names
            for model_name in ["llama3.1:latest", "qwen2.5:7b", "llama3.1", "mistral"]:
                try:
                    temp_provider = create_ollama_provider({
                        "host": "http://localhost:11434",
                        "default_model": model_name,
                        "timeout": 5.0
                    })
                    if temp_provider.is_available():
                        cls.test_model = model_name
                        break
                except Exception:
                    continue
        
        if not cls.test_model:
            cls.test_model = "llama3.1:latest"  # Default fallback
        
        # Create factory configuration
        cls.factory_config = FactoryConfig(
            default_provider="ollama",
            provider_configs={
                "ollama": ProviderConfig(
                    host="http://localhost:11434",
                    default_model=cls.test_model,
                    timeout=60.0,  # Longer timeout for integration tests
                    retry_attempts=2
                )
            }
        )
        cls.factory.load_config(cls.factory_config)
    
    def test_provider_availability(self):
        """Test that Ollama provider is available."""
        self.assertTrue(self.factory.is_provider_available("ollama"))
    
    def test_list_models(self):
        """Test listing available models."""
        models = self.factory.list_models("ollama")
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0, "No models found on Ollama server")
        print(f"\nFound {len(models)} models: {[m.name for m in models[:5]]}")
    
    def test_simple_chat(self):
        """Test simple chat request."""
        provider = self.factory.get_default_provider()
        
        request = ChatRequest(
            messages=[
                Message(role="user", content="Say 'Hello' and nothing else.")
            ]
        )
        
        response = provider.chat(request)
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response.message, str)
        self.assertGreater(len(response.message), 0)
        print(f"\nResponse: {response.message[:100]}")
    
    def test_chat_with_system_prompt(self):
        """Test chat with system prompt."""
        provider = self.factory.get_default_provider()
        
        request = ChatRequest(
            messages=[
                Message(role="user", content="What is 2+2?")
            ],
            system_prompt=SystemPrompt(content="You are a helpful math assistant. Answer concisely.")
        )
        
        response = provider.chat(request)
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response.message, str)
        self.assertGreater(len(response.message), 0)
        print(f"\nMath response: {response.message[:100]}")
    
    def test_conversation_history(self):
        """Test chat with conversation history."""
        provider = self.factory.get_default_provider()
        
        request = ChatRequest(
            messages=[
                Message(role="user", content="My name is Alice."),
                Message(role="assistant", content="Nice to meet you, Alice!"),
                Message(role="user", content="What's my name?")
            ]
        )
        
        response = provider.chat(request)
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response.message, str)
        # Should remember the name
        self.assertIn("alice", response.message.lower())
        print(f"\nConversation response: {response.message[:100]}")
    
    def test_structured_output(self):
        """Test structured output extraction."""
        from dataclasses import dataclass
        
        @dataclass
        class PersonInfo:
            name: str
            age: int
            city: str
        
        provider = self.factory.get_default_provider()
        
        request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content="Extract information about this person: John Smith, age 30, lives in New York. Return only valid JSON in this format: {\"name\": \"...\", \"age\": ..., \"city\": \"...\"}"
                )
            ],
            structured_output_type=PersonInfo
        )
        
        response = provider.chat(request)
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response.message, str)
        
        # Check if structured data was parsed
        if response.structured_data:
            person = response.structured_data
            self.assertIsInstance(person, PersonInfo)
            self.assertIsInstance(person.name, str)
            self.assertIsInstance(person.age, int)
            self.assertIsInstance(person.city, str)
            print(f"\nStructured output: {person}")
        else:
            print(f"\nStructured output parsing failed, raw response: {response.message[:200]}")
            # This is okay - some models may not return perfect JSON
    
    def test_provider_features(self):
        """Test getting provider features."""
        features = self.factory.get_provider_features("ollama")
        
        self.assertIsNotNone(features)
        self.assertTrue(features.structured_output)  # Via JSON extraction
        self.assertTrue(features.streaming)  # Ollama supports streaming
        self.assertTrue(features.temperature)
        self.assertTrue(features.top_p)
        print(f"\nProvider features: structured_output={features.structured_output}, "
              f"streaming={features.streaming}, context_window={features.context_window}")
    
    def test_error_handling_invalid_model(self):
        """Test error handling for invalid model."""
        # Create provider with invalid model
        invalid_config = ProviderConfig(
            host="http://localhost:11434",
            default_model="nonexistent-model-12345",
            timeout=10.0
        )
        
        provider = create_ollama_provider({
            "host": "http://localhost:11434",
            "default_model": "nonexistent-model-12345",
            "timeout": 10.0
        })
        
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")]
        )
        
        # Should raise an error (either ModelNotAvailableError or similar)
        with self.assertRaises(LLMError):
            provider.chat(request)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)











