"""
Unit tests for provider factory.
"""

import unittest
from unittest.mock import Mock
from llm_provider.factory import ProviderFactory
from llm_provider.models import FactoryConfig, ProviderConfig, Model
from llm_provider.provider import Provider
from llm_provider.errors import InvalidConfigurationError


class MockProvider(Provider):
    """Mock provider for testing."""
    
    def __init__(self, name: str = "mock"):
        self._name = name
    
    def chat(self, request):
        from llm_provider.models import ChatResponse
        return ChatResponse(message="Test response")
    
    async def achat(self, request):
        from llm_provider.models import ChatResponse
        return ChatResponse(message="Test response")
    
    def list_models(self):
        return [Model(name="test-model")]
    
    def name(self):
        return self._name
    
    def is_available(self):
        return True
    
    def supported_features(self):
        from llm_provider.models import ProviderFeatures
        return ProviderFeatures()


class TestFactory(unittest.TestCase):
    """Test cases for provider factory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = ProviderFactory()
        
        # Register mock provider
        def create_mock(config_dict):
            return MockProvider(config_dict.get("name", "mock"))
        
        self.factory.register_provider("mock", create_mock)
    
    def test_register_provider(self):
        """Test provider registration."""
        providers = self.factory.list_registered_providers()
        self.assertIn("mock", providers)
    
    def test_create_provider(self):
        """Test creating a provider from config."""
        config = ProviderConfig(
            host="http://localhost",
            default_model="test-model"
        )
        provider = self.factory.create_provider("mock", config)
        self.assertIsInstance(provider, Provider)
        self.assertEqual(provider.name(), "mock")
    
    def test_create_provider_not_registered(self):
        """Test creating provider that's not registered."""
        config = ProviderConfig(
            host="http://localhost",
            default_model="test-model"
        )
        with self.assertRaises(InvalidConfigurationError):
            self.factory.create_provider("nonexistent", config)
    
    def test_load_config(self):
        """Test loading factory configuration."""
        config = FactoryConfig(
            default_provider="mock",
            provider_configs={
                "mock": ProviderConfig(
                    host="http://localhost",
                    default_model="test-model"
                )
            }
        )
        self.factory.load_config(config)
        self.assertEqual(self.factory.get_config(), config)
    
    def test_get_default_provider(self):
        """Test getting default provider."""
        config = FactoryConfig(
            default_provider="mock",
            provider_configs={
                "mock": ProviderConfig(
                    host="http://localhost",
                    default_model="test-model"
                )
            }
        )
        self.factory.load_config(config)
        provider = self.factory.get_default_provider()
        self.assertIsInstance(provider, Provider)
    
    def test_list_models(self):
        """Test listing models for a provider."""
        config = FactoryConfig(
            default_provider="mock",
            provider_configs={
                "mock": ProviderConfig(
                    host="http://localhost",
                    default_model="test-model"
                )
            }
        )
        self.factory.load_config(config)
        models = self.factory.list_models("mock")
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "test-model")
    
    def test_is_provider_available(self):
        """Test checking provider availability."""
        config = FactoryConfig(
            default_provider="mock",
            provider_configs={
                "mock": ProviderConfig(
                    host="http://localhost",
                    default_model="test-model"
                )
            }
        )
        self.factory.load_config(config)
        self.assertTrue(self.factory.is_provider_available("mock"))
    
    def test_clear_cache(self):
        """Test clearing factory cache."""
        config = FactoryConfig(
            default_provider="mock",
            provider_configs={
                "mock": ProviderConfig(
                    host="http://localhost",
                    default_model="test-model"
                )
            }
        )
        self.factory.load_config(config)
        
        # Create provider to populate cache
        provider1 = self.factory.get_default_provider()
        
        # Clear cache
        self.factory.clear_cache()
        
        # Should still work (will recreate)
        provider2 = self.factory.get_default_provider()
        self.assertIsInstance(provider2, Provider)


if __name__ == '__main__':
    unittest.main()

