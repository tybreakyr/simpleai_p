"""
Factory for creating and managing LLM providers.
"""

import threading
from typing import Dict, List, Callable, Optional, Any, TypeVar

from .models import FactoryConfig, ProviderConfig, Model, ProviderFeatures
from .provider import Provider
from .errors import InvalidConfigurationError, ValidationError
from .config import validate_factory_config, validate_provider_config


T = TypeVar('T')


class ProviderFactory:
    """
    Factory for creating and managing LLM provider instances.
    
    Thread-safe implementation with provider and model caching.
    """
    
    def __init__(self, config: Optional[FactoryConfig] = None):
        """
        Initialize the factory.
        
        Args:
            config: Optional factory configuration
        """
        self._lock = threading.RLock()
        self._provider_registry: Dict[str, Callable[[Dict[str, Any]], Provider]] = {}
        self._provider_cache: Dict[str, Provider] = {}
        self._model_cache: Dict[str, List[Model]] = {}
        self._config: Optional[FactoryConfig] = None
        
        if config:
            self.load_config(config)
    
    def register_provider(self, name: str, constructor: Callable[[Dict[str, Any]], Provider]) -> None:
        """
        Register a provider constructor function.
        
        Args:
            name: Provider name identifier
            constructor: Factory function that takes a config dict and returns a Provider
        """
        with self._lock:
            self._provider_registry[name] = constructor
            # Clear cache for this provider
            if name in self._provider_cache:
                del self._provider_cache[name]
            if name in self._model_cache:
                del self._model_cache[name]
    
    def create_provider(self, name: str, config: ProviderConfig) -> Provider:
        """
        Create a provider instance from configuration.
        
        Args:
            name: Provider name (must be registered)
            config: Provider configuration
            
        Returns:
            Provider instance
            
        Raises:
            InvalidConfigurationError: If provider not registered or config invalid
        """
        # Validate configuration
        validate_provider_config(config)
        
        with self._lock:
            # Check if provider is registered
            if name not in self._provider_registry:
                raise InvalidConfigurationError(
                    f"Provider '{name}' is not registered. "
                    f"Available providers: {list(self._provider_registry.keys())}"
                )
            
            # Check cache first
            cache_key = f"{name}:{id(config)}"
            if cache_key in self._provider_cache:
                return self._provider_cache[cache_key]
            
            # Create provider
            constructor = self._provider_registry[name]
            
            # Convert ProviderConfig to dict for constructor
            config_dict = {
                "host": config.host,
                "default_model": config.default_model,
                "timeout": config.timeout,
                "retry_attempts": config.retry_attempts,
                "api_key": config.api_key,
                "rate_limit": config.rate_limit,
                "extra_settings": config.extra_settings
            }
            
            try:
                provider = constructor(config_dict)
                self._provider_cache[cache_key] = provider
                return provider
            except Exception as e:
                raise InvalidConfigurationError(
                    f"Failed to create provider '{name}': {str(e)}",
                    cause=e
                ) from e
    
    def create_provider_by_name(self, name: str) -> Provider:
        """
        Create a provider instance from factory's stored configuration.
        
        Args:
            name: Provider name (must exist in factory config)
            
        Returns:
            Provider instance
            
        Raises:
            InvalidConfigurationError: If factory config not loaded or provider not found
        """
        with self._lock:
            if not self._config:
                raise InvalidConfigurationError(
                    "Factory configuration not loaded. Call load_config() first."
                )
            
            if name not in self._config.provider_configs:
                raise InvalidConfigurationError(
                    f"Provider '{name}' not found in factory configuration. "
                    f"Available providers: {list(self._config.provider_configs.keys())}"
                )
            
            config = self._config.provider_configs[name]
            return self.create_provider(name, config)
    
    def get_default_provider(self) -> Provider:
        """
        Get the default provider instance.
        
        Returns:
            Default provider instance
            
        Raises:
            InvalidConfigurationError: If factory config not loaded
        """
        with self._lock:
            if not self._config:
                raise InvalidConfigurationError(
                    "Factory configuration not loaded. Call load_config() first."
                )
            
            return self.create_provider_by_name(self._config.default_provider)
    
    def list_registered_providers(self) -> List[str]:
        """
        List all registered provider names.
        
        Returns:
            List of registered provider names
        """
        with self._lock:
            return list(self._provider_registry.keys())
    
    def list_models(self, provider_name: str) -> List[Model]:
        """
        List available models for a provider.
        
        Args:
            provider_name: Provider name
            
        Returns:
            List of available models (cached)
            
        Raises:
            InvalidConfigurationError: If provider not found
        """
        with self._lock:
            # Check cache first
            if provider_name in self._model_cache:
                return self._model_cache[provider_name]
            
            # Get provider instance
            provider = self.create_provider_by_name(provider_name)
            
            # List models
            try:
                models = provider.list_models()
                self._model_cache[provider_name] = models
                return models
            except Exception as e:
                raise InvalidConfigurationError(
                    f"Failed to list models for provider '{provider_name}': {str(e)}",
                    cause=e
                ) from e
    
    def is_provider_available(self, name: str) -> bool:
        """
        Check if a provider is available.
        
        Args:
            name: Provider name
            
        Returns:
            True if provider is available, False otherwise
        """
        try:
            provider = self.create_provider_by_name(name)
            return provider.is_available()
        except Exception:
            return False
    
    def get_provider_features(self, name: str) -> ProviderFeatures:
        """
        Get provider features.
        
        Args:
            name: Provider name
            
        Returns:
            ProviderFeatures describing capabilities
            
        Raises:
            InvalidConfigurationError: If provider not found
        """
        provider = self.create_provider_by_name(name)
        return provider.supported_features()
    
    def load_config(self, config: FactoryConfig) -> None:
        """
        Load factory configuration.
        
        Args:
            config: Factory configuration
            
        Raises:
            ValidationError: If configuration is invalid
        """
        validate_factory_config(config)
        
        with self._lock:
            self._config = config
            # Clear caches when config changes
            self._provider_cache.clear()
            self._model_cache.clear()
    
    def get_config(self) -> Optional[FactoryConfig]:
        """
        Get current factory configuration.
        
        Returns:
            Factory configuration or None if not loaded
        """
        with self._lock:
            return self._config
    
    def clear_cache(self) -> None:
        """
        Clear provider and model caches.
        """
        with self._lock:
            self._provider_cache.clear()
            self._model_cache.clear()

