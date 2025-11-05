"""
Provider interface for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic

from .models import ChatRequest, ChatResponse, Model, ProviderFeatures


T = TypeVar('T')


class Provider(ABC, Generic[T]):
    """
    Abstract base class for LLM providers.
    
    All provider implementations must implement this interface.
    """
    
    @abstractmethod
    def chat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        """
        Send a chat request to the provider.
        
        Args:
            request: Chat request containing messages and optional system prompt
            
        Returns:
            Chat response with message and optional structured data
            
        Raises:
            LLMError: If the request fails
        """
        pass
    
    @abstractmethod
    def list_models(self) -> List[Model]:
        """
        List available models for this provider.
        
        Returns:
            List of available models
            
        Raises:
            LLMError: If the operation fails
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """
        Get the provider's identifier name.
        
        Returns:
            Provider name string
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is currently reachable and operational.
        
        Returns:
            True if provider is available, False otherwise
        """
        pass
    
    @abstractmethod
    def supported_features(self) -> ProviderFeatures:
        """
        Get the provider's supported features.
        
        Returns:
            ProviderFeatures describing capabilities
        """
        pass

