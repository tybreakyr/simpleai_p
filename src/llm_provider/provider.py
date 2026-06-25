"""
Provider interface for LLM providers.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .models import (
    ChatRequest,
    ChatResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    Model,
    ProviderFeatures,
)

T = TypeVar("T")


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
    async def achat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        """
        Send a chat request to the provider asynchronously.

        Args:
            request: Chat request containing messages and optional system prompt

        Returns:
            Chat response with message and optional structured data

        Raises:
            LLMError: If the request fails
        """
        pass

    @abstractmethod
    def list_models(self) -> list[Model]:
        """
        List available models for this provider.

        Returns:
            List of available models

        Raises:
            LLMError: If the operation fails
        """
        pass

    async def alist_models(self) -> list[Model]:
        """
        List available models for this provider asynchronously.

        Returns:
            List of available models

        Raises:
            LLMError: If the operation fails
        """
        return await asyncio.to_thread(self.list_models)

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate or edit image(s).

        With only a ``prompt`` this is text→image. Supplying a source ``image`` on
        the request switches to img2img: ``image`` + ``prompt`` routes to the
        provider's edit endpoint (add a ``mask`` for OpenAI inpainting), and
        ``image`` alone (no prompt) routes to OpenAI variations.

        Default implementation raises, so providers without an image API
        (Anthropic, Ollama, mlx-lm) report a clear error. OpenAI and Gemini
        override this. Returns images as base64 ``ImagePart``s.

        Raises:
            ValidationError: If the provider/model does not support the request.
            LLMError: If the request fails.
        """
        from .errors import ValidationError

        raise ValidationError(message=f"{self.name()} does not support image generation")

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Async variant of :meth:`generate_image`.

        Defaults to running the sync method in a thread; providers with a native
        async client override this.
        """
        return await asyncio.to_thread(self.generate_image, request)

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
