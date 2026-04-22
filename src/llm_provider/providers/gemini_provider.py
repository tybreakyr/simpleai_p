"""
Gemini provider implementation via the google-genai SDK.

Uses the current (non-deprecated) ``google.genai`` package (not the legacy
``google-generativeai``). The client is initialised with an API key from
Google AI Studio (https://aistudio.google.com/).

Install:
    pip install google-genai>=1.0.0
    # or
    pip install "llm-provider[gemini]"

Supported features:
    - Chat with system prompts (via system_instruction)
    - Structured output (JSON extraction)
    - Vision / multimodal (model-dependent)
    - Temperature and top_p sampling
    - Up to ~1M token context window (model-dependent)
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, TypeVar

from ..models import (
    ChatRequest, ChatResponse, Model, ProviderConfig,
    ProviderFeatures,
)
from ..provider import Provider
from ..errors import (
    LLMError, ConnectionFailedError, TimeoutError,
    InvalidResponseError,
)
from ..json_extractor import parse_structured_output
from .base_provider import BaseProvider


T = TypeVar('T')

_GENERATE_CONTENT_METHOD = "generateContent"


class GeminiProvider(BaseProvider[T]):
    """
    Google Gemini provider via google-genai SDK.

    api_key is required. host is stored in ProviderConfig for interface
    consistency but the SDK uses the API key directly.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("GeminiProvider requires an 'api_key' in ProviderConfig")

        # Import here so the module can be imported without the package installed
        # (tests can mock it)
        from google import genai
        self._client = genai.Client(api_key=config.api_key)

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def chat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        """Send a chat request to Gemini."""

        def _chat() -> ChatResponse[T]:
            from google.genai import types

            model_name = request.model or self._config.default_model

            messages = list(request.messages)
            if not messages:
                raise InvalidResponseError(
                    "ChatRequest must contain at least one message", operation="chat"
                )

            # Build contents list from messages.
            # Gemini SDK uses "user" / "model" roles (not "assistant").
            contents: list[types.Content] = []

            # Prepend system prompt as a user turn if provided
            # (google-genai supports system_instruction in GenerateContentConfig)
            system_instruction: Optional[str] = (
                request.system_prompt.content if request.system_prompt else None
            )

            for msg in messages:
                gemini_role = "model" if msg.role == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part(text=msg.content)],
                    )
                )

            # Build generation config
            gen_cfg_kwargs: Dict[str, Any] = {}
            if request.temperature is not None:
                gen_cfg_kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                gen_cfg_kwargs["top_p"] = request.top_p
            if system_instruction:
                gen_cfg_kwargs["system_instruction"] = system_instruction

            gen_config = types.GenerateContentConfig(**gen_cfg_kwargs) if gen_cfg_kwargs else None

            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
                )

                message_content: str = response.text or ""

                structured_data: Optional[T] = None
                if request.structured_output_type:
                    try:
                        structured_data = parse_structured_output(
                            message_content, request.structured_output_type
                        )
                    except Exception:
                        pass

                return ChatResponse(
                    message=message_content,
                    structured_data=structured_data,
                )

            except Exception as e:
                self._classify_and_raise_error(e, "chat")

        return self._execute_with_retry(_chat, "chat")

    def list_models(self) -> List[Model]:
        """List Gemini models that support content generation."""

        def _list_models() -> List[Model]:
            try:
                return [
                    Model(name=m.name)
                    for m in self._client.models.list()
                    if m.name and "gemini" in m.name.lower()
                ]
            except Exception as e:
                self._classify_and_raise_error(e, "list_models")

        return self._execute_with_retry(_list_models, "list_models")

    def name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        try:
            next(iter(self._client.models.list()))
            return True
        except Exception:
            return False

    def supported_features(self) -> ProviderFeatures:
        return ProviderFeatures(
            structured_output=True,
            streaming=True,
            vision=True,
            context_window=1_000_000,   # Gemini 1.5 Pro
            supported_roles=["user", "model"],
            function_calling=True,
            temperature=True,
            top_p=True,
        )


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------

def create_gemini_provider(config_dict: Dict[str, Any]) -> Provider:
    """
    Factory function to create a GeminiProvider.

    Args:
        config_dict: keys:
            - api_key:        Google Generative AI API key (required)
            - default_model:  e.g. "gemini-1.5-pro" or "gemini-2.0-flash" (required)
            - timeout:        seconds (default 60.0)
            - retry_attempts: (default 3)
            - rate_limit:     optional requests-per-minute cap
            - extra_settings: optional extra dict
    """
    api_key = config_dict.get("api_key")
    default_model = config_dict.get("default_model", "gemini-1.5-flash")
    timeout = float(config_dict.get("timeout", 60.0))
    retry_attempts = int(config_dict.get("retry_attempts", 3))
    rate_limit = config_dict.get("rate_limit")
    extra_settings = config_dict.get("extra_settings", {})

    if not api_key:
        raise ValueError("Gemini provider requires 'api_key' in configuration")
    if not default_model:
        raise ValueError("Gemini provider requires 'default_model' in configuration")

    provider_config = ProviderConfig(
        host="https://generativelanguage.googleapis.com",
        default_model=default_model,
        timeout=timeout,
        retry_attempts=retry_attempts,
        api_key=api_key,
        rate_limit=rate_limit,
        extra_settings=extra_settings,
    )

    return GeminiProvider(provider_config)
