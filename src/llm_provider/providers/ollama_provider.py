"""
Ollama provider implementation.

Communicates with an Ollama server via its REST API (/api/chat, /api/tags).
Supports local and remote instances, optional bearer-token auth, and
model-specific options passed through ``extra_settings``.

Extra settings are forwarded as top-level fields in the /api/chat payload,
which allows caller-controlled Ollama options such as:

- ``think: false``   — disable chain-of-thought on thinking models (e.g. Qwen3)
- ``keep_alive: "10m"`` — keep the model loaded between requests

Requires: ``requests``
"""

import requests
from typing import Dict, Any, List, Optional, TypeVar
from urllib.parse import urljoin

from ..models import (
    ChatRequest, ChatResponse, Model, ProviderConfig, 
    ProviderFeatures, Message, SystemPrompt
)
from ..provider import Provider
from ..errors import (
    LLMError, ConnectionFailedError, TimeoutError, 
    InvalidResponseError, ModelNotAvailableError, classify_error
)
from ..retry import RetryConfig
from ..json_extractor import parse_structured_output
from .base_provider import BaseProvider


T = TypeVar('T')


class OllamaProvider(BaseProvider[T]):
    """
    Ollama provider — wraps the Ollama REST API (/api/chat, /api/tags).

    Supports local and remote instances. Extra settings in ``ProviderConfig``
    are forwarded as top-level payload fields on every chat request, enabling
    model-specific options like ``think`` (disable CoT on thinking models) and
    ``keep_alive`` (memory retention between requests).
    """
    
    def __init__(self, config: ProviderConfig):
        """Initialize Ollama provider."""
        super().__init__(config)
        
        # Ensure host ends with /api
        self._base_url = config.host.rstrip('/')
        if not self._base_url.endswith('/api'):
            self._base_url = urljoin(self._base_url, '/api')
        
        self._session = requests.Session()
        if config.api_key:
            # Ollama doesn't typically use API keys, but we can add custom headers
            self._session.headers.update({"Authorization": f"Bearer {config.api_key}"})
    
    def _build_payload(self, request: ChatRequest[T]) -> Dict[str, Any]:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt.content})
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": request.model or self._config.default_model,
            "messages": messages,
            "stream": False
        }

        if request.temperature is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["temperature"] = request.temperature

        if request.top_p is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["top_p"] = request.top_p

        if self._config.extra_settings:
            for key, value in self._config.extra_settings.items():
                payload[key] = value

        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                    }
                }
                for tool in request.tools
            ]
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice

        return payload

    def _parse_response(self, data: Dict[str, Any], request: ChatRequest[T]) -> ChatResponse[T]:
        if "message" not in data:
            raise InvalidResponseError(
                "Ollama response missing 'message' field",
                operation="chat"
            )

        message_data = data["message"]
        message_content = message_data.get("content", "")

        from ..models import ToolCall
        tool_calls = []
        if "tool_calls" in message_data:
            for tc in message_data["tool_calls"]:
                func = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=ToolCall.make_id(),
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {})
                ))

        structured_data: Optional[T] = None
        if request.structured_output_type and not tool_calls:
            try:
                structured_data = parse_structured_output(
                    message_content,
                    request.structured_output_type
                )
            except Exception:
                pass

        return ChatResponse(
            message=message_content,
            structured_data=structured_data,
            tool_calls=tool_calls if tool_calls else None
        )

    def chat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        """Send a chat request to Ollama."""
        def _chat():
            payload = self._build_payload(request)
            url = f"{self._base_url}/chat"
            
            try:
                response = self._session.post(
                    url,
                    json=payload,
                    timeout=self._config.timeout
                )
                response.raise_for_status()
                return self._parse_response(response.json(), request)
            
            except requests.exceptions.Timeout as e:
                raise TimeoutError(
                    f"Request to Ollama timed out after {self._config.timeout}s",
                    operation="chat",
                    cause=e
                ) from e
            
            except requests.exceptions.ConnectionError as e:
                raise ConnectionFailedError(
                    f"Failed to connect to Ollama at {self._base_url}",
                    operation="chat",
                    cause=e
                ) from e
            
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else None
                
                if status_code == 404:
                    raise ModelNotAvailableError(
                        f"Model '{payload['model']}' not found in Ollama",
                        operation="chat",
                        cause=e
                    ) from e
                
                error_type, retryable = classify_error(str(e), status_code, e)
                raise LLMError(
                    error_type=error_type,
                    message=f"Ollama API error: {str(e)}",
                    retryable=retryable,
                    operation="chat",
                    cause=e
                ) from e
            
            except Exception as e:
                self._classify_and_raise_error(e, "chat")
        
        return self._execute_with_retry(_chat, "chat")

    async def achat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        from ..retry import _async_retry_with_backoff
        import httpx
        
        if not hasattr(self, "_async_client"):
            headers = {"Authorization": f"Bearer {self._config.api_key}"} if self._config.api_key else {}
            self._async_client = httpx.AsyncClient(
                headers=headers,
                timeout=self._config.timeout,
            )
            
        async def _achat() -> ChatResponse[T]:
            payload = self._build_payload(request)
            url = f"{self._base_url}/chat"
            
            try:
                response = await self._async_client.post(
                    url,
                    json=payload
                )
                response.raise_for_status()
                return self._parse_response(response.json(), request)
            
            except httpx.TimeoutException as e:
                raise TimeoutError(
                    f"Request to Ollama timed out after {self._config.timeout}s",
                    operation="achat",
                    cause=e
                ) from e
            
            except httpx.RequestError as e:
                raise ConnectionFailedError(
                    f"Failed to connect to Ollama at {self._base_url}",
                    operation="achat",
                    cause=e
                ) from e
                
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                
                if status_code == 404:
                    raise ModelNotAvailableError(
                        f"Model '{payload['model']}' not found in Ollama",
                        operation="achat",
                        cause=e
                    ) from e
                
                error_type, retryable = classify_error(str(e), status_code, e)
                raise LLMError(
                    error_type=error_type,
                    message=f"Ollama API error: {str(e)}",
                    retryable=retryable,
                    operation="achat",
                    cause=e
                ) from e
            
            except Exception as e:
                self._classify_and_raise_error(e, "achat")

        return await _async_retry_with_backoff(_achat, self._retry_config, "achat")
    
    def list_models(self) -> List[Model]:
        """List available models from Ollama."""
        def _list_models():
            url = f"{self._base_url}/tags"
            
            try:
                response = self._session.get(
                    url,
                    timeout=min(self._config.timeout, 10.0)  # Use shorter timeout for list
                )
                response.raise_for_status()
                
                data = response.json()
                
                if "models" not in data:
                    raise InvalidResponseError(
                        "Ollama response missing 'models' field",
                        operation="list_models"
                    )
                
                models = []
                for model_data in data["models"]:
                    model_name = model_data.get("name", "")
                    if model_name:
                        models.append(Model(name=model_name))
                
                return models
            
            except requests.exceptions.Timeout as e:
                raise TimeoutError(
                    f"Request to Ollama timed out",
                    operation="list_models",
                    cause=e
                ) from e
            
            except requests.exceptions.ConnectionError as e:
                raise ConnectionFailedError(
                    f"Failed to connect to Ollama at {self._base_url}",
                    operation="list_models",
                    cause=e
                ) from e
            
            except Exception as e:
                self._classify_and_raise_error(e, "list_models")
        
        return self._execute_with_retry(_list_models, "list_models")
    
    def name(self) -> str:
        """Get provider name."""
        return "ollama"
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            # Try to list models with short timeout
            url = f"{self._base_url}/tags"
            response = self._session.get(url, timeout=5.0)
            response.raise_for_status()
            return True
        except Exception:
            return False
    
    def supported_features(self) -> ProviderFeatures:
        """Get supported features."""
        return ProviderFeatures(
            structured_output=True,  # Via JSON extraction
            streaming=True,  # Ollama supports streaming
            vision=False,  # Ollama doesn't natively support vision in chat API
            context_window=8192,  # Typical for Ollama models
            supported_roles=["system", "user", "assistant"],
            function_calling=True,
            temperature=True,
            top_p=True,
            async_supported=True,
        )


def create_ollama_provider(config_dict: Dict[str, Any]) -> Provider:
    """
    Factory function to create an Ollama provider.
    
    Args:
        config_dict: Configuration dictionary with keys:
            - host: Ollama server URL (default: http://localhost:11434)
            - default_model: Default model name (required)
            - timeout: Request timeout in seconds (default: 30.0)
            - retry_attempts: Number of retry attempts (default: 3)
            - api_key: Optional API key for authentication
            - rate_limit: Optional rate limit (requests per minute)
            - extra_settings: Optional extra settings dict
    
    Returns:
        OllamaProvider instance
    """
    # Extract configuration with defaults
    host = config_dict.get("host", "http://localhost:11434")
    default_model = config_dict.get("default_model", "")
    timeout = config_dict.get("timeout", 30.0)
    retry_attempts = config_dict.get("retry_attempts", 3)
    api_key = config_dict.get("api_key")
    rate_limit = config_dict.get("rate_limit")
    extra_settings = config_dict.get("extra_settings", {})
    
    # Validate required fields
    if not default_model:
        raise ValueError("Ollama provider requires 'default_model' in configuration")
    
    # Create ProviderConfig
    provider_config = ProviderConfig(
        host=host,
        default_model=default_model,
        timeout=timeout,
        retry_attempts=retry_attempts,
        api_key=api_key,
        rate_limit=rate_limit,
        extra_settings=extra_settings
    )
    
    return OllamaProvider(provider_config)

