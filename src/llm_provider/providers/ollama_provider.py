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

from typing import Any, TypeVar
from urllib.parse import urljoin

import requests

from ..errors import (
    ConnectionFailedError,
    InvalidResponseError,
    LLMError,
    ModelNotAvailableError,
    TimeoutError,
    classify_error,
)
from ..json_extractor import parse_structured_output
from ..models import (
    ChatRequest,
    ChatResponse,
    Model,
    ProviderConfig,
    ProviderFeatures,
    SystemPrompt,
)
from ..provider import Provider
from .base_provider import BaseProvider

T = TypeVar("T")

# extra_settings keys this library interprets itself rather than forwarding as
# top-level Ollama payload fields (which would be ignored, or worse, rejected).
_CONTROL_EXTRA_SETTINGS = frozenset({"disable_thinking", "structured_output_format"})


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
        self._base_url = config.host.rstrip("/")
        if not self._base_url.endswith("/api"):
            self._base_url = urljoin(self._base_url, "/api")

        self._session = requests.Session()
        if config.api_key:
            # Ollama doesn't typically use API keys, but we can add custom headers
            self._session.headers.update({"Authorization": f"Bearer {config.api_key}"})

    def _build_payload(self, request: ChatRequest[T]) -> dict[str, Any]:
        messages = []
        system_content = self._maybe_no_think(
            request.system_prompt.content if request.system_prompt else None
        )
        if system_content:
            messages.append({"role": "system", "content": system_content})
        for msg in request.messages:
            if msg.role == "assistant" and msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
            elif msg.role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            else:
                messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": request.model or self._config.default_model,
            "messages": messages,
            "stream": False,
        }

        if request.temperature is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["temperature"] = request.temperature

        if request.top_p is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["top_p"] = request.top_p

        if self._config.extra_settings:
            for key, value in self._config.extra_settings.items():
                # Control keys consumed by this library (not Ollama wire fields).
                if key in _CONTROL_EXTRA_SETTINGS:
                    continue
                payload[key] = value

        if request.response_schema is not None:
            # Ollama enforces structured output natively via the `format` field,
            # which accepts a full JSON Schema.
            payload["format"] = request.response_schema

        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice

        if request.extra_body:
            payload.update(request.extra_body.get("ollama", {}))

        return payload

    def _parse_response(self, data: dict[str, Any], request: ChatRequest[T]) -> ChatResponse[T]:
        if "message" not in data:
            raise InvalidResponseError("Ollama response missing 'message' field", operation="chat")

        message_data = data["message"]
        message_content = message_data.get("content", "")

        from ..models import ToolCall

        tool_calls = []
        if "tool_calls" in message_data:
            for tc in message_data["tool_calls"]:
                func = tc.get("function", {})
                tool_calls.append(
                    ToolCall(
                        id=ToolCall.make_id(),
                        name=func.get("name", ""),
                        arguments=func.get("arguments", {}),
                    )
                )

        structured_data: T | None = None
        if request.response_schema is not None and not tool_calls:
            structured_data = self._decode_structured_dict(message_content)
        elif request.structured_output_type and not tool_calls:
            try:
                structured_data = parse_structured_output(
                    message_content, request.structured_output_type
                )
            except Exception:
                pass

        return ChatResponse(
            message=message_content,
            structured_data=structured_data,
            tool_calls=tool_calls if tool_calls else None,
        )

    _REQUIRED_NUDGE = (
        "You MUST call exactly one of the provided tools. "
        "Do not respond with prose; emit a tool call."
    )

    def _needs_required_emulation(self, request: ChatRequest[T], response: ChatResponse[T]) -> bool:
        """Ollama has no native 'required' tool mode; emulate it via a retry nudge."""
        return request.tool_choice == "any" and bool(request.tools) and not response.tool_calls

    def _with_required_nudge(self, request: ChatRequest[T]) -> ChatRequest[T]:
        """Build a copy of ``request`` with a stricter system prompt forcing a tool call."""
        if request.system_prompt:
            new_sp = SystemPrompt(
                content=request.system_prompt.content + "\n\n" + self._REQUIRED_NUDGE
            )
        else:
            new_sp = SystemPrompt(content=self._REQUIRED_NUDGE)
        return ChatRequest(
            messages=list(request.messages),
            system_prompt=new_sp,
            structured_output_type=request.structured_output_type,
            response_schema=request.response_schema,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            tools=request.tools,
            tool_choice=request.tool_choice,
            extra_body=request.extra_body,
        )

    def chat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        """Send a chat request to Ollama."""
        request, _flat_map = self._maybe_flatten_tools(request)

        def _chat_for(req: ChatRequest[T]) -> ChatResponse[T]:
            payload = self._build_payload(req)
            url = f"{self._base_url}/chat"

            try:
                response = self._session.post(url, json=payload, timeout=self._config.timeout)
                response.raise_for_status()
                return self._parse_response(response.json(), req)

            except requests.exceptions.Timeout as e:
                raise TimeoutError(
                    f"Request to Ollama timed out after {self._config.timeout}s",
                    operation="chat",
                    cause=e,
                ) from e

            except requests.exceptions.ConnectionError as e:
                raise ConnectionFailedError(
                    f"Failed to connect to Ollama at {self._base_url}", operation="chat", cause=e
                ) from e

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else None

                if status_code == 404:
                    raise ModelNotAvailableError(
                        f"Model '{payload['model']}' not found in Ollama", operation="chat", cause=e
                    ) from e

                error_type, retryable = classify_error(str(e), status_code, e)
                raise LLMError(
                    error_type=error_type,
                    message=f"Ollama API error: {str(e)}",
                    retryable=retryable,
                    operation="chat",
                    cause=e,
                ) from e

            except Exception as e:
                self._classify_and_raise_error(e, "chat")

        result = self._execute_with_retry(lambda: _chat_for(request), "chat")
        if self._needs_required_emulation(request, result):
            nudged = self._with_required_nudge(request)
            result = self._execute_with_retry(lambda: _chat_for(nudged), "chat")
            if not result.tool_calls:
                raise InvalidResponseError(
                    "Ollama did not return a tool call after required-mode retry",
                    operation="chat",
                )
        return self._maybe_renest_tool_calls(result, _flat_map)

    async def achat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        import httpx

        if not hasattr(self, "_async_client"):
            headers = (
                {"Authorization": f"Bearer {self._config.api_key}"} if self._config.api_key else {}
            )
            self._async_client = httpx.AsyncClient(
                headers=headers,
                timeout=self._config.timeout,
            )

        request, _flat_map = self._maybe_flatten_tools(request)

        async def _achat_for(req: ChatRequest[T]) -> ChatResponse[T]:
            payload = self._build_payload(req)
            url = f"{self._base_url}/chat"

            try:
                response = await self._async_client.post(url, json=payload)
                response.raise_for_status()
                return self._parse_response(response.json(), req)

            except httpx.TimeoutException as e:
                raise TimeoutError(
                    f"Request to Ollama timed out after {self._config.timeout}s",
                    operation="achat",
                    cause=e,
                ) from e

            except httpx.RequestError as e:
                raise ConnectionFailedError(
                    f"Failed to connect to Ollama at {self._base_url}", operation="achat", cause=e
                ) from e

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                if status_code == 404:
                    raise ModelNotAvailableError(
                        f"Model '{payload['model']}' not found in Ollama",
                        operation="achat",
                        cause=e,
                    ) from e

                error_type, retryable = classify_error(str(e), status_code, e)
                raise LLMError(
                    error_type=error_type,
                    message=f"Ollama API error: {str(e)}",
                    retryable=retryable,
                    operation="achat",
                    cause=e,
                ) from e

            except Exception as e:
                self._classify_and_raise_error(e, "achat")

        result = await self._arun_with_limit(lambda: _achat_for(request), "achat")
        if self._needs_required_emulation(request, result):
            nudged = self._with_required_nudge(request)
            result = await self._arun_with_limit(lambda: _achat_for(nudged), "achat")
            if not result.tool_calls:
                raise InvalidResponseError(
                    "Ollama did not return a tool call after required-mode retry",
                    operation="achat",
                )
        return self._maybe_renest_tool_calls(result, _flat_map)

    def list_models(self) -> list[Model]:
        """List available models from Ollama."""

        def _list_models():
            url = f"{self._base_url}/tags"

            try:
                response = self._session.get(
                    url,
                    timeout=min(self._config.timeout, 10.0),  # Use shorter timeout for list
                )
                response.raise_for_status()

                data = response.json()

                if "models" not in data:
                    raise InvalidResponseError(
                        "Ollama response missing 'models' field", operation="list_models"
                    )

                models = []
                for model_data in data["models"]:
                    model_name = model_data.get("name", "")
                    if model_name:
                        models.append(Model(name=model_name))

                return models

            except requests.exceptions.Timeout as e:
                raise TimeoutError(
                    "Request to Ollama timed out", operation="list_models", cause=e
                ) from e

            except requests.exceptions.ConnectionError as e:
                raise ConnectionFailedError(
                    f"Failed to connect to Ollama at {self._base_url}",
                    operation="list_models",
                    cause=e,
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
            supported_roles=["system", "user", "assistant", "tool"],
            function_calling=True,
            temperature=True,
            top_p=True,
            async_supported=True,
        )


def create_ollama_provider(config_dict: dict[str, Any]) -> Provider:
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
    max_concurrent = config_dict.get("max_concurrent")
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
        max_concurrent=max_concurrent,
        extra_settings=extra_settings,
    )

    return OllamaProvider(provider_config)
