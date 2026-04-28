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
import re
import time
import threading

from ..errors import (
    LLMError, ConnectionFailedError, TimeoutError,
    InvalidResponseError, RateLimitExceededError,
)
from ..json_extractor import parse_structured_output
from .base_provider import BaseProvider


T = TypeVar('T')

_GENERATE_CONTENT_METHOD = "generateContent"


def _parse_gemini_retry_delay(error_str: str) -> Optional[float]:
    """
    Extract the suggested retry delay (seconds) from a Gemini 429 error string.

    Gemini embeds this in two ways:
      - JSON body: 'retryDelay': '9s'
      - Human-readable: "Please retry in 9.3758776s"
    Returns None if no delay is found.
    """
    # JSON-style: 'retryDelay': '9.3s'  or  "retryDelay": "9s"
    m = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+(?:\.\d+)?)s['\"]", error_str)
    if m:
        return float(m.group(1)) + 1.0  # +1s buffer

    # Prose: "retry in 9.3758776s"
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", error_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 1.0

    return None


def _is_daily_quota(error_str: str) -> bool:
    """Return True if the 429 is a daily quota exhaustion (not per-minute)."""
    return "PerDay" in error_str or "per_day" in error_str.lower()


class GeminiProvider(BaseProvider[T]):
    """
    Google Gemini provider via google-genai SDK.

    api_key is required. host is stored in ProviderConfig for interface
    consistency but the SDK uses the API key directly.

    Rate limiting: set rate_limit (requests per minute) in ProviderConfig to
    enforce a minimum interval between requests and avoid quota exhaustion.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("GeminiProvider requires an 'api_key' in ProviderConfig")

        # Import here so the module can be imported without the package installed
        # (tests can mock it)
        from google import genai
        self._client = genai.Client(api_key=config.api_key)

        # Rate limiter state
        self._rate_limit_rpm: Optional[float] = config.rate_limit
        self._last_request_at: float = 0.0
        self._rate_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def _enforce_rate_limit(self) -> None:
        """Sleep if needed to respect the configured requests-per-minute limit."""
        if not self._rate_limit_rpm:
            return
        min_interval = 60.0 / self._rate_limit_rpm
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_request_at = time.monotonic()

    def _build_kwargs(self, request: ChatRequest[T]) -> tuple[str, list, Any]:
        from google.genai import types

        model_name = request.model or self._config.default_model

        messages = list(request.messages)
        if not messages:
            raise InvalidResponseError(
                "ChatRequest must contain at least one message", operation="chat"
            )

        contents: list[types.Content] = []
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

        gen_cfg_kwargs: Dict[str, Any] = {}
        if request.temperature is not None:
            gen_cfg_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            gen_cfg_kwargs["top_p"] = request.top_p
        if system_instruction:
            gen_cfg_kwargs["system_instruction"] = system_instruction
            
        if request.tools:
            gen_cfg_kwargs["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema
                        }
                        for tool in request.tools
                    ]
                }
            ]
            
        if request.tool_choice:
            if request.tool_choice == "auto":
                gen_cfg_kwargs["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}
            elif request.tool_choice == "none":
                gen_cfg_kwargs["tool_config"] = {"function_calling_config": {"mode": "NONE"}}
            elif request.tool_choice == "any":
                gen_cfg_kwargs["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
            else:
                gen_cfg_kwargs["tool_config"] = {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [request.tool_choice]
                    }
                }

        gen_config = types.GenerateContentConfig(**gen_cfg_kwargs) if gen_cfg_kwargs else None
        return model_name, contents, gen_config

    def _parse_response(self, response: Any, request: ChatRequest[T]) -> ChatResponse[T]:
        message_content: str = response.text or ""
        
        from ..models import ToolCall
        tool_calls = []
        if getattr(response, "function_calls", None):
            for fc in response.function_calls:
                call_id = getattr(fc, "id", None) or ToolCall.make_id()
                args = fc.args
                if hasattr(args, "items"):
                    args = dict(args.items())
                elif not isinstance(args, dict):
                    try:
                        import json
                        args = json.loads(args) if isinstance(args, str) else {}
                    except Exception:
                        args = {}
                tool_calls.append(ToolCall(id=call_id, name=fc.name, arguments=args))

        structured_data: Optional[T] = None
        if request.structured_output_type and not tool_calls:
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
            stop_reason=None
        )

    def chat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        """Send a chat request to Gemini."""
        def _chat() -> ChatResponse[T]:
            self._enforce_rate_limit()
            model_name, contents, gen_config = self._build_kwargs(request)

            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
                )
                return self._parse_response(response, request)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    daily = _is_daily_quota(error_str)
                    retry_after = None if daily else _parse_gemini_retry_delay(error_str)
                    raise RateLimitExceededError(
                        message=(
                            "Gemini daily quota exhausted — will not retry until quota resets. "
                            "Consider upgrading to a paid tier or reducing pipeline frequency."
                            if daily else
                            f"Gemini rate limit exceeded. "
                            f"Retry after {retry_after:.0f}s." if retry_after else
                            "Gemini rate limit exceeded."
                        ),
                        operation="chat",
                        cause=e,
                        retry_after=retry_after,
                        retryable=not daily,
                    ) from e
                self._classify_and_raise_error(e, "chat")

        return self._execute_with_retry(_chat, "chat")

    async def achat(self, request: ChatRequest[T]) -> ChatResponse[T]:
        from ..retry import _async_retry_with_backoff
            
        async def _achat() -> ChatResponse[T]:
            self._enforce_rate_limit()
            model_name, contents, gen_config = self._build_kwargs(request)

            try:
                response = await self._client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
                )
                return self._parse_response(response, request)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    daily = _is_daily_quota(error_str)
                    retry_after = None if daily else _parse_gemini_retry_delay(error_str)
                    from ..errors import RateLimitExceededError
                    raise RateLimitExceededError(
                        message=(
                            "Gemini daily quota exhausted — will not retry until quota resets. "
                            if daily else
                            f"Gemini rate limit exceeded. Retry after {retry_after:.0f}s." if retry_after else
                            "Gemini rate limit exceeded."
                        ),
                        operation="achat",
                        cause=e,
                        retry_after=retry_after,
                        retryable=not daily,
                    ) from e
                self._classify_and_raise_error(e, "achat")

        return await _async_retry_with_backoff(_achat, self._retry_config, "achat")

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
            async_supported=True,
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
