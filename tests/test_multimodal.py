"""
Tests for multimodal (image) content support across the data model and providers.
"""

import base64
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

from llm_provider.errors import ValidationError
from llm_provider.models import (
    ChatRequest,
    ImagePart,
    ImageUrl,
    Message,
    ProviderConfig,
    TextPart,
    as_content_parts,
    message_has_images,
)
from llm_provider.retry import RetryConfig

# A tiny 1x1 PNG worth of bytes (content is irrelevant for these unit tests).
RAW_IMAGE = b"\x89PNG\r\n\x1a\n_fake_png_bytes_"
B64_IMAGE = base64.b64encode(RAW_IMAGE).decode("ascii")


# ----------------------------------------------------------------------------
# Data-model tests
# ----------------------------------------------------------------------------


class TestContentParts(unittest.TestCase):
    def test_textpart_requires_string(self):
        with self.assertRaises(ValueError):
            TextPart(123)  # type: ignore[arg-type]

    def test_imagepart_from_bytes_roundtrips(self):
        part = ImagePart.from_bytes(RAW_IMAGE, "image/png")
        self.assertEqual(part.data, B64_IMAGE)
        self.assertEqual(base64.b64decode(part.data), RAW_IMAGE)
        self.assertEqual(part.media_type, "image/png")

    def test_imagepart_rejects_empty_data(self):
        with self.assertRaises(ValueError):
            ImagePart("", "image/png")

    def test_imagepart_rejects_non_image_media_type(self):
        with self.assertRaises(ValueError):
            ImagePart(B64_IMAGE, "application/pdf")

    def test_imageurl_accepts_http_schemes(self):
        self.assertEqual(ImageUrl("https://example.com/a.png").url, "https://example.com/a.png")
        self.assertEqual(ImageUrl("http://example.com/a.png").url, "http://example.com/a.png")

    def test_imageurl_rejects_unsafe_schemes(self):
        for bad in ("file:///etc/passwd", "data:image/png;base64,AAAA", "ftp://x/y.png"):
            with self.assertRaises(ValueError):
                ImageUrl(bad)


class TestMessageContentValidation(unittest.TestCase):
    def test_string_content_still_valid(self):
        msg = Message(role="user", content="hello")
        self.assertEqual(msg.content, "hello")

    def test_list_content_valid(self):
        msg = Message(
            role="user", content=[TextPart("hi"), ImagePart.from_bytes(RAW_IMAGE, "image/png")]
        )
        self.assertEqual(len(msg.content), 2)

    def test_empty_list_rejected(self):
        with self.assertRaises(ValueError):
            Message(role="user", content=[])

    def test_non_part_items_rejected(self):
        with self.assertRaises(ValueError):
            Message(role="user", content=["just a string"])  # type: ignore[list-item]

    def test_list_content_rejected_on_tool_result(self):
        with self.assertRaises(ValueError):
            Message(role="tool", content=[TextPart("x")], tool_call_id="call_1")

    def test_list_content_rejected_with_tool_calls(self):
        from llm_provider.models import ToolCall

        with self.assertRaises(ValueError):
            Message(
                role="assistant",
                content=[TextPart("x")],
                tool_calls=[ToolCall(id="call_1", name="f", arguments={})],
            )


class TestContentHelpers(unittest.TestCase):
    def test_as_content_parts_wraps_string(self):
        parts = as_content_parts("hello")
        self.assertEqual(len(parts), 1)
        self.assertIsInstance(parts[0], TextPart)
        self.assertEqual(parts[0].text, "hello")

    def test_as_content_parts_passes_list_through(self):
        original = [TextPart("a")]
        self.assertIs(as_content_parts(original), original)

    def test_message_has_images(self):
        no_imgs = [
            Message(role="user", content="hi"),
            Message(role="user", content=[TextPart("x")]),
        ]
        self.assertFalse(message_has_images(no_imgs))
        with_part = [Message(role="user", content=[ImagePart.from_bytes(RAW_IMAGE, "image/png")])]
        self.assertTrue(message_has_images(with_part))
        with_url = [Message(role="user", content=[ImageUrl("https://x/y.png")])]
        self.assertTrue(message_has_images(with_url))


# ----------------------------------------------------------------------------
# Provider translation helpers
# ----------------------------------------------------------------------------


def _anthropic_provider():
    from llm_provider.providers.anthropic_provider import AnthropicProvider

    config = ProviderConfig(
        host="https://api.anthropic.com", default_model="claude-sonnet-4-6", api_key="k"
    )
    with patch.dict(sys.modules, {"anthropic": MagicMock()}):
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._config = config
        provider._client = MagicMock()
        provider._max_tokens = 8192
        provider._retry_config = RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.01)
        return provider


def _openai_provider():
    from llm_provider.providers.openai_provider import OpenAIProvider

    config = ProviderConfig(host="https://api.openai.com", default_model="gpt-4o", api_key="k")
    with patch.dict(sys.modules, {"openai": MagicMock()}):
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._config = config
        provider._client = MagicMock()
        provider._retry_config = RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.01)
        return provider


def _ollama_provider(**extra_settings):
    from llm_provider.providers.ollama_provider import OllamaProvider

    config = ProviderConfig(
        host="http://localhost:11434",
        default_model="llava",
        extra_settings=extra_settings or {},
    )
    return OllamaProvider(config)


def _gemini_provider():
    from llm_provider.providers.gemini_provider import GeminiProvider

    config = ProviderConfig(
        host="https://gen.googleapis.com", default_model="gemini-1.5-flash", api_key="k"
    )
    provider = GeminiProvider.__new__(GeminiProvider)
    provider._config = config
    provider._client = MagicMock()
    provider._rate_limit_rpm = None
    provider._last_request_at = 0.0
    provider._rate_lock = threading.Lock()
    provider._retry_config = RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.01)
    return provider


def _gemini_genai_patch():
    """Return (patcher, mock_genai) so the genai SDK is mocked while building."""
    mock_genai = MagicMock()
    patcher = patch.dict(
        sys.modules,
        {
            "google": MagicMock(),
            "google.genai": mock_genai,
            "google.genai.types": mock_genai.types,
        },
    )
    return patcher, mock_genai


class TestAnthropicImageTranslation(unittest.TestCase):
    def test_text_only_stays_bare_string(self):
        provider = _anthropic_provider()
        kwargs = provider._build_kwargs(ChatRequest(messages=[Message(role="user", content="hi")]))
        self.assertEqual(kwargs["messages"][0], {"role": "user", "content": "hi"})

    def test_image_part_block(self):
        provider = _anthropic_provider()
        msg = Message(
            role="user", content=[TextPart("what is this?"), ImagePart(B64_IMAGE, "image/png")]
        )
        kwargs = provider._build_kwargs(ChatRequest(messages=[msg]))
        self.assertEqual(
            kwargs["messages"][0]["content"],
            [
                {"type": "text", "text": "what is this?"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": B64_IMAGE},
                },
            ],
        )

    def test_image_url_block(self):
        provider = _anthropic_provider()
        msg = Message(role="user", content=[ImageUrl("https://x/y.png")])
        kwargs = provider._build_kwargs(ChatRequest(messages=[msg]))
        self.assertEqual(
            kwargs["messages"][0]["content"],
            [{"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}],
        )


class TestOpenAIImageTranslation(unittest.TestCase):
    def test_text_only_stays_bare_string(self):
        provider = _openai_provider()
        kwargs = provider._build_kwargs(ChatRequest(messages=[Message(role="user", content="hi")]))
        self.assertEqual(kwargs["messages"][0], {"role": "user", "content": "hi"})

    def test_image_part_data_uri(self):
        provider = _openai_provider()
        msg = Message(
            role="user", content=[TextPart("describe"), ImagePart(B64_IMAGE, "image/png")]
        )
        kwargs = provider._build_kwargs(ChatRequest(messages=[msg]))
        self.assertEqual(
            kwargs["messages"][0]["content"],
            [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{B64_IMAGE}"}},
            ],
        )

    def test_image_url_forwarded(self):
        provider = _openai_provider()
        msg = Message(role="user", content=[ImageUrl("https://x/y.png")])
        kwargs = provider._build_kwargs(ChatRequest(messages=[msg]))
        self.assertEqual(
            kwargs["messages"][0]["content"],
            [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
        )


class TestGeminiImageTranslation(unittest.TestCase):
    def test_image_part_builds_inline_blob(self):
        provider = _gemini_provider()
        patcher, mock_genai = _gemini_genai_patch()
        msg = Message(role="user", content=[TextPart("hi"), ImagePart(B64_IMAGE, "image/png")])
        with patcher:
            provider._build_kwargs(ChatRequest(messages=[msg]))
        mock_genai.types.Blob.assert_called_once_with(mime_type="image/png", data=RAW_IMAGE)

    def test_image_url_raises(self):
        provider = _gemini_provider()
        patcher, _ = _gemini_genai_patch()
        msg = Message(role="user", content=[ImageUrl("https://x/y.png")])
        with patcher, self.assertRaises(ValidationError):
            provider._build_kwargs(ChatRequest(messages=[msg]))


class TestOllamaImageTranslation(unittest.TestCase):
    def test_image_part_with_vision_enabled(self):
        provider = _ollama_provider(vision=True)
        msg = Message(
            role="user",
            content=[TextPart("hi "), TextPart("there"), ImagePart(B64_IMAGE, "image/png")],
        )
        payload = provider._build_payload(ChatRequest(messages=[msg]))
        # last appended message is the user turn
        user_msg = payload["messages"][-1]
        self.assertEqual(user_msg["role"], "user")
        self.assertEqual(user_msg["content"], "hi there")
        self.assertEqual(user_msg["images"], [B64_IMAGE])

    def test_image_url_raises(self):
        provider = _ollama_provider(vision=True)
        msg = Message(role="user", content=[ImageUrl("https://x/y.png")])
        with self.assertRaises(ValidationError):
            provider._build_payload(ChatRequest(messages=[msg]))

    def test_image_without_vision_flag_raises(self):
        provider = _ollama_provider()  # vision defaults off
        msg = Message(role="user", content=[ImagePart(B64_IMAGE, "image/png")])
        with self.assertRaises(ValidationError):
            provider._build_payload(ChatRequest(messages=[msg]))


class TestVisionGate(unittest.TestCase):
    def test_non_vision_provider_raises(self):
        provider = _ollama_provider()  # vision off
        request = ChatRequest(
            messages=[Message(role="user", content=[ImagePart(B64_IMAGE, "image/png")])]
        )
        with self.assertRaises(ValidationError):
            provider._assert_image_support(request)

    def test_vision_provider_passes(self):
        provider = _anthropic_provider()  # vision=True
        request = ChatRequest(
            messages=[Message(role="user", content=[ImagePart(B64_IMAGE, "image/png")])]
        )
        provider._assert_image_support(request)  # should not raise

    def test_text_only_never_gated(self):
        provider = _ollama_provider()  # vision off
        request = ChatRequest(messages=[Message(role="user", content="hello")])
        provider._assert_image_support(request)  # should not raise


if __name__ == "__main__":
    unittest.main()
