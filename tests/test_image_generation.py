"""
Tests for text-to-image generation across the data model and providers.
"""

import base64
import sys
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from llm_provider.errors import ValidationError
from llm_provider.models import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePart,
    ProviderConfig,
)
from llm_provider.retry import RetryConfig

RAW_IMAGE = b"\x89PNG\r\n\x1a\n_fake_png_bytes_"
B64_IMAGE = base64.b64encode(RAW_IMAGE).decode("ascii")


# ----------------------------------------------------------------------------
# Data-model tests
# ----------------------------------------------------------------------------


class TestImageGenerationModels(unittest.TestCase):
    def test_request_defaults(self):
        req = ImageGenerationRequest(prompt="a cat")
        self.assertEqual(req.n, 1)
        self.assertIsNone(req.size)

    def test_request_rejects_empty_prompt(self):
        with self.assertRaises(ValueError):
            ImageGenerationRequest(prompt="")

    def test_request_rejects_n_below_one(self):
        with self.assertRaises(ValueError):
            ImageGenerationRequest(prompt="a cat", n=0)

    def test_request_rejects_unknown_extra_body_key(self):
        with self.assertRaises(ValueError):
            ImageGenerationRequest(prompt="a cat", extra_body={"bogus": {}})

    def test_request_accepts_known_extra_body(self):
        req = ImageGenerationRequest(
            prompt="a cat", extra_body={"gemini": {"aspect_ratio": "16:9"}}
        )
        self.assertEqual(req.extra_body["gemini"]["aspect_ratio"], "16:9")

    def test_response_holds_image_parts(self):
        resp = ImageGenerationResponse(images=[ImagePart(B64_IMAGE, "image/png")])
        self.assertEqual(len(resp.images), 1)

    def test_response_rejects_empty(self):
        with self.assertRaises(ValueError):
            ImageGenerationResponse(images=[])

    def test_edit_request_image_plus_prompt(self):
        req = ImageGenerationRequest(prompt="make it night", image=ImagePart(B64_IMAGE, "image/png"))
        self.assertTrue(req.is_edit)
        self.assertFalse(req.is_variation)

    def test_variation_request_image_no_prompt(self):
        req = ImageGenerationRequest(image=ImagePart(B64_IMAGE, "image/png"))
        self.assertTrue(req.is_variation)
        self.assertFalse(req.is_edit)

    def test_no_image_no_prompt_raises(self):
        with self.assertRaises(ValueError):
            ImageGenerationRequest()

    def test_mask_without_image_raises(self):
        with self.assertRaises(ValueError):
            ImageGenerationRequest(prompt="x", mask=ImagePart(B64_IMAGE, "image/png"))

    def test_mask_without_prompt_raises(self):
        with self.assertRaises(ValueError):
            ImageGenerationRequest(
                image=ImagePart(B64_IMAGE, "image/png"),
                mask=ImagePart(B64_IMAGE, "image/png"),
            )

    def test_mask_with_image_and_prompt_ok(self):
        req = ImageGenerationRequest(
            prompt="replace the sky",
            image=ImagePart(B64_IMAGE, "image/png"),
            mask=ImagePart(B64_IMAGE, "image/png"),
        )
        self.assertTrue(req.is_edit)

    def test_image_must_be_image_part(self):
        with self.assertRaises(ValueError):
            ImageGenerationRequest(prompt="x", image="not-an-image-part")


# ----------------------------------------------------------------------------
# Provider construction helpers
# ----------------------------------------------------------------------------


def _openai_provider(**extra_settings):
    from llm_provider.providers.openai_provider import OpenAIProvider

    config = ProviderConfig(
        host="https://api.openai.com",
        default_model="gpt-4o",
        api_key="k",
        extra_settings=extra_settings or {},
    )
    with patch.dict(sys.modules, {"openai": MagicMock()}):
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._config = config
        provider._client = MagicMock()
        provider._semaphore = None
        provider._retry_config = RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.01)
        return provider


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
    provider._semaphore = None
    provider._retry_config = RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.01)
    return provider


def _gemini_genai_patch():
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


def _ollama_provider():
    from llm_provider.providers.ollama_provider import OllamaProvider

    return OllamaProvider(ProviderConfig(host="http://localhost:11434", default_model="llama2"))


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


def _openai_image_response(b64=B64_IMAGE, revised=None):
    datum = MagicMock()
    datum.b64_json = b64
    datum.revised_prompt = revised
    resp = MagicMock()
    resp.data = [datum]
    return resp


# ----------------------------------------------------------------------------
# OpenAI
# ----------------------------------------------------------------------------


class TestOpenAIImageGeneration(unittest.TestCase):
    def test_gpt_image_1_omits_response_format(self):
        provider = _openai_provider()
        provider._client.images.generate.return_value = _openai_image_response()
        provider.generate_image(ImageGenerationRequest(prompt="a cat", size="1024x1024"))
        kwargs = provider._client.images.generate.call_args[1]
        self.assertEqual(kwargs["model"], "gpt-image-1")
        self.assertEqual(kwargs["prompt"], "a cat")
        self.assertEqual(kwargs["size"], "1024x1024")
        self.assertNotIn("response_format", kwargs)

    def test_dalle_sets_b64_response_format(self):
        provider = _openai_provider(image_model="dall-e-3")
        provider._client.images.generate.return_value = _openai_image_response(
            revised="a fluffy cat"
        )
        resp = provider.generate_image(ImageGenerationRequest(prompt="a cat"))
        kwargs = provider._client.images.generate.call_args[1]
        self.assertEqual(kwargs["model"], "dall-e-3")
        self.assertEqual(kwargs["response_format"], "b64_json")
        self.assertEqual(resp.revised_prompt, "a fluffy cat")

    def test_parses_b64_into_image_part(self):
        provider = _openai_provider()
        provider._client.images.generate.return_value = _openai_image_response()
        resp = provider.generate_image(ImageGenerationRequest(prompt="a cat"))
        self.assertIsInstance(resp.images[0], ImagePart)
        self.assertEqual(resp.images[0].data, B64_IMAGE)
        self.assertEqual(resp.images[0].media_type, "image/png")

    def test_extra_body_and_quality_flow_through(self):
        provider = _openai_provider()
        provider._client.images.generate.return_value = _openai_image_response()
        provider.generate_image(
            ImageGenerationRequest(
                prompt="a cat", quality="hd", n=2, extra_body={"openai": {"style": "vivid"}}
            )
        )
        kwargs = provider._client.images.generate.call_args[1]
        self.assertEqual(kwargs["quality"], "hd")
        self.assertEqual(kwargs["n"], 2)
        self.assertEqual(kwargs["style"], "vivid")

    def test_mlx_disables_generation(self):
        from llm_provider.providers import create_mlx_provider

        with patch.dict(sys.modules, {"openai": MagicMock()}):
            provider = create_mlx_provider({"default_model": "mlx-community/Qwen3"})
        self.assertFalse(provider.supported_features().image_generation)
        with self.assertRaises(ValidationError):
            provider.generate_image(ImageGenerationRequest(prompt="a cat"))

    def test_async_generation(self):
        import asyncio

        provider = _openai_provider()
        provider._async_client = MagicMock()
        provider._async_client.images.generate = AsyncMock(return_value=_openai_image_response())
        resp = asyncio.run(provider.agenerate_image(ImageGenerationRequest(prompt="a cat")))
        self.assertEqual(resp.images[0].data, B64_IMAGE)


# ----------------------------------------------------------------------------
# OpenAI img2img (edit / inpaint / variation)
# ----------------------------------------------------------------------------


class TestOpenAIImageEditing(unittest.TestCase):
    def test_edit_routes_to_edit_endpoint(self):
        provider = _openai_provider()
        provider._client.images.edit.return_value = _openai_image_response()
        resp = provider.generate_image(
            ImageGenerationRequest(prompt="make it night", image=ImagePart(B64_IMAGE, "image/png"))
        )
        provider._client.images.generate.assert_not_called()
        kwargs = provider._client.images.edit.call_args[1]
        self.assertEqual(kwargs["model"], "gpt-image-1")
        self.assertEqual(kwargs["prompt"], "make it night")
        # image passed as a (name, bytes, mime) upload tuple
        self.assertEqual(kwargs["image"][2], "image/png")
        self.assertEqual(kwargs["image"][1], RAW_IMAGE)
        self.assertNotIn("mask", kwargs)
        self.assertEqual(resp.images[0].data, B64_IMAGE)

    def test_inpaint_passes_mask(self):
        provider = _openai_provider()
        provider._client.images.edit.return_value = _openai_image_response()
        provider.generate_image(
            ImageGenerationRequest(
                prompt="replace sky",
                image=ImagePart(B64_IMAGE, "image/png"),
                mask=ImagePart(B64_IMAGE, "image/png"),
            )
        )
        kwargs = provider._client.images.edit.call_args[1]
        self.assertEqual(kwargs["mask"][2], "image/png")

    def test_edit_multiple_source_images(self):
        provider = _openai_provider()
        provider._client.images.edit.return_value = _openai_image_response()
        provider.generate_image(
            ImageGenerationRequest(
                prompt="combine",
                image=[ImagePart(B64_IMAGE, "image/png"), ImagePart(B64_IMAGE, "image/png")],
            )
        )
        kwargs = provider._client.images.edit.call_args[1]
        self.assertEqual(len(kwargs["image"]), 2)

    def test_dalle_edit_forces_b64(self):
        provider = _openai_provider(image_model="dall-e-2")
        provider._client.images.edit.return_value = _openai_image_response()
        provider.generate_image(
            ImageGenerationRequest(prompt="x", image=ImagePart(B64_IMAGE, "image/png"))
        )
        kwargs = provider._client.images.edit.call_args[1]
        self.assertEqual(kwargs["response_format"], "b64_json")

    def test_variation_routes_to_variation_endpoint(self):
        provider = _openai_provider()
        provider._client.images.create_variation.return_value = _openai_image_response()
        resp = provider.generate_image(
            ImageGenerationRequest(image=ImagePart(B64_IMAGE, "image/png"), n=2)
        )
        provider._client.images.edit.assert_not_called()
        kwargs = provider._client.images.create_variation.call_args[1]
        self.assertEqual(kwargs["model"], "dall-e-2")
        self.assertEqual(kwargs["n"], 2)
        self.assertEqual(kwargs["response_format"], "b64_json")
        self.assertNotIn("prompt", kwargs)
        self.assertEqual(resp.images[0].data, B64_IMAGE)

    def test_variation_rejects_multiple_images(self):
        provider = _openai_provider()
        with self.assertRaises(ValidationError):
            provider.generate_image(
                ImageGenerationRequest(
                    image=[ImagePart(B64_IMAGE, "image/png"), ImagePart(B64_IMAGE, "image/png")]
                )
            )

    def test_async_edit(self):
        import asyncio

        provider = _openai_provider()
        provider._async_client = MagicMock()
        provider._async_client.images.edit = AsyncMock(return_value=_openai_image_response())
        resp = asyncio.run(
            provider.agenerate_image(
                ImageGenerationRequest(prompt="x", image=ImagePart(B64_IMAGE, "image/png"))
            )
        )
        self.assertEqual(resp.images[0].data, B64_IMAGE)


# ----------------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------------


def _imagen_response(raw=RAW_IMAGE):
    gen = MagicMock()
    gen.image.image_bytes = raw
    resp = MagicMock()
    resp.generated_images = [gen]
    return resp


class TestGeminiImageGeneration(unittest.TestCase):
    def test_generate_images_called_with_config(self):
        provider = _gemini_provider()
        patcher, mock_genai = _gemini_genai_patch()
        provider._client.models.generate_images.return_value = _imagen_response()
        with patcher:
            resp = provider.generate_image(
                ImageGenerationRequest(
                    prompt="a cat", n=3, extra_body={"gemini": {"aspect_ratio": "16:9"}}
                )
            )
        mock_genai.types.GenerateImagesConfig.assert_called_once_with(
            number_of_images=3, aspect_ratio="16:9"
        )
        call = provider._client.models.generate_images.call_args[1]
        self.assertEqual(call["model"], "imagen-3.0-generate-002")
        self.assertEqual(call["prompt"], "a cat")
        self.assertEqual(resp.images[0].data, B64_IMAGE)
        self.assertEqual(resp.images[0].media_type, "image/png")

    def test_supports_image_generation(self):
        provider = _gemini_provider()
        self.assertTrue(provider.supported_features().image_generation)


def _gemini_edit_response(raw=RAW_IMAGE, mime="image/png"):
    part = MagicMock()
    part.inline_data.data = raw
    part.inline_data.mime_type = mime
    cand = MagicMock()
    cand.content.parts = [part]
    resp = MagicMock()
    resp.candidates = [cand]
    return resp


class TestGeminiImageEditing(unittest.TestCase):
    def test_edit_uses_generate_content(self):
        provider = _gemini_provider()
        patcher, mock_genai = _gemini_genai_patch()
        provider._client.models.generate_content.return_value = _gemini_edit_response()
        with patcher:
            resp = provider.generate_image(
                ImageGenerationRequest(
                    prompt="make it night", image=ImagePart(B64_IMAGE, "image/png")
                )
            )
        provider._client.models.generate_images.assert_not_called()
        call = provider._client.models.generate_content.call_args[1]
        self.assertEqual(call["model"], "gemini-3.1-flash-image")
        mock_genai.types.GenerateContentConfig.assert_called_once_with(
            response_modalities=["TEXT", "IMAGE"]
        )
        self.assertEqual(resp.images[0].data, B64_IMAGE)
        self.assertEqual(resp.images[0].media_type, "image/png")

    def test_edit_mask_raises(self):
        provider = _gemini_provider()
        patcher, _ = _gemini_genai_patch()
        with patcher, self.assertRaises(ValidationError):
            provider.generate_image(
                ImageGenerationRequest(
                    prompt="x",
                    image=ImagePart(B64_IMAGE, "image/png"),
                    mask=ImagePart(B64_IMAGE, "image/png"),
                )
            )

    def test_variation_without_prompt_raises(self):
        provider = _gemini_provider()
        patcher, _ = _gemini_genai_patch()
        with patcher, self.assertRaises(ValidationError):
            provider.generate_image(ImageGenerationRequest(image=ImagePart(B64_IMAGE, "image/png")))


# ----------------------------------------------------------------------------
# Unsupported providers
# ----------------------------------------------------------------------------


class TestUnsupportedProviders(unittest.TestCase):
    def test_anthropic_raises(self):
        provider = _anthropic_provider()
        with self.assertRaises(ValidationError):
            provider.generate_image(ImageGenerationRequest(prompt="a cat"))

    def test_ollama_raises(self):
        provider = _ollama_provider()
        with self.assertRaises(ValidationError):
            provider.generate_image(ImageGenerationRequest(prompt="a cat"))

    def test_openai_generation_on_by_default(self):
        provider = _openai_provider()
        self.assertTrue(provider.supported_features().image_generation)


if __name__ == "__main__":
    unittest.main()
