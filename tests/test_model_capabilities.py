import unittest

from llm_provider.model_capabilities import (
    get_model_capabilities,
    supports_nested_tool_params,
)


class TestModelCapabilities(unittest.TestCase):
    def test_qwen3_does_not_support_nested_tool_params(self):
        self.assertFalse(supports_nested_tool_params("mlx-community/Qwen3.5-9B-MLX-4bit"))
        # Case-insensitive substring match.
        self.assertFalse(supports_nested_tool_params("QWEN3-7B"))

    def test_unknown_model_is_assumed_capable(self):
        self.assertTrue(supports_nested_tool_params("gpt-4o"))
        self.assertTrue(supports_nested_tool_params("claude-3-5-sonnet"))
        self.assertTrue(supports_nested_tool_params("llama3.1:latest"))

    def test_empty_or_none_model_is_capable(self):
        self.assertTrue(supports_nested_tool_params(None))
        self.assertTrue(supports_nested_tool_params(""))

    def test_get_model_capabilities_returns_dataclass(self):
        caps = get_model_capabilities("qwen3")
        self.assertFalse(caps.supports_nested_tool_params)
        caps2 = get_model_capabilities("gpt-4o")
        self.assertTrue(caps2.supports_nested_tool_params)


if __name__ == "__main__":
    unittest.main()
