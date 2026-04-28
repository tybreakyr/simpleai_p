"""
Unit tests for data models.
"""

import unittest
from llm_provider.models import (
    Message,
    SystemPrompt,
    ChatRequest,
    ChatResponse,
    Model,
    ProviderFeatures,
    ProviderConfig,
    FactoryConfig,
    ToolSchema,
    ToolCall,
)


class TestModels(unittest.TestCase):
    """Test cases for data models."""
    
    def test_message(self):
        """Test Message model."""
        msg = Message(role="user", content="Hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")
        
        # Empty role should raise error
        with self.assertRaises(ValueError):
            Message(role="", content="Hello")
    
    def test_system_prompt(self):
        """Test SystemPrompt model."""
        prompt = SystemPrompt(content="You are a helpful assistant")
        self.assertEqual(prompt.content, "You are a helpful assistant")
    
    def test_chat_request(self):
        """Test ChatRequest model."""
        messages = [Message(role="user", content="Hello")]
        request = ChatRequest(messages=messages)
        self.assertEqual(len(request.messages), 1)
        
        # Empty messages should raise error
        with self.assertRaises(ValueError):
            ChatRequest(messages=[])
    
    def test_chat_response(self):
        """Test ChatResponse model."""
        response = ChatResponse(message="Hello, how can I help?")
        self.assertEqual(response.message, "Hello, how can I help?")
        self.assertIsNone(response.structured_data)
    
    def test_model(self):
        """Test Model model."""
        model = Model(name="gpt-4")
        self.assertEqual(model.name, "gpt-4")
        
        # Empty name should raise error
        with self.assertRaises(ValueError):
            Model(name="")
    
    def test_provider_features(self):
        """Test ProviderFeatures model."""
        features = ProviderFeatures(
            structured_output=True,
            streaming=True,
            context_window=8192
        )
        self.assertTrue(features.structured_output)
        self.assertTrue(features.streaming)
        self.assertEqual(features.context_window, 8192)
        
        # Negative context window should raise error
        with self.assertRaises(ValueError):
            ProviderFeatures(context_window=-1)
    
    def test_provider_config(self):
        """Test ProviderConfig model."""
        config = ProviderConfig(
            host="http://localhost:8000",
            default_model="gpt-4",
            timeout=30.0,
            retry_attempts=3
        )
        self.assertEqual(config.host, "http://localhost:8000")
        self.assertEqual(config.default_model, "gpt-4")
        self.assertEqual(config.timeout, 30.0)
        
        # Invalid timeout should raise error
        with self.assertRaises(ValueError):
            ProviderConfig(host="http://localhost", default_model="model", timeout=-1)
    
    def test_factory_config(self):
        """Test FactoryConfig model."""
        provider_config = ProviderConfig(
            host="http://localhost",
            default_model="model"
        )
        factory_config = FactoryConfig(
            default_provider="provider1",
            provider_configs={"provider1": provider_config}
        )
        self.assertEqual(factory_config.default_provider, "provider1")
        self.assertEqual(len(factory_config.provider_configs), 1)
        
        # Default provider must exist in configs
        with self.assertRaises(ValueError):
            FactoryConfig(
                default_provider="nonexistent",
                provider_configs={"provider1": provider_config}
            )


class TestToolSchema(unittest.TestCase):
    """Test cases for ToolSchema."""

    def test_basic(self):
        schema = ToolSchema(
            name="get_weather",
            description="Get current weather",
            input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
        )
        self.assertEqual(schema.name, "get_weather")
        self.assertIsInstance(schema.input_schema, dict)

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            ToolSchema(name="", description="d", input_schema={})

    def test_non_dict_schema_raises(self):
        with self.assertRaises(ValueError):
            ToolSchema(name="fn", description="d", input_schema="bad")  # type: ignore


class TestToolCall(unittest.TestCase):
    """Test cases for ToolCall."""

    def test_basic(self):
        tc = ToolCall(id="call_abc123", name="get_weather", arguments={"location": "NYC"})
        self.assertEqual(tc.id, "call_abc123")
        self.assertEqual(tc.name, "get_weather")
        self.assertEqual(tc.arguments["location"], "NYC")

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            ToolCall(id="x", name="", arguments={})

    def test_non_dict_arguments_raises(self):
        with self.assertRaises(ValueError):
            ToolCall(id="x", name="fn", arguments="bad")  # type: ignore

    def test_make_id_format(self):
        id1 = ToolCall.make_id()
        id2 = ToolCall.make_id()
        self.assertTrue(id1.startswith("call_"))
        self.assertEqual(len(id1), len("call_") + 12)
        self.assertNotEqual(id1, id2)


class TestChatRequestToolFields(unittest.TestCase):
    """Test new tool fields on ChatRequest."""

    def _req(self, **kwargs):
        return ChatRequest(messages=[Message(role="user", content="hi")], **kwargs)

    def test_defaults_none(self):
        req = self._req()
        self.assertIsNone(req.tools)
        self.assertIsNone(req.tool_choice)

    def test_with_tools(self):
        tools = [ToolSchema(name="fn", description="d", input_schema={})]
        req = self._req(tools=tools, tool_choice="auto")
        self.assertEqual(len(req.tools), 1)
        self.assertEqual(req.tool_choice, "auto")


class TestChatResponseToolFields(unittest.TestCase):
    """Test new tool_calls / stop_reason fields on ChatResponse."""

    def test_defaults_none(self):
        resp = ChatResponse(message="hello")
        self.assertIsNone(resp.tool_calls)
        self.assertIsNone(resp.stop_reason)

    def test_with_tool_calls(self):
        tc = ToolCall(id="call_1", name="fn", arguments={"x": 1})
        resp = ChatResponse(message="", tool_calls=[tc], stop_reason="tool_use")
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.stop_reason, "tool_use")


class TestProviderFeaturesAsync(unittest.TestCase):
    """Test async_supported feature flag."""

    def test_default_false(self):
        features = ProviderFeatures()
        self.assertFalse(features.async_supported)

    def test_can_set_true(self):
        features = ProviderFeatures(async_supported=True)
        self.assertTrue(features.async_supported)


if __name__ == '__main__':
    unittest.main()

