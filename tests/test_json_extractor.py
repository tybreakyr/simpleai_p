"""
Unit tests for JSON extraction utility.
"""

import unittest
from llm_provider.json_extractor import extract_json, parse_structured_output
from llm_provider.errors import JSONParseFailedError


class TestJSONExtractor(unittest.TestCase):
    """Test cases for JSON extraction."""
    
    def test_direct_validation(self):
        """Test direct JSON validation."""
        json_str = '{"key": "value"}'
        result = extract_json(json_str)
        self.assertEqual(result, json_str)
    
    def test_direct_validation_with_whitespace(self):
        """Test direct validation with whitespace."""
        json_str = '  {"key": "value"}  '
        result = extract_json(json_str)
        self.assertEqual(result.strip(), '{"key": "value"}')
    
    def test_markdown_extraction_json(self):
        """Test extraction from markdown code blocks with json."""
        text = '''
        Here's the JSON:
        ```json
        {"key": "value"}
        ```
        '''
        result = extract_json(text)
        self.assertEqual(result.strip(), '{"key": "value"}')
    
    def test_markdown_extraction_generic(self):
        """Test extraction from generic markdown code blocks."""
        text = '''
        Here's the JSON:
        ```
        {"key": "value"}
        ```
        '''
        result = extract_json(text)
        self.assertEqual(result.strip(), '{"key": "value"}')
    
    def test_json_with_unescaped_newlines(self):
        """Test JSON repair for unescaped newlines."""
        text = '{"key": "value\nwith newline"}'
        result = extract_json(text)
        # Should be able to parse it
        import json
        parsed = json.loads(result)
        self.assertEqual(parsed["key"], "value\nwith newline")
    
    def test_brace_counting(self):
        """Test brace counting extraction."""
        text = 'Some text before {"key": "value"} some text after'
        result = extract_json(text)
        self.assertIn('{"key": "value"}', result)
    
    def test_aggressive_cleanup(self):
        """Test aggressive cleanup with prefixes."""
        text = 'Here\'s the JSON: {"key": "value"}'
        result = extract_json(text)
        self.assertIn('{"key": "value"}', result)
    
    def test_nested_json(self):
        """Test nested JSON extraction."""
        json_str = '{"outer": {"inner": "value"}}'
        result = extract_json(json_str)
        self.assertEqual(result, json_str)
    
    def test_json_array(self):
        """Test JSON array extraction."""
        json_str = '[{"key": "value"}, {"key2": "value2"}]'
        result = extract_json(json_str)
        self.assertEqual(result, json_str)
    
    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises error."""
        text = 'This is not JSON at all'
        with self.assertRaises(JSONParseFailedError):
            extract_json(text)
    
    def test_parse_structured_output_dict(self):
        """Test parsing structured output to dict."""
        text = '{"name": "John", "age": 30}'
        result = parse_structured_output(text, dict)
        self.assertEqual(result["name"], "John")
        self.assertEqual(result["age"], 30)
    
    def test_parse_structured_output_dataclass(self):
        """Test parsing structured output to dataclass."""
        from dataclasses import dataclass
        
        @dataclass
        class Person:
            name: str
            age: int
        
        text = '{"name": "John", "age": 30}'
        result = parse_structured_output(text, Person)
        self.assertEqual(result.name, "John")
        self.assertEqual(result.age, 30)


if __name__ == '__main__':
    unittest.main()

