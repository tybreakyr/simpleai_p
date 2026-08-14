"""Unit tests for the double-encoded tool-argument decoder."""

import json
import unittest

from llm_provider.tool_arg_decode import decode_tool_arguments

QUESTIONNAIRE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


class TestDecodeToolArguments(unittest.TestCase):
    def test_double_encoded_array_is_decoded(self):
        questions = [
            {"text": "Who are you?", "options": ["a", "b"]},
            {"text": "Why here?", "options": ["c"]},
        ]
        args = {"title": "Wizard", "questions": json.dumps(questions)}

        out = decode_tool_arguments(args, QUESTIONNAIRE_SCHEMA)

        self.assertIsInstance(out["questions"], list)
        self.assertEqual(out["questions"], questions)
        self.assertEqual(out["title"], "Wizard")

    def test_correctly_encoded_array_is_untouched(self):
        questions = [{"text": "Who are you?", "options": ["a"]}]
        out = decode_tool_arguments({"questions": questions}, QUESTIONNAIRE_SCHEMA)
        self.assertEqual(out["questions"], questions)

    def test_nested_double_encoding_inside_items(self):
        args = {
            "questions": [
                {"text": "Who?", "options": json.dumps(["a", "b"])},
            ]
        }
        out = decode_tool_arguments(args, QUESTIONNAIRE_SCHEMA)
        self.assertEqual(out["questions"][0]["options"], ["a", "b"])

    def test_string_property_holding_json_is_not_coerced(self):
        schema = {"type": "object", "properties": {"note": {"type": "string"}}}
        out = decode_tool_arguments({"note": '["not", "a", "list"]'}, schema)
        self.assertEqual(out["note"], '["not", "a", "list"]')

    def test_union_allowing_string_is_not_coerced(self):
        schema = {
            "type": "object",
            "properties": {"payload": {"type": ["string", "array"], "items": {"type": "string"}}},
        }
        out = decode_tool_arguments({"payload": '["a"]'}, schema)
        self.assertEqual(out["payload"], '["a"]')

    def test_decoded_type_must_match_declaration(self):
        # Declared array, but the string parses to an object -> keep original.
        out = decode_tool_arguments({"questions": '{"text": "x"}'}, QUESTIONNAIRE_SCHEMA)
        self.assertEqual(out["questions"], '{"text": "x"}')

    def test_invalid_json_string_is_kept(self):
        out = decode_tool_arguments({"questions": "[not json"}, QUESTIONNAIRE_SCHEMA)
        self.assertEqual(out["questions"], "[not json")

    def test_non_json_looking_string_is_kept(self):
        out = decode_tool_arguments({"questions": "none provided"}, QUESTIONNAIRE_SCHEMA)
        self.assertEqual(out["questions"], "none provided")

    def test_double_encoded_object_property(self):
        schema = {
            "type": "object",
            "properties": {"meta": {"type": "object", "properties": {"genre": {"type": "string"}}}},
        }
        out = decode_tool_arguments({"meta": '{"genre": "horror"}'}, schema)
        self.assertEqual(out["meta"], {"genre": "horror"})

    def test_map_style_object_values_are_decoded(self):
        schema = {
            "type": "object",
            "properties": {
                "costs": {
                    "type": "object",
                    "additionalProperties": {"type": "array", "items": {"type": "string"}},
                }
            },
        }
        out = decode_tool_arguments({"costs": {"a": '["x"]'}}, schema)
        self.assertEqual(out["costs"], {"a": ["x"]})

    def test_unknown_property_and_missing_schema_are_noops(self):
        self.assertEqual(decode_tool_arguments({"x": "[1]"}, QUESTIONNAIRE_SCHEMA), {"x": "[1]"})
        self.assertEqual(decode_tool_arguments({"x": "[1]"}, None), {"x": "[1]"})
        self.assertEqual(decode_tool_arguments("nope", QUESTIONNAIRE_SCHEMA), "nope")


if __name__ == "__main__":
    unittest.main()
