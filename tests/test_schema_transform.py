import unittest

from llm_provider.schema_transform import (
    flatten_tool_schema,
    renest_arguments,
    schema_has_flattenable_nesting,
)


def _nested_intentions_schema():
    """An object-of-objects, like storyteller's submit_intentions."""

    def _obj(with_volume=False):
        props = {
            "text": {"type": "string"},
            "determination": {"type": "integer", "minimum": 1, "maximum": 100},
        }
        required = ["text", "determination"]
        if with_volume:
            props["volume"] = {"type": "string", "enum": ["soft", "shout"]}
            required.append("volume")
        return {"type": "object", "properties": props, "required": required}

    return {
        "type": "object",
        "properties": {
            "say": _obj(with_volume=True),
            "do": _obj(),
        },
        "required": ["say", "do"],
    }


class TestFlattenRoundTrip(unittest.TestCase):
    def test_flattens_nested_objects_to_scalar_fields(self):
        flat, mapping = flatten_tool_schema(_nested_intentions_schema())
        self.assertEqual(
            set(flat["properties"]),
            {"say__text", "say__determination", "say__volume", "do__text", "do__determination"},
        )
        # The nested object schemas are gone — only scalar leaves remain.
        for prop in flat["properties"].values():
            self.assertNotEqual(prop.get("type"), "object")
        # Leaf schema is carried through verbatim.
        self.assertEqual(flat["properties"]["say__volume"]["enum"], ["soft", "shout"])

    def test_required_propagates_only_when_every_ancestor_required(self):
        schema = _nested_intentions_schema()
        # 'do' itself is required and both its leaves are required.
        flat, _ = flatten_tool_schema(schema)
        self.assertIn("do__text", flat["required"])
        self.assertIn("say__volume", flat["required"])

        # Make 'say' optional at the top: its leaves must drop out of required.
        schema["required"] = ["do"]
        flat2, _ = flatten_tool_schema(schema)
        self.assertNotIn("say__text", flat2["required"])
        self.assertIn("do__text", flat2["required"])

    def test_round_trip_renests_arguments(self):
        _, mapping = flatten_tool_schema(_nested_intentions_schema())
        flat_args = {
            "say__text": "hi",
            "say__determination": 80,
            "say__volume": "shout",
            "do__text": "run",
            "do__determination": 50,
        }
        nested = renest_arguments(flat_args, mapping)
        self.assertEqual(
            nested,
            {
                "say": {"text": "hi", "determination": 80, "volume": "shout"},
                "do": {"text": "run", "determination": 50},
            },
        )

    def test_collision_raises(self):
        # A literal 'a__b' top-level key plus a nested a.b would collide.
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "object", "properties": {"b": {"type": "string"}}},
                "a__b": {"type": "string"},
            },
        }
        with self.assertRaises(ValueError):
            flatten_tool_schema(schema)


class TestPassThrough(unittest.TestCase):
    def test_arrays_of_objects_pass_through_unchanged(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"x": {"type": "integer"}}},
                },
            },
        }
        flat, mapping = flatten_tool_schema(schema)
        self.assertEqual(flat, schema)
        self.assertEqual(mapping, {})
        self.assertFalse(schema_has_flattenable_nesting(schema))

    def test_additional_properties_map_passes_through(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "object", "additionalProperties": {"type": "integer"}},
                "name": {"type": "string"},
            },
        }
        flat, mapping = flatten_tool_schema(schema)
        self.assertEqual(flat, schema)
        self.assertEqual(mapping, {})

    def test_all_flat_schema_is_noop(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        }
        flat, mapping = flatten_tool_schema(schema)
        self.assertEqual(flat, schema)
        self.assertEqual(mapping, {})

    def test_renest_passes_through_unmapped_keys(self):
        # Top-level scalars and unexpected keys survive re-nesting.
        _, mapping = flatten_tool_schema(
            {
                "type": "object",
                "properties": {
                    "say": {"type": "object", "properties": {"text": {"type": "string"}}},
                    "loose": {"type": "string"},
                },
            }
        )
        nested = renest_arguments({"say__text": "hi", "loose": "x", "surprise": 1}, mapping)
        self.assertEqual(nested, {"say": {"text": "hi"}, "loose": "x", "surprise": 1})


class TestDeepNesting(unittest.TestCase):
    def test_recurses_multiple_levels(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {"type": "object", "properties": {"c": {"type": "string"}}},
                    },
                },
            },
        }
        flat, mapping = flatten_tool_schema(schema)
        self.assertIn("a__b__c", flat["properties"])
        self.assertEqual(mapping["a__b__c"], ["a", "b", "c"])
        nested = renest_arguments({"a__b__c": "v"}, mapping)
        self.assertEqual(nested, {"a": {"b": {"c": "v"}}})


if __name__ == "__main__":
    unittest.main()
