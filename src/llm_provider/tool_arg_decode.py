"""Repair double-encoded tool-call arguments from OpenAI-compatible endpoints.

Some local OpenAI-compatible servers (observed on ``mlx-lm`` serving Qwen3.5)
emit *array-of-object* tool arguments double-encoded: the whole ``arguments``
blob is JSON, and the value of an array property inside it is itself a JSON
*string* rather than a list. After a single ``json.loads`` a caller sees::

    {"questions": "[{\\"text\\": \\"...\\"}, ...]"}   # str, not list

and iterating it walks the string character by character (the classic
``'['`` first element). This module decodes that second layer.

Deliberately conservative — coercion is **gated on the declared tool schema**:

- Only properties whose schema declares ``array`` or ``object`` are considered.
- A property that *also* allows ``string`` is never coerced, so a legitimately
  string-valued argument that happens to look like JSON is left alone.
- The decoded value must match the declared type (a string declared ``array``
  must parse to a ``list``), otherwise the original string is kept.
- Recurses through arrays and objects so nesting at any depth is repaired.

Unrelated to :mod:`schema_transform`, which is a nested↔flat codec for
fixed-key objects and deliberately leaves arrays untouched.

All functions here are pure and provider-agnostic.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["decode_tool_arguments"]


def _declared_types(prop_schema: Any) -> set[str]:
    """Collect the JSON Schema types a property may take.

    Handles ``type`` as a string or list, and unions expressed via
    ``anyOf``/``oneOf``. Returns an empty set when nothing is declared.
    """
    if not isinstance(prop_schema, dict):
        return set()
    types: set[str] = set()
    declared = prop_schema.get("type")
    if isinstance(declared, str):
        types.add(declared)
    elif isinstance(declared, list):
        types.update(t for t in declared if isinstance(t, str))
    for key in ("anyOf", "oneOf"):
        for sub in prop_schema.get(key) or []:
            types |= _declared_types(sub)
    return types


def _subschema_for_type(prop_schema: dict[str, Any], wanted: str) -> dict[str, Any]:
    """Pick the branch of a union schema that declares ``wanted``.

    Returns ``prop_schema`` itself when it declares the type directly.
    """
    if wanted in _declared_types(
        {k: v for k, v in prop_schema.items() if k not in ("anyOf", "oneOf")}
    ):
        return prop_schema
    for key in ("anyOf", "oneOf"):
        for sub in prop_schema.get(key) or []:
            if isinstance(sub, dict) and wanted in _declared_types(sub):
                return sub
    return prop_schema


def _coerce(value: Any, prop_schema: Any) -> Any:
    """Decode ``value`` one layer if the schema says it should be structured."""
    if not isinstance(prop_schema, dict):
        return value

    types = _declared_types(prop_schema)
    structured = types & {"array", "object"}

    if isinstance(value, str):
        # Only touch strings the schema says should not be strings.
        if not structured or "string" in types:
            return value
        text = value.strip()
        if not text or text[0] not in "[{":
            return value
        try:
            decoded = json.loads(text)
        except (ValueError, TypeError):
            return value
        if isinstance(decoded, list) and "array" in types:
            return _coerce(decoded, _subschema_for_type(prop_schema, "array"))
        if isinstance(decoded, dict) and "object" in types:
            return _coerce(decoded, _subschema_for_type(prop_schema, "object"))
        return value

    if isinstance(value, list) and "array" in types:
        items = _subschema_for_type(prop_schema, "array").get("items")
        return [_coerce(item, items) for item in value]

    if isinstance(value, dict) and "object" in types:
        obj_schema = _subschema_for_type(prop_schema, "object")
        props = obj_schema.get("properties")
        extra = obj_schema.get("additionalProperties")
        out: dict[str, Any] = {}
        for key, item in value.items():
            sub = None
            if isinstance(props, dict):
                sub = props.get(key)
            if sub is None and isinstance(extra, dict):
                sub = extra
            out[key] = _coerce(item, sub)
        return out

    return value


def decode_tool_arguments(arguments: Any, input_schema: Any) -> Any:
    """Return ``arguments`` with schema-declared structures decoded.

    ``arguments`` is the once-decoded tool-call argument dict; ``input_schema``
    is the tool's declared JSON Schema (as sent on the wire). A no-op when
    either is missing or the schema declares no properties.
    """
    if not isinstance(arguments, dict) or not isinstance(input_schema, dict):
        return arguments
    props = input_schema.get("properties")
    if not isinstance(props, dict):
        return arguments
    return {key: _coerce(value, props.get(key)) for key, value in arguments.items()}
