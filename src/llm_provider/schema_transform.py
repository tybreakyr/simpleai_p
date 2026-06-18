"""Bidirectional nested↔flat codec for tool-call JSON Schemas.

Some models mishandle nested objects in tool-call parameters (see
``model_capabilities``). For those, this module rewrites a tool's ``input_schema``
so that nested *fixed-key objects* become flat scalar fields before the request is
sent, and re-nests the returned argument dict afterwards — so callers can always
declare the natural nested schema and receive natural nested arguments.

Scope, deliberately narrow:
- **Flattened:** properties whose ``type == "object"`` *and* which carry a fixed
  ``properties`` dict (recursively). The flat key is the dotted path joined by
  ``"__"``.
- **Left untouched:** scalars, enums, arrays (including arrays-of-objects), and
  objects that are maps (``additionalProperties`` with no fixed ``properties``).
  These don't flatten to a fixed field set, so they pass through unchanged — which
  is what keeps batch/map tool schemas safe.

All functions here are pure and provider-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Path separator for flattened keys. Two underscores so a single ``_`` inside an
# existing property name can't accidentally collide with a flattened path.
SEP = "__"

# A mapping records, per produced flat key, the nested path needed to rebuild it.
FlattenMapping = Dict[str, List[str]]


def _is_flattenable_object(prop_schema: Any) -> bool:
    """True for a fixed-key object (``type: object`` with a ``properties`` dict).

    Map-style objects (``additionalProperties`` only) and non-objects are not
    flattenable and return False.
    """
    return (
        isinstance(prop_schema, dict)
        and prop_schema.get("type") == "object"
        and isinstance(prop_schema.get("properties"), dict)
    )


def schema_has_flattenable_nesting(input_schema: Any) -> bool:
    """Cheap guard: does the schema have any nested fixed-key object to flatten?

    Only the top level need be checked — a deeper nested object is always reached
    through a top-level flattenable object.
    """
    if not isinstance(input_schema, dict):
        return False
    props = input_schema.get("properties")
    if not isinstance(props, dict):
        return False
    return any(_is_flattenable_object(child) for child in props.values())


def flatten_tool_schema(input_schema: Dict[str, Any]) -> Tuple[Dict[str, Any], FlattenMapping]:
    """Flatten nested fixed-key objects in a tool ``input_schema``.

    Returns ``(flat_schema, mapping)``. ``mapping`` maps each *flattened* leaf key
    (path length >= 2) to its nested path; top-level scalars are absent from it and
    re-nest as themselves. Raises ``ValueError`` on a flat-key collision so a bad
    schema fails loudly rather than dropping a field silently.
    """
    if not schema_has_flattenable_nesting(input_schema):
        return input_schema, {}

    flat_props: Dict[str, Any] = {}
    flat_required: List[str] = []
    mapping: FlattenMapping = {}

    def _walk(obj_schema: Dict[str, Any], prefix_key: str, prefix_path: List[str], ancestors_required: bool) -> None:
        sub_props: Dict[str, Any] = obj_schema.get("properties", {}) or {}
        sub_required = set(obj_schema.get("required", []) or [])
        for name, child in sub_props.items():
            flat_key = f"{prefix_key}{SEP}{name}" if prefix_key else name
            path = prefix_path + [name]
            this_required = ancestors_required and (name in sub_required)
            if _is_flattenable_object(child):
                _walk(child, flat_key, path, this_required)
            else:
                if flat_key in flat_props:
                    raise ValueError(f"schema flatten collision on key {flat_key!r}")
                flat_props[flat_key] = child
                if len(path) >= 2:
                    mapping[flat_key] = path
                if this_required:
                    flat_required.append(flat_key)

    _walk(input_schema, "", [], True)

    flat_schema = {
        k: v for k, v in input_schema.items() if k not in ("properties", "required")
    }
    flat_schema["properties"] = flat_props
    if flat_required:
        flat_schema["required"] = flat_required
    return flat_schema, mapping


def renest_arguments(flat_args: Dict[str, Any], mapping: FlattenMapping) -> Dict[str, Any]:
    """Rebuild a nested argument dict from flat args using ``mapping``.

    Keys absent from ``mapping`` (top-level scalars, plus anything the model
    returned that wasn't in the schema) pass through unchanged.
    """
    if not mapping:
        return flat_args
    out: Dict[str, Any] = {}
    for key, value in flat_args.items():
        path = mapping.get(key)
        if not path:
            out[key] = value
            continue
        cursor = out
        for seg in path[:-1]:
            nxt = cursor.get(seg)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[seg] = nxt
            cursor = nxt
        cursor[path[-1]] = value
    return out
