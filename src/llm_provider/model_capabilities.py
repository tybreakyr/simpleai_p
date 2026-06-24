"""Per-model capability registry.

``ProviderFeatures`` describes a *provider* (OpenAI, Gemini, …). Some quirks are
finer-grained than that: a single OpenAI-compatible endpoint (mlx-lm, Ollama) can
serve a tiny local model that mishandles nested tool-call parameters *and* a large
model that handles them fine. Those are **per-model** facts, keyed by model name.

The registry is intentionally tiny and conservative: a model is assumed fully
capable unless a known-weak pattern says otherwise. Add a pattern here when a model
is found to need accommodation — callers never have to know about it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities of a single model, as far as this library needs to adapt to them.

    Defaults describe a fully-capable model, so an unknown model is left untouched.
    """

    # Whether the model reliably handles nested objects in tool-call parameters.
    # When False, the library flattens nested tool schemas on the way out and
    # re-nests the returned arguments (see ``schema_transform``).
    supports_nested_tool_params: bool = True


_FULLY_CAPABLE = ModelCapabilities()

# Ordered (substring, capabilities) rules, matched case-insensitively against the
# model name. First match wins. Seeded only with what we've actually observed:
# Qwen3-family models raise on nested tool params under multi-call load.
_REGISTRY: list[tuple[str, ModelCapabilities]] = [
    ("qwen3", ModelCapabilities(supports_nested_tool_params=False)),
]


def get_model_capabilities(model_name: str | None) -> ModelCapabilities:
    """Look up capabilities for ``model_name`` (first matching pattern wins).

    Unknown / empty names return the fully-capable defaults so the library never
    degrades a model it has no information about.
    """
    if not model_name:
        return _FULLY_CAPABLE
    lowered = model_name.lower()
    for pattern, caps in _REGISTRY:
        if pattern in lowered:
            return caps
    return _FULLY_CAPABLE


def supports_nested_tool_params(model_name: str | None) -> bool:
    """Convenience predicate for the nested-tool-params capability."""
    return get_model_capabilities(model_name).supports_nested_tool_params
