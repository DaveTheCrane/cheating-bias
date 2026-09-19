"""Model registry.

Specs are ``<provider>:<id>``:

- ``mock:honest``, ``mock:hardcoder``, ... — scripted, no API key needed
- ``anthropic:claude-sonnet-4-6``
- ``openai:gpt-4.1``
"""

from __future__ import annotations

from harness.config import TaskSpec
from harness.models.base import (
    Message,
    Model,
    ModelError,
    ModelResponse,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from harness.models.mock import PERSONALITIES, MockModel

__all__ = [
    "Message",
    "Model",
    "ModelError",
    "ModelResponse",
    "ToolCall",
    "ToolResult",
    "ToolSchema",
    "MockModel",
    "PERSONALITIES",
    "build_model",
    "parse_spec",
]

PROVIDERS = ("mock", "anthropic", "openai")


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``provider:id``, defaulting bare specs to the mock provider."""
    text = str(spec).strip()
    if ":" not in text:
        raise ValueError(
            f"model spec {spec!r} must be '<provider>:<id>'; providers are {list(PROVIDERS)}"
        )
    provider, _, model_id = text.partition(":")
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r} in {spec!r}; have {list(PROVIDERS)}")
    return provider, model_id.strip()


def build_model(
    spec: str,
    task: TaskSpec | None = None,
    temperature: float = 1.0,
    max_tokens: int = 4096,
) -> Model:
    """Construct a model from its spec string."""
    provider, model_id = parse_spec(spec)

    if provider == "mock":
        model = MockModel(model_id, task=task)
        return model

    if provider == "anthropic":
        from harness.models.anthropic_model import AnthropicModel

        return AnthropicModel(model_id, temperature=temperature, max_tokens=max_tokens)

    from harness.models.openai_model import OpenAIModel

    return OpenAIModel(model_id, temperature=temperature, max_tokens=max_tokens)
