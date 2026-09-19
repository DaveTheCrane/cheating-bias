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
    "requires_credentials",
    "is_local",
]

PROVIDERS = ("mock", "anthropic", "openai", "ollama")

# Providers that reach a hosted API and therefore need a key in the environment.
CREDENTIALLED = ("anthropic", "openai")

# Providers served from this machine. Grids against these should not be run
# concurrently: a local server processes requests serially by default, so extra
# workers only queue and contend for memory.
LOCAL = ("ollama",)


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``provider:id`` on the first colon.

    Only the first colon separates, because Ollama tags contain one of their own:
    ``ollama:qwen3.5:latest`` is provider ``ollama``, model ``qwen3.5:latest``.
    """
    text = str(spec).strip()
    if ":" not in text:
        raise ValueError(
            f"model spec {spec!r} must be '<provider>:<id>'; providers are {list(PROVIDERS)}"
        )
    provider, _, model_id = text.partition(":")
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r} in {spec!r}; have {list(PROVIDERS)}")
    if not model_id.strip():
        raise ValueError(f"model spec {spec!r} names no model after the provider")
    return provider, model_id.strip()


def requires_credentials(spec: str) -> bool:
    """Whether this spec will need an API key."""
    return parse_spec(spec)[0] in CREDENTIALLED


def is_local(spec: str) -> bool:
    """Whether this spec is served from this machine."""
    return parse_spec(spec)[0] in LOCAL


def build_model(
    spec: str,
    task: TaskSpec | None = None,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    provider_options: dict[str, dict] | None = None,
) -> Model:
    """Construct a model from its spec string.

    Args:
        spec: ``<provider>:<id>``.
        task: Bound to mock models, which script themselves per task.
        temperature: Sampling temperature.
        max_tokens: Output cap.
        provider_options: Extra constructor keyword arguments, keyed by provider
            name. Lets a grid config carry provider-specific settings (Ollama's
            host, think flag, num_ctx) without the runner knowing about them.
    """
    provider, model_id = parse_spec(spec)
    extra = dict((provider_options or {}).get(provider, {}))

    if provider == "mock":
        return MockModel(model_id, task=task)

    if provider == "anthropic":
        from harness.models.anthropic_model import AnthropicModel

        return AnthropicModel(
            model_id, temperature=temperature, max_tokens=max_tokens, **extra
        )

    if provider == "ollama":
        from harness.models.ollama_model import OllamaModel

        return OllamaModel(model_id, temperature=temperature, max_tokens=max_tokens, **extra)

    from harness.models.openai_model import OpenAIModel

    return OpenAIModel(model_id, temperature=temperature, max_tokens=max_tokens, **extra)
