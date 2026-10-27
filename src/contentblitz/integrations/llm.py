"""The only module that talks to model providers. See docs/adr/0002."""

import litellm

from contentblitz.core.config import ModelConfig, get_model_config, get_settings

_callbacks_configured = False


def _configure_callbacks() -> None:
    global _callbacks_configured
    if not _callbacks_configured and get_settings().langsmith_tracing:
        litellm.success_callback = [*litellm.success_callback, "langsmith"]
    _callbacks_configured = True


def complete(agent: str, messages: list[dict], config: ModelConfig | None = None, **kwargs) -> str:
    """Run a chat completion for `agent`, falling back through its configured models."""
    _configure_callbacks()
    config = config or get_model_config()
    models = config.for_agent(agent)
    response = litellm.completion(
        model=models.primary,
        fallbacks=models.fallbacks or None,
        messages=messages,
        num_retries=config.llm.num_retries,
        timeout=config.llm.timeout_seconds,
        **kwargs,
    )
    return response.choices[0].message.content or ""
