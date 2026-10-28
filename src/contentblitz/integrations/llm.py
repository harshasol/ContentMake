"""The only module that talks to model providers. See docs/adr/0002."""

import litellm
from pydantic import BaseModel, ValidationError

from contentblitz.core.config import ModelConfig, get_model_config, get_settings

_callbacks_configured = False


class StructuredOutputError(ValueError):
    """The model's reply did not fit the schema, even after one retry."""


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


def complete_structured[T: BaseModel](
    agent: str, messages: list[dict], schema: type[T], config: ModelConfig | None = None
) -> T:
    """Complete into `schema`, retrying once with the validation error if the reply doesn't fit.

    Raises StructuredOutputError if the second reply is also invalid. Provider errors propagate.
    """
    reply = complete(agent, messages, config=config, response_format=schema)
    try:
        return schema.model_validate_json(reply)
    except ValidationError as error:
        retry = [
            *messages,
            {"role": "assistant", "content": reply},
            {
                "role": "user",
                "content": f"That reply did not match the schema:\n{error}\nReply again with "
                "valid JSON only.",
            },
        ]
    reply = complete(agent, retry, config=config, response_format=schema)
    try:
        return schema.model_validate_json(reply)
    except ValidationError as error:
        raise StructuredOutputError(str(error)) from error
