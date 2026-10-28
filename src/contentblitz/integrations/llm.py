"""The only module that talks to model providers. See docs/adr/0002."""

from dataclasses import dataclass

import litellm
from pydantic import BaseModel, ValidationError

from contentblitz.core.config import ModelConfig, get_model_config, get_settings

_callbacks_configured = False


class StructuredOutputError(ValueError):
    """The model's reply did not fit the schema, even after one retry."""


@dataclass(frozen=True)
class Reply[T]:
    """A model's reply, with the model that gave it (a fallback, if the primary failed)."""

    value: T
    model: str


def _configure_callbacks() -> None:
    global _callbacks_configured
    if not _callbacks_configured and get_settings().langsmith_tracing:
        litellm.success_callback = [*litellm.success_callback, "langsmith"]
    _callbacks_configured = True


def complete(agent: str, messages: list[dict], config: ModelConfig | None = None, **kwargs) -> str:
    """Run a chat completion for `agent`, falling back through its configured models."""
    return _complete(agent, messages, config, **kwargs).value


def _complete(agent: str, messages: list[dict], config: ModelConfig | None, **kwargs) -> Reply[str]:
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
    answered_by = getattr(response, "model", None) or models.primary
    return Reply(response.choices[0].message.content or "", answered_by)


def complete_structured[T: BaseModel](
    agent: str, messages: list[dict], schema: type[T], config: ModelConfig | None = None
) -> T:
    """Complete into `schema`, retrying once with the validation error if the reply doesn't fit.

    Raises StructuredOutputError if the second reply is also invalid. Provider errors propagate.
    """
    return complete_structured_reply(agent, messages, schema, config).value


def complete_structured_reply[T: BaseModel](
    agent: str, messages: list[dict], schema: type[T], config: ModelConfig | None = None
) -> Reply[T]:
    """Like complete_structured, but also says which model gave the reply that fit."""
    reply = _complete(agent, messages, config, response_format=schema)
    try:
        return Reply(schema.model_validate_json(reply.value), reply.model)
    except ValidationError as error:
        retry = [
            *messages,
            {"role": "assistant", "content": reply.value},
            {
                "role": "user",
                "content": f"That reply did not match the schema:\n{error}\nReply again with "
                "valid JSON only.",
            },
        ]
    reply = _complete(agent, retry, config, response_format=schema)
    try:
        return Reply(schema.model_validate_json(reply.value), reply.model)
    except ValidationError as error:
        raise StructuredOutputError(str(error)) from error
