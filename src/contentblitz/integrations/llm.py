"""The only module that talks to model providers. See docs/adr/0002."""

import base64
import io
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict

import httpx
import litellm
from PIL import Image, ImageStat
from pydantic import BaseModel, ValidationError

from contentblitz.core import call_context, tracing, usage
from contentblitz.core.config import Aspect, ModelConfig, get_model_config, get_settings


class StructuredOutputError(ValueError):
    """The model's reply did not fit the schema, even after one retry."""


@dataclass(frozen=True)
class Reply[T]:
    """A model's reply, with the model that gave it (a fallback, if the primary failed)."""

    value: T
    model: str


def _tokens(response, field: str) -> int:
    return int(getattr(getattr(response, "usage", None), field, 0) or 0)


def _cost(response) -> float:
    """LiteLLM's cost for the response; 0 when it has none or can't price the model."""
    cost = (getattr(response, "_hidden_params", None) or {}).get("response_cost")
    if cost is None:
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:  # noqa: BLE001 - an unpriced model must not fail the call
            cost = 0.0
    return float(cost or 0.0)


def _annotate(span, model: str, response) -> None:
    """Put the model, tokens, cost and cache hit on the model call's span."""
    tokens_in = _tokens(response, "prompt_tokens")
    tokens_out = _tokens(response, "completion_tokens")
    hidden = getattr(response, "_hidden_params", None) or {}
    span.add_metadata(
        {
            "model": model,
            "ls_model_name": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": _cost(response),
            "cache_hit": bool(hidden.get("cache_hit")),
        }
    )
    span.end(
        outputs={
            "usage_metadata": {
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            }
        }
    )


def _record_completion(agent: str, primary: str, answered_by: str, response, started: float):
    """Append the call to the usage ledger, tagged from the call context (spec #24)."""
    context = call_context.current()
    usage.record(
        usage.UsageEvent(
            ts=datetime.now(UTC).isoformat(),
            run_id=context.run_id,
            conversation_id=context.conversation_id,
            agent=agent,
            kind="completion",
            model=answered_by,
            primary=primary,
            tokens_in=_tokens(response, "prompt_tokens"),
            tokens_out=_tokens(response, "completion_tokens"),
            cost_usd=_cost(response),
            latency_s=round(time.perf_counter() - started, 3),
        )
    )


def complete(agent: str, messages: list[dict], config: ModelConfig | None = None, **kwargs) -> str:
    """Run a chat completion for `agent`, falling back through its configured models."""
    return _complete(agent, messages, config, **kwargs).value


def _complete(agent: str, messages: list[dict], config: ModelConfig | None, **kwargs) -> Reply[str]:
    config = config or get_model_config()
    models = config.for_agent(agent)
    started = time.perf_counter()
    with tracing.span(
        f"{agent} model call", "llm", inputs={"messages": messages}, metadata={"agent": agent}
    ) as span:
        response = litellm.completion(
            model=models.primary,
            fallbacks=models.fallbacks or None,
            messages=messages,
            num_retries=config.llm.num_retries,
            timeout=config.llm.timeout_seconds,
            **kwargs,
        )
        answered_by = getattr(response, "model", None) or models.primary
        _annotate(span, answered_by, response)
    _record_completion(agent, models.primary, answered_by, response, started)
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


class ImageGenerationError(RuntimeError):
    """An image model gave no usable image: no data, or a URL that couldn't be downloaded."""


class ImageCheckError(ImageGenerationError):
    """An image model's result failed a check: not a PNG, JPEG or WebP, too small, or blank."""


class ModelOutcome(TypedDict):
    """How one image model did when it was tried: why it failed, or "used"."""

    model: str
    outcome: str


USED = "used"


class NoUsableImage(ImageGenerationError):
    """Every image model failed; `fallbacks` says how each one did, in the order tried."""

    def __init__(self, fallbacks: tuple[ModelOutcome, ...]):
        reasons = "; ".join(f"{f['model']}: {f['outcome']}" for f in fallbacks)
        super().__init__(f"no image model gave a usable image: {reasons}")
        self.fallbacks = fallbacks


@dataclass(frozen=True)
class GeneratedImage:
    """An image model's result: the image bytes, the model that made it, the provider's URL
    when it returned one (provider URLs expire, so the bytes are what gets kept), and the
    models tried and failed before it, each with its outcome."""

    data: bytes
    model: str
    provider_url: str | None = None
    fallbacks: tuple[ModelOutcome, ...] = ()

    def models_tried(self) -> list[ModelOutcome]:
        """Each model tried, in order, with its outcome: the failed ones, then the one used."""
        return [*self.fallbacks, {"model": self.model, "outcome": USED}]


IMAGE_FORMATS = {"PNG": "PNG", "JPEG": "JPEG", "WEBP": "WebP"}
MIN_IMAGE_SIDE = 512
MIN_PIXEL_STDDEV = 2.0  # below this, the image is nearly a single colour
OUTCOME_CHARS = 200


def check_image(data: bytes) -> None:
    """Raises ImageCheckError unless `data` decodes as a PNG, JPEG or WebP image whose sides
    are both at least 512 px and which isn't nearly a single colour."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            kind, (width, height) = image.format, image.size
            sample = image.convert("RGB").resize((64, 64))
    except Exception as error:  # noqa: BLE001 - any decoding failure means it isn't an image
        raise ImageCheckError("not an image") from error
    if kind not in IMAGE_FORMATS:
        raise ImageCheckError(f"{kind} is not PNG, JPEG or WebP")
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ImageCheckError(f"{width}×{height} is smaller than {MIN_IMAGE_SIDE} px")
    # Each colour channel is measured, so colours of equal brightness still count as different.
    if max(ImageStat.Stat(sample).stddev) < MIN_PIXEL_STDDEV:
        raise ImageCheckError("nearly a single colour")


def has_api_key(model: str) -> bool:
    """Whether the model's provider has an API key in the settings. A provider the settings
    don't name is assumed configured; its call fails over to the next model if not."""
    provider = model.split("/", 1)[0]
    field = f"{provider}_api_key"
    settings = get_settings()
    return bool(getattr(settings, field)) if field in type(settings).model_fields else True


def _outcome(model: str, error: Exception) -> ModelOutcome:
    if isinstance(error, ImageCheckError):
        outcome = f"failed check: {error}"
    elif isinstance(error, ImageGenerationError):
        outcome = str(error)
    else:
        outcome = f"{type(error).__name__}: {error}"
    return {"model": model, "outcome": outcome[:OUTCOME_CHARS]}


def generate_image(
    prompt: str,
    aspect: Aspect,
    config: ModelConfig | None = None,
    client: httpx.Client | None = None,
) -> GeneratedImage:
    """Generate one image, trying the configured image models in order, each at its own size
    for `aspect`, until one gives an image that passes `check_image`. A model whose provider
    has no API key is skipped. Raises NoUsableImage, listing each outcome, when none does."""
    config = config or get_model_config()
    fallbacks: list[ModelOutcome] = []
    for model in config.image.models:
        if not has_api_key(model):
            fallbacks.append({"model": model, "outcome": "no API key configured"})
            continue
        try:
            image = generate_image_with(model, prompt, aspect, config, client)
            check_image(image.data)
        except Exception as error:  # noqa: BLE001 - any failure moves on to the next model
            fallbacks.append(_outcome(model, error))
            continue
        return GeneratedImage(image.data, image.model, image.provider_url, tuple(fallbacks))
    raise NoUsableImage(tuple(fallbacks))


def generate_image_with(
    model: str,
    prompt: str,
    aspect: Aspect,
    config: ModelConfig | None = None,
    client: httpx.Client | None = None,
) -> GeneratedImage:
    """Generate one image with `model`, unchecked. Asks for base64 output, and downloads the
    image when the model returns a URL instead. Raises ImageGenerationError for no image."""
    config = config or get_model_config()
    # gpt-image models always return base64 and reject response_format; DALL·E needs asking.
    base64_format = {} if "gpt-image" in model else {"response_format": "b64_json"}
    with tracing.span(
        "image model call", "llm", inputs={"prompt": prompt}, metadata={"model": model}
    ):
        response = litellm.image_generation(
            model=model,
            prompt=prompt,
            n=1,
            size=config.image.size_for(model, aspect),
            quality=config.image.quality_for(model),
            timeout=config.llm.timeout_seconds,
            **base64_format,
        )
    result = next(iter(getattr(response, "data", None) or []), None)
    if b64 := getattr(result, "b64_json", None):
        return GeneratedImage(base64.b64decode(b64), model)
    if url := getattr(result, "url", None):
        return GeneratedImage(_download(url, config.llm.timeout_seconds, client), model, url)
    raise ImageGenerationError(f"{model} returned no image")


def _download(url: str, timeout: float, client: httpx.Client | None) -> bytes:
    try:
        if client is not None:
            response = client.get(url, timeout=timeout)
        else:
            response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise ImageGenerationError(f"couldn't download the image: {error}") from error
    return response.content
