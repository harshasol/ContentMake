"""The only module that talks to model providers. See docs/adr/0002."""

import base64
import io
from dataclasses import dataclass

import httpx
import litellm
from PIL import Image, ImageStat
from pydantic import BaseModel, ValidationError

from contentblitz.core.config import Aspect, ModelConfig, get_model_config, get_settings

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


class ImageGenerationError(RuntimeError):
    """An image model gave no usable image: no data, or a URL that couldn't be downloaded."""


class ImageCheckError(ImageGenerationError):
    """An image model's result failed a check: not a PNG, JPEG or WebP, too small, or blank."""


class NoUsableImage(ImageGenerationError):
    """Every image model failed; `fallbacks` says how each one did, in the order tried."""

    def __init__(self, fallbacks: tuple[str, ...]):
        super().__init__("no image model gave a usable image: " + "; ".join(fallbacks))
        self.fallbacks = fallbacks


@dataclass(frozen=True)
class GeneratedImage:
    """An image model's result: the image bytes, the model that made it, the provider's URL
    when it returned one (provider URLs expire, so the bytes are what gets kept), and the
    models tried and failed before it, each as "model: outcome"."""

    data: bytes
    model: str
    provider_url: str | None = None
    fallbacks: tuple[str, ...] = ()


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
            grey = image.convert("L").resize((64, 64))
    except Exception as error:  # noqa: BLE001 - any decoding failure means it isn't an image
        raise ImageCheckError("not an image") from error
    if kind not in IMAGE_FORMATS:
        raise ImageCheckError(f"{kind} is not PNG, JPEG or WebP")
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ImageCheckError(f"{width}×{height} is smaller than {MIN_IMAGE_SIDE} px")
    if ImageStat.Stat(grey).stddev[0] < MIN_PIXEL_STDDEV:
        raise ImageCheckError("nearly a single colour")


def has_api_key(model: str) -> bool:
    """Whether the model's provider has an API key in the settings. A provider the settings
    don't name is assumed configured; its call fails over to the next model if not."""
    provider = model.split("/", 1)[0]
    field = f"{provider}_api_key"
    settings = get_settings()
    return bool(getattr(settings, field)) if field in type(settings).model_fields else True


def _outcome(model: str, error: Exception) -> str:
    if isinstance(error, ImageCheckError):
        outcome = f"failed check: {error}"
    elif isinstance(error, ImageGenerationError):
        outcome = str(error)
    else:
        outcome = f"{type(error).__name__}: {error}"
    return f"{model}: {outcome}"[:OUTCOME_CHARS]


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
    fallbacks: list[str] = []
    for model in config.image.models:
        if not has_api_key(model):
            fallbacks.append(f"{model}: no API key configured")
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
    _configure_callbacks()
    config = config or get_model_config()
    # gpt-image models always return base64 and reject response_format; DALL·E needs asking.
    base64_format = {} if "gpt-image" in model else {"response_format": "b64_json"}
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
