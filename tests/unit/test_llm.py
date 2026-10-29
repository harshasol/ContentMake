import base64
import io
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from pydantic import BaseModel

from contentblitz.core.config import DEFAULT_CONFIG_PATH, load_model_config
from contentblitz.integrations import llm


def _reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_complete_tries_the_primary_first_and_leaves_fallbacks_to_the_loop(monkeypatch):
    calls = {}

    def fake_completion(**kwargs):
        calls.update(kwargs)
        return _reply("hi")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.agents["blog_writer"].models = ["openai/gpt-4o", "anthropic/claude-sonnet-4-20250514"]

    assert llm.complete("blog_writer", [{"role": "user", "content": "x"}], config=config) == "hi"
    assert calls["model"] == "openai/gpt-4o"
    assert "fallbacks" not in calls


class Answer(BaseModel):
    value: int


def _scripted_completion(monkeypatch, replies):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _reply(replies[len(calls) - 1])

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    return calls


def test_complete_structured_sends_schema_and_returns_validated_object(monkeypatch):
    calls = _scripted_completion(monkeypatch, ['{"value": 42}'])
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.agents["query_handler"].models = ["openai/gpt-4o-mini", "gemini/gemini-2.0-flash"]

    result = llm.complete_structured(
        "query_handler", [{"role": "user", "content": "x"}], Answer, config=config
    )

    assert result == Answer(value=42)
    assert calls[0]["response_format"] is Answer
    assert calls[0]["model"] == "openai/gpt-4o-mini"


def test_complete_structured_retries_once_with_the_validation_error(monkeypatch):
    calls = _scripted_completion(monkeypatch, ['{"value": "lots"}', '{"value": 7}'])

    result = llm.complete_structured("query_handler", [{"role": "user", "content": "x"}], Answer)

    assert result == Answer(value=7)
    assert len(calls) == 2
    retry = calls[1]["messages"]
    assert retry[:2] == [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": '{"value": "lots"}'},
    ]
    assert retry[2]["role"] == "user" and "value" in retry[2]["content"]


def test_complete_structured_gives_up_after_a_second_invalid_reply(monkeypatch):
    calls = _scripted_completion(monkeypatch, ["not json", '{"other": 1}', '{"value": 1}'])

    with pytest.raises(llm.StructuredOutputError):
        llm.complete_structured("query_handler", [{"role": "user", "content": "x"}], Answer)
    assert len(calls) == 2


def test_a_structured_reply_names_the_model_that_answered(monkeypatch):
    replies = [RuntimeError("down"), _reply('{"value": 7}')]

    def fake_completion(**kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.agents["query_handler"].models = ["openai/gpt-4o-mini", "claude-sonnet-4-20250514"]

    reply = llm.complete_structured_reply(
        "query_handler", [{"role": "user", "content": "x"}], Answer, config=config
    )

    assert reply == llm.Reply(value=Answer(value=7), model="claude-sonnet-4-20250514")


def test_a_structured_reply_falls_back_to_the_configured_model_when_none_is_named(monkeypatch):
    _scripted_completion(monkeypatch, ['{"value": 42}'])
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.agents["query_handler"].models = ["openai/gpt-4o-mini"]

    reply = llm.complete_structured_reply(
        "query_handler", [{"role": "user", "content": "x"}], Answer, config=config
    )

    assert reply.model == "openai/gpt-4o-mini"


def _png(size=(1536, 1024), blank=False, format="PNG"):
    """Image bytes as a provider would return them, made with Pillow."""
    buffer = io.BytesIO()
    image = Image.new("RGB", size, (30, 90, 160))
    if not blank:
        image.paste((240, 200, 60), (0, 0, size[0] // 2, size[1] // 2))
    image.save(buffer, format=format)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def openai_key(monkeypatch):
    """The image models are OpenAI's, which are only tried with a key configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _image_response(b64_json=None, url=None):
    return SimpleNamespace(data=[SimpleNamespace(b64_json=b64_json, url=url)])


def _scripted_image_generation(monkeypatch, response):
    calls = []

    def fake_image_generation(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(llm.litellm, "image_generation", fake_image_generation)
    return calls


def test_generate_image_decodes_a_base64_result_from_the_first_image_model(monkeypatch):
    png = _png()
    calls = _scripted_image_generation(
        monkeypatch, _image_response(b64_json=base64.b64encode(png).decode())
    )

    image = llm.generate_image("a robot doctor, no text", "landscape")

    assert image == llm.GeneratedImage(data=png, model="openai/gpt-image-1", provider_url=None)
    [call] = calls
    assert call["model"] == "openai/gpt-image-1"
    assert call["prompt"] == "a robot doctor, no text"
    assert (call["size"], call["quality"]) == ("1536x1024", "medium")


def test_generate_image_asks_dall_e_for_base64_at_its_own_size_and_quality(monkeypatch):
    calls = _scripted_image_generation(
        monkeypatch, _image_response(b64_json=base64.b64encode(_png()).decode())
    )
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.image.models = ["openai/dall-e-3"]

    llm.generate_image("a robot doctor", "portrait", config=config)

    [call] = calls
    assert call["model"] == "openai/dall-e-3"
    assert call["response_format"] == "b64_json"
    assert (call["size"], call["quality"]) == ("1024x1792", "standard")


def test_generate_image_downloads_a_url_result_and_keeps_the_provider_url(monkeypatch):
    png = _png((1024, 1024))
    url = "https://images.example.com/abc.png"
    _scripted_image_generation(monkeypatch, _image_response(url=url))
    requested = []

    def serve(request):
        requested.append(str(request.url))
        return httpx.Response(200, content=png, headers={"content-type": "image/png"})

    client = httpx.Client(transport=httpx.MockTransport(serve))
    image = llm.generate_image("a robot doctor", "square", client=client)

    assert requested == [url]
    assert image == llm.GeneratedImage(data=png, model="openai/gpt-image-1", provider_url=url)


def test_generate_image_raises_when_the_download_fails(monkeypatch):
    _scripted_image_generation(monkeypatch, _image_response(url="https://images.example.com/x"))
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(404)))

    with pytest.raises(llm.ImageGenerationError):
        llm.generate_image("a robot doctor", "square", client=client)


def test_generate_image_raises_when_the_result_has_no_image(monkeypatch):
    _scripted_image_generation(monkeypatch, SimpleNamespace(data=[]))

    with pytest.raises(llm.ImageGenerationError):
        llm.generate_image("a robot doctor", "square")


def _b64(data):
    return _image_response(b64_json=base64.b64encode(data).decode())


def _per_model_image_generation(monkeypatch, results):
    """Each image model returns (or raises) its own scripted result; the calls are kept."""
    calls = []

    def fake_image_generation(**kwargs):
        calls.append(kwargs["model"])
        result = results[kwargs["model"]]
        if isinstance(result, Exception):
            raise result
        return _b64(result)

    monkeypatch.setattr(llm.litellm, "image_generation", fake_image_generation)
    return calls


def _three_models():
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.image.models = ["openai/gpt-image-1", "openai/dall-e-3", "openai/dall-e-2"]
    return config


def test_image_models_are_tried_in_config_order_until_one_gives_a_usable_image(monkeypatch):
    png = _png()
    calls = _per_model_image_generation(
        monkeypatch,
        {
            "openai/gpt-image-1": RuntimeError("quota exceeded"),
            "openai/dall-e-3": RuntimeError("service unavailable"),
            "openai/dall-e-2": png,
        },
    )

    image = llm.generate_image("a robot doctor", "landscape", config=_three_models())

    assert calls == ["openai/gpt-image-1", "openai/dall-e-3", "openai/dall-e-2"]
    assert (image.data, image.model) == (png, "openai/dall-e-2")
    assert image.fallbacks == (
        {"model": "openai/gpt-image-1", "outcome": "RuntimeError: quota exceeded"},
        {"model": "openai/dall-e-3", "outcome": "RuntimeError: service unavailable"},
    )


def test_the_first_usable_image_stops_the_fallback(monkeypatch):
    calls = _per_model_image_generation(
        monkeypatch, {"openai/gpt-image-1": _png(), "openai/dall-e-3": _png()}
    )

    image = llm.generate_image("a robot doctor", "square")

    assert calls == ["openai/gpt-image-1"]
    assert image.fallbacks == ()


@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        (_png(blank=True), "failed check: nearly a single colour"),
        (_png((400, 1024)), "failed check: 400×1024 is smaller than 512 px"),
        (b"not an image", "failed check: not an image"),
        (_png(format="GIF"), "failed check: GIF is not PNG, JPEG or WebP"),
    ],
    ids=["blank", "tiny", "undecodable", "gif"],
)
def test_an_unusable_result_counts_as_a_failure_and_the_next_model_is_tried(
    monkeypatch, result, outcome
):
    png = _png()
    _per_model_image_generation(monkeypatch, {"openai/gpt-image-1": result, "openai/dall-e-3": png})

    image = llm.generate_image("a robot doctor", "square")

    assert (image.data, image.model) == (png, "openai/dall-e-3")
    assert image.fallbacks == ({"model": "openai/gpt-image-1", "outcome": f"{outcome}"},)


def test_an_image_in_colours_of_equal_brightness_is_not_rejected_as_blank():
    # Red and green halves that are the same grey once converted to greyscale.
    buffer = io.BytesIO()
    image = Image.new("RGB", (1024, 1024), (200, 0, 0))
    image.paste((0, 102, 0), (0, 0, 512, 1024))
    image.save(buffer, format="PNG")
    assert len(set(image.convert("L").getdata())) == 1

    llm.check_image(buffer.getvalue())


@pytest.mark.parametrize("format", ["JPEG", "WEBP"])
def test_jpeg_and_webp_results_are_accepted(monkeypatch, format):
    data = _png(format=format)
    _per_model_image_generation(monkeypatch, {"openai/gpt-image-1": data})

    assert llm.generate_image("a robot doctor", "square").data == data


def test_when_every_model_fails_the_error_lists_each_outcome(monkeypatch):
    _per_model_image_generation(
        monkeypatch,
        {"openai/gpt-image-1": RuntimeError("quota exceeded"), "openai/dall-e-3": _png((300, 300))},
    )

    with pytest.raises(llm.NoUsableImage) as raised:
        llm.generate_image("a robot doctor", "square")

    assert raised.value.fallbacks == (
        {"model": "openai/gpt-image-1", "outcome": "RuntimeError: quota exceeded"},
        {"model": "openai/dall-e-3", "outcome": "failed check: 300×300 is smaller than 512 px"},
    )


def test_a_model_without_its_api_key_is_skipped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    calls = _per_model_image_generation(monkeypatch, {"openai/gpt-image-1": _png()})

    with pytest.raises(llm.NoUsableImage) as raised:
        llm.generate_image("a robot doctor", "square")

    assert calls == []
    assert raised.value.fallbacks == (
        {"model": "openai/gpt-image-1", "outcome": "no API key configured"},
        {"model": "openai/dall-e-3", "outcome": "no API key configured"},
    )
