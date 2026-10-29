"""Fallbacks, backoff and estimated prices in the usage ledger (spec #24, ticket #72)."""

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from contentblitz.core.config import DEFAULT_CONFIG_PATH, TokenPrice, load_model_config
from contentblitz.core.usage import get_ledger
from contentblitz.integrations import llm

MESSAGES = [{"role": "user", "content": "x"}]


def _response(content="hi", tokens=(100, 50), cost=0.002):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model="gpt-4o-2024-08-06",  # providers report dated names, never the configured one
        usage=SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1]),
        _hidden_params={"response_cost": cost},
    )


def _config():
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.agents["blog_writer"].models = ["openai/gpt-4o", "anthropic/claude-x"]
    return config


def _scripted(monkeypatch, *outcomes):
    """Each litellm.completion call pops an outcome: a response, or an exception to raise."""
    queue = list(outcomes)
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm.litellm, "completion", fake)
    return calls


def test_a_failed_primary_falls_back_and_the_event_captures_its_error(monkeypatch):
    calls = _scripted(monkeypatch, RuntimeError("429 rate limited"), _response())

    assert llm.complete("blog_writer", MESSAGES, config=_config()) == "hi"

    assert [c["model"] for c in calls] == ["openai/gpt-4o", "anthropic/claude-x"]
    (event,) = get_ledger().events
    assert (event.model, event.primary) == ("anthropic/claude-x", "openai/gpt-4o")
    assert event.failed == [{"provider": "openai/gpt-4o", "error": "429 rate limited"}]
    assert event.fallback


def test_a_dated_model_name_from_the_provider_is_not_a_fallback(monkeypatch):
    _scripted(monkeypatch, _response())

    llm.complete("blog_writer", MESSAGES, config=_config())

    (event,) = get_ledger().events
    assert (event.model, event.fallback) == ("openai/gpt-4o", False)


def test_every_model_failing_raises_the_last_error_and_records_nothing(monkeypatch):
    _scripted(monkeypatch, RuntimeError("first"), ValueError("second"))

    with pytest.raises(ValueError, match="second"):
        llm.complete("blog_writer", MESSAGES, config=_config())

    assert get_ledger().events == []


def test_retries_use_the_configured_backoff(monkeypatch):
    calls = _scripted(monkeypatch, _response())
    config = _config()
    config.llm.num_retries = 3
    config.llm.retry_strategy = "constant_retry"

    llm.complete("blog_writer", MESSAGES, config=config)

    assert (calls[0]["num_retries"], calls[0]["retry_strategy"]) == (3, "constant_retry")


def test_the_default_retry_strategy_is_exponential_backoff():
    assert load_model_config(DEFAULT_CONFIG_PATH).llm.retry_strategy == "exponential_backoff_retry"


def test_an_unpriced_model_is_estimated_from_the_configured_per_1k_prices(monkeypatch):
    _scripted(monkeypatch, _response(tokens=(2000, 1000), cost=0.0))
    config = _config()
    config.llm.token_prices = {"openai/gpt-4o": TokenPrice(input_per_1k=0.01, output_per_1k=0.03)}

    llm.complete("blog_writer", MESSAGES, config=config)

    (event,) = get_ledger().events
    assert event.cost_usd == pytest.approx(0.05)  # 2 * 0.01 + 1 * 0.03
    assert event.estimated


def test_a_priced_model_is_not_flagged_as_estimated(monkeypatch):
    _scripted(monkeypatch, _response())
    config = _config()
    config.llm.token_prices = {"openai/gpt-4o": TokenPrice(input_per_1k=9, output_per_1k=9)}

    llm.complete("blog_writer", MESSAGES, config=config)

    (event,) = get_ledger().events
    assert (event.cost_usd, event.estimated) == (0.002, False)


def test_an_unpriced_model_with_no_configured_price_costs_nothing_unflagged(monkeypatch):
    _scripted(monkeypatch, _response(cost=None))

    llm.complete("blog_writer", MESSAGES, config=_config())

    (event,) = get_ledger().events
    assert (event.cost_usd, event.estimated) == (0.0, False)


# Images


def _png():
    buffer = io.BytesIO()
    image = Image.new("RGB", (1536, 1024), (30, 90, 160))
    image.paste((240, 200, 60), (0, 0, 700, 500))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_response(cost=0.042):
    result = SimpleNamespace(b64_json=base64.b64encode(_png()).decode(), url=None)
    return SimpleNamespace(data=[result], _hidden_params={"response_cost": cost})


def _fake_images(monkeypatch, outcomes):
    def fake(**kwargs):
        outcome = outcomes[kwargs["model"]]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm.litellm, "image_generation", fake)


@pytest.fixture(autouse=True)
def openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def test_an_image_records_an_image_event_with_its_model_and_cost(monkeypatch):
    _fake_images(monkeypatch, {"openai/gpt-image-1": _image_response(0.042)})

    llm.generate_image("a cat", "landscape")

    (event,) = get_ledger().events
    assert (event.kind, event.model, event.cost_usd) == ("image", "openai/gpt-image-1", 0.042)
    assert (event.agent, event.estimated, event.fallback) == ("image_generator", False, False)


def test_an_image_fallback_captures_the_failed_models_error(monkeypatch):
    _fake_images(
        monkeypatch,
        {"openai/gpt-image-1": RuntimeError("content policy"), "openai/dall-e-3": _image_response()},
    )

    llm.generate_image("a cat", "landscape")

    (event,) = get_ledger().events
    assert (event.model, event.primary) == ("openai/dall-e-3", "openai/gpt-image-1")
    assert event.failed == [{"provider": "openai/gpt-image-1", "error": "RuntimeError: content policy"}]


def test_an_unpriced_image_uses_the_configured_per_image_price_as_an_estimate(monkeypatch):
    _fake_images(monkeypatch, {"openai/gpt-image-1": _image_response(cost=0.0)})
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.image.prices = {"openai/gpt-image-1": 0.04}

    llm.generate_image("a cat", "landscape", config)

    (event,) = get_ledger().events
    assert (event.cost_usd, event.estimated) == (0.04, True)


def test_the_shipped_config_prices_every_image_model():
    config = load_model_config(DEFAULT_CONFIG_PATH)
    assert set(config.image.models) <= set(config.image.prices)
