"""The model response cache (spec #24, ticket #74): LiteLLM's disk cache under the data folder."""

import base64
import io
import os
import subprocess
import sys
import time
import types

import diskcache.core
import litellm
import pytest
from litellm import CustomLLM, ModelResponse, Usage
from litellm.types.utils import ImageObject
from PIL import Image
from pydantic import BaseModel

from contentblitz.core import cache
from contentblitz.core.config import DEFAULT_CONFIG_PATH, get_settings, load_model_config
from contentblitz.core.usage import get_ledger
from contentblitz.integrations import llm

MESSAGES = [{"role": "user", "content": "hello"}]
REAL_IMAGE_GENERATION = litellm.image_generation  # tests/conftest.py blocks it for every test


class FakeProvider(CustomLLM):
    """The provider below LiteLLM's cache; `calls` counts what actually reaches it."""

    def __init__(self):
        self.calls = []
        self.images = 0
        self.text = "ok"

    def completion(self, model, messages, **kwargs):
        self.calls.append(model)
        return ModelResponse(
            model=model,
            choices=[{"message": {"role": "assistant", "content": self.text}}],
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )

    async def acompletion(self, model, messages, **kwargs):
        return self.completion(model, messages, **kwargs)

    def image_generation(self, model, prompt, model_response, optional_params, logging_obj, **kw):
        self.images += 1
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, "PNG")
        model_response.data = [ImageObject(b64_json=base64.b64encode(buffer.getvalue()).decode())]
        return model_response


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "fake", "custom_handler": fake}]
    )
    return fake


def _config(**options):
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.agents["blog_writer"].models = ["fake/primary"]
    config.cache.model_responses = config.cache.model_responses.model_copy(update=options)
    return config


def test_a_second_identical_completion_is_served_from_the_cache_at_zero_cost(provider):
    config = _config()

    first = llm.complete("blog_writer", MESSAGES, config)
    second = llm.complete("blog_writer", MESSAGES, config)

    assert first == second == "ok"
    assert len(provider.calls) == 1
    fresh, hit = get_ledger().events
    assert (fresh.cached, hit.cached) == (False, True)
    assert hit.cost_usd == 0.0
    assert (hit.agent, hit.kind) == ("blog_writer", "completion")


def test_a_different_request_is_not_served_from_the_cache(provider):
    config = _config()
    llm.complete("blog_writer", MESSAGES, config)
    llm.complete("blog_writer", [{"role": "user", "content": "other"}], config)

    assert len(provider.calls) == 2


class Answer(BaseModel):
    value: int


def test_structured_completions_are_cached_too(provider):
    provider.text = '{"value": 3}'
    config = _config()

    for _ in range(2):
        assert llm.complete_structured("blog_writer", MESSAGES, Answer, config) == Answer(value=3)

    assert len(provider.calls) == 1
    assert [e.cached for e in get_ledger().events] == [False, True]


def test_switching_the_cache_off_sends_every_call_to_the_provider(provider):
    config = _config(enabled=False)

    llm.complete("blog_writer", MESSAGES, config)
    llm.complete("blog_writer", MESSAGES, config)

    assert len(provider.calls) == 2
    assert not any(e.cached for e in get_ledger().events)


def test_switching_the_cache_off_after_use_stops_serving_hits(provider):
    llm.complete("blog_writer", MESSAGES, _config())
    llm.complete("blog_writer", MESSAGES, _config(enabled=False))

    assert len(provider.calls) == 2


def test_cached_entries_expire_after_the_configured_time(provider, monkeypatch):
    config = _config(expiry_hours=1)
    llm.complete("blog_writer", MESSAGES, config)

    real = time.time
    later = types.SimpleNamespace(
        time=lambda: real() + 2 * 3600, sleep=time.sleep, monotonic=time.monotonic
    )
    monkeypatch.setattr(diskcache.core, "time", later)
    llm.complete("blog_writer", MESSAGES, config)

    assert len(provider.calls) == 2


def test_the_cache_lives_under_the_data_folder_and_survives_a_new_process(provider):
    llm.complete("blog_writer", MESSAGES, _config())

    data = get_settings().contentblitz_data_dir
    folder = data / "cache" / "model_responses"
    assert folder.is_dir() and any(folder.iterdir())
    assert cache.cache_dir("model_responses") == folder

    script = (
        "import litellm\n"
        "from contentblitz.core.config import DEFAULT_CONFIG_PATH, load_model_config\n"
        "from contentblitz.integrations import llm\n"
        "class P(litellm.CustomLLM):\n"
        "    def completion(self, *a, **k): raise SystemExit('reached the provider')\n"
        "    async def acompletion(self, *a, **k): raise SystemExit('reached the provider')\n"
        "litellm.custom_provider_map = [{'provider': 'fake', 'custom_handler': P()}]\n"
        "c = load_model_config(DEFAULT_CONFIG_PATH)\n"
        "c.agents['blog_writer'].models = ['fake/primary']\n"
        "print(llm.complete('blog_writer', [{'role': 'user', 'content': 'hello'}], c))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "CONTENTBLITZ_DATA_DIR": str(data), "PYTHONPATH": "src"},
    )
    assert result.stdout.strip().endswith("ok"), result.stderr[-500:]


def test_image_generation_is_never_served_from_the_cache(provider, monkeypatch):
    llm.complete("blog_writer", MESSAGES, _config())  # the cache is on
    monkeypatch.setattr(litellm, "image_generation", REAL_IMAGE_GENERATION)

    for _ in range(2):
        litellm.image_generation(model="fake/img", prompt="a cat")

    assert provider.images == 2


def test_cache_settings_have_a_switch_a_folder_and_an_expiry_per_cache():
    config = load_model_config(DEFAULT_CONFIG_PATH)

    assert set(cache.CACHE_NAMES) == {"model_responses", "search", "page_text", "research_brief"}
    expiry = {name: cache.options(name, config).expiry_seconds for name in cache.CACHE_NAMES}
    assert expiry == {
        "model_responses": 24 * 3600,
        "search": 24 * 3600,
        "page_text": 7 * 24 * 3600,
        "research_brief": None,  # its key changes whenever the research does
    }
    for name in cache.CACHE_NAMES:
        assert cache.options(name, config).enabled
        assert cache.cache_dir(name) == get_settings().contentblitz_data_dir / "cache" / name


def test_the_cache_folder_name_is_configurable():
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.cache.folder = "elsewhere"

    assert cache.cache_dir("search", config) == (
        get_settings().contentblitz_data_dir / "elsewhere" / "search"
    )


def test_concurrent_calls_build_the_litellm_cache_once(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    built = []
    real = litellm.Cache

    def slow_cache(*args, **kwargs):
        time.sleep(0.05)
        built.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(litellm, "Cache", slow_cache)
    config = load_model_config(DEFAULT_CONFIG_PATH)

    with ThreadPoolExecutor(8) as pool:
        for future in [pool.submit(llm._configure_cache, config) for _ in range(8)]:
            future.result()

    assert len(built) == 1
