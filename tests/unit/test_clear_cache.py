"""Clear cache (spec #24, ticket #77): the workspace action empties every cache, not the ledger."""

import litellm
import pytest
from litellm import CustomLLM, ModelResponse, Usage

from contentblitz.core import cache
from contentblitz.core.config import DEFAULT_CONFIG_PATH, load_model_config
from contentblitz.core.usage import UsageEvent, get_ledger
from contentblitz.integrations import llm, search_cache
from contentblitz.workflow import brief_cache
from contentblitz.workflow.workspace import Workspace

MESSAGES = [{"role": "user", "content": "hello"}]


class FakeProvider(CustomLLM):
    def __init__(self):
        self.calls = 0

    def completion(self, model, messages, **kwargs):
        self.calls += 1
        return ModelResponse(
            model=model,
            choices=[{"message": {"role": "assistant", "content": "ok"}}],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def acompletion(self, model, messages, **kwargs):
        return self.completion(model, messages, **kwargs)


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "fake", "custom_handler": fake}]
    )
    return fake


def test_clear_cache_sends_the_next_identical_model_call_to_the_provider(provider):
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.agents["blog_writer"].models = ["fake/primary"]
    llm.complete("blog_writer", MESSAGES, config)
    llm.complete("blog_writer", MESSAGES, config)
    assert provider.calls == 1

    Workspace(None, "t").clear_cache()
    llm.complete("blog_writer", MESSAGES, config)

    assert provider.calls == 2


def test_clear_cache_empties_the_search_and_page_text_caches():
    search_cache.put("search", "k", ["result"])
    search_cache.put("page_text", "u", "text")

    Workspace(None, "t").clear_cache()

    assert search_cache.get("search", "k") is None
    assert search_cache.get("page_text", "u") is None


def test_clear_cache_empties_the_brief_cache():
    brief_cache._cache().set("k", "brief")

    Workspace(None, "t").clear_cache()

    assert brief_cache._cache().get("k") is None


def test_clear_cache_leaves_the_usage_ledger_alone():
    event = UsageEvent(ts="2026-11-04T10:00:00+00:00", agent="a", kind="completion", model="m")
    get_ledger().record(event)
    get_ledger().record_run("r1", "a request", "2026-11-04T10:00:00+00:00", 1.0)

    Workspace(None, "t").clear_cache()

    assert len(get_ledger().events) == 1
    assert get_ledger().runs


def test_clear_cache_closes_every_handle_the_shared_opener_holds():
    search_handle = cache.open_cache("search")
    brief_handle = cache.open_cache("research_brief")
    assert search_handle is cache.open_cache("search")

    Workspace(None, "t").clear_cache()

    assert cache._open == {}
    assert cache.open_cache("search") is not search_handle
    assert brief_handle is not cache.open_cache("research_brief")


def test_clear_cache_works_when_no_cache_exists_yet():
    Workspace(None, "t").clear_cache()
    assert not any(p.exists() for p in cache.cache_dirs())
