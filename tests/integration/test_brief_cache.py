"""The Research Brief cache (spec #24, ticket #76).

A Content Strategist step whose key (Sources, instructions, Strategist model) matches a cached
Brief reuses it without a model call. Helpers come from the Research Brief tests.
"""

import pytest
from test_research_brief import (
    HEALTHCARE,
    RESEARCH_ONLY,
    FakeStrategist,
    draft,
    research_record,
    run_plan,
    topic,
)

from contentblitz.core import cache
from contentblitz.core import config as core_config
from contentblitz.core.usage import get_ledger
from contentblitz.integrations import llm
from contentblitz.workflow import brief_cache


@pytest.fixture
def model(monkeypatch):
    fake = FakeStrategist(draft())
    monkeypatch.setattr(llm, "complete_structured_reply", fake)
    return fake


def with_config(monkeypatch, agents=None, **brief_changes):
    """Make the cache and the Strategist read the config changed as asked."""
    base = core_config.get_model_config()
    changes = {
        "cache": base.cache.model_copy(
            update={"research_brief": base.cache.research_brief.model_copy(update=brief_changes)}
        )
    }
    if agents is not None:
        changes["agents"] = {**base.agents, **agents}
    updated = base.model_copy(update=changes)
    monkeypatch.setattr(cache, "get_model_config", lambda: updated)
    monkeypatch.setattr(brief_cache, "get_model_config", lambda: updated)
    return updated


def brief_of(state):
    return state["topics"]["topic-1"]["brief"]


def test_a_matching_key_reuses_the_cached_brief_without_a_model_call(model):
    _, first = run_plan(RESEARCH_ONLY, {"topic-1": topic()})
    assert len(model.calls) == 1

    _, second = run_plan(RESEARCH_ONLY, {"topic-1": topic()}, thread="t2")

    assert len(model.calls) == 1
    assert brief_of(second) == brief_of(first)


def test_different_sources_instructions_or_model_miss_the_cache(model, monkeypatch):
    run_plan(RESEARCH_ONLY, {"topic-1": topic()})
    assert len(model.calls) == 1

    other_text = research_record("who.int", "nature.com", "fda.gov")
    other_text["sources"][1]["page_text"] = "New text."
    run_plan(RESEARCH_ONLY, {"topic-1": topic(record=other_text)}, thread="t2")
    assert len(model.calls) == 2

    other_url = research_record("who.int", "nature.com", "cdc.gov")
    other_url["sources"][1]["page_text"] = HEALTHCARE["sources"][1]["page_text"]
    run_plan(RESEARCH_ONLY, {"topic-1": topic(record=other_url)}, thread="t3")
    assert len(model.calls) == 3

    focused = [{"agent": "content_strategist", "instructions": "focus on regulation"}]
    run_plan(focused, {"topic-1": topic()}, thread="t4")
    assert len(model.calls) == 4

    strategist = core_config.get_model_config().agents["content_strategist"]
    other = strategist.model_copy(update={"models": ["anthropic/claude-sonnet-4"]})
    with_config(monkeypatch, agents={"content_strategist": other})
    run_plan(RESEARCH_ONLY, {"topic-1": topic()}, thread="t5")
    assert len(model.calls) == 5


def test_changing_only_the_fallback_models_still_hits(model, monkeypatch):
    run_plan(RESEARCH_ONLY, {"topic-1": topic()})
    strategist = core_config.get_model_config().agents["content_strategist"]
    other = strategist.model_copy(update={"models": [strategist.models[0], "other/fallback"]})
    with_config(monkeypatch, agents={"content_strategist": other})

    run_plan(RESEARCH_ONLY, {"topic-1": topic()}, thread="t2")

    assert len(model.calls) == 1


def test_a_brief_cache_hit_is_recorded_as_a_cached_free_completion(model):
    run_plan(RESEARCH_ONLY, {"topic-1": topic()})
    assert get_ledger().events == []  # the fake model call records nothing, and a miss neither

    run_plan(RESEARCH_ONLY, {"topic-1": topic()}, thread="t2")

    [hit] = get_ledger().events
    primary = core_config.get_model_config().for_agent("content_strategist").primary
    assert (hit.agent, hit.kind, hit.model, hit.primary) == (
        "content_strategist",
        "completion",
        primary,
        primary,
    )
    assert (hit.cached, hit.cost_usd, hit.tokens_in, hit.tokens_out) == (True, 0.0, 0, 0)
    assert get_ledger().cache_hit_rate() == 1.0
    assert [(b.name, b.cached) for b in get_ledger().by_agent()] == [("content_strategist", 1)]


def test_the_brief_cache_survives_a_new_process(model):
    run_plan(RESEARCH_ONLY, {"topic-1": topic()})

    cache.close_all()  # a new process has opened nothing yet
    run_plan(RESEARCH_ONLY, {"topic-1": topic()}, thread="t2")

    assert len(model.calls) == 1


def test_the_brief_cache_can_be_switched_off_in_config(model, monkeypatch):
    with_config(monkeypatch, enabled=False)

    run_plan(RESEARCH_ONLY, {"topic-1": topic()})
    run_plan(RESEARCH_ONLY, {"topic-1": topic()}, thread="t2")

    assert len(model.calls) == 2
    assert not cache.cache_dir("research_brief").exists()


def test_a_failed_brief_is_not_cached(model):
    model.reply = RuntimeError("down")
    run_plan(RESEARCH_ONLY, {"topic-1": topic()})
    model.reply = draft()
    run_plan(RESEARCH_ONLY, {"topic-1": topic()}, thread="t2")

    assert len(model.calls) == 2


@pytest.fixture(autouse=True)
def _close_caches():
    yield
    cache.close_all()
