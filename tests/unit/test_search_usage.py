"""Searches in the usage ledger (spec #24, ticket #72)."""

import pytest

from contentblitz.core import call_context
from contentblitz.core.config import get_model_config
from contentblitz.core.usage import get_ledger
from contentblitz.integrations.search import (
    SearchResult,
    SearchUnavailableError,
    search_with_fallback,
)


class Failing:
    name = "SerpAPI"

    def search(self, query, max_results=10, fresh=False):
        raise RuntimeError("quota exceeded")


class Working:
    name = "Tavily"

    def search(self, query, max_results=10, fresh=False):
        return [SearchResult(title="t", url="https://example.com", snippet="s")]


def test_a_search_records_the_provider_that_answered_at_zero_cost():
    with call_context.run("conv", "run-1"), call_context.agent("research"):
        search_with_fallback([Working()], "ai")

    (event,) = get_ledger().events
    assert (event.kind, event.model, event.primary) == ("search", "Tavily", "Tavily")
    assert (event.agent, event.run_id, event.cost_usd, event.failed) == ("research", "run-1", 0, [])
    assert event.latency_s >= 0
    assert not event.fallback


def test_a_search_fallback_records_the_failed_provider():
    search_with_fallback([Failing(), Working()], "ai")

    (event,) = get_ledger().events
    assert (event.model, event.primary) == ("Tavily", "SerpAPI")
    assert event.failed == [{"provider": "SerpAPI", "error": "quota exceeded"}]
    assert event.fallback


def test_a_search_no_provider_answers_is_still_recorded():
    with pytest.raises(SearchUnavailableError):
        search_with_fallback([Failing()], "ai")

    (event,) = get_ledger().events
    assert (event.model, event.failed) == (
        "none",
        [{"provider": "SerpAPI", "error": "quota exceeded"}],
    )


def test_a_search_costs_the_configured_per_search_price():
    get_model_config().search.prices["tavily"] = 0.008

    search_with_fallback([Working()], "ai")

    (event,) = get_ledger().events
    assert (event.cost_usd, event.estimated) == (0.008, False)


def test_a_search_outside_a_run_gets_a_placeholder_agent():
    search_with_fallback([Working()], "ai")

    assert get_ledger().events[0].agent == "search"
