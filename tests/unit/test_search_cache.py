"""The search and page text caches (spec #24, ticket #75)."""

import httpx
import pytest

from contentblitz.core import cache
from contentblitz.core.config import get_model_config
from contentblitz.core.usage import get_ledger
from contentblitz.integrations import search, search_cache
from contentblitz.integrations.search import SearchResult, search_with_fallback

HOUR = 3600


class Counting:
    def __init__(self, name="Tavily", url="https://example.com/a"):
        self.name = name
        self.url = url
        self.calls = []

    def search(self, query, max_results=10, fresh=False):
        self.calls.append((query, max_results, fresh))
        return [SearchResult(title="t", url=self.url, snippet="s")]


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(search_cache, "clock", fake)
    return fake


@pytest.fixture(autouse=True)
def close_caches():
    yield
    search_cache.close_all()


def switch_off(monkeypatch, name):
    options = getattr(get_model_config().cache, name)
    monkeypatch.setattr(options, "enabled", False)


def test_a_repeated_search_comes_from_the_cache_and_is_recorded_as_cached():
    provider = Counting()
    first = search_with_fallback([provider], "AI in healthcare")
    second = search_with_fallback([provider], "AI in healthcare")

    assert len(provider.calls) == 1
    assert second.results == first.results and second.provider == "Tavily"
    live, hit = get_ledger().events
    assert (live.cached, hit.cached) == (False, True)
    assert (hit.kind, hit.model, hit.cost_usd) == ("search", "Tavily", 0)


def test_the_query_is_normalised_but_provider_and_max_results_are_part_of_the_key():
    provider = Counting()
    search_with_fallback([provider], "AI  in Healthcare ")
    search_with_fallback([provider], "ai in healthcare")
    assert len(provider.calls) == 1
    search_with_fallback([provider], "ai in healthcare", max_results=5)
    search_with_fallback([Counting(name="SerpAPI")], "ai in healthcare")
    assert len(provider.calls) == 2


def test_a_search_entry_expires_after_its_configured_time(clock):
    provider = Counting()
    expiry = get_model_config().cache.search.expiry_hours
    search_with_fallback([provider], "ai")
    clock.now += (expiry - 1) * HOUR
    search_with_fallback([provider], "ai")
    assert len(provider.calls) == 1
    clock.now += 2 * HOUR
    search_with_fallback([provider], "ai")
    assert len(provider.calls) == 2


def test_a_fresh_search_skips_the_cache_and_overwrites_the_entry():
    provider = Counting()
    search_with_fallback([provider], "ai")
    provider.url = "https://example.com/new"
    fresh = search_with_fallback([provider], "ai", fresh=True)
    assert len(provider.calls) == 2 and provider.calls[1][2] is True
    assert fresh.results[0].url == "https://example.com/new"

    again = search_with_fallback([provider], "ai")
    assert len(provider.calls) == 2
    assert again.results[0].url == "https://example.com/new"


def test_the_search_cache_can_be_switched_off(monkeypatch):
    switch_off(monkeypatch, "search")
    provider = Counting()
    search_with_fallback([provider], "ai")
    search_with_fallback([provider], "ai")
    assert len(provider.calls) == 2
    assert not cache.cache_dir("search").exists()


def test_a_failed_search_is_not_cached():
    class Down:
        name = "Tavily"

        def search(self, *args, **kwargs):
            raise RuntimeError("down")

    with pytest.raises(search.SearchUnavailableError):
        search_with_fallback([Down()], "ai")
    provider = Counting()
    search_with_fallback([provider], "ai")
    assert len(provider.calls) == 1


HTML = (
    "<html><body><article>"
    + "".join(
        f"<p>Paragraph {i} about hospitals using AI triage tools to cut waits.</p>"
        for i in range(8)
    )
    + "</article></body></html>"
)
URL = "https://www.example.com/page"


def counting_client():
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(200, headers={"Content-Type": "text/html"}, content=HTML)

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


def test_repeated_page_text_comes_from_the_cache():
    client, requests = counting_client()
    first = search.fetch_page(URL, 5, 10_000, client=client)
    second = search.fetch_page(URL, 5, 10_000, client=client)
    assert first and second == first
    assert len(requests) == 1


def test_a_cached_page_is_cut_to_the_callers_max_chars():
    client, requests = counting_client()
    search.fetch_page(URL, 5, 10_000, client=client)
    short = search.fetch_page(URL, 5, 40, client=client)
    assert len(short) <= 40 and len(requests) == 1


def test_page_text_expires_after_its_configured_time(clock):
    client, requests = counting_client()
    expiry = get_model_config().cache.page_text.expiry_hours
    search.fetch_page(URL, 5, 10_000, client=client)
    clock.now += (expiry - 1) * HOUR
    search.fetch_page(URL, 5, 10_000, client=client)
    assert len(requests) == 1
    clock.now += 2 * HOUR
    search.fetch_page(URL, 5, 10_000, client=client)
    assert len(requests) == 2


def test_a_fresh_page_fetch_skips_the_cache_and_overwrites_the_entry():
    client, requests = counting_client()
    search.fetch_page(URL, 5, 10_000, client=client)
    search.fetch_page(URL, 5, 10_000, client=client, fresh=True)
    assert len(requests) == 2
    search.fetch_page(URL, 5, 10_000, client=client)
    assert len(requests) == 2


def test_an_unreadable_page_is_not_cached():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert search.fetch_page(URL, 5, 100, client=client) == ""
    search.fetch_page(URL, 5, 100, client=client)
    assert len(calls) == 2


def test_the_page_text_cache_can_be_switched_off(monkeypatch):
    switch_off(monkeypatch, "page_text")
    client, requests = counting_client()
    search.fetch_page(URL, 5, 10_000, client=client)
    search.fetch_page(URL, 5, 10_000, client=client)
    assert len(requests) == 2
    assert not cache.cache_dir("page_text").exists()


def test_an_empty_search_answer_is_not_cached():
    class Empty(Counting):
        def search(self, query, max_results=10, fresh=False):
            self.calls.append(query)
            return []

    provider = Empty()
    search_with_fallback([provider], "ai")
    search_with_fallback([provider], "ai")
    assert len(provider.calls) == 2
