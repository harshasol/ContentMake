import pytest

from contentblitz.integrations.search import (
    SearchResult,
    SearchUnavailableError,
    search_with_fallback,
)


class Failing:
    name = "failing"

    def search(self, query, max_results=10, fresh=False):
        raise RuntimeError("quota exceeded")


class Working:
    name = "working"

    def search(self, query, max_results=10, fresh=False):
        return [SearchResult(title="t", url="https://example.com", snippet="s")]


def test_falls_back_to_next_provider():
    response = search_with_fallback([Failing(), Working()], "ai")
    assert response.results[0].url == "https://example.com"
    assert response.provider == "working"
    assert response.failures == {"failing": "quota exceeded"}


def test_raises_when_every_provider_fails():
    with pytest.raises(SearchUnavailableError, match="quota exceeded"):
        search_with_fallback([Failing()], "ai")


class Recording:
    """Fails or answers, and records the fresh flag it was called with."""

    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail
        self.fresh = []

    def search(self, query, max_results=10, fresh=False):
        self.fresh.append(fresh)
        if self.fail:
            raise RuntimeError("quota exceeded")
        return [SearchResult(title="t", url="https://example.com", snippet="s")]


def test_fresh_flag_reaches_every_provider_tried():
    first, second = Recording("first", fail=True), Recording("second")
    search_with_fallback([first, second], "ai", fresh=True)
    assert (first.fresh, second.fresh) == ([True], [True])


def test_searches_are_not_fresh_by_default():
    provider = Recording("only")
    search_with_fallback([provider], "ai")
    assert provider.fresh == [False]
