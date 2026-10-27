import pytest

from contentblitz.integrations.search import (
    SearchResult,
    SearchUnavailableError,
    search_with_fallback,
)


class Failing:
    name = "failing"

    def search(self, query, max_results=10):
        raise RuntimeError("quota exceeded")


class Working:
    name = "working"

    def search(self, query, max_results=10):
        return [SearchResult(title="t", url="https://example.com", snippet="s")]


def test_falls_back_to_next_provider():
    results = search_with_fallback([Failing(), Working()], "ai")
    assert results[0].url == "https://example.com"


def test_raises_when_every_provider_fails():
    with pytest.raises(SearchUnavailableError, match="quota exceeded"):
        search_with_fallback([Failing()], "ai")
