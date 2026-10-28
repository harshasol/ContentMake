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
    response = search_with_fallback([Failing(), Working()], "ai")
    assert response.results[0].url == "https://example.com"
    assert response.provider == "working"
    assert response.failures == {"failing": "quota exceeded"}


def test_raises_when_every_provider_fails():
    with pytest.raises(SearchUnavailableError, match="quota exceeded"):
        search_with_fallback([Failing()], "ai")
