"""Web search behind one interface; providers are tried in the order set in models.yaml."""

from typing import Protocol

from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]: ...


class SearchUnavailableError(RuntimeError):
    pass


def search_with_fallback(
    providers: list[SearchProvider], query: str, max_results: int = 10
) -> list[SearchResult]:
    errors = []
    for provider in providers:
        try:
            return provider.search(query, max_results)
        except Exception as exc:  # noqa: BLE001 - any provider failure moves to the next one
            errors.append(f"{provider.name}: {exc}")
    raise SearchUnavailableError("; ".join(errors) or "no search providers configured")
