"""Web search behind one interface; providers are tried in the order set in models.yaml.

SerpAPI and Tavily are called over HTTP with httpx; there are no vendor SDKs.
"""

from typing import Protocol

import httpx
from pydantic import BaseModel

from contentblitz.core.config import ModelConfig, Settings, get_model_config, get_settings

REQUEST_TIMEOUT_SECONDS = 15


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 10, fresh: bool = False) -> list[SearchResult]:
        """`fresh` asks for new results, so a provider must not answer from a cache."""
        ...


class SearchProviderError(RuntimeError):
    """One provider could not answer: an HTTP error, a timeout, or an error in its reply."""


class SearchUnavailableError(RuntimeError):
    pass


class _HttpProvider:
    name = ""

    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self._api_key = api_key
        self._client = client

    def _send(self, method: str, url: str, **kwargs) -> dict:
        """Send the request and return its JSON body, raising SearchProviderError on failure."""
        try:
            if self._client:
                response = self._client.request(method, url, **kwargs)
            else:
                with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                    response = client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise SearchProviderError("timed out") from exc
        except httpx.HTTPError as exc:
            raise SearchProviderError(f"request failed: {exc}") from exc
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.is_error:
            raise SearchProviderError(
                f"returned {response.status_code}: {self._error(body) or response.text}"
            )
        return body

    @staticmethod
    def _error(body: dict) -> str | None:
        return None


class SerpApiProvider(_HttpProvider):
    """Google results through SerpAPI. Only the organic results are used."""

    name = "SerpAPI"
    URL = "https://serpapi.com/search.json"
    NO_RESULTS = "hasn't returned any results"

    def search(self, query: str, max_results: int = 10, fresh: bool = False) -> list[SearchResult]:
        params = {"engine": "google", "q": query, "api_key": self._api_key, "num": max_results}
        body = self._send("GET", self.URL, params=params)
        error = self._error(body)
        if error and self.NO_RESULTS not in error:
            raise SearchProviderError(error)
        return [
            SearchResult(title=r["title"], url=r["link"], snippet=r.get("snippet", ""))
            for r in body.get("organic_results", [])[:max_results]
            if r.get("title") and r.get("link")
        ]

    @staticmethod
    def _error(body: dict) -> str | None:
        return body.get("error")


class TavilyProvider(_HttpProvider):
    """Tavily's basic search."""

    name = "Tavily"
    URL = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = 10, fresh: bool = False) -> list[SearchResult]:
        payload = {"query": query, "search_depth": "basic", "max_results": max_results}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = self._send("POST", self.URL, json=payload, headers=headers)
        return [
            SearchResult(title=r["title"], url=r["url"], snippet=r.get("content") or "")
            for r in body.get("results", [])[:max_results]
            if r.get("title") and r.get("url")
        ]

    @staticmethod
    def _error(body: dict) -> str | None:
        detail = body.get("detail")
        return detail.get("error") if isinstance(detail, dict) else detail


# Config name → (provider class, the Settings field holding its API key)
PROVIDERS: dict[str, tuple[type[_HttpProvider], str]] = {
    "serpapi": (SerpApiProvider, "serpapi_api_key"),
    "tavily": (TavilyProvider, "tavily_api_key"),
}


def configured_providers(
    settings: Settings | None = None,
    config: ModelConfig | None = None,
    client: httpx.Client | None = None,
) -> list[SearchProvider]:
    """The providers in config order, leaving out any without an API key in settings."""
    settings = settings or get_settings()
    config = config or get_model_config()
    providers: list[SearchProvider] = []
    for name in config.search.providers:
        provider_class, key_field = PROVIDERS[name]
        if api_key := getattr(settings, key_field):
            providers.append(provider_class(api_key, client))
    return providers


class SearchResponse(BaseModel):
    """One query's results, the provider that answered, and why earlier providers failed."""

    provider: str
    results: list[SearchResult]
    failures: dict[str, str] = {}  # provider name → reason


def search_with_fallback(
    providers: list[SearchProvider], query: str, max_results: int = 10, fresh: bool = False
) -> SearchResponse:
    """`fresh` is passed to every provider tried, so none answers from a cache."""
    failures: dict[str, str] = {}
    for provider in providers:
        try:
            results = provider.search(query, max_results, fresh=fresh)
        except Exception as exc:  # noqa: BLE001 - any provider failure moves to the next one
            failures[provider.name] = str(exc)
            continue
        return SearchResponse(provider=provider.name, results=results, failures=failures)
    reasons = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
    raise SearchUnavailableError(reasons or "no search providers configured")
