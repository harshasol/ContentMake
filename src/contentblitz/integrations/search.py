"""Web search behind one interface; providers are tried in the order set in models.yaml.

SerpAPI and Tavily are called over HTTP with httpx; there are no vendor SDKs.
Page text is read with fetch_page, which extracts the readable article with trafilatura.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import NamedTuple, Protocol

import httpx
import trafilatura
from pydantic import BaseModel

from contentblitz.core import call_context, tracing, usage
from contentblitz.core.config import ModelConfig, Settings, get_model_config, get_settings

REQUEST_TIMEOUT_SECONDS = 10  # per search request; queries run in parallel, providers in turn
MAX_PAGE_BYTES = 2_000_000  # HTML read per page; anything past this is ignored
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


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
    """No provider answered. `failures` maps each provider tried to why it failed."""

    def __init__(self, failures: dict[str, str]):
        reasons = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
        super().__init__(reasons or "no search providers configured")
        self.failures = failures


@contextmanager
def _http_client(client: httpx.Client | None, timeout: float) -> Iterator[httpx.Client]:
    """The injected client, else a short-lived one with this timeout, closed afterwards."""
    if client:
        yield client
    else:
        with httpx.Client(timeout=timeout) as own_client:
            yield own_client


class _HttpProvider:
    name: str  # set by each provider

    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self._api_key = api_key
        self._client = client

    def _send(self, method: str, url: str, **kwargs) -> dict:
        """Send the request and return its JSON body, raising SearchProviderError on failure."""
        try:
            with _http_client(self._client, REQUEST_TIMEOUT_SECONDS) as client:
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
        """The provider's error message in a reply body, if any. Each provider reads its own."""
        return None


class SerpApiProvider(_HttpProvider):
    """Google results through SerpAPI. Only the organic results are used."""

    name = "SerpAPI"
    URL = "https://serpapi.com/search.json"
    NO_RESULTS = "hasn't returned any results"

    def search(self, query: str, max_results: int = 10, fresh: bool = False) -> list[SearchResult]:
        params = {"engine": "google", "q": query, "api_key": self._api_key, "num": max_results}
        if fresh:
            params["no_cache"] = "true"  # SerpAPI otherwise may answer from its 1-hour cache
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
        # Tavily has no cache parameter, so `fresh` needs nothing here.
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


class ProviderEntry(NamedTuple):
    provider_class: type[_HttpProvider]
    key_field: str  # the Settings field holding its API key


# Config name → its provider
PROVIDERS: dict[str, ProviderEntry] = {
    "serpapi": ProviderEntry(SerpApiProvider, "serpapi_api_key"),
    "tavily": ProviderEntry(TavilyProvider, "tavily_api_key"),
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


def missing_key_names(config: ModelConfig | None = None) -> list[str]:
    """The environment variables that would configure each provider, in config order."""
    config = config or get_model_config()
    return [PROVIDERS[name].key_field.upper() for name in config.search.providers]


class SearchResponse(BaseModel):
    """One query's results, the provider that answered, and why earlier providers failed."""

    provider: str
    results: list[SearchResult]
    failures: dict[str, str] = {}  # provider name → reason


def search_with_fallback(
    providers: list[SearchProvider], query: str, max_results: int = 10, fresh: bool = False
) -> SearchResponse:
    """`fresh` is passed to every provider tried, so none answers from a cache. Each query is
    recorded in the usage ledger, with the provider that answered and those that failed."""
    failures: dict[str, str] = {}
    started = time.perf_counter()
    for provider in providers:
        try:
            with tracing.span(
                f"search: {provider.name}",
                "retriever",
                inputs={"query": query},
                metadata={"provider": provider.name, "query": query, "fresh": fresh},
            ) as span:
                results = provider.search(query, max_results, fresh=fresh)
                span.end(outputs={"results": len(results)})
        except Exception as exc:  # noqa: BLE001 - any provider failure moves to the next one
            failures[provider.name] = str(exc)
            continue
        _record_search(providers, provider.name, failures, started)
        return SearchResponse(provider=provider.name, results=results, failures=failures)
    _record_search(providers, NO_PROVIDER, failures, started)
    raise SearchUnavailableError(failures)


NO_PROVIDER = "none"  # the ledger's provider for a query no provider answered


def _record_search(
    providers: list[SearchProvider], answered_by: str, failures: dict[str, str], started: float
) -> None:
    context = call_context.current()
    price = get_model_config().search.prices.get(answered_by.lower(), 0.0)
    usage.record(
        usage.UsageEvent(
            ts=datetime.now(UTC).isoformat(),
            run_id=context.run_id,
            conversation_id=context.conversation_id,
            agent=context.agent or "search",
            kind="search",
            model=answered_by,
            primary=providers[0].name if providers else None,
            cost_usd=price,
            latency_s=round(time.perf_counter() - started, 3),
            failed=[{"provider": name, "error": error} for name, error in failures.items()],
        )
    )


def fetch_page(url: str, timeout: float, max_chars: int, client: httpx.Client | None = None) -> str:
    """The readable article text of a web page, cut to max_chars.

    Returns "" instead of raising when the page fails, takes longer than timeout seconds in
    all, isn't HTML, or has nothing readable. Only the first MAX_PAGE_BYTES are read.
    """
    with tracing.span("fetch page", "tool", inputs={"url": url}) as span:
        try:
            with _http_client(client, timeout) as http_client:
                html = _read_html(http_client, url, timeout)
            text = trafilatura.extract(html, include_comments=False) if html else None
        except Exception:  # noqa: BLE001 - an unreadable page falls back to its snippet
            text = None
        page = (text or "").strip()[:max_chars].rstrip()
        span.end(outputs={"chars": len(page)})
    return page


def _read_html(client: httpx.Client, url: str, timeout: float) -> str:
    """The page's HTML, or "" if it fails, isn't HTML, or isn't read within timeout seconds.

    A page cut off by the deadline gives "", not the partial HTML read so far.
    """
    deadline = time.monotonic() + timeout
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    with client.stream(
        "GET", url, headers=headers, timeout=timeout, follow_redirects=True
    ) as response:
        content_type = response.headers.get("Content-Type", "")
        if response.is_error or "html" not in content_type.lower():
            return ""
        body = bytearray()
        if not _time_left(response, deadline):
            return ""
        for chunk in response.iter_bytes():
            body += chunk
            if not _time_left(response, deadline):
                return ""
            if len(body) >= MAX_PAGE_BYTES:
                break
        return bytes(body[:MAX_PAGE_BYTES]).decode(response.encoding or "utf-8", errors="replace")


def _time_left(response: httpx.Response, deadline: float) -> bool:
    """Whether the deadline is still ahead; if so, the next read may wait only until then.

    httpx has no total timeout, but httpcore reads the request's timeout settings before every
    read, so lowering "read" here keeps a stalled body from running past the deadline.
    """
    left = deadline - time.monotonic()
    timeouts = response.request.extensions.get("timeout")
    if isinstance(timeouts, dict):
        timeouts["read"] = max(left, 0.001)
    return left > 0
