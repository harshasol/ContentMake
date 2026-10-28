"""Merging search results from several queries into ranked, de-duplicated Sources."""

from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from contentblitz.integrations.search import SearchResponse
from contentblitz.workflow.state import ResearchSource

MAX_PER_SITE = 2
_TRACKING_PARAMS = {"gclid", "fbclid"}


@dataclass
class QuerySearch:
    """One query's search: what the fallback chain returned, and when."""

    query: str
    response: SearchResponse
    retrieved_at: str


@dataclass
class _Candidate:
    source: ResearchSource
    first_seen: int
    positions: list[int] = field(default_factory=list)

    def rank_key(self) -> tuple[int, int, int]:
        """Found by more searches first, then higher up the results, then found earlier."""
        return (-len(self.positions), min(self.positions), self.first_seen)


def normalise_url(url: str) -> str:
    """Lower-cased host, no fragment, no utm_*/gclid/fbclid parameters, no trailing slash."""
    parts = urlsplit(url.strip())
    params = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not (key.lower().startswith("utm_") or key.lower() in _TRACKING_PARAMS)
    ]
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(params), ""))


def site_of(url: str) -> str:
    host = urlsplit(url).hostname or ""
    return host.removeprefix("www.")


def merge_sources(searches: list[QuerySearch], max_sources: int) -> list[ResearchSource]:
    """De-duplicate by normalised URL, rank, keep at most two per site, and cap the total."""
    candidates: dict[str, _Candidate] = {}
    for search in searches:
        for position, result in enumerate(search.response.results):
            url = normalise_url(result.url)
            if url not in candidates:
                source: ResearchSource = {
                    "title": result.title,
                    "url": url,
                    "site": site_of(url),
                    "snippet": result.snippet,
                    "page_text": "",
                    "query": search.query,
                    "provider": search.response.provider,
                    "retrieved_at": search.retrieved_at,
                }
                candidates[url] = _Candidate(source, first_seen=len(candidates))
            candidates[url].positions.append(position)

    kept: list[ResearchSource] = []
    per_site: dict[str, int] = {}
    for candidate in sorted(candidates.values(), key=_Candidate.rank_key):
        site = candidate.source["site"]
        if per_site.get(site, 0) < MAX_PER_SITE:
            per_site[site] = per_site.get(site, 0) + 1
            kept.append(candidate.source)
    return kept[:max_sources]
