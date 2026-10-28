"""Deep Research Agent: searches the web and gathers Sources for a Topic. See spec #5.

It plans a few search queries with the `research` model, runs each through the search
fallback chain, merges the results into ranked Sources and saves them on the active Topic
as its Research Record. It does not summarise: that is the Content Strategist's job.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TypedDict

from langsmith import traceable
from pydantic import BaseModel, Field, field_validator

from contentblitz.core.config import SearchOptions, get_model_config
from contentblitz.integrations import llm, search
from contentblitz.integrations.search import SearchUnavailableError
from contentblitz.workflow.plan import current_step
from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.sources import QuerySearch, merge_sources
from contentblitz.workflow.state import ContentBlitzState, ResearchRecord, ResearchSource
from contentblitz.workflow.topics import active_topic

MIN_QUERIES = 3

SYSTEM_PROMPT = """\
You plan web research for a content marketer. Write 3 or 4 diverse Google search queries
about the Topic, each from a different angle: for example recent news, statistics and data,
expert analysis, and real-world examples or case studies. Every query must respect the
marketer's instructions (a focus, a market or a time frame). Keep each query short, the way
a person types into a search box."""

NO_PROVIDERS = (
    "I couldn't research *{topic}*: no search provider is configured, so I stopped the plan. "
    "Set {keys} in the environment."
)
NO_SOURCES = (
    "I couldn't find any Sources for *{topic}*, so I stopped the plan. "
    "Try rephrasing the Topic or your instructions."
)
NO_TOPIC = "Deep Research Agent: there's no active Topic to research."


class QueryPlan(BaseModel):
    queries: list[str] = Field(description="3 or 4 diverse web search queries.")

    @field_validator("queries")
    @classmethod
    def _distinct_and_enough(cls, queries: list[str]) -> list[str]:
        distinct = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
        if len(distinct) < MIN_QUERIES:
            raise ValueError(f"give at least {MIN_QUERIES} different queries")
        return distinct


def _plan_queries(topic: str, instructions: str | None, limit: int) -> list[str]:
    """3–4 queries from the `research` model; the Topic name alone if planning fails."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Topic: {topic}\nInstructions: {instructions or 'none'}"},
    ]
    try:
        plan = llm.complete_structured("research", messages, QueryPlan)
    except Exception:  # noqa: BLE001 - a failed plan still lets research run on the Topic name
        return [f"{topic} {instructions}" if instructions else topic]
    return plan.queries[:limit]


class ProviderOutcome(TypedDict):
    """How one query fared across the search providers, for the trace."""

    query: str
    provider: str | None  # the provider that answered, or None if none did
    results: int
    failures: dict[str, str]  # provider name → why it failed


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _search_all(
    providers: list[search.SearchProvider], queries: list[str], fresh: bool
) -> tuple[list[QuerySearch], list[ProviderOutcome]]:
    """Every query through the fallback chain at once. A query no provider answers is skipped."""

    def run(query: str) -> tuple[QuerySearch | None, ProviderOutcome]:
        try:
            response = search.search_with_fallback(providers, query, fresh=fresh)
        except SearchUnavailableError as error:
            failed = ProviderOutcome(query=query, provider=None, results=0, failures=error.failures)
            return None, failed
        outcome = ProviderOutcome(
            query=query,
            provider=response.provider,
            results=len(response.results),
            failures=response.failures,
        )
        return QuerySearch(query, response, retrieved_at=_now()), outcome

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        runs = list(pool.map(run, queries))
    return [s for s, _ in runs if s], [outcome for _, outcome in runs]


def _read_pages(sources: list[ResearchSource], options: SearchOptions) -> list[ResearchSource]:
    """The top Sources with their page text, read in parallel; the rest keep only their snippet.

    A page that can't be read gives no text, so its Source falls back to its snippet.
    """
    top = sources[: options.pages_to_read]
    if not top:
        return sources

    def read(source: ResearchSource) -> ResearchSource:
        text = search.fetch_page(
            source["url"], timeout=options.page_timeout_seconds, max_chars=options.text_cap_chars
        )
        return {**source, "page_text": text}

    with ThreadPoolExecutor(max_workers=len(top)) as pool:
        return [*pool.map(read, top), *sources[len(top) :]]


@traceable(name="research searches")
def _trace(queries: list[str], provider_outcomes: list[ProviderOutcome], source_count: int) -> dict:
    """Puts the queries, provider outcomes and Source count in the research step's trace."""
    return {
        "queries": queries,
        "provider_outcomes": provider_outcomes,
        "source_count": source_count,
    }


def _plural(count: int, word: str, plural: str | None = None) -> str:
    return f"{count} {word if count == 1 else plural or word + 's'}"


def _providers(searches: list[QuerySearch]) -> list[str]:
    """The providers that answered, in the order they first did."""
    return list(dict.fromkeys(s.response.provider for s in searches))


def _report(
    topic: str, queries: list[str], searches: list[QuerySearch], sources: list[ResearchSource]
) -> str:
    """The progress message. It counts every query run, including any no provider answered."""
    answered = _providers(searches)
    failed = {name for s in searches for name in s.response.failures}
    via = " and ".join(answered)
    if failed := sorted(failed - set(answered)):
        via += f" ({' and '.join(failed)} failed)"
    sites = len({s["site"] for s in sources})
    return (
        f"Researched *{topic}*: {_plural(len(queries), 'search', 'searches')}, "
        f"{_plural(len(sources), 'Source')} from {_plural(sites, 'site')} via {via}"
    )


def _missing_keys() -> str:
    keys = [search.PROVIDERS[name][1].upper() for name in get_model_config().search.providers]
    return " or ".join(keys)


def research(state: ContentBlitzState) -> ContentBlitzState:
    topic = active_topic(state)
    if not topic:
        return {"messages": [progress_message(NO_TOPIC)]}
    name = topic["name"]
    providers = search.configured_providers()
    if not providers:
        _trace([], [], 0)
        return {"abort_reason": NO_PROVIDERS.format(topic=name, keys=_missing_keys())}

    options = get_model_config().search
    step = current_step(state) or {}
    queries = _plan_queries(name, step.get("instructions"), options.queries_per_run)
    searches, outcomes = _search_all(providers, queries, fresh=step.get("fresh", False))
    sources = merge_sources(searches, options.max_sources)
    _trace(queries, outcomes, len(sources))
    if not sources:
        return {"abort_reason": NO_SOURCES.format(topic=name)}
    sources = _read_pages(sources, options)

    record: ResearchRecord = {
        "queries": queries,
        "providers": _providers(searches),
        "gathered_at": _now(),
        "sources": sources,
    }
    topics = dict(state["topics"])
    # New research (first or fresh) replaces the record and makes any existing Brief stale;
    # the next content_strategist step rebuilds it. A failed run returned above, keeping both.
    topics[state["active_topic_id"]] = {**topic, "research": record, "brief": None}
    return {
        "messages": [progress_message(_report(name, queries, searches, sources))],
        "topics": topics,
    }
