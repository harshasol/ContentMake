"""Research runs through the compiled workflow, with the model and search providers faked."""

from datetime import datetime

import pytest
from langchain_core.messages import HumanMessage
from langsmith import tracing_context

from contentblitz.agents import query_handler as qh
from contentblitz.core.config import Settings
from contentblitz.integrations import search
from contentblitz.integrations.search import SearchProviderError, SearchResult
from contentblitz.workflow.graph import build_graph
from contentblitz.workflow.routing import RoutingDecision

NO_SOURCES = (
    "I couldn't find any Sources for *AI in healthcare*, so I stopped the plan. "
    "Try rephrasing the Topic or your instructions."
)
QUERIES = ["AI in healthcare 2026", "AI diagnosis statistics", "AI healthcare regulation"]


class FakeModel:
    """Stands in for the model: a scripted routing decision, and a scripted query plan."""

    def __init__(self, decision, queries):
        self.decision = decision
        self.queries = queries
        self.calls = []

    def __call__(self, agent, messages, schema):
        self.calls.append({"agent": agent, "messages": messages, "schema": schema})
        if agent == "research":
            return schema(queries=self.queries)
        return self.decision


class FakeProvider:
    """A search provider with scripted results per query, or one that always fails."""

    def __init__(self, name, results=None, fail=False):
        self.name = name
        self.results = results or {}
        self.fail = fail
        self.queries = []
        self.fresh = []  # the fresh flag each search was called with

    def search(self, query, max_results=10, fresh=False):
        self.queries.append(query)
        self.fresh.append(fresh)
        if self.fail:
            raise SearchProviderError("quota exceeded")
        outcome = self.results.get(query, [])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome[:max_results]


def hit(url, title=None):
    return SearchResult(title=title or f"Page at {url}", url=url, snippet=f"About {url}")


@pytest.fixture(autouse=True)
def with_api_key(monkeypatch):
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key="x"))


def fake_model(monkeypatch, decision=None, queries=QUERIES):
    model = FakeModel(decision or research_request(), queries)
    monkeypatch.setattr(qh.llm, "complete_structured", model)
    return model


def fake_search(monkeypatch, *providers):
    monkeypatch.setattr(search, "configured_providers", lambda: list(providers))


def research_request(topic_name="AI in healthcare", instructions=None, same_topic=False):
    return RoutingDecision(
        intent="new_content",
        topic_name=topic_name,
        same_topic=same_topic,
        formats=["research"],
        instructions=instructions,
    )


def config(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def run(graph, text="research AI in healthcare", start=None, thread="t1"):
    """Send one message; return the replies it produced and the state afterwards."""
    before = len(graph.get_state(config(thread)).values.get("messages", []))
    message = {"messages": [HumanMessage(content=text)], **(start or {})}
    result = graph.invoke(message, config=config(thread))
    return [m.content for m in result["messages"][before + 1 :]], result


def research_record(state, topic_id="topic-1"):
    return state["topics"][topic_id]["research"]


def urls(record):
    return [s["url"] for s in record["sources"]]


def test_research_saves_a_research_record_on_the_topic_and_reports_it(monkeypatch):
    fake_model(monkeypatch)
    fake_search(
        monkeypatch,
        FakeProvider(
            "SerpAPI",
            {
                QUERIES[0]: [hit("https://who.int/ai", "WHO on AI"), hit("https://nature.com/a")],
                QUERIES[1]: [hit("https://nature.com/b"), hit("https://who.int/ai")],
                QUERIES[2]: [hit("https://fda.gov/ai")],
            },
        ),
    )
    replies, state = run(build_graph())

    assert (
        replies[1]
        == "Researched *AI in healthcare*: 3 searches, 4 Sources from 3 sites via SerpAPI"
    )
    record = research_record(state)
    assert record["queries"] == QUERIES
    assert record["providers"] == ["SerpAPI"]
    assert urls(record) == [
        "https://who.int/ai",  # found by two searches
        "https://nature.com/b",  # first in its results
        "https://fda.gov/ai",  # first in its results, found later
        "https://nature.com/a",  # second in its results
    ]
    first = record["sources"][0]
    retrieved_at = first.pop("retrieved_at")
    assert first == {
        "title": "WHO on AI",
        "url": "https://who.int/ai",
        "site": "who.int",
        "snippet": "About https://who.int/ai",
        "page_text": "",
        "query": QUERIES[0],
        "provider": "SerpAPI",
    }
    assert datetime.fromisoformat(retrieved_at).tzinfo is not None
    assert datetime.fromisoformat(record["gathered_at"]) >= datetime.fromisoformat(retrieved_at)


def test_query_planning_uses_the_research_model_with_the_topic_and_instructions(monkeypatch):
    model = fake_model(
        monkeypatch,
        research_request(instructions="focus on 2026 regulation"),
        queries=["q1", "q2", "q3", "q4", "q5"],
    )
    provider = FakeProvider("SerpAPI", {"q1": [hit("https://who.int/ai")]})
    fake_search(monkeypatch, provider)
    run(build_graph(), "research AI in healthcare, focus on 2026 regulation")

    [planning] = [c for c in model.calls if c["agent"] == "research"]
    prompt = "\n".join(m["content"] for m in planning["messages"])
    assert "AI in healthcare" in prompt
    assert "focus on 2026 regulation" in prompt
    assert provider.queries == ["q1", "q2", "q3", "q4"]


def test_the_same_page_found_twice_or_with_tracking_parameters_is_one_source(monkeypatch):
    fake_model(monkeypatch)
    fake_search(
        monkeypatch,
        FakeProvider(
            "SerpAPI",
            {
                QUERIES[0]: [hit("https://example.com/guide"), hit("https://other.org/x")],
                QUERIES[1]: [hit("https://EXAMPLE.com/guide/?utm_source=news&utm_medium=x#top")],
                QUERIES[2]: [
                    hit("https://example.com/guide?gclid=abc&fbclid=def"),
                    hit("https://example.com/guide?page=2&utm_campaign=y"),
                ],
            },
        ),
    )
    replies, state = run(build_graph())

    assert urls(research_record(state)) == [
        "https://example.com/guide",
        "https://other.org/x",
        "https://example.com/guide?page=2",
    ]
    assert "3 searches, 3 Sources from 2 sites" in replies[1]


def test_at_most_two_sources_come_from_one_site(monkeypatch):
    fake_model(monkeypatch)
    many = [hit(f"https://www.bigsite.com/page-{n}") for n in range(5)]
    fake_search(
        monkeypatch,
        FakeProvider("SerpAPI", {QUERIES[0]: many, QUERIES[1]: [hit("https://small.org/a")]}),
    )
    _, state = run(build_graph())

    record = research_record(state)
    assert urls(record) == [
        "https://www.bigsite.com/page-0",
        "https://small.org/a",
        "https://www.bigsite.com/page-1",
    ]
    assert [s["site"] for s in record["sources"]] == ["bigsite.com", "small.org", "bigsite.com"]


def test_sources_found_by_more_searches_and_higher_up_rank_first(monkeypatch):
    fake_model(monkeypatch)
    fake_search(
        monkeypatch,
        FakeProvider(
            "SerpAPI",
            {
                QUERIES[0]: [
                    hit("https://a.com/1"),
                    hit("https://b.com/1"),
                    hit("https://c.com/1"),
                ],
                QUERIES[1]: [hit("https://d.com/1"), hit("https://c.com/1")],
                QUERIES[2]: [
                    hit("https://e.com/1"),
                    hit("https://c.com/1"),
                    hit("https://b.com/1"),
                ],
            },
        ),
    )
    _, state = run(build_graph())

    assert urls(research_record(state)) == [
        "https://c.com/1",  # found by all three searches
        "https://b.com/1",  # found by two
        "https://a.com/1",  # found once, first in its results
        "https://d.com/1",
        "https://e.com/1",
    ]


def test_sources_are_capped_at_ten_by_default(monkeypatch):
    fake_model(monkeypatch)
    results = {
        q: [hit(f"https://site{n}-{i}.com/") for n in range(6)] for i, q in enumerate(QUERIES)
    }
    fake_search(monkeypatch, FakeProvider("SerpAPI", results))
    replies, state = run(build_graph())

    assert len(research_record(state)["sources"]) == 10
    assert "3 searches, 10 Sources from 10 sites" in replies[1]


def test_limits_come_from_the_config(monkeypatch, tmp_path):
    from contentblitz.core.config import DEFAULT_CONFIG_PATH

    path = tmp_path / "models.yaml"
    path.write_text(
        DEFAULT_CONFIG_PATH.read_text()
        .replace("queries_per_run: 4", "queries_per_run: 3")
        .replace("max_sources: 10", "max_sources: 2")
    )
    monkeypatch.setenv("CONTENTBLITZ_CONFIG", str(path))
    fake_model(monkeypatch, queries=["q1", "q2", "q3", "q4"])
    provider = FakeProvider("SerpAPI", {"q1": [hit(f"https://s{n}.com/") for n in range(5)]})
    fake_search(monkeypatch, provider)
    _, state = run(build_graph())

    assert provider.queries == ["q1", "q2", "q3"]
    assert len(research_record(state)["sources"]) == 2


def test_research_carries_on_with_the_next_provider_when_the_first_fails(monkeypatch):
    fake_model(monkeypatch)
    tavily = FakeProvider(
        "Tavily", {q: [hit(f"https://site{i}.com/")] for i, q in enumerate(QUERIES)}
    )
    fake_search(monkeypatch, FakeProvider("SerpAPI", fail=True), tavily)
    replies, state = run(build_graph())

    expected = "Researched *AI in healthcare*: 3 searches, 3 Sources from 3 sites"
    assert replies[1] == f"{expected} via Tavily (SerpAPI failed)"
    record = research_record(state)
    assert record["providers"] == ["Tavily"]
    assert {s["provider"] for s in record["sources"]} == {"Tavily"}


def test_a_search_that_fails_on_every_provider_is_skipped(monkeypatch):
    fake_model(monkeypatch)
    results = {
        QUERIES[0]: [hit("https://a.com/")],
        QUERIES[1]: SearchProviderError("SerpAPI timed out"),
        QUERIES[2]: [hit("https://b.com/")],
    }
    fake_search(monkeypatch, FakeProvider("SerpAPI", results))
    replies, state = run(build_graph())

    assert (
        replies[1]
        == "Researched *AI in healthcare*: 2 searches, 2 Sources from 2 sites via SerpAPI"
    )
    assert research_record(state)["queries"] == QUERIES


def test_with_no_search_provider_configured_the_marketer_is_told_which_keys_to_set(monkeypatch):
    fake_model(monkeypatch)
    fake_search(monkeypatch)
    replies, state = run(build_graph())

    assert "SERPAPI_API_KEY" in replies[1] and "TAVILY_API_KEY" in replies[1]
    assert not state["topics"]["topic-1"].get("research")


def test_when_no_search_finds_anything_no_research_is_saved(monkeypatch):
    fake_model(monkeypatch)
    fake_search(monkeypatch, FakeProvider("SerpAPI"))
    replies, state = run(build_graph())

    assert replies[1:] == [NO_SOURCES]
    assert not state["topics"]["topic-1"].get("research")


def blog_request(topic_name="AI in healthcare", fresh_research=False, same_topic=False):
    return RoutingDecision(
        intent="new_content",
        topic_name=topic_name,
        same_topic=same_topic,
        formats=["blog", "linkedin_post"],
        fresh_research=fresh_research,
    )


def test_when_no_search_finds_anything_the_rest_of_the_plan_stops(monkeypatch):
    fake_model(monkeypatch, blog_request())
    fake_search(monkeypatch, FakeProvider("SerpAPI"))
    replies, state = run(build_graph(), "write a blog and a LinkedIn post on AI in healthcare")

    assert replies == [
        "Plan for *AI in healthcare*: research → brief → blog → LinkedIn post",
        NO_SOURCES,
    ]
    assert state["completed_steps"] == [{"agent": "research", "instructions": None}]


def test_with_no_search_provider_configured_the_rest_of_the_plan_stops(monkeypatch):
    fake_model(monkeypatch, blog_request())
    fake_search(monkeypatch)
    replies, state = run(build_graph(), "write a blog and a LinkedIn post on AI in healthcare")

    assert replies[1:] == [
        "I couldn't research *AI in healthcare*: no search provider is configured, "
        "so I stopped the plan. Set SERPAPI_API_KEY or TAVILY_API_KEY in the environment."
    ]
    assert state["completed_steps"] == [{"agent": "research", "instructions": None}]


def test_failed_fresh_research_leaves_the_topics_research_record_and_brief_as_they_were(
    monkeypatch,
):
    fake_model(monkeypatch, blog_request(fresh_research=True, same_topic=True))
    fake_search(monkeypatch, FakeProvider("SerpAPI", fail=True))
    record = {
        "queries": ["AI in healthcare"],
        "providers": ["Tavily"],
        "gathered_at": "2026-10-01T09:00:00+00:00",
        "sources": [],
    }
    brief = {"summary": "AI speeds up diagnosis.", "key_findings": [], "sources": []}
    topic = {"name": "AI in healthcare", "research": record, "brief": brief, "piece_ids": []}
    start = {"topics": {"topic-1": topic}, "active_topic_id": "topic-1"}
    replies, state = run(build_graph(), "find newer sources, then a blog", start=start)

    assert replies[1:] == [NO_SOURCES]
    assert state["topics"]["topic-1"] == {
        "name": "AI in healthcare",
        "research": {
            "queries": ["AI in healthcare"],
            "providers": ["Tavily"],
            "gathered_at": "2026-10-01T09:00:00+00:00",
            "sources": [],
        },
        "brief": {"summary": "AI speeds up diagnosis.", "key_findings": [], "sources": []},
        "piece_ids": [],
    }


def researched_topics():
    """Two Topics seeded with an old research record and a Research Brief; the first is active."""

    def topic(name, url):
        record = {
            "queries": [name],
            "providers": ["Tavily"],
            "gathered_at": "2026-10-01T09:00:00+00:00",
            "sources": [
                {
                    "title": "Old page",
                    "url": url,
                    "site": url.split("/")[2],
                    "snippet": "Old.",
                    "page_text": "",
                    "query": name,
                    "provider": "Tavily",
                    "retrieved_at": "2026-10-01T09:00:00+00:00",
                }
            ],
        }
        brief = {"summary": f"Old Brief on {name}.", "key_findings": [], "sources": []}
        return {"name": name, "research": record, "brief": brief, "piece_ids": []}

    return {
        "topics": {
            "topic-1": topic("AI in healthcare", "https://old.org/ai"),
            "topic-2": topic("AI in finance", "https://old.org/finance"),
        },
        "active_topic_id": "topic-1",
    }


def test_fresh_research_replaces_the_research_record_and_clears_the_brief(monkeypatch):
    fake_model(monkeypatch, blog_request(fresh_research=True, same_topic=True))
    provider = FakeProvider("SerpAPI", {QUERIES[0]: [hit("https://new.org/ai")]})
    fake_search(monkeypatch, provider)
    start = researched_topics()
    replies, state = run(build_graph(), "find newer sources, then a blog", start=start)

    assert replies[1] == (
        "Researched *AI in healthcare*: 3 searches, 1 Source from 1 site via SerpAPI"
    )
    topic = state["topics"]["topic-1"]
    assert topic["research"]["queries"] == QUERIES
    assert urls(topic["research"]) == ["https://new.org/ai"]
    assert topic["brief"] is None
    assert state["topics"]["topic-2"] == researched_topics()["topics"]["topic-2"]


def test_fresh_research_tells_every_search_to_skip_cached_results(monkeypatch):
    fake_model(monkeypatch, blog_request(fresh_research=True, same_topic=True))
    serpapi = FakeProvider("SerpAPI", fail=True)
    tavily = FakeProvider("Tavily", {QUERIES[0]: [hit("https://new.org/ai")]})
    fake_search(monkeypatch, serpapi, tavily)
    run(build_graph(), "find newer sources, then a blog", start=researched_topics())

    assert serpapi.fresh == [True, True, True]
    assert tavily.fresh == [True, True, True]


def test_research_that_is_not_fresh_may_use_cached_results(monkeypatch):
    fake_model(monkeypatch)
    provider = FakeProvider("SerpAPI", {QUERIES[0]: [hit("https://new.org/ai")]})
    fake_search(monkeypatch, provider)
    run(build_graph())

    assert provider.fresh == [False, False, False]


def test_after_a_stopped_plan_the_next_request_runs_its_whole_plan(monkeypatch):
    fake_model(monkeypatch, blog_request(same_topic=True))
    provider = FakeProvider("SerpAPI")
    fake_search(monkeypatch, provider)
    graph = build_graph()
    run(graph, "write a blog and a LinkedIn post on AI in healthcare")

    provider.results = {QUERIES[0]: [hit("https://who.int/ai")]}
    replies, _ = run(graph, "try again")

    assert replies[1:] == [
        "Researched *AI in healthcare*: 3 searches, 1 Source from 1 site via SerpAPI",
        "Content Strategist: done (stub).",
        "SEO Blog Writer: done (stub).",
        "LinkedIn Post Writer: done (stub).",
    ]


def test_if_query_planning_fails_research_searches_for_the_topic_itself(monkeypatch):
    model = fake_model(monkeypatch, research_request(instructions="UK market only"))

    def planning_fails(agent, messages, schema):
        if agent == "research":
            raise ConnectionError("every model failed")
        return model(agent, messages, schema)

    monkeypatch.setattr(qh.llm, "complete_structured", planning_fails)
    provider = FakeProvider("SerpAPI", {"AI in healthcare UK market only": [hit("https://a.uk/")]})
    fake_search(monkeypatch, provider)
    _, state = run(build_graph())

    assert provider.queries == ["AI in healthcare UK market only"]
    assert research_record(state)["queries"] == ["AI in healthcare UK market only"]


def test_each_topics_research_is_kept_separately(monkeypatch):
    model = fake_model(monkeypatch)
    provider = FakeProvider("SerpAPI", {QUERIES[0]: [hit("https://healthcare.org/")]})
    fake_search(monkeypatch, provider)
    graph = build_graph()
    run(graph)

    model.decision = research_request("AI in finance")
    provider.results = {QUERIES[0]: [hit("https://finance.org/")]}
    _, state = run(graph, "research AI in finance", start={"active_topic_id": None})

    assert urls(research_record(state, "topic-1")) == ["https://healthcare.org"]
    assert urls(research_record(state, "topic-2")) == ["https://finance.org"]


class RecordingClient:
    """A LangSmith client that keeps the runs it is sent instead of uploading them."""

    def __init__(self):
        self.runs = {}

    def create_run(self, **run):
        self.runs[run["id"]] = run

    def update_run(self, run_id, **update):
        self.runs.setdefault(run_id, {}).update(update)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_queries_provider_outcomes_and_source_count_are_traced(monkeypatch):
    fake_model(monkeypatch)
    fake_search(
        monkeypatch,
        FakeProvider("SerpAPI", fail=True),
        FakeProvider("Tavily", {QUERIES[0]: [hit("https://a.com/"), hit("https://b.com/")]}),
    )
    client = RecordingClient()
    with tracing_context(enabled=True, client=client):
        run(build_graph())

    outputs = [r.get("outputs") or {} for r in client.runs.values()]
    [traced] = [o for o in outputs if "source_count" in o]
    assert traced["queries"] == QUERIES
    assert traced["source_count"] == 2
    assert traced["provider_outcomes"][0] == {
        "query": QUERIES[0],
        "provider": "Tavily",
        "results": 2,
        "failures": {"SerpAPI": "quota exceeded"},
    }
