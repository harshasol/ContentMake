"""Searches run on worker threads, which must still carry the run and agent (spec #24, #72)."""

from test_research import (  # noqa: F401 - the two fixtures are autouse there
    QUERIES,
    FakeProvider,
    build_graph,
    fake_model,
    fake_search,
    hit,
    pages,
    run,
    with_api_key,
)

from contentblitz.core import call_context
from contentblitz.core.usage import get_ledger


def test_each_search_in_a_research_step_is_recorded_with_the_run_and_agent(monkeypatch):
    fake_model(monkeypatch)
    fake_search(
        monkeypatch, FakeProvider("SerpAPI", {q: [hit("https://who.int/a")] for q in QUERIES})
    )

    with call_context.run("conv-1", "run-1"):
        run(build_graph())

    searches = [e for e in get_ledger().events if e.kind == "search"]
    assert len(searches) == len(QUERIES)
    assert {(e.run_id, e.agent, e.model) for e in searches} == {("run-1", "research", "SerpAPI")}
