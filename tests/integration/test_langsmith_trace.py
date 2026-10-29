"""One LangSmith trace per run (#24, ticket #78), against a fake LangSmith client.

Runs the real compiled workflow through the Workspace with LiteLLM faked at its lowest level and a
scripted Query Handler that posts a one-step blog plan.
"""

import test_blog_writer as blog
from test_usage_tagging import fake_litellm

from contentblitz.core import tracing
from contentblitz.core.config import get_model_config, get_settings
from contentblitz.integrations import llm, search
from contentblitz.workflow import graph as workflow
from contentblitz.workflow.workspace import Workspace


class RecordingClient:
    """A LangSmith client that keeps the runs it is sent instead of uploading them."""

    def __init__(self):
        self.runs = {}

    def create_run(self, **run):
        self.runs[str(run["id"])] = run

    def update_run(self, run_id, **update):
        self.runs.setdefault(str(run_id), {}).update(update)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def scripted_workspace(monkeypatch):
    def query_handler(state):
        return {
            "plan": blog.BLOG,
            "completed_steps": [],
            "abort_reason": None,
            "topics": {"topic-1": blog.topic()},
            "active_topic_id": "topic-1",
        }

    monkeypatch.setattr(workflow, "query_handler", query_handler)
    return Workspace(workflow.build_graph(), "conv-1")


def switch_tracing(monkeypatch, on=True, key="ls-key"):
    monkeypatch.setenv("LANGSMITH_TRACING", "true" if on else "false")
    if key:
        monkeypatch.setenv("LANGSMITH_API_KEY", key)
    else:
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    get_settings.cache_clear()
    client = RecordingClient()
    monkeypatch.setattr(tracing, "make_client", lambda settings: client)
    return client


def by_name(client, name):
    return [r for r in client.runs.values() if r["name"] == name]


def metadata(run):
    return (run.get("extra") or {}).get("metadata") or {}


def roots(client):
    return [r for r in client.runs.values() if not r.get("parent_run_id")]


def test_a_run_is_one_trace_with_agent_steps_and_provider_calls_nested_under_them(monkeypatch):
    client = switch_tracing(monkeypatch)
    fake_litellm(monkeypatch)
    workspace = scripted_workspace(monkeypatch)

    run_id = workspace.send("write a blog")

    runs = list(client.runs.values())
    [root] = roots(client)
    assert str(root["id"]).replace("-", "") == run_id
    assert {r["trace_id"] for r in runs} == {root["id"]}
    [step] = by_name(client, "blog_writer")
    assert step["parent_run_id"] == root["id"]
    calls = [r for r in runs if r["run_type"] == "llm"]
    assert len(calls) == 8  # the draft and 3 revisions, each reviewed
    assert all(str(step["id"]) in c["dotted_order"] for c in calls)


def test_the_root_carries_conversation_run_topic_and_brand_profile(monkeypatch):
    client = switch_tracing(monkeypatch)
    fake_litellm(monkeypatch)
    workspace = scripted_workspace(monkeypatch)
    workspace.send("write a blog")  # the first run makes the Topic active
    client.runs.clear()

    run_id = workspace.send("another blog")

    [root] = roots(client)
    meta = metadata(root)
    assert meta["conversation_id"] == "conv-1"
    assert meta["run_id"] == run_id
    assert meta["topic"] == "AI in healthcare"
    assert meta["brand_profile_id"] == "default"


def test_model_spans_carry_model_tokens_cost_and_cache_hit(monkeypatch):
    client = switch_tracing(monkeypatch)
    fake_litellm(monkeypatch)
    scripted_workspace(monkeypatch).send("write a blog")

    calls = [r for r in client.runs.values() if r["run_type"] == "llm"]
    first = min(calls, key=lambda r: r["start_time"])
    meta = metadata(first)
    assert meta["model"] == get_model_config().for_agent("blog_writer").primary
    assert (meta["tokens_in"], meta["tokens_out"], meta["cost_usd"]) == (100, 40, 0.01)
    assert meta["cache_hit"] is False


def test_quality_loop_attempts_and_scores_appear_as_steps(monkeypatch):
    client = switch_tracing(monkeypatch)
    fake_litellm(monkeypatch)
    scripted_workspace(monkeypatch).send("write a blog")

    attempts = by_name(client, "blog attempt")
    assert len(attempts) == 4
    assert all("quality_score" in (a.get("outputs") or {}) for a in attempts)


def test_a_search_is_a_span_under_the_step_that_made_it(monkeypatch):
    client = switch_tracing(monkeypatch)

    class Provider:
        name = "Fake"

        def search(self, query, max_results=10, fresh=False):
            return [search.SearchResult(title="t", url="https://x.test", snippet="s")]

    with tracing.tracing(), tracing.span("step"):
        search.search_with_fallback([Provider()], "q")

    [span] = by_name(client, "search: Fake")
    assert metadata(span)["query"] == "q"
    assert span["parent_run_id"] == by_name(client, "step")[0]["id"]


def test_tracing_off_sends_nothing(monkeypatch):
    client = switch_tracing(monkeypatch, on=False)
    fake_litellm(monkeypatch)
    scripted_workspace(monkeypatch).send("write a blog")

    assert client.runs == {}


def test_no_api_key_sends_nothing(monkeypatch):
    client = switch_tracing(monkeypatch, on=True, key=None)
    fake_litellm(monkeypatch)
    scripted_workspace(monkeypatch).send("write a blog")

    assert client.runs == {}


def test_the_litellm_langsmith_callback_is_not_installed(monkeypatch):
    switch_tracing(monkeypatch)
    fake_litellm(monkeypatch)
    scripted_workspace(monkeypatch).send("write a blog")

    assert "langsmith" not in llm.litellm.success_callback


def test_the_routing_decision_and_plan_are_on_the_query_handler_span(monkeypatch):
    from contentblitz.agents import query_handler as qh
    from contentblitz.workflow.routing import RoutingDecision

    client = switch_tracing(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    decision = RoutingDecision(intent="new_content", topic_name="AI", formats=["blog"])
    monkeypatch.setattr(qh.llm, "complete_structured", lambda *a, **k: decision)
    monkeypatch.setitem(workflow.AGENT_NODES, "research", lambda state: {})
    monkeypatch.setitem(workflow.AGENT_NODES, "blog_writer", lambda state: {})
    monkeypatch.setitem(workflow.AGENT_NODES, "content_strategist", lambda state: {})

    Workspace(workflow.build_graph(), "conv-2").send("write a blog on AI")

    [handler] = by_name(client, "query_handler")
    meta = metadata(handler)
    assert meta["routing_decision"]["intent"] == "new_content"
    assert meta["plan"][-1] == "blog_writer"
    assert meta["topic"] == "AI"
