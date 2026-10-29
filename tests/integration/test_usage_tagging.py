"""Usage ledger (#24): every model call is tagged with its run, conversation and agent.

Runs the real compiled workflow with LiteLLM's completion faked at the lowest level, reusing the
blog writer tests' Topic, Brief and drafts.
"""

import json
from types import SimpleNamespace

import test_blog_writer as blog

from contentblitz.core import call_context
from contentblitz.core.usage import get_ledger
from contentblitz.integrations import llm
from contentblitz.workflow import graph as workflow


def fake_litellm(monkeypatch):
    """Scripted JSON per schema, with usage, cost and the answering model."""

    def completion(**kwargs):
        schema = kwargs["response_format"].__name__
        content = {"BlogDraft": blog.blog_draft(), "Verdict": blog.verdict()}[schema]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))],
            model=kwargs["model"],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40),
            _hidden_params={"response_cost": 0.01},
        )

    monkeypatch.setattr(llm.litellm, "completion", completion)


def test_every_model_call_in_a_run_carries_its_run_conversation_and_agent(monkeypatch):
    fake_litellm(monkeypatch)

    with call_context.run("conv-7", "run-7"):
        blog.run_plan(blog.BLOG, {"topic-1": blog.topic()})

    events = get_ledger().events
    # The draft and its 3 revisions, each reviewed: the quality loop's attempts are all tagged.
    assert [e.agent for e in events] == ["blog_writer", "quality_judge"] * 4
    assert {(e.run_id, e.conversation_id) for e in events} == {("run-7", "conv-7")}
    assert get_ledger().caption("run-7") == "This run: $0.080 · 1.1k tokens"


def test_a_call_outside_a_run_records_the_agent_and_no_run(monkeypatch):
    fake_litellm(monkeypatch)

    blog.run_plan(blog.BLOG, {"topic-1": blog.topic()})

    assert {(e.run_id, e.conversation_id) for e in get_ledger().events} == {(None, None)}
    assert get_ledger().events[0].agent == "blog_writer"


def test_each_plan_step_sets_the_current_agent_for_the_calls_it_makes(monkeypatch):
    seen = []

    def writer(state):
        seen.append(call_context.current().agent)
        return {}

    monkeypatch.setitem(workflow.AGENT_NODES, "blog_writer", writer)

    blog.run_plan(blog.BLOG, {"topic-1": blog.topic()})

    assert seen == ["blog_writer"]
    assert call_context.current().agent is None  # nothing leaks out of the step
