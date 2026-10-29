"""The workspace module works on one conversation of the real compiled graph (#23, #59).

The model is faked at the LLM integration module and the data folder is a temporary directory.
"""

import io

import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from contentblitz.agents import query_handler as qh
from contentblitz.core.config import Settings
from contentblitz.integrations import llm
from contentblitz.workflow import graph as workflow
from contentblitz.workflow.graph import build_graph
from contentblitz.workflow.report import render_report
from contentblitz.workflow.routing import RoutingDecision
from contentblitz.workflow.workspace import ChatMessage, Workspace, new_thread_id


@pytest.fixture
def model(monkeypatch):
    """The routing model answers every message with a chat reply `reply to <message>`."""
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key="x"))

    def complete(agent, messages, schema):
        text = messages[-1]["content"]
        return RoutingDecision(intent="chat", reply=f"reply to {text}")

    monkeypatch.setattr(qh.llm, "complete_structured", complete)


def test_a_new_conversation_has_no_messages():
    assert Workspace(build_graph(), new_thread_id()).messages() == []


def test_thread_ids_are_unique():
    assert new_thread_id() != new_thread_id()


def test_messages_are_read_from_the_graph_state_in_order(model):
    workspace = Workspace(build_graph(), "t1")

    workspace.send("first")
    workspace.send("second")

    assert [(m.role, m.text) for m in workspace.messages()] == [
        ("user", "first"),
        ("assistant", "reply to first"),
        ("user", "second"),
        ("assistant", "reply to second"),
    ]


def test_another_workspace_on_the_same_thread_sees_the_conversation(model):
    graph = build_graph()
    Workspace(graph, "t1").send("hello")

    again = Workspace(graph, "t1").messages()

    assert [m.text for m in again] == ["hello", "reply to hello"]
    assert Workspace(graph, "other").messages() == []


def test_echo_mode_works_without_an_api_key(monkeypatch):
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key=None))
    workspace = Workspace(build_graph(), "t1")

    workspace.send("hello")

    assert [m.text for m in workspace.messages()] == [
        "hello",
        "(No model API key configured, echoing) hello",
    ]


def test_every_message_of_a_multi_message_run_is_listed(monkeypatch):
    def two_messages(state):
        return {"messages": [AIMessage(content="plan"), AIMessage(content="step")]}

    monkeypatch.setattr(workflow, "query_handler", two_messages)
    workspace = Workspace(build_graph(), "t1")

    workspace.send("go")

    assert [m.text for m in workspace.messages()] == ["go", "plan", "step"]


def test_an_image_message_carries_the_images_metadata(monkeypatch):
    def generate(prompt, aspect, config=None, client=None):
        buffer = io.BytesIO()
        Image.new("RGB", (1024, 1024), (20, 120, 140)).save(buffer, format="PNG")
        return llm.GeneratedImage(buffer.getvalue(), "openai/gpt-image-1")

    def complete(agent, messages, schema):
        return schema(
            prompt="A robot doctor in a bright clinic.",
            use="general",
            aspect=None,
            alt_text="A robot doctor",
        )

    monkeypatch.setattr(llm, "generate_image", generate)
    monkeypatch.setattr(llm, "complete_structured", complete)
    graph = build_graph()
    config = {"configurable": {"thread_id": "t1"}}
    graph.update_state(
        config,
        {
            "messages": [("human", "an image of a robot doctor")],
            "topics": {"topic-1": {"name": "AI", "research": None, "brief": None, "piece_ids": []}},
            "active_topic_id": "topic-1",
            "plan": [{"agent": "image_generator", "instructions": "a robot doctor"}],
            "completed_steps": [],
            "abort_reason": None,
        },
        as_node="query_handler",
    )
    graph.invoke(None, config)

    messages = Workspace(graph, "t1").messages()

    [image_message] = [m for m in messages if m.image]
    assert image_message.role == "assistant"
    assert "robot doctor" in image_message.image["prompt"]
    assert image_message.image["model"] == "openai/gpt-image-1"
    assert all(isinstance(m, ChatMessage) for m in messages)


def _brief(**changes):
    source = {
        "number": 1,
        "title": "AI in radiology",
        "url": "https://example.com/ai",
        "site": "example.com",
        "retrieved_at": "2026-10-28T09:00:00+00:00",
    }
    brief = {
        "summary": "AI helps radiologists.",
        "key_findings": [{"statement": "Reads scans faster.", "source_numbers": [1]}],
        "statistics": [],
        "angles": [{"title": "Speed", "hook": "Faster reads"}],
        "target_keyword": "ai radiology",
        "secondary_keywords": ["scans"],
        "sources": [source],
        "created_at": "2026-10-28T09:05:00+00:00",
        "model": "gpt-4o",
        "research_gathered_at": "2026-10-28T09:00:00+00:00",
    }
    return {**brief, **changes}


def _seed(brief, active="topic-1"):
    graph = build_graph()
    topic = {"name": "AI in radiology", "research": None, "brief": brief, "piece_ids": []}
    graph.update_state(
        {"configurable": {"thread_id": "t1"}},
        {"topics": {"topic-1": topic}, "active_topic_id": active},
        as_node="query_handler",
    )
    return Workspace(graph, "t1")


def test_research_is_the_active_topics_brief_and_rendered_report():
    brief = _brief()

    research = _seed(brief).research()

    assert research.topic_name == "AI in radiology"
    assert research.brief == brief
    assert research.report == render_report("AI in radiology", brief)
    assert "[[1]](https://example.com/ai)" in research.report


def test_research_is_none_without_a_brief_or_an_active_topic():
    assert Workspace(build_graph(), "new").research() is None
    assert _seed(None).research() is None
    assert _seed(_brief(), active=None).research() is None

def test_the_configuration_shows_which_keys_are_set_and_never_their_values():
    from contentblitz.core.config import get_model_config

    settings = Settings(
        _env_file=None, openai_api_key="sk-secret-openai", tavily_api_key="tvly-secret"
    )
    view = Workspace(build_graph(), new_thread_id()).configuration(settings, get_model_config())

    assert view.provider_keys == {
        "OpenAI": True,
        "Anthropic": False,
        "Gemini": False,
        "SerpAPI": False,
        "Tavily": True,
    }
    assert "secret" not in repr(view)


def test_the_configuration_describes_models_search_images_and_thresholds():
    from contentblitz.core.config import get_model_config

    config = get_model_config()
    view = Workspace(build_graph(), new_thread_id()).configuration(
        Settings(_env_file=None), config
    )

    assert view.agents["blog_writer"] == (config.agents["blog_writer"].primary, [])
    assert set(view.agents) == set(config.agents)
    assert view.search_providers == config.search.providers
    assert view.image_models == config.image.models
    assert view.thresholds["Pass mark"] == "7"
    assert view.thresholds["Blog words"] == "900–1800"
