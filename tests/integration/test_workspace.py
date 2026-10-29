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
