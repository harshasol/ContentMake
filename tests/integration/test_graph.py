from langchain_core.messages import HumanMessage

from contentblitz.agents import query_handler as qh
from contentblitz.core.config import Settings
from contentblitz.workflow.graph import build_graph

CONFIG = {"configurable": {"thread_id": "t1"}}


def fake_complete(agent, messages):
    return f"reply to {messages[-1]['content']}"


def test_echoes_without_api_key(monkeypatch):
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key=None))
    result = build_graph().invoke({"messages": [HumanMessage(content="hello")]}, config=CONFIG)
    assert "hello" in result["messages"][-1].content


def test_conversation_is_kept_per_thread(monkeypatch):
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key="x"))
    monkeypatch.setattr(qh.llm, "complete", fake_complete)
    graph = build_graph()
    graph.invoke({"messages": [HumanMessage(content="one")]}, config=CONFIG)
    result = graph.invoke({"messages": [HumanMessage(content="two")]}, config=CONFIG)
    assert [m.content for m in result["messages"]] == ["one", "reply to one", "two", "reply to two"]
