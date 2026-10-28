import pytest
from langchain_core.messages import HumanMessage

from contentblitz.agents import query_handler as qh
from contentblitz.core.config import Settings
from contentblitz.integrations.llm import StructuredOutputError
from contentblitz.workflow.graph import build_graph
from contentblitz.workflow.routing import RoutingDecision


def config(thread="t1"):
    return {"configurable": {"thread_id": thread}}


class FakeRouter:
    """Stands in for the model: returns a scripted routing decision per message."""

    def __init__(self, decide):
        self.decide = decide
        self.calls = []

    def __call__(self, agent, messages, schema):
        self.calls.append({"agent": agent, "messages": messages, "schema": schema})
        return self.decide(messages[-1]["content"])


@pytest.fixture
def with_api_key(monkeypatch):
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key="x"))


def fake_router(monkeypatch, decide):
    router = FakeRouter(decide)
    monkeypatch.setattr(qh.llm, "complete_structured", router)
    return router


def send(graph, text, thread="t1"):
    """Send one message; return the assistant replies that run produced."""
    before = len(graph.get_state(config(thread)).values.get("messages", []))
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config(thread))
    return [m.content for m in result["messages"][before + 1 :]]


def chat(reply):
    return lambda text: RoutingDecision(intent="chat", reply=reply.format(text=text))


def test_echoes_without_api_key(monkeypatch):
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key=None))
    assert send(build_graph(), "hello") == ["(No model API key configured, echoing) hello"]


def test_chat_gets_the_decisions_reply_and_nothing_else(monkeypatch, with_api_key):
    fake_router(monkeypatch, chat("I research topics and write blogs and LinkedIn posts."))
    replies = send(build_graph(), "what can you do?")
    assert replies == ["I research topics and write blogs and LinkedIn posts."]


def test_unclear_gets_the_clarifying_question(monkeypatch, with_api_key):
    question = (
        "What would you like me to do? For example: 'write a blog about it' or "
        "'research this topic'."
    )
    unclear = RoutingDecision(intent="unclear", clarifying_question=question)
    fake_router(monkeypatch, lambda _: unclear)
    assert send(build_graph(), "do something with this") == [question]


@pytest.mark.parametrize("intent", ["new_content", "refine"])
def test_content_and_refine_requests_are_not_supported_yet(monkeypatch, with_api_key, intent):
    fake_router(monkeypatch, lambda _: RoutingDecision(intent=intent, subject="AI in healthcare"))
    [reply] = send(build_graph(), "write a blog about AI in healthcare")
    assert "not supported yet" in reply


def test_invalid_routing_reply_gets_a_generic_clarifying_question(monkeypatch, with_api_key):
    def invalid(_):
        raise StructuredOutputError("bad reply")

    fake_router(monkeypatch, invalid)
    [reply] = send(build_graph(), "hmm")
    assert reply == qh.GENERIC_CLARIFYING_QUESTION


def test_provider_failure_gets_an_apology(monkeypatch, with_api_key):
    def down(_):
        raise ConnectionError("every provider failed")

    fake_router(monkeypatch, down)
    [reply] = send(build_graph(), "hello")
    assert reply == qh.APOLOGY


def test_routing_sees_the_last_ten_messages_with_the_query_handler_model(monkeypatch, with_api_key):
    router = fake_router(monkeypatch, chat("reply to {text}"))
    graph = build_graph()
    for n in range(7):
        send(graph, f"message {n}")

    call = router.calls[-1]
    assert call["agent"] == "query_handler"
    assert call["schema"] is RoutingDecision
    conversation = [m for m in call["messages"] if m["role"] != "system"]
    assert conversation == [
        {"role": "assistant", "content": "reply to message 1"},
        {"role": "user", "content": "message 2"},
        {"role": "assistant", "content": "reply to message 2"},
        {"role": "user", "content": "message 3"},
        {"role": "assistant", "content": "reply to message 3"},
        {"role": "user", "content": "message 4"},
        {"role": "assistant", "content": "reply to message 4"},
        {"role": "user", "content": "message 5"},
        {"role": "assistant", "content": "reply to message 5"},
        {"role": "user", "content": "message 6"},
    ]


def test_conversation_is_kept_per_thread(monkeypatch, with_api_key):
    fake_router(monkeypatch, chat("reply to {text}"))
    graph = build_graph()
    send(graph, "one", thread="a")
    send(graph, "other", thread="b")
    send(graph, "two", thread="a")

    def history(thread):
        return [m.content for m in graph.get_state(config(thread)).values["messages"]]

    assert history("a") == ["one", "reply to one", "two", "reply to two"]
    assert history("b") == ["other", "reply to other"]
