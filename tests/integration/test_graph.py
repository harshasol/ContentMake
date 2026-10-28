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


def test_refine_requests_are_not_supported_yet(monkeypatch, with_api_key):
    fake_router(monkeypatch, lambda _: RoutingDecision(intent="refine", instructions="shorter"))
    [reply] = send(build_graph(), "make it shorter")
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


def new_content(*formats, subject="AI in healthcare", fresh_research=False, same_subject=True):
    decision = RoutingDecision(
        intent="new_content",
        subject=subject,
        same_subject=same_subject,
        formats=list(formats),
        fresh_research=fresh_research,
    )
    return lambda _: decision


RESEARCH = "Deep Research Agent: done (stub)."
BRIEF = "Content Strategist: done (stub)."
BLOG = "SEO Blog Writer: done (stub)."
LINKEDIN = "LinkedIn Post Writer: done (stub)."
IMAGE = "Image Generation Agent: done (stub)."


def test_content_request_posts_the_plan_then_runs_research_first(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("blog", "linkedin_post"))
    replies = send(
        build_graph(), "Research AI in healthcare, then write a blog and a LinkedIn post"
    )
    assert replies == [
        "Plan for *AI in healthcare*: research → brief → blog → LinkedIn post",
        RESEARCH,
        BRIEF,
        BLOG,
        LINKEDIN,
    ]


def test_first_content_request_creates_the_active_topic_and_records_the_steps(
    monkeypatch, with_api_key
):
    fake_router(monkeypatch, new_content("blog"))
    graph = build_graph()
    send(graph, "write a blog about AI in healthcare")

    state = graph.get_state(config()).values
    assert state["topics"] == {"topic-1": {"name": "AI in healthcare", "brief": None}}
    assert state["active_topic_id"] == "topic-1"
    assert state["intent"] == "new_content"
    steps = [{"agent": "research"}, {"agent": "content_strategist"}]
    steps.append({"agent": "blog_writer", "instructions": None})
    assert state["plan"] == steps
    assert state["completed_steps"] == steps


def researched_topic(**brief):
    brief = {"summary": "AI speeds up diagnosis.", "key_findings": [], "sources": [], **brief}
    return {
        "topics": {"topic-1": {"name": "AI in healthcare", "brief": brief}},
        "active_topic_id": "topic-1",
    }


def send_with(graph, text, start, thread="t1"):
    result = graph.invoke({"messages": [HumanMessage(content=text)], **start}, config(thread))
    return [m.content for m in result["messages"][1:]]


def test_later_pieces_reuse_the_topics_research_brief(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("linkedin_post", "blog"))
    replies = send_with(build_graph(), "now a LinkedIn post and a blog", researched_topic())
    assert replies == ["Plan for *AI in healthcare*: LinkedIn post → blog", LINKEDIN, BLOG]


def test_fresh_research_reruns_research_even_with_a_brief(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("blog", fresh_research=True))
    replies = send_with(build_graph(), "find newer sources, then a blog", researched_topic())
    assert replies[1:] == [RESEARCH, BRIEF, BLOG]


def test_research_only_runs_research_and_the_brief(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("research"))
    replies = send(build_graph(), "research AI in healthcare")
    assert replies == ["Plan for *AI in healthcare*: research → brief", RESEARCH, BRIEF]


def test_image_only_runs_without_research(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("image", subject="a robot doctor"))
    replies = send(build_graph(), "make an image of a robot doctor")
    assert replies == ["Plan for *a robot doctor*: image", IMAGE]


def test_duplicate_formats_are_removed_from_the_plan(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content(*["blog"] * 3, *["linkedin_post"] * 5, "image"))
    replies = send(build_graph(), "three blogs and five posts and an image")
    assert replies[1:] == [RESEARCH, BRIEF, BLOG, LINKEDIN, IMAGE]


def test_questions_about_research_are_grounded_in_the_active_brief(monkeypatch, with_api_key):
    router = fake_router(monkeypatch, chat("It used the WHO report."))
    start = researched_topic(
        summary="AI cuts radiology reading times.",
        key_findings=["Diagnosis is 30% faster", "Adoption is slowest in rural clinics"],
        sources=[{"title": "WHO report on AI", "url": "https://who.int/ai"}],
    )
    replies = send_with(build_graph(), "what sources did you use?", start)

    assert replies == ["It used the WHO report."]
    context = "\n".join(m["content"] for m in router.calls[-1]["messages"] if m["role"] == "system")
    for fact in [
        "AI in healthcare",
        "AI cuts radiology reading times.",
        "Diagnosis is 30% faster",
        "Adoption is slowest in rural clinics",
        "WHO report on AI",
        "https://who.int/ai",
    ]:
        assert fact in context


def test_routing_is_told_when_the_topic_has_no_brief(monkeypatch, with_api_key):
    router = fake_router(monkeypatch, chat("Not yet."))
    start = {"topics": {"topic-1": {"name": "AI in healthcare"}}, "active_topic_id": "topic-1"}
    send_with(build_graph(), "what did the research say?", start)
    context = "\n".join(m["content"] for m in router.calls[-1]["messages"] if m["role"] == "system")
    assert "AI in healthcare" in context
    assert "no Research Brief yet" in context


def test_each_run_starts_a_fresh_plan(monkeypatch, with_api_key):
    decisions = iter([new_content("blog")(None), RoutingDecision(intent="chat", reply="Hi!")])
    fake_router(monkeypatch, lambda _: next(decisions))
    graph = build_graph()
    send(graph, "write a blog about AI in healthcare")
    assert send(graph, "hello") == ["Hi!"]
    state = graph.get_state(config()).values
    assert (state["intent"], state["plan"], state["completed_steps"]) == ("chat", [], [])


def routes(**decisions):
    """A fake router that picks a scripted decision by the message text."""
    return lambda text: decisions[text]


ASK = "a blog on AI in radiology"
RADIOLOGY = RoutingDecision(
    intent="new_content", subject="AI in radiology", same_subject=False, formats=["blog"]
)


def test_a_different_subject_asks_the_topic_question_and_runs_nothing(monkeypatch, with_api_key):
    fake_router(monkeypatch, routes(**{ASK: RADIOLOGY}))
    graph = build_graph()
    [question] = send_with(graph, ASK, researched_topic())

    assert "Start a new Topic" in question
    assert "AI in healthcare" in question and "AI in radiology" in question
    state = graph.get_state(config()).values
    assert state["completed_steps"] == []
    assert state["pending_topic_question"] == {
        "name": "AI in radiology",
        "plan": [{"agent": "blog_writer", "instructions": None}],
    }
    assert state["active_topic_id"] == "topic-1"


def answer(topic_answer):
    return RoutingDecision(intent="chat", topic_answer=topic_answer)


def test_answering_new_creates_the_topic_and_runs_the_held_plan_research_first(
    monkeypatch, with_api_key
):
    fake_router(monkeypatch, routes(**{ASK: RADIOLOGY, "a new one": answer("new")}))
    graph = build_graph()
    send_with(graph, ASK, researched_topic())
    replies = send(graph, "a new one")

    assert replies == ["Plan for *AI in radiology*: research → brief → blog", RESEARCH, BRIEF, BLOG]
    state = graph.get_state(config()).values
    assert state["active_topic_id"] == "topic-2"
    assert state["topics"]["topic-2"] == {"name": "AI in radiology", "brief": None}
    assert state["topics"]["topic-1"]["name"] == "AI in healthcare"
    assert state["pending_topic_question"] is None


def test_answering_continue_runs_the_held_plan_on_the_current_topic_and_its_brief(
    monkeypatch, with_api_key
):
    keep_going = "keep going with healthcare"
    fake_router(monkeypatch, routes(**{ASK: RADIOLOGY, keep_going: answer("continue")}))
    graph = build_graph()
    send_with(graph, ASK, researched_topic())
    replies = send(graph, keep_going)

    assert replies == ["Plan for *AI in healthcare*: blog", BLOG]
    state = graph.get_state(config()).values
    assert state["active_topic_id"] == "topic-1"
    assert list(state["topics"]) == ["topic-1"]
    assert state["pending_topic_question"] is None


def test_any_other_reply_drops_the_held_plan_and_is_handled_normally(monkeypatch, with_api_key):
    linkedin = new_content("linkedin_post")(None)
    fake_router(monkeypatch, routes(**{ASK: RADIOLOGY, "actually, a LinkedIn post": linkedin}))
    graph = build_graph()
    send_with(graph, ASK, researched_topic())
    replies = send(graph, "actually, a LinkedIn post")

    assert replies == ["Plan for *AI in healthcare*: LinkedIn post", LINKEDIN]
    assert graph.get_state(config()).values["pending_topic_question"] is None


def test_routing_is_told_about_the_pending_topic_question(monkeypatch, with_api_key):
    router = fake_router(monkeypatch, routes(**{ASK: RADIOLOGY, "hmm": answer(None)}))
    graph = build_graph()
    send_with(graph, ASK, researched_topic())
    send(graph, "hmm")

    context = "\n".join(m["content"] for m in router.calls[-1]["messages"] if m["role"] == "system")
    assert "Pending Topic question" in context
    assert "AI in radiology" in context
    assert "topic_answer" in context


def test_a_narrower_angle_the_decision_calls_the_same_subject_runs_without_asking(
    monkeypatch, with_api_key
):
    router = fake_router(monkeypatch, new_content("blog", subject="AI in radiology"))
    replies = send_with(build_graph(), "a blog on AI in radiology", researched_topic())

    assert replies == ["Plan for *AI in healthcare*: blog", BLOG]
    context = "\n".join(m["content"] for m in router.calls[-1]["messages"] if m["role"] == "system")
    assert "same_subject" in context


def two_topics(active):
    return {
        "topics": {
            **researched_topic()["topics"],
            "topic-2": {"name": "AI in radiology", "brief": None},
        },
        "active_topic_id": active,
    }


def test_the_active_topic_set_in_the_input_is_respected(monkeypatch, with_api_key):
    router = fake_router(monkeypatch, new_content("linkedin_post"))
    graph = build_graph()
    send_with(graph, "start", two_topics("topic-2"))
    replies = send_with(graph, "now a LinkedIn post", {"active_topic_id": "topic-1"})

    assert replies[-2:] == ["Plan for *AI in healthcare*: LinkedIn post", LINKEDIN]
    assert "Active Topic: AI in healthcare" in router.calls[-1]["messages"][1]["content"]


def test_with_no_active_topic_a_content_request_creates_a_topic_without_asking(
    monkeypatch, with_api_key
):
    fake_router(monkeypatch, new_content("blog", subject="AI in finance", same_subject=False))
    graph = build_graph()
    replies = send_with(graph, "a blog on AI in finance", two_topics(None))

    assert replies == ["Plan for *AI in finance*: research → brief → blog", RESEARCH, BRIEF, BLOG]
    state = graph.get_state(config()).values
    assert state["active_topic_id"] == "topic-3"
    names = [t["name"] for t in state["topics"].values()]
    assert names == ["AI in healthcare", "AI in radiology", "AI in finance"]
