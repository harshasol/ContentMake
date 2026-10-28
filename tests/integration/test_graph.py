import pytest
from langchain_core.messages import HumanMessage

from contentblitz.agents import query_handler as qh
from contentblitz.core.config import Settings
from contentblitz.integrations import llm, search
from contentblitz.integrations.llm import StructuredOutputError
from contentblitz.integrations.search import SearchResult
from contentblitz.workflow import graph as workflow
from contentblitz.workflow.graph import build_graph
from contentblitz.workflow.routing import RoutingDecision
from contentblitz.workflow.topics import add_piece


def config(thread="t1"):
    return {"configurable": {"thread_id": thread}}


BRIEF_DRAFT = {
    "summary": "AI is speeding up diagnosis.",
    "key_findings": [
        {"statement": "AI reads scans faster.", "source_numbers": [1]},
        {"statement": "Regulators are catching up.", "source_numbers": [1]},
    ],
    "statistics": [],
    "angles": [
        {"title": "The new colleague", "hook": "Your scan may be read twice."},
        {"title": "Regulation catches up", "hook": "Approvals keep coming."},
        {"title": "The rural gap", "hook": "The clinics that need AI most get it last."},
    ],
    "target_keyword": "AI in healthcare",
    "secondary_keywords": ["AI diagnosis", "medical AI regulation", "AI radiology"],
}

BLOG_DRAFT = {
    "template": "how-to",
    "title": "AI in healthcare",
    "meta_description": "How AI in healthcare speeds up diagnosis.",
    "body": "# AI in healthcare\n\nAI reads scans faster [1].",
}
VERDICT = {"engagement": 8, "suggestions": []}


class FakeRouter:
    """Stands in for the model: returns a scripted routing decision per message.

    Research query planning gets three fixed queries, the Content Strategist a scripted draft
    Brief, the blog writer a scripted blog and the reviewer a scripted verdict; only routing
    and Content Strategist calls are recorded.
    """

    def __init__(self, decide):
        self.decide = decide
        self.calls = []
        self.strategist_calls = []

    def __call__(self, agent, messages, schema):
        if agent == "research":
            return schema(queries=["q1", "q2", "q3"])
        if agent == "content_strategist":
            self.strategist_calls.append(messages)
            return schema.model_validate(BRIEF_DRAFT)
        if agent == "blog_writer":
            return schema.model_validate(BLOG_DRAFT)
        if agent == "quality_judge":
            return schema.model_validate(VERDICT)
        self.calls.append({"agent": agent, "messages": messages, "schema": schema})
        return self.decide(messages[-1]["content"])


class OneResultSearch:
    """A search provider that finds the same page for every query."""

    name = "FakeSearch"

    def __init__(self):
        self.fresh = []  # the fresh flag each search was called with

    def search(self, query, max_results=10, fresh=False):
        self.fresh.append(fresh)
        return [SearchResult(title="AI report", url="https://example.com/ai", snippet="AI.")]


@pytest.fixture(autouse=True)
def fake_web_search(monkeypatch):
    provider = OneResultSearch()
    monkeypatch.setattr(search, "configured_providers", lambda: [provider])
    monkeypatch.setattr(search, "fetch_page", lambda url, timeout, max_chars: "")
    return provider


@pytest.fixture
def with_api_key(monkeypatch):
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key="x"))


def fake_router(monkeypatch, decide):
    router = FakeRouter(decide)
    monkeypatch.setattr(qh.llm, "complete_structured", router)
    answer_through_complete_structured(monkeypatch)
    return router


def answer_through_complete_structured(monkeypatch):
    """The Content Strategist asks for the model's reply with its name; route that call through
    whatever fake stands in for `complete_structured`, naming the configured model."""

    def reply(agent, messages, schema):
        return llm.Reply(llm.complete_structured(agent, messages, schema), "openai/gpt-4o")

    monkeypatch.setattr(llm, "complete_structured_reply", reply)


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


def test_progress_and_plan_messages_do_not_use_up_the_history_window(monkeypatch, with_api_key):
    everything = new_content("blog", "linkedin_post", "image")(None)
    router = fake_router(
        monkeypatch,
        lambda text: everything if text == "the works" else chat("reply to {text}")(text),
    )
    graph = build_graph()
    send(graph, "first")
    send(graph, "second")
    assert len(send(graph, "the works")) == 6
    send(graph, "last")

    conversation = [m for m in router.calls[-1]["messages"] if m["role"] != "system"]
    assert conversation == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply to first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply to second"},
        {"role": "user", "content": "the works"},
        {"role": "user", "content": "last"},
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


def new_content(*formats, topic_name="AI in healthcare", fresh_research=False, same_topic=False):
    decision = RoutingDecision(
        intent="new_content",
        topic_name=topic_name,
        same_topic=same_topic,
        formats=list(formats),
        fresh_research=fresh_research,
    )
    return lambda _: decision


def researched(topic="AI in healthcare"):
    return f"Researched *{topic}*: 3 searches, 1 Source from 1 site via FakeSearch"


RESEARCH = researched()
BRIEF = (
    "Research Brief ready: 2 findings, 0 statistics, 1 Source. "
    "Ask 'show me the research' to read it."
)
BLOG = (
    "# AI in healthcare\n\nAI reads scans faster [1].\n\n---\n\n"
    "**Meta description:** How AI in healthcare speeds up diagnosis.\n\n"
    "**Template:** how-to\n\n"
    "**Quality Score:** 36/100\n\n"
    "- keyword density 42.9% ✗\n"
    "- meta description 41 chars ✗\n"
    "- headings 1 H1, 0 H2s ✗\n"
    "- readability grade 0.9 ✓\n"
    "- length 7 words ✗\n"
    "- citations 1 Source cited ✗\n\n"
    "**Automatic revisions:** 3 of 3\n\n"
    "**Engagement:** 8/10"
)
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
    assert list(state["topics"]) == ["topic-1"]
    topic = state["topics"]["topic-1"]
    assert (topic["name"], topic["piece_ids"]) == ("AI in healthcare", ["piece-1"])
    assert [s["url"] for s in topic["research"]["sources"]] == ["https://example.com/ai"]
    assert [s["url"] for s in topic["brief"]["sources"]] == ["https://example.com/ai"]
    assert state["active_topic_id"] == "topic-1"
    assert state["intent"] == "new_content"
    steps = [
        {"agent": "research", "instructions": None},
        {"agent": "content_strategist", "instructions": None},
        {"agent": "blog_writer", "instructions": None},
    ]
    assert state["plan"] == steps
    assert state["completed_steps"] == steps


def researched_topic(**brief):
    brief = {
        "summary": "AI speeds up diagnosis.",
        "key_findings": [],
        "statistics": [],
        "angles": [],
        "target_keyword": "AI in healthcare",
        "secondary_keywords": [],
        "sources": [
            {
                "number": 1,
                "title": "AI report",
                "url": "https://example.com/ai",
                "site": "example.com",
                "retrieved_at": "2026-10-20T09:25:00+00:00",
            }
        ],
        "created_at": "2026-10-20T10:00:00+00:00",
        "model": "openai/gpt-4o",
        "research_gathered_at": "2026-10-20T09:30:00+00:00",
        **brief,
    }
    return {
        "topics": {"topic-1": {"name": "AI in healthcare", "brief": brief}},
        "active_topic_id": "topic-1",
    }


def send_with(graph, text, start, thread="t1"):
    result = graph.invoke({"messages": [HumanMessage(content=text)], **start}, config(thread))
    return [m.content for m in result["messages"][1:]]


def test_later_pieces_reuse_the_topics_research_brief(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("linkedin_post", "blog", same_topic=True))
    replies = send_with(build_graph(), "now a LinkedIn post and a blog", researched_topic())
    assert replies == ["Plan for *AI in healthcare*: LinkedIn post → blog", LINKEDIN, BLOG]


def test_fresh_research_reruns_research_even_with_a_brief(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("blog", fresh_research=True, same_topic=True))
    replies = send_with(build_graph(), "find newer sources, then a blog", researched_topic())
    assert replies[1:] == [RESEARCH, BRIEF, BLOG]


def topic_without_brief(research):
    """A Topic whose Research Brief was never built, as a failed Content Strategist leaves it."""
    return {
        "topics": {
            "topic-1": {"name": "AI in healthcare", "research": research, "brief": None},
        },
        "active_topic_id": "topic-1",
    }


GATHERED = {
    "queries": ["q1"],
    "providers": ["FakeSearch"],
    "gathered_at": "2026-10-20T09:30:00+00:00",
    "sources": [
        {
            "title": "AI report",
            "url": "https://example.com/ai",
            "site": "example.com",
            "snippet": "AI.",
            "page_text": "",
            "query": "q1",
            "provider": "FakeSearch",
            "retrieved_at": "2026-10-20T09:25:00+00:00",
        }
    ],
}


def test_research_without_a_brief_rebuilds_only_the_brief_before_writers(
    monkeypatch, with_api_key, fake_web_search
):
    fake_router(monkeypatch, new_content("blog", "linkedin_post", same_topic=True))
    graph = build_graph()
    replies = send_with(graph, "write a blog and a LinkedIn post", topic_without_brief(GATHERED))

    assert replies == [
        "Plan for *AI in healthcare*: brief → blog → LinkedIn post",
        BRIEF,
        BLOG,
        LINKEDIN,
    ]
    assert fake_web_search.fresh == []
    topic = graph.get_state(config()).values["topics"]["topic-1"]
    assert topic["research"] == GATHERED
    assert topic["brief"]["sources"][0]["url"] == "https://example.com/ai"


def test_a_topic_with_no_research_still_plans_research_then_the_brief(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("blog", same_topic=True))
    replies = send_with(build_graph(), "write a blog", topic_without_brief(None))
    assert replies == [
        "Plan for *AI in healthcare*: research → brief → blog",
        RESEARCH,
        BRIEF,
        BLOG,
    ]


def test_fresh_research_on_a_topic_without_a_brief_plans_research_again(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("blog", fresh_research=True, same_topic=True))
    replies = send_with(
        build_graph(), "find newer sources, then a blog", topic_without_brief(GATHERED)
    )
    assert replies[1:] == [RESEARCH, BRIEF, BLOG]


def test_research_only_runs_research_and_shows_the_research_report(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("research"))
    replies = send(build_graph(), "research AI in healthcare")

    assert replies[:2] == ["Plan for *AI in healthcare*: research → brief", RESEARCH]
    [report] = replies[2:]
    assert report.startswith("# AI in healthcare\n")
    assert "- AI reads scans faster. [[1]](https://example.com/ai)" in report
    assert "1. [AI report](https://example.com/ai), example.com, retrieved " in report


def test_the_marketers_instructions_reach_the_content_strategist(monkeypatch, with_api_key):
    decision = RoutingDecision(
        intent="new_content",
        topic_name="AI in healthcare",
        formats=["research"],
        instructions="focus on regulation",
    )
    router = fake_router(monkeypatch, lambda _: decision)
    send(build_graph(), "research AI in healthcare, focus on regulation")

    [messages] = router.strategist_calls
    assert "focus on regulation" in "\n".join(m["content"] for m in messages)


def test_any_agent_can_stop_the_rest_of_the_plan_with_a_reason(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("blog", "linkedin_post"))
    reason = "The brief couldn't be written, so I stopped the plan."
    monkeypatch.setitem(
        workflow.AGENT_NODES, "content_strategist", lambda _: {"abort_reason": reason}
    )
    graph = build_graph()
    replies = send(graph, "write a blog and a LinkedIn post about AI in healthcare")

    assert replies == [
        "Plan for *AI in healthcare*: research → brief → blog → LinkedIn post",
        RESEARCH,
        reason,
    ]
    state = graph.get_state(config()).values
    assert [s["agent"] for s in state["completed_steps"]] == ["research", "content_strategist"]


def test_image_only_runs_without_research(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("image", topic_name="a robot doctor"))
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
        key_findings=[
            {"statement": "Diagnosis is 30% faster", "source_numbers": [1]},
            {"statement": "Adoption is slowest in rural clinics", "source_numbers": [1, 2]},
        ],
        statistics=[
            {"figure": "30%", "description": "faster diagnosis", "source_numbers": [1]},
        ],
        sources=[
            {
                "number": 1,
                "title": "WHO report on AI",
                "url": "https://who.int/ai",
                "site": "who.int",
                "retrieved_at": "2026-10-20T09:21:00+00:00",
            },
            {
                "number": 2,
                "title": "Rural clinics survey",
                "url": "https://nature.com/rural",
                "site": "nature.com",
                "retrieved_at": "2026-10-20T09:22:00+00:00",
            },
        ],
    )
    replies = send_with(build_graph(), "what sources did you use?", start)

    assert replies == ["It used the WHO report."]
    context = "\n".join(m["content"] for m in router.calls[-1]["messages"] if m["role"] == "system")
    for fact in [
        "AI in healthcare",
        "AI cuts radiology reading times.",
        "- Diagnosis is 30% faster [1]",
        "- Adoption is slowest in rural clinics [1, 2]",
        "- 30%: faster diagnosis [1]",
        "[1] WHO report on AI (https://who.int/ai)",
        "[2] Rural clinics survey (https://nature.com/rural)",
    ]:
        assert fact in context


def test_routing_is_told_when_the_topic_has_no_brief(monkeypatch, with_api_key):
    router = fake_router(monkeypatch, chat("Not yet."))
    start = {"topics": {"topic-1": {"name": "AI in healthcare"}}, "active_topic_id": "topic-1"}
    send_with(build_graph(), "what did the research say?", start)
    context = "\n".join(m["content"] for m in router.calls[-1]["messages"] if m["role"] == "system")
    assert "AI in healthcare" in context
    assert "no Research Brief yet" in context


def show_research(text):
    return RoutingDecision(intent="chat", show_research_report=True)


WHO_SOURCE = {
    "number": 1,
    "title": "WHO report on AI",
    "url": "https://who.int/ai",
    "site": "who.int",
    "retrieved_at": "2026-10-20T09:21:00+00:00",
}


def test_show_me_the_research_replies_with_the_active_topics_full_report(monkeypatch, with_api_key):
    fake_router(monkeypatch, show_research)
    start = researched_topic(
        summary="AI cuts radiology reading times.",
        key_findings=[{"statement": "Diagnosis is 30% faster.", "source_numbers": [1]}],
        sources=[WHO_SOURCE],
    )
    graph = build_graph()
    replies = send_with(graph, "show me the research", start)

    [report] = replies
    assert report.startswith("# AI in healthcare\n")
    assert "*Research gathered on 20 October 2026.*" in report
    assert "AI cuts radiology reading times." in report
    assert "- Diagnosis is 30% faster. [[1]](https://who.int/ai)" in report
    assert "1. [WHO report on AI](https://who.int/ai), who.int, retrieved 20 October 2026" in report
    state = graph.get_state(config()).values
    assert state["plan"] == []
    assert state["completed_steps"] == []


NO_RESEARCH_YET = (
    "There's no research yet. Ask me to research a Topic first, "
    'for example "research AI in healthcare".'
)


@pytest.mark.parametrize(
    "start",
    [
        {"topics": {"topic-1": {"name": "AI in healthcare", "brief": None}}},
        {"topics": {}, "active_topic_id": None},
    ],
    ids=["no Research Brief", "no active Topic"],
)
def test_show_me_the_research_before_any_exists_suggests_a_research_request(
    monkeypatch, with_api_key, start
):
    fake_router(monkeypatch, show_research)
    graph = build_graph()
    start = {"active_topic_id": "topic-1", **start}
    assert send_with(graph, "what did you find?", start) == [NO_RESEARCH_YET]
    assert graph.get_state(config()).values["completed_steps"] == []


def test_show_me_the_research_shows_only_the_active_topics_report(monkeypatch, with_api_key):
    fake_router(monkeypatch, show_research)
    healthcare = researched_topic(summary="AI cuts radiology reading times.")["topics"]["topic-1"]
    radiology = researched_topic(
        summary="Radiologists read twice as many scans.", target_keyword="AI radiology"
    )
    radiology = {**radiology["topics"]["topic-1"], "name": "AI in radiology"}
    start = {"topics": {"topic-1": healthcare, "topic-2": radiology}, "active_topic_id": "topic-2"}

    [report] = send_with(build_graph(), "show me the research", start)

    assert report.startswith("# AI in radiology\n")
    assert "Radiologists read twice as many scans." in report
    assert "AI in healthcare" not in report
    assert "AI cuts radiology reading times." not in report


def test_each_run_starts_a_fresh_plan(monkeypatch, with_api_key):
    decisions = iter([new_content("blog")(None), RoutingDecision(intent="chat", reply="Hi!")])
    fake_router(monkeypatch, lambda _: next(decisions))
    graph = build_graph()
    send(graph, "write a blog about AI in healthcare")
    assert send(graph, "hello") == ["Hi!"]
    state = graph.get_state(config()).values
    assert (state["intent"], state["plan"], state["completed_steps"]) == ("chat", [], [])


def piece(kind, title, topic_id="topic-1"):
    return {"kind": kind, "topic_id": topic_id, "title": title, "revisions": [{"text": "draft"}]}


def with_pieces(**pieces):
    """A researched active Topic plus a second Topic, seeded with pieces the way agents add them."""
    state = researched_topic()
    state["topics"]["topic-2"] = {"name": "Remote work", "brief": None}
    for piece_id, content_piece in pieces.items():
        state.update(add_piece(state, piece_id, content_piece))
    return state


def refine(instructions, target=None, formats=()):
    decision = RoutingDecision(
        intent="refine", refine_target=target, instructions=instructions, formats=list(formats)
    )
    return lambda _: decision


def test_refine_runs_one_step_to_the_writer_of_the_named_piece(monkeypatch, with_api_key):
    fake_router(monkeypatch, refine("make it punchier", target="piece-2"))
    start = with_pieces(
        **{
            "piece-1": piece("blog", "AI in the clinic"),
            "piece-2": piece("linkedin_post", "AI hype"),
        }
    )
    graph = build_graph()
    replies = send_with(graph, "make the LinkedIn post punchier", start)

    assert replies == [
        "Refining *AI hype* (LinkedIn post)",
        'LinkedIn Post Writer: refining piece-2 with "make it punchier" (stub).',
    ]
    state = graph.get_state(config()).values
    step = {"agent": "linkedin_writer", "target": "piece-2", "instructions": "make it punchier"}
    assert (state["intent"], state["plan"], state["completed_steps"]) == ("refine", [step], [step])


def test_refine_without_a_target_uses_the_only_piece_of_that_kind_on_the_active_topic(
    monkeypatch, with_api_key
):
    fake_router(monkeypatch, refine("shorter", formats=["linkedin_post"]))
    start = with_pieces(
        **{
            "piece-1": piece("blog", "AI in the clinic"),
            "piece-2": piece("linkedin_post", "AI hype"),
            "piece-3": piece("linkedin_post", "Old post", topic_id="topic-2"),
        }
    )
    replies = send_with(build_graph(), "shorten the LinkedIn post", start)
    assert replies == [
        "Refining *AI hype* (LinkedIn post)",
        'LinkedIn Post Writer: refining piece-2 with "shorter" (stub).',
    ]


def test_refine_it_uses_the_only_piece_on_the_active_topic(monkeypatch, with_api_key):
    fake_router(monkeypatch, refine("brighter colours"))
    start = with_pieces(
        **{
            "piece-1": piece("image", "Robot doctor"),
            "piece-2": piece("blog", "Old blog", topic_id="topic-2"),
        }
    )
    replies = send_with(build_graph(), "make it brighter", start)
    assert replies == [
        "Refining *Robot doctor* (image)",
        'Image Generation Agent: refining piece-1 with "brighter colours" (stub).',
    ]


def test_routing_is_given_a_summary_of_the_content_pieces(monkeypatch, with_api_key):
    router = fake_router(monkeypatch, chat("Sure."))
    start = with_pieces(
        **{
            "piece-1": piece("blog", "AI in the clinic"),
            "piece-2": piece("linkedin_post", "Old post", topic_id="topic-2"),
        }
    )
    send_with(build_graph(), "which pieces do I have?", start)

    context = "\n".join(m["content"] for m in router.calls[-1]["messages"] if m["role"] == "system")
    assert "- piece-1: blog on AI in healthcare, titled AI in the clinic" in context
    assert "- piece-2: linkedin_post on Remote work, titled Old post" in context


def test_refine_with_several_candidates_asks_which_piece_and_runs_nothing(
    monkeypatch, with_api_key
):
    fake_router(monkeypatch, refine("make it punchier"))
    start = with_pieces(
        **{
            "piece-1": piece("blog", "AI in the clinic"),
            "piece-2": piece("linkedin_post", "AI hype"),
        }
    )
    graph = build_graph()
    [question] = send_with(graph, "make it punchier", start)

    assert question == (
        "Which piece would you like me to refine?\n"
        "- *AI in the clinic* (blog)\n"
        "- *AI hype* (LinkedIn post)"
    )
    state = graph.get_state(config()).values
    assert (state["intent"], state["plan"], state["completed_steps"]) == ("refine", [], [])


def test_refine_with_no_pieces_says_there_is_nothing_to_refine_yet(monkeypatch, with_api_key):
    fake_router(monkeypatch, refine("shorter", target="piece-9", formats=["blog"]))
    graph = build_graph()
    [reply] = send_with(graph, "shorten the blog", researched_topic())

    assert reply == qh.NOTHING_TO_REFINE
    state = graph.get_state(config()).values
    assert (state["plan"], state["completed_steps"]) == ([], [])


def test_refine_of_a_missing_kind_lists_the_pieces_the_topic_has(monkeypatch, with_api_key):
    fake_router(monkeypatch, refine("shorter", formats=["linkedin_post"]))
    start = with_pieces(
        **{
            "piece-1": piece("blog", "AI in the clinic"),
            "piece-2": piece("image", "Robot doctor"),
        }
    )
    graph = build_graph()
    [reply] = send_with(graph, "shorten the LinkedIn post", start)

    assert reply == (
        "There's no LinkedIn post on *AI in healthcare* to refine. It has:\n"
        "- *AI in the clinic* (blog)\n"
        "- *Robot doctor* (image)"
    )
    state = graph.get_state(config()).values
    assert (state["plan"], state["completed_steps"]) == ([], [])


def test_refine_ignores_pieces_on_other_topics_unless_named(monkeypatch, with_api_key):
    start = with_pieces(**{"piece-1": piece("linkedin_post", "Old post", topic_id="topic-2")})
    fake_router(monkeypatch, refine("shorter", formats=["linkedin_post"]))
    assert send_with(build_graph(), "shorten the post", start) == [qh.NOTHING_TO_REFINE]

    fake_router(monkeypatch, refine("shorter", target="piece-1"))
    replies = send_with(build_graph(), "shorten the old post", start)
    assert replies == [
        "Refining *Old post* (LinkedIn post)",
        'LinkedIn Post Writer: refining piece-1 with "shorter" (stub).',
    ]


def test_the_requests_keyword_template_and_length_reach_a_blog_refine_step(
    monkeypatch, with_api_key
):
    decision = refine("as a 600-word listicle on AI diagnostics", formats=["blog"])(None)
    fake_router(monkeypatch, lambda _: decision.model_copy(update=BLOG_OPTIONS))
    start = with_pieces(**{"piece-1": piece("blog", "AI in the clinic")})
    graph = build_graph()
    send_with(graph, "make it a 600-word listicle, target keyword: AI diagnostics", start)

    state = graph.get_state(config()).values
    instructions = "as a 600-word listicle on AI diagnostics"
    assert state["plan"] == [
        {"agent": "blog_writer", "target": "piece-1", "instructions": instructions, **BLOG_OPTIONS}
    ]
    draft, refined = state["content_pieces"]["piece-1"]["revisions"]
    assert draft == {"text": "draft"}
    assert (refined["origin"], refined["target_keyword"], refined["template"]) == (
        "refinement",
        "AI diagnostics",
        "listicle",
    )


def routes(**decisions):
    """A fake router that picks a scripted decision by the message text."""
    return lambda text: decisions[text]


ASK = "a blog on AI in radiology"
RADIOLOGY = RoutingDecision(
    intent="new_content", topic_name="AI in radiology", same_topic=False, formats=["blog"]
)


def test_a_different_topic_asks_the_topic_question_and_runs_nothing(monkeypatch, with_api_key):
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

    research = researched("AI in radiology")
    assert replies == ["Plan for *AI in radiology*: research → brief → blog", research, BRIEF, BLOG]
    state = graph.get_state(config()).values
    assert state["active_topic_id"] == "topic-2"
    new_topic = state["topics"]["topic-2"]
    assert (new_topic["name"], new_topic["piece_ids"]) == ("AI in radiology", ["piece-1"])
    assert new_topic["research"]["sources"]
    assert new_topic["brief"]["sources"]
    assert state["topics"]["topic-1"] == researched_topic()["topics"]["topic-1"]
    assert state["pending_topic_question"] is None


def test_a_held_plan_asking_for_fresh_research_stays_fresh_on_the_new_topic(
    monkeypatch, with_api_key, fake_web_search
):
    fresh = RADIOLOGY.model_copy(update={"fresh_research": True})
    fake_router(monkeypatch, routes(**{ASK: fresh, "a new one": answer("new")}))
    graph = build_graph()
    send_with(graph, ASK, researched_topic())
    send(graph, "a new one")

    assert fake_web_search.fresh == [True, True, True]


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
    linkedin = new_content("linkedin_post", same_topic=True)(None)
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


def test_a_request_on_the_same_topic_runs_without_asking(monkeypatch, with_api_key):
    fake_router(monkeypatch, new_content("blog", same_topic=True))
    replies = send_with(build_graph(), "a blog on AI in healthcare", researched_topic())
    assert replies == ["Plan for *AI in healthcare*: blog", BLOG]


def test_a_narrower_angle_asks_the_topic_question(monkeypatch, with_api_key):
    narrower = new_content("blog", topic_name="AI in radiology", same_topic=False)
    fake_router(monkeypatch, narrower)
    graph = build_graph()
    [question] = send_with(graph, "a blog on AI in radiology", researched_topic())

    assert question == (
        "This sounds like a different Topic: *AI in radiology*. "
        "Start a new Topic, or continue with *AI in healthcare*?"
    )
    assert graph.get_state(config()).values["completed_steps"] == []


def test_routing_is_told_that_a_narrower_angle_is_not_the_same_topic(monkeypatch, with_api_key):
    router = fake_router(monkeypatch, chat("Sure."))
    send_with(build_graph(), "hello", researched_topic())

    context = "\n".join(m["content"] for m in router.calls[-1]["messages"] if m["role"] == "system")
    assert "same_topic" in context
    assert "only a request on the active Topic itself counts as the same Topic" in context
    assert "a narrower or different angle on it is not the same Topic" in context


def two_topics(active):
    return {
        "topics": {
            **researched_topic()["topics"],
            "topic-2": {"name": "AI in radiology", "brief": None},
        },
        "active_topic_id": active,
    }


def test_the_active_topic_set_in_the_input_is_respected(monkeypatch, with_api_key):
    router = fake_router(monkeypatch, new_content("linkedin_post", same_topic=True))
    graph = build_graph()
    send_with(graph, "start", two_topics("topic-2"))
    replies = send_with(graph, "now a LinkedIn post", {"active_topic_id": "topic-1"})

    assert replies[-2:] == ["Plan for *AI in healthcare*: LinkedIn post", LINKEDIN]
    assert "Active Topic: AI in healthcare" in router.calls[-1]["messages"][1]["content"]


def test_with_no_active_topic_a_content_request_creates_a_topic_without_asking(
    monkeypatch, with_api_key
):
    fake_router(monkeypatch, new_content("blog", topic_name="AI in finance"))
    graph = build_graph()
    replies = send_with(graph, "a blog on AI in finance", two_topics(None))

    research = researched("AI in finance")
    assert replies == ["Plan for *AI in finance*: research → brief → blog", research, BRIEF, BLOG]
    state = graph.get_state(config()).values
    assert state["active_topic_id"] == "topic-3"
    names = [t["name"] for t in state["topics"].values()]
    assert names == ["AI in healthcare", "AI in radiology", "AI in finance"]


def test_new_topic_ids_follow_the_highest_topic_number_and_never_overwrite(
    monkeypatch, with_api_key
):
    fake_router(monkeypatch, new_content("image", topic_name="AI in finance"))
    healthcare = {"name": "AI in healthcare", "brief": None}
    notes = {"name": "Notes", "brief": None}
    start = {"topics": {"topic-3": healthcare, "notes": notes}, "active_topic_id": None}
    graph = build_graph()
    send_with(graph, "an image about AI in finance", start)

    state = graph.get_state(config()).values
    assert state["active_topic_id"] == "topic-4"
    assert state["topics"] == {
        "topic-3": healthcare,
        "notes": notes,
        "topic-4": {"name": "AI in finance", "brief": None, "piece_ids": []},
    }


BLOG_OPTIONS = {"target_keyword": "AI diagnostics", "template": "listicle", "length": 600}


def blog_revision(state):
    [piece_id] = state["topics"][state["active_topic_id"]]["piece_ids"]
    [revision] = state["content_pieces"][piece_id]["revisions"]
    return revision


def test_the_requests_keyword_template_and_length_reach_the_blog_step(monkeypatch, with_api_key):
    decision = new_content("blog", "linkedin_post", same_topic=True)(None)
    fake_router(monkeypatch, lambda _: decision.model_copy(update=BLOG_OPTIONS))
    graph = build_graph()
    send_with(graph, "a 600-word listicle, target keyword: AI diagnostics", researched_topic())

    state = graph.get_state(config()).values
    assert state["plan"] == [
        {"agent": "blog_writer", "instructions": None, **BLOG_OPTIONS},
        {"agent": "linkedin_writer", "instructions": None},
    ]
    revision = blog_revision(state)
    assert (revision["target_keyword"], revision["template"]) == ("AI diagnostics", "listicle")


def test_the_requests_blog_options_carry_over_when_a_held_plan_is_rebuilt_for_a_new_topic(
    monkeypatch, with_api_key
):
    asked = RADIOLOGY.model_copy(update=BLOG_OPTIONS)
    fake_router(monkeypatch, routes(**{ASK: asked, "a new one": answer("new")}))
    graph = build_graph()
    send_with(graph, ASK, researched_topic())
    send(graph, "a new one")

    state = graph.get_state(config()).values
    assert state["plan"] == [
        {"agent": "research", "instructions": None},
        {"agent": "content_strategist", "instructions": None},
        {"agent": "blog_writer", "instructions": None, **BLOG_OPTIONS},
    ]
    revision = blog_revision(state)
    assert (revision["target_keyword"], revision["template"]) == ("AI diagnostics", "listicle")
