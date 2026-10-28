"""The SEO Blog Writer writes a blog from the Research Brief through the compiled workflow.

Each test seeds a Topic that already has a Research Brief, and a plan with a blog step, as the
Query Handler would leave them, then runs the workflow from the dispatcher. The model is faked
at the LLM integration module and returns a scripted blog draft.
"""

from datetime import datetime

import pytest
from langchain_core.messages import HumanMessage

from contentblitz.integrations import llm
from contentblitz.workflow.graph import build_graph

BRIEF_CREATED_AT = "2026-10-20T10:00:00+00:00"


def source(n, site):
    return {
        "number": n,
        "title": f"Report {n} from {site}",
        "url": f"https://{site}/report-{n}",
        "site": site,
        "retrieved_at": "2026-10-20T09:21:00+00:00",
    }


BRIEF = {
    "summary": "AI is speeding up diagnosis. Regulators are catching up.",
    "key_findings": [
        {"statement": "AI reads scans faster than radiologists.", "source_numbers": [1, 2]},
        {"statement": "The FDA has cleared over 900 AI devices.", "source_numbers": [3]},
    ],
    "statistics": [
        {
            "figure": "30%",
            "description": "faster scan reading",
            "source_numbers": [2],
            "verified": True,
        },
        {
            "figure": "950",
            "description": "AI devices cleared by the FDA",
            "source_numbers": [3],
            "verified": False,
        },
    ],
    "angles": [
        {"title": "The radiologist's new colleague", "hook": "Your next scan may be read twice."},
        {"title": "Regulation catches up", "hook": "950 approvals and counting."},
    ],
    "target_keyword": "AI in healthcare",
    "secondary_keywords": ["AI diagnosis", "medical AI regulation"],
    "sources": [source(1, "who.int"), source(2, "nature.com"), source(3, "fda.gov")],
    "created_at": BRIEF_CREATED_AT,
    "model": "openai/gpt-4o",
    "research_gathered_at": "2026-10-20T09:30:00+00:00",
}

BODY = """\
# AI in Healthcare: How Hospitals Read Scans Faster

AI in healthcare is changing how scans are read [1].

## Faster diagnosis

Radiologists using AI read scans 30% faster [2].

## Regulation

The FDA has cleared over 900 AI devices [3].

## Sources

1. [Report 1 from who.int](https://who.int/report-1)
2. [Report 2 from nature.com](https://nature.com/report-2)
3. [Report 3 from fda.gov](https://fda.gov/report-3)"""

META = "AI in healthcare helps radiologists read scans faster, and regulators are catching up."


def blog_draft(**overrides):
    """A blog draft as the `blog_writer` model would return it."""
    return {
        "template": "thought leadership",
        "title": "AI in Healthcare: How Hospitals Read Scans Faster",
        "meta_description": META,
        "body": BODY,
        **overrides,
    }


SUGGESTIONS = [
    "Open with the 30% statistic to hook the reader.",
    "Add a call to action for hospital CIOs.",
    "Split the regulation section into shorter paragraphs.",
    "Name one cleared device as an example.",
    "End with what readers should do next.",
]


def verdict(**overrides):
    """A reviewer verdict as the `quality_judge` model would return it."""
    return {"engagement": 8, "suggestions": SUGGESTIONS, **overrides}


class FakeModel:
    """Stands in for every model: scripted blog drafts and reviewer verdicts, one per call.

    The blog writer gets `reply` and the reviewer gets `verdict`; an exception is raised.
    Other agents (the LinkedIn Writer is still a stub) aren't expected to call.
    """

    def __init__(self, reply, verdict):
        self.reply = reply
        self.verdict = verdict
        self.calls = []

    def __call__(self, agent, messages, schema):
        self.calls.append({"agent": agent, "messages": messages, "schema": schema})
        reply = self.verdict if agent == "quality_judge" else self.reply
        if isinstance(reply, Exception):
            raise reply
        return llm.Reply(schema.model_validate(reply), "openai/gpt-4o")

    def calls_to(self, agent):
        return [call for call in self.calls if call["agent"] == agent]


@pytest.fixture
def model(monkeypatch):
    fake = FakeModel(blog_draft(), verdict())
    monkeypatch.setattr(llm, "complete_structured_reply", fake)
    return fake


def topic(name="AI in healthcare"):
    return {"name": name, "research": None, "brief": BRIEF, "piece_ids": []}


def config(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def run_plan(plan, topics=None, graph=None, thread="t1"):
    """Run `plan` from the dispatcher, as if the Query Handler had just posted it.

    Returns the replies the plan produced and the state afterwards.
    """
    graph = graph or build_graph()
    before = len(graph.get_state(config(thread)).values.get("messages", []))
    seeded = {
        "messages": [HumanMessage(content="go")],
        "plan": plan,
        "completed_steps": [],
        "abort_reason": None,
    }
    if topics is not None:
        seeded |= {"topics": topics, "active_topic_id": "topic-1"}
    graph.update_state(config(thread), seeded, as_node="query_handler")
    result = graph.invoke(None, config(thread))
    return [m.content for m in result["messages"][before + 1 :]], result


BLOG = [{"agent": "blog_writer", "instructions": None}]


def test_a_blog_step_posts_the_blog_with_its_meta_description_and_template_set_apart(model):
    [reply], _ = run_plan(BLOG, {"topic-1": topic()})

    assert reply.startswith(BODY)
    after_body = reply.removeprefix(BODY)
    assert META in after_body
    assert "thought leadership" in after_body


def test_the_blog_is_saved_as_a_content_piece_on_the_topic_with_its_first_revision(model):
    _, state = run_plan(BLOG, {"topic-1": topic()})

    [piece_id] = state["topics"]["topic-1"]["piece_ids"]
    piece = state["content_pieces"][piece_id]
    assert (piece["kind"], piece["topic_id"], piece["title"]) == (
        "blog",
        "topic-1",
        "AI in Healthcare: How Hospitals Read Scans Faster",
    )
    [revision] = piece["revisions"]
    created_at = revision.pop("created_at")
    assert datetime.fromisoformat(created_at) > datetime.fromisoformat(BRIEF_CREATED_AT)
    revision.pop("quality_score")  # see the Quality Score tests
    assert revision == {
        "text": BODY,
        "meta_description": META,
        "target_keyword": "AI in healthcare",
        "template": "thought leadership",
        "origin": "draft",
    }


# 155 characters, with the target keyword
META_155 = (
    "AI in healthcare helps radiologists read scans faster and spot disease earlier, while "
    "regulators race to keep up with hundreds of newly cleared AI devices."
)


def revision_of(state):
    """The only Revision of the only blog on the Topic."""
    [piece_id] = state["topics"]["topic-1"]["piece_ids"]
    [revision] = state["content_pieces"][piece_id]["revisions"]
    return revision


def test_the_blog_is_checked_and_its_quality_score_is_posted_and_saved_on_the_revision(model):
    # BODY, leaving out its Sources section and Markdown: 35 words, "AI in healthcare" twice
    # (2 x 3 / 35 = 17.1%); one H1 with the keyword but only 2 H2s, neither with it. The
    # meta description and readability pass, but it is too short. It cites and lists Sources
    # 1-3, so citations pass: 3 of 6 checks, 70 x 3/6 = 35 check points. The reviewer's
    # engagement 8 is worth 8/10 x 30 = 24 points: 59 in all.
    model.reply = blog_draft(meta_description=META_155)
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    after_body = reply.removeprefix(BODY)
    assert "Quality Score: 59/100" in after_body.replace("*", "")
    for line in [
        "keyword density 17.1% ✗",
        "meta description 155 chars ✓",
        "headings 1 H1, 2 H2s ✗",
        "readability grade 7.0 ✓",
        "length 35 words ✗",
        "citations 3 Sources cited ✓",
    ]:
        assert line in after_body, line

    quality = revision_of(state)["quality_score"]
    assert quality["score"] == 59
    assert [(c["name"], c["passed"], c["measured"]) for c in quality["checks"]] == [
        ("keyword density", False, "17.1%"),
        ("meta description", True, "155 chars"),
        ("headings", False, "1 H1, 2 H2s"),
        ("readability", True, "grade 7.0"),
        ("length", False, "35 words"),
        ("citations", True, "3 Sources cited"),
    ]
    assert all(c["reason"] for c in quality["checks"])


# Each section has a lead sentence with "AI in healthcare" and 34 short sentences of 7 words:
# 3 + 4 + 1 + 1 heading words and 5 + 5 + 6 + 5 + 4 x 238 sentence words = 982 words, with
# the keyword 6 times: 6 x 3 / 982 = 1.8%. One H1 and three H2s with the keyword in the H1 and
# an H2. Sources 1-3 are each cited, and listed.
FILLER = " ".join(["Doctors use this tool to read scans."] * 34)
PASSING_BODY = (
    "# AI in healthcare\n\n"
    f"AI in healthcare is here [1]. {FILLER}\n\n"
    "## AI in healthcare today\n\n"
    f"AI in healthcare reads scans [2]. {FILLER}\n\n"
    "## Diagnosis\n\n"
    f"AI in healthcare finds disease early. {FILLER}\n\n"
    "## Regulation\n\n"
    f"AI in healthcare needs rules [3]. {FILLER}\n\n"
    "## Sources\n\n"
    "1. [Report 1 from who.int](https://who.int/report-1)\n"
    "2. [Report 2 from nature.com](https://nature.com/report-2)\n"
    "3. [Report 3 from fda.gov](https://fda.gov/report-3)"
)


def test_a_blog_that_passes_every_check_with_full_marks_from_the_reviewer_scores_100(model):
    model.reply = blog_draft(body=PASSING_BODY, meta_description=META_155)
    model.verdict = verdict(engagement=10)
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    assert "Quality Score: 100/100" in reply.replace("*", "")
    quality = revision_of(state)["quality_score"]
    assert (quality["score"], quality["passed"]) == (100, True)


@pytest.mark.parametrize(
    ("engagement", "score", "passed"),
    [(7, 91, True), (6, 88, False)],  # 70 check points + engagement / 10 x 30
)
def test_a_blog_passes_when_every_check_passes_and_the_reviewer_reaches_the_pass_mark(
    model, engagement, score, passed
):
    model.reply = blog_draft(body=PASSING_BODY, meta_description=META_155)
    model.verdict = verdict(engagement=engagement)
    _, state = run_plan(BLOG, {"topic-1": topic()})

    quality = revision_of(state)["quality_score"]
    assert (quality["score"], quality["passed"]) == (score, passed)


def test_a_blog_that_fails_a_check_does_not_pass_even_with_full_marks_from_the_reviewer(model):
    model.reply = blog_draft(meta_description=META_155)
    model.verdict = verdict(engagement=10)
    _, state = run_plan(BLOG, {"topic-1": topic()})

    assert revision_of(state)["quality_score"]["passed"] is False


def test_the_reviewer_rating_and_top_3_suggestions_are_posted_and_all_are_saved(model):
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    after_body = reply.removeprefix(BODY)
    assert "Engagement: 8/10" in after_body.replace("*", "")
    for suggestion in SUGGESTIONS[:3]:
        assert suggestion in after_body, suggestion
    for suggestion in SUGGESTIONS[3:]:
        assert suggestion not in after_body, suggestion
    assert revision_of(state)["quality_score"]["review"] == {
        "engagement": 8,
        "suggestions": SUGGESTIONS,
    }


def test_the_reviewer_is_given_the_blog_and_its_judging_criteria(model):
    run_plan(BLOG, {"topic-1": topic()})

    [call] = model.calls_to("quality_judge")
    prompt = "\n".join(m["content"] for m in call["messages"])
    for fact in [BODY, META, "hook", "flow", "clarity", "call to action"]:
        assert fact.lower() in prompt.lower(), fact


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("every provider is down"), llm.StructuredOutputError("not a verdict")],
)
def test_when_the_reviewer_fails_the_score_comes_from_the_checks_alone(model, failure):
    # 3 of 6 checks pass: 70 x 3/6 check points, scaled to 100.
    model.reply = blog_draft(meta_description=META_155)
    model.verdict = failure
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    assert reply.startswith(BODY)
    after_body = reply.replace("*", "").removeprefix(BODY)
    assert "Quality Score: 50/100" in after_body
    assert "reviewer unavailable" in after_body
    quality = revision_of(state)["quality_score"]
    assert (quality["score"], quality["passed"], quality["review"]) == (50, False, None)


def test_when_the_reviewer_fails_a_blog_passing_every_check_still_passes(model):
    model.reply = blog_draft(body=PASSING_BODY, meta_description=META_155)
    model.verdict = RuntimeError("every provider is down")
    _, state = run_plan(BLOG, {"topic-1": topic()})

    quality = revision_of(state)["quality_score"]
    assert (quality["score"], quality["passed"]) == (100, True)


def test_a_second_blog_on_the_same_topic_is_a_separate_content_piece(model):
    graph = build_graph()
    run_plan(BLOG, {"topic-1": topic()}, graph=graph)
    model.reply = blog_draft(title="Regulating AI in Healthcare", template="how-to")
    _, state = run_plan(BLOG, graph=graph)

    piece_ids = state["topics"]["topic-1"]["piece_ids"]
    assert len(piece_ids) == 2
    titles = [state["content_pieces"][pid]["title"] for pid in piece_ids]
    assert titles == [
        "AI in Healthcare: How Hospitals Read Scans Faster",
        "Regulating AI in Healthcare",
    ]
    assert [len(state["content_pieces"][pid]["revisions"]) for pid in piece_ids] == [1, 1]


def test_the_writer_model_is_given_the_whole_brief_keywords_templates_length_and_instructions(
    model,
):
    plan = [{"agent": "blog_writer", "instructions": "aimed at hospital CIOs"}]
    run_plan(plan, {"topic-1": topic()})

    [call] = model.calls_to("blog_writer")
    prompt = "\n".join(m["content"] for m in call["messages"])
    for fact in [
        "aimed at hospital CIOs",
        # the Brief: summary, findings, statistics with their verified flags, angles, Sources
        "AI is speeding up diagnosis. Regulators are catching up.",
        "AI reads scans faster than radiologists. [1, 2]",
        "30%: faster scan reading [2] (verified)",
        "950: AI devices cleared by the FDA [3] (unverified)",
        "prefer verified statistics",
        "The radiologist's new colleague: Your next scan may be read twice.",
        "[1] Report 1 from who.int (who.int): https://who.int/report-1",
        "[3] Report 3 from fda.gov (fda.gov): https://fda.gov/report-3",
        # the keywords
        "Target keyword: AI in healthcare",
        "Secondary keywords: AI diagnosis, medical AI regulation",
        # the default word range
        "900-1800 words",
        # the four template guides
        "how-to: the problem, then the steps to solve it, then tips",
        "listicle: numbered items, one H2 each",
        "comparison: the options, the criteria they are compared on, then a verdict",
        "thought leadership: a thesis, the evidence for it, then its implications",
    ]:
        assert fact.lower() in prompt.lower(), fact


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("every provider is down"), llm.StructuredOutputError("not a blog")],
)
def test_a_failed_draft_posts_an_error_saves_nothing_and_the_plan_carries_on(model, failure):
    model.reply = failure
    plan = [*BLOG, {"agent": "linkedin_writer", "instructions": None}]
    replies, state = run_plan(plan, {"topic-1": topic()})

    assert replies == [
        "I couldn't write the blog on *AI in healthcare*: the model didn't give a usable reply. "
        "The rest of the plan carries on, so ask again to retry the blog.",
        "LinkedIn Post Writer: done (stub).",
    ]
    assert state["topics"]["topic-1"]["piece_ids"] == []
    assert not state.get("content_pieces")
    assert not state["abort_reason"]
    assert [s["agent"] for s in state["completed_steps"]] == ["blog_writer", "linkedin_writer"]


# Options from the request: target keyword, template and length


def prompt_of(call):
    return "\n".join(m["content"] for m in call["messages"])


def test_a_target_keyword_from_the_request_is_drafted_for_checked_and_saved(model):
    # The H1 becomes "AI diagnostics: How Hospitals Read Scans Faster", so BODY has 34 words
    # with "AI diagnostics" once: 1 x 2 / 34 = 5.9%.
    body = BODY.replace("# AI in Healthcare:", "# AI diagnostics:")
    model.reply = blog_draft(body=body)
    plan = [{"agent": "blog_writer", "instructions": None, "target_keyword": "AI diagnostics"}]
    [reply], state = run_plan(plan, {"topic-1": topic()})

    [call] = model.calls_to("blog_writer")
    assert "Target keyword: AI diagnostics" in prompt_of(call)
    assert "keyword density 5.9% ✗" in reply
    assert revision_of(state)["target_keyword"] == "AI diagnostics"


def test_without_a_keyword_in_the_request_the_briefs_suggested_keyword_is_used(model):
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    [call] = model.calls_to("blog_writer")
    assert "Target keyword: AI in healthcare" in prompt_of(call)
    assert "keyword density 17.1% ✗" in reply
    assert revision_of(state)["target_keyword"] == "AI in healthcare"


def test_a_template_named_in_the_request_is_used_and_recorded(model):
    model.reply = blog_draft(template="listicle")
    plan = [{"agent": "blog_writer", "instructions": None, "template": "listicle"}]
    [reply], state = run_plan(plan, {"topic-1": topic()})

    [call] = model.calls_to("blog_writer")
    assert "Template: listicle" in prompt_of(call)
    assert "Template: listicle" in reply.replace("*", "")
    assert revision_of(state)["template"] == "listicle"


def test_without_a_template_in_the_request_the_writer_picks_one_and_it_is_recorded(model):
    model.reply = blog_draft(template="comparison")
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    [call] = model.calls_to("blog_writer")
    assert "pick the template that fits best" in prompt_of(call).lower()
    assert "Template: comparison" in reply.replace("*", "")
    assert revision_of(state)["template"] == "comparison"


def length_check(state):
    [result] = [c for c in revision_of(state)["quality_score"]["checks"] if c["name"] == "length"]
    return result


def test_a_length_in_the_request_replaces_the_default_range_with_15_percent_either_side(model):
    # PASSING_BODY's 982 words are within the default 900-1800, but not 600 ± 15% (510-690).
    model.reply = blog_draft(body=PASSING_BODY)
    plan = [{"agent": "blog_writer", "instructions": "cut it to 600 words", "length": 600}]
    [reply], state = run_plan(plan, {"topic-1": topic()})

    [call] = model.calls_to("blog_writer")
    prompt = prompt_of(call)
    assert "Target length: 600 words" in prompt
    assert "900-1800" not in prompt
    assert "length 982 words ✗" in reply
    result = length_check(state)
    assert not result["passed"]
    assert "510-690" in result["reason"]


def test_without_a_length_in_the_request_the_default_range_applies(model):
    model.reply = blog_draft(body=PASSING_BODY)
    _, state = run_plan(BLOG, {"topic-1": topic()})

    result = length_check(state)
    assert result["passed"]
    assert "900-1800" in result["reason"]
