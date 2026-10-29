"""The SEO Blog Writer writes a blog from the Research Brief through the compiled workflow.

Each test seeds a Topic that already has a Research Brief, and a plan with a blog step, as the
Query Handler would leave them, then runs the workflow from the dispatcher. The model is faked
at the LLM integration module and returns a scripted blog draft.
"""

from datetime import datetime

import pytest
from langchain_core.messages import HumanMessage
from langsmith import tracing_context

from contentblitz.core.config import DEFAULT_CONFIG_PATH
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

    The blog writer gets `reply` and the reviewer gets `verdict`; an exception is raised. Either
    may be a list, one per Attempt: the nth call to that agent gets the nth item, and the last
    item repeats.
    """

    def __init__(self, reply, verdict):
        self.reply = reply
        self.verdict = verdict
        self.calls = []

    def __call__(self, agent, messages, schema):
        self.calls.append({"agent": agent, "messages": messages, "schema": schema})
        reply = self.verdict if agent == "quality_judge" else self.reply
        if isinstance(reply, list):
            reply = reply[min(len(self.calls_to(agent)), len(reply)) - 1]
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


def run_plan(plan, topics=None, graph=None, thread="t1", pieces=None):
    """Run `plan` from the dispatcher, as if the Query Handler had just posted it.

    `pieces` seeds the Content Pieces. Returns the replies the plan produced and the state
    afterwards.
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
    if pieces is not None:
        seeded["content_pieces"] = pieces
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
    assert quality["reviewer_unavailable"] is False


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


def test_the_reviewers_engagement_score_and_top_3_suggestions_are_posted_and_all_are_saved(model):
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

    call = model.calls_to("quality_judge")[0]
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
    assert quality["reviewer_unavailable"] is True


def test_when_the_reviewer_fails_a_blog_passing_every_check_does_not_pass_and_is_revised(model):
    # A pass needs the reviewer's average, so the loop revises up to the maximum (3) and keeps
    # the best Attempt.
    model.reply = blog_draft(body=PASSING_BODY, meta_description=META_155)
    model.verdict = RuntimeError("every provider is down")
    _, state = run_plan(BLOG, {"topic-1": topic()})

    quality = revision_of(state)["quality_score"]
    assert (quality["score"], quality["passed"], quality["revisions_used"]) == (100, False, 3)
    assert len(model.calls_to("blog_writer")) == 4


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

    call = model.calls_to("blog_writer")[0]  # the first draft
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
    plan = [*BLOG, {"agent": "image_generator", "instructions": None}]
    replies, state = run_plan(plan, {"topic-1": topic()})

    # The second reply is the image step's.
    assert len(replies) == 2
    assert replies[0] == (
        "I couldn't write the blog on *AI in healthcare*: the model didn't give a usable reply. "
        "The rest of the plan carries on, so ask again to retry the blog."
    )
    assert state["topics"]["topic-1"]["piece_ids"] == []
    assert not state.get("content_pieces")
    assert not state["abort_reason"]
    assert [s["agent"] for s in state["completed_steps"]] == ["blog_writer", "image_generator"]


def test_a_first_draft_that_passes_is_finalised_with_no_revisions_used(model):
    model.reply = blog_draft(body=PASSING_BODY, meta_description=META_155)
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    assert len(model.calls_to("blog_writer")) == 1
    assert "Automatic revisions: 0 of 3" in reply.replace("*", "")
    quality = revision_of(state)["quality_score"]
    assert (quality["passed"], quality["revisions_used"]) == (True, 0)


PASSING = blog_draft(body=PASSING_BODY, meta_description=META_155)
FAILING = blog_draft(meta_description=META_155)  # fails 3 checks: see the Quality Score tests


def test_a_failing_draft_is_revised_until_one_passes_and_the_loop_stops_there(model):
    model.reply = [FAILING, FAILING, PASSING, FAILING]
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    assert len(model.calls_to("blog_writer")) == 3
    assert len(model.calls_to("quality_judge")) == 3
    assert reply.startswith(PASSING_BODY)
    assert "Automatic revisions: 2 of 3" in reply.replace("*", "")
    quality = revision_of(state)["quality_score"]
    assert (quality["passed"], quality["revisions_used"]) == (True, 2)


# Passes every check but the meta description, which is too short: 5 of 6 checks.
NEARLY = blog_draft(body=PASSING_BODY, title="Nearly there")


def test_the_revise_call_is_given_the_current_draft_its_failed_checks_and_the_suggestions(model):
    model.reply = [FAILING, PASSING]
    run_plan(BLOG, {"topic-1": topic()})

    revise = prompt_of(model.calls_to("blog_writer")[1])
    assert BODY in revise and META_155 in revise  # the current draft, to change
    # the failed checks, with their measured values and targets
    for fact in ["keyword density", "17.1%", "1-2%", "headings", "at least 3 H2s"]:
        assert fact in revise, fact
    assert "length" in revise and "35 words" in revise and "900-1800" in revise
    for passed in ["155 chars", "grade 7.0", "3 Sources cited"]:
        assert passed not in revise, passed
    for suggestion in SUGGESTIONS:
        assert suggestion in revise, suggestion
    assert "simpler words" not in revise  # readability passed


def test_a_failed_readability_check_asks_for_shorter_sentences_and_simpler_words(model):
    long_words = (
        "Sophisticated computational methodologies revolutionise diagnostic interpretation."
    )
    model.reply = [blog_draft(body=PASSING_BODY.replace(FILLER, long_words * 20)), PASSING]
    run_plan(BLOG, {"topic-1": topic()})

    revise = prompt_of(model.calls_to("blog_writer")[1]).lower()
    assert "readability" in revise
    assert "shorter sentences and simpler words" in revise


def test_the_loop_stops_after_the_maximum_revisions_and_keeps_the_highest_scoring_attempt(model):
    # 3 of 6 checks: 35 + 24 = 59. 5 of 6 checks: 58.3 + 24 = 82.
    model.reply = [FAILING, NEARLY, FAILING, FAILING, PASSING]
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    assert len(model.calls_to("blog_writer")) == 4  # the first draft and 3 revisions
    assert len(model.calls_to("quality_judge")) == 4
    assert reply.startswith(PASSING_BODY)
    assert "Automatic revisions: 3 of 3" in reply.replace("*", "")
    [piece_id] = state["topics"]["topic-1"]["piece_ids"]
    piece = state["content_pieces"][piece_id]
    assert piece["title"] == "Nearly there"
    [revision] = piece["revisions"]  # the Attempts are not Revisions
    quality = revision["quality_score"]
    assert (quality["score"], quality["passed"], quality["revisions_used"]) == (82, False, 3)


def test_an_attempt_the_reviewer_missed_counts_no_reviewer_points_when_choosing_the_best(model):
    # Both pass 5 of 6 checks. Reviewed with engagement 6: 58.3 + 18 = 76. Unreviewed, its
    # score is 83 (its check points scaled to 100), but it is compared on its 58.3 check points.
    model.reply = [NEARLY, blog_draft(body=PASSING_BODY, title="Unreviewed")]
    model.verdict = [verdict(engagement=6), RuntimeError("every provider is down")]
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    [piece_id] = state["topics"]["topic-1"]["piece_ids"]
    assert state["content_pieces"][piece_id]["title"] == "Nearly there"
    assert "Quality Score: 76/100" in reply.replace("*", "")


def test_when_a_revise_call_fails_the_best_attempt_so_far_is_kept(model):
    model.reply = [FAILING, NEARLY, RuntimeError("every provider is down")]
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    assert len(model.calls_to("blog_writer")) == 3
    assert reply.startswith(PASSING_BODY)
    assert "Automatic revisions: 1 of 3" in reply.replace("*", "")
    [piece_id] = state["topics"]["topic-1"]["piece_ids"]
    assert state["content_pieces"][piece_id]["title"] == "Nearly there"


class RecordingClient:
    """A LangSmith client that keeps the runs it is sent instead of uploading them."""

    def __init__(self):
        self.runs = {}

    def create_run(self, **run):
        self.runs[run["id"]] = run

    def update_run(self, run_id, **update):
        self.runs.setdefault(run_id, {}).update(update)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_each_attempts_checks_reviewer_scores_and_quality_score_are_traced(model):
    model.reply = [FAILING, PASSING]
    model.verdict = [verdict(engagement=8), verdict(engagement=9)]
    client = RecordingClient()
    with tracing_context(enabled=True, client=client):
        _, state = run_plan(BLOG, {"topic-1": topic()})

    runs = sorted(client.runs.values(), key=lambda r: r["start_time"])
    traced = [(r.get("outputs") or {}) for r in runs if r.get("name") == "blog attempt"]
    assert [t["attempt"] for t in traced] == [1, 2]
    first, second = (t["quality_score"] for t in traced)
    assert (first["score"], first["passed"], first["review"]["engagement"]) == (59, False, 8)
    assert [c["passed"] for c in first["checks"]] == [False, True, False, True, False, True]
    assert (second["score"], second["passed"], second["review"]["engagement"]) == (97, True, 9)
    assert all(c["passed"] for c in second["checks"])


def test_the_maximum_revisions_comes_from_config(model, tmp_path, monkeypatch):
    models = tmp_path / "models.yaml"
    defaults = DEFAULT_CONFIG_PATH.read_text()
    models.write_text(defaults.replace("max_revisions: 3", "max_revisions: 1"))
    monkeypatch.setenv("CONTENTBLITZ_CONFIG", str(models))
    model.reply = FAILING
    [reply], _ = run_plan(BLOG, {"topic-1": topic()})

    assert len(model.calls_to("blog_writer")) == 2
    assert "Automatic revisions: 1 of 1" in reply.replace("*", "")


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

    call = model.calls_to("blog_writer")[0]  # the first draft
    assert "Target keyword: AI diagnostics" in prompt_of(call)
    assert "keyword density 5.9% ✗" in reply
    assert revision_of(state)["target_keyword"] == "AI diagnostics"


def test_without_a_keyword_in_the_request_the_briefs_suggested_keyword_is_used(model):
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    call = model.calls_to("blog_writer")[0]  # the first draft
    assert "Target keyword: AI in healthcare" in prompt_of(call)
    assert "keyword density 17.1% ✗" in reply
    assert revision_of(state)["target_keyword"] == "AI in healthcare"


def test_a_template_named_in_the_request_is_used_and_recorded(model):
    model.reply = blog_draft(template="listicle")
    plan = [{"agent": "blog_writer", "instructions": None, "template": "listicle"}]
    [reply], state = run_plan(plan, {"topic-1": topic()})

    call = model.calls_to("blog_writer")[0]  # the first draft
    assert "Template: listicle" in prompt_of(call)
    assert "Template: listicle" in reply.replace("*", "")
    assert revision_of(state)["template"] == "listicle"


def test_a_revised_attempt_keeps_the_template_named_in_the_request(model):
    model.reply = [FAILING, {**PASSING, "template": "comparison"}]
    plan = [{"agent": "blog_writer", "instructions": None, "template": "listicle"}]
    _, state = run_plan(plan, {"topic-1": topic()})

    revision = revision_of(state)
    assert (revision["quality_score"]["revisions_used"], revision["template"]) == (1, "listicle")


def test_without_a_template_in_the_request_the_writer_picks_one_and_it_is_recorded(model):
    model.reply = blog_draft(template="comparison")
    [reply], state = run_plan(BLOG, {"topic-1": topic()})

    call = model.calls_to("blog_writer")[0]  # the first draft
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

    call = model.calls_to("blog_writer")[0]  # the first draft
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


# Refining a blog: a refine step targets a blog and starts from its latest Revision

HAND_EDIT_BODY = BODY.replace(
    "AI in healthcare is changing how scans are read [1].",
    "I rewrote this intro myself: AI in healthcare is changing how scans are read [1].",
)
DRAFT_REVISION = {
    "text": BODY,
    "meta_description": META,
    "target_keyword": "AI in healthcare",
    "template": "thought leadership",
    "origin": "draft",
    "created_at": "2026-10-21T10:00:00+00:00",
}
HAND_EDIT = {
    "text": HAND_EDIT_BODY,
    "meta_description": META_155,
    "target_keyword": "AI in healthcare",
    "template": "thought leadership",
    "origin": "hand_edit",
    "created_at": "2026-10-22T10:00:00+00:00",
}


def refine_blog(
    instructions="make the intro shorter", revisions=(DRAFT_REVISION, HAND_EDIT), **options
):
    """Run a refine step on the seeded blog piece-1, by default with a hand edit as its latest
    Revision. `options` are the request's blog options on the step."""
    plan = [{"agent": "blog_writer", "target": "piece-1", "instructions": instructions, **options}]
    blog = {
        "kind": "blog",
        "topic_id": "topic-1",
        "title": "AI in Healthcare: How Hospitals Read Scans Faster",
        "revisions": list(revisions),
    }
    seeded = topic() | {"piece_ids": ["piece-1"]}
    return run_plan(plan, {"topic-1": seeded}, pieces={"piece-1": blog})


def test_refining_a_blog_revises_its_latest_revision_even_a_hand_edit(model):
    model.reply = PASSING
    refine_blog()

    refine = prompt_of(model.calls_to("blog_writer")[0])
    assert HAND_EDIT_BODY in refine and META_155 in refine  # the latest Revision, to change
    assert "make the intro shorter" in refine


def test_a_refined_blog_is_added_as_a_new_refinement_revision_on_the_same_piece(model):
    model.reply = PASSING
    _, state = refine_blog()

    assert list(state["content_pieces"]) == ["piece-1"]  # no new Content Piece
    assert state["topics"]["topic-1"]["piece_ids"] == ["piece-1"]
    draft, hand_edit, refined = state["content_pieces"]["piece-1"]["revisions"]
    assert (draft, hand_edit) == (DRAFT_REVISION, HAND_EDIT)  # earlier Revisions are unchanged
    assert datetime.fromisoformat(refined.pop("created_at")) > datetime.fromisoformat(
        HAND_EDIT["created_at"]
    )
    quality = refined.pop("quality_score")
    assert (quality["passed"], quality["revisions_used"]) == (True, 0)
    assert refined == {
        "text": PASSING_BODY,
        "meta_description": META_155,
        "target_keyword": "AI in healthcare",
        "template": "thought leadership",
        "origin": "refinement",
    }


def test_a_refinement_that_fails_a_check_is_revised_until_one_passes(model):
    model.reply = [FAILING, PASSING]
    [reply], state = refine_blog()

    calls = model.calls_to("blog_writer")
    assert len(calls) == 2  # the refinement, then one automatic revision
    assert len(model.calls_to("quality_judge")) == 2
    revise = prompt_of(calls[1])
    assert BODY in revise and "keyword density" in revise  # the refined draft and what failed
    assert "Automatic revisions: 1 of 3" in reply.replace("*", "")
    *_, refined = state["content_pieces"]["piece-1"]["revisions"]
    assert (refined["text"], refined["quality_score"]["revisions_used"]) == (PASSING_BODY, 1)


def test_the_chat_shows_the_refined_blog_its_meta_description_and_quality_score(model):
    model.reply = PASSING
    model.verdict = verdict(engagement=10)
    [reply], _ = refine_blog()

    assert reply.startswith(PASSING_BODY)
    after_body = reply.removeprefix(PASSING_BODY).replace("*", "")
    assert f"Meta description: {META_155}" in after_body
    assert "Quality Score: 100/100" in after_body
    assert "keyword density 1.8% ✓" in after_body


WRITTEN_FOR = {"target_keyword": "AI diagnostics", "template": "listicle"}


def test_a_refinement_keeps_the_keyword_and_template_of_the_latest_revision(model):
    # The Brief suggests "AI in healthcare"; the blog was written for "AI diagnostics".
    model.reply = blog_draft(template="comparison")  # the template the model says it used
    _, state = refine_blog(revisions=[DRAFT_REVISION | WRITTEN_FOR, HAND_EDIT | WRITTEN_FOR])

    refine = prompt_of(model.calls_to("blog_writer")[0])
    assert "Target keyword: AI diagnostics" in refine
    assert "Template: listicle" in refine
    *_, refined = state["content_pieces"]["piece-1"]["revisions"]
    assert (refined["target_keyword"], refined["template"]) == ("AI diagnostics", "listicle")


def test_a_hand_edit_that_records_no_keyword_or_template_keeps_the_ones_before_it(model):
    bare_edit = {k: v for k, v in HAND_EDIT.items() if k not in WRITTEN_FOR}
    _, state = refine_blog(revisions=[DRAFT_REVISION | WRITTEN_FOR, bare_edit])

    *_, refined = state["content_pieces"]["piece-1"]["revisions"]
    assert (refined["target_keyword"], refined["template"]) == ("AI diagnostics", "listicle")


def test_a_refinement_that_sets_a_keyword_template_and_length_uses_them(model):
    model.reply = blog_draft(body=PASSING_BODY)
    _, state = refine_blog(
        "cut it to 600 words, as a how-to on AI radiology",
        target_keyword="AI radiology",
        template="how-to",
        length=600,
    )

    refine = prompt_of(model.calls_to("blog_writer")[0])
    assert "Target keyword: AI radiology" in refine
    assert "Template: how-to" in refine
    assert "Target length: 600 words" in refine
    *_, refined = state["content_pieces"]["piece-1"]["revisions"]
    assert (refined["target_keyword"], refined["template"]) == ("AI radiology", "how-to")
    [length] = [c for c in refined["quality_score"]["checks"] if c["name"] == "length"]
    assert "510-690" in length["reason"]


def test_a_failed_refinement_posts_an_error_and_leaves_the_revisions_unchanged(model):
    model.reply = RuntimeError("every provider is down")
    [reply], state = refine_blog()

    assert reply == (
        "I couldn't refine *AI in Healthcare: How Hospitals Read Scans Faster*: the model "
        "didn't give a usable reply. Its latest Revision is unchanged, so ask again to retry."
    )
    assert state["content_pieces"]["piece-1"]["revisions"] == [DRAFT_REVISION, HAND_EDIT]


def test_refining_a_blog_whose_topic_has_no_brief_says_so_and_changes_nothing(model):
    # Fresh research clears the Brief; if the Content Strategist then fails, there is none.
    plan = [{"agent": "blog_writer", "target": "piece-1", "instructions": "shorter"}]
    seeded = topic() | {"brief": None, "piece_ids": ["piece-1"]}
    blog = {"kind": "blog", "topic_id": "topic-1", "title": "AI now", "revisions": [HAND_EDIT]}
    [reply], state = run_plan(plan, {"topic-1": seeded}, pieces={"piece-1": blog})

    assert reply == (
        "I can't refine *AI now* yet: *AI in healthcare* has no Research Brief to check it "
        "against. Ask me to research it again first."
    )
    assert model.calls == []
    assert state["content_pieces"]["piece-1"]["revisions"] == [HAND_EDIT]
