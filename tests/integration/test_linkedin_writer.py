"""The LinkedIn Writer writes a LinkedIn post from the Research Brief through the compiled workflow.

Each test seeds a Topic that already has a Research Brief, and a plan with a LinkedIn step, as
the Query Handler would leave them, then runs the workflow from the dispatcher. The model is
faked at the LLM integration module and returns scripted drafts and reviewer verdicts.
"""

from datetime import datetime

import pytest
from langchain_core.messages import HumanMessage
from langsmith import tracing_context

from contentblitz.integrations import llm
from contentblitz.workflow.graph import build_graph

# What a Revision records when the conversation's active Brand Profile is Default (#52).
DEFAULT_RECORD = {"id": "default", "name": "Default", "updated_at": "2026-10-29T00:00:00+00:00"}

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

HOOK = "Your next scan may be read twice: once by a radiologist, once by AI."
QUESTION = "Where do you see AI helping your team first? Share your thoughts in the comments."
# Eight short paragraphs, the figures 30% and 900 from the Brief, ending with a question.
TEXT = "\n\n".join(
    [
        HOOK,
        "AI now reads scans 30% faster than radiologists working alone. That is not a lab "
        "result. It is happening in hospitals today, and it changes how care teams plan their "
        "week.",
        "Regulators are catching up too. The FDA has cleared over 900 AI devices, and that "
        "number keeps climbing as more tools move from pilot to practice.",
        "So what does this mean for the people who run hospitals? Faster reads mean shorter "
        "queues, fewer missed findings and more time for the patients who need a human "
        "conversation most.",
        "Think about a rural clinic with one radiologist covering three towns. A tool that flags "
        "the urgent scans first can decide who gets seen this morning and who waits until next "
        "week.",
        "But speed is only half the story. Trust matters just as much. Clinicians want to know "
        "how a tool was tested, where it fails and who is accountable when it does.",
        "The winners will be the teams that pair AI with clear rules, good training and honest "
        "measurement of what changes for patients, not just for the budget. That takes "
        "patience, and it takes leaders who ask hard questions early.",
        QUESTION,
    ]
)
HASHTAGS = [
    "#AIinHealthcare",
    "#AIDiagnosis",
    "#Radiology",
    "#MedTech",
    "#HealthTech",
    "#DigitalHealth",
    "#Hospitals",
    "#FDA",
    "#Innovation",
    "#FutureOfCare",
]
POST = f"{TEXT}\n\n{' '.join(HASHTAGS)}"  # 1,341 characters


def linkedin_draft(**overrides):
    """A LinkedIn draft as the `linkedin_writer` model would return it."""
    return {"template": "industry insights", "text": TEXT, "hashtags": HASHTAGS, **overrides}


SUGGESTIONS = [
    "Open with the 30% statistic.",
    "Cut the fourth paragraph.",
    "Name a hospital that uses it.",
    "Ask a sharper question.",
]


def verdict(**overrides):
    return {"engagement": 8, "suggestions": SUGGESTIONS, **overrides}


class FakeModel:
    """Stands in for every model: scripted drafts for the writers and verdicts for the reviewer.

    Either may be a list, one per call to that agent, with the last item repeating; an
    exception is raised.
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
    fake = FakeModel(linkedin_draft(), verdict())
    monkeypatch.setattr(llm, "complete_structured_reply", fake)
    return fake


def topic(name="AI in healthcare"):
    return {"name": name, "research": None, "brief": BRIEF, "piece_ids": []}


def config(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def run_plan(plan, topics=None, graph=None, thread="t1", pieces=None):
    """Run `plan` from the dispatcher, as if the Query Handler had just posted it. Returns the
    replies the plan produced and the state afterwards."""
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


def prompt_of(call):
    return "\n".join(m["content"] for m in call["messages"])


def plain(reply):
    return reply.replace("*", "")


def revision_of(state):
    """The only Revision of the only LinkedIn post on the Topic."""
    [piece_id] = state["topics"]["topic-1"]["piece_ids"]
    [revision] = state["content_pieces"][piece_id]["revisions"]
    return revision


LINKEDIN = [{"agent": "linkedin_writer", "instructions": None}]


def test_a_linkedin_step_posts_the_post_ready_to_copy_with_its_character_count(model):
    [reply], _ = run_plan(LINKEDIN, {"topic-1": topic()})

    assert reply.startswith(f"```text\n{POST}\n```")
    after_post = plain(reply.removeprefix(f"```text\n{POST}\n```"))
    assert "Characters: 1,341" in after_post
    assert "Template: industry insights" in after_post


def test_the_post_is_saved_as_a_linkedin_content_piece_with_its_first_revision(model):
    _, state = run_plan(LINKEDIN, {"topic-1": topic()})

    [piece_id] = state["topics"]["topic-1"]["piece_ids"]
    piece = state["content_pieces"][piece_id]
    assert (piece["kind"], piece["topic_id"]) == ("linkedin_post", "topic-1")
    assert piece["title"].startswith("Your next scan may be read twice")
    [revision] = piece["revisions"]
    created_at = revision.pop("created_at")
    assert datetime.fromisoformat(created_at) > datetime.fromisoformat(BRIEF_CREATED_AT)
    quality = revision.pop("quality_score")
    assert quality["checks"]
    assert revision == {
        "text": POST,
        "template": "industry insights",
        "origin": "draft",
        "brand_profile": DEFAULT_RECORD,
    }


def test_hashtags_without_a_hash_are_given_one_and_end_the_post_on_one_line(model):
    model.reply = linkedin_draft(hashtags=[h.removeprefix("#") for h in HASHTAGS])
    _, state = run_plan(LINKEDIN, {"topic-1": topic()})

    assert revision_of(state)["text"] == POST


def test_the_post_is_checked_and_its_quality_score_is_posted_and_saved(model):
    # Every check passes: 70 check points. Engagement 8 is worth 24: 94 in all.
    [reply], state = run_plan(LINKEDIN, {"topic-1": topic()})

    after_post = plain(reply.split("```")[-1])
    assert "Quality Score: 94/100" in after_post
    assert "length 1,341 chars ✓" in after_post
    assert "hashtags 10 hashtags ✓" in after_post
    for line in [
        "hook 68 chars ✓",
        "paragraphs 8 paragraphs ✓",
        'call to action "comment" ✓',
        "plain text no Markdown ✓",
        "grounded figures 2 figures, all in the Brief ✓",
    ]:
        assert line in after_post, line
    assert "Automatic revisions: 0 of 3" in after_post
    assert "Engagement: 8/10" in after_post
    for suggestion in SUGGESTIONS[:3]:
        assert suggestion in after_post
    assert SUGGESTIONS[3] not in after_post
    quality = revision_of(state)["quality_score"]
    assert (quality["score"], quality["passed"], quality["revisions_used"]) == (94, True, 0)
    assert quality["checks"][:2] == [
        {
            "name": "length",
            "passed": True,
            "measured": "1,341 chars",
            "reason": "1341 characters, hashtags included; the target is 1300-1600",
        },
        {
            "name": "hashtags",
            "passed": True,
            "measured": "10 hashtags",
            "reason": "10 unique hashtags, within 8-12, all in the closing line",
        },
    ]


def test_the_writer_model_is_given_the_whole_brief_the_template_guides_and_instructions(model):
    plan = [{"agent": "linkedin_writer", "instructions": "aimed at founders"}]
    run_plan(plan, {"topic-1": topic()})

    prompt = prompt_of(model.calls_to("linkedin_writer")[0]).lower()
    for fact in [
        "aimed at founders",
        "AI is speeding up diagnosis. Regulators are catching up.",
        "AI reads scans faster than radiologists.",
        "30%: faster scan reading (verified)",
        "950: AI devices cleared by the FDA (unverified)",
        "prefer verified statistics",
        "The radiologist's new colleague: Your next scan may be read twice.",
        "Keywords: AI in healthcare, AI diagnosis, medical AI regulation",
        "hashtags from the keywords",
        "1300-1600 characters",
        "8-12 hashtags",
        "at most 150 characters",  # the opening line
        "300 characters",  # the longest paragraph
        "industry insights: a striking statistic or claim, why it matters, a takeaway",
        "personal story: a narrative arc",
        "tips: a hook, then 3-5 numbered tips as short paragraphs",
        "announcement: the news, the benefit, a call to action",
    ]:
        assert fact.lower() in prompt, fact


def test_the_reviewer_is_given_the_post_and_linkedins_judging_criteria(model):
    run_plan(LINKEDIN, {"topic-1": topic()})

    prompt = prompt_of(model.calls_to("quality_judge")[0]).lower()
    assert POST.lower() in prompt
    for criterion in ["hook", "readability on a phone", "professional tone", "call to action"]:
        assert criterion in prompt, criterion


def test_a_second_post_on_the_same_topic_is_a_separate_content_piece(model):
    graph = build_graph()
    run_plan(LINKEDIN, {"topic-1": topic()}, graph=graph)
    model.reply = linkedin_draft(text=TEXT.replace(HOOK, "AI is rewriting radiology."))
    _, state = run_plan(LINKEDIN, graph=graph)

    piece_ids = state["topics"]["topic-1"]["piece_ids"]
    assert len(piece_ids) == 2
    titles = [state["content_pieces"][pid]["title"] for pid in piece_ids]
    assert titles[1] == "AI is rewriting radiology."
    assert [len(state["content_pieces"][pid]["revisions"]) for pid in piece_ids] == [1, 1]


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("every provider is down"), llm.StructuredOutputError("not a post")],
)
def test_a_failed_draft_posts_an_error_saves_nothing_and_the_plan_carries_on(model, failure):
    model.reply = failure
    plan = [*LINKEDIN, {"agent": "image_generator", "instructions": None}]
    replies, state = run_plan(plan, {"topic-1": topic()})

    # The second reply is the image step's.
    assert len(replies) == 2
    assert replies[0] == (
        "I couldn't write the LinkedIn post on *AI in healthcare*: the model didn't give a "
        "usable reply. The rest of the plan carries on, so ask again to retry the post."
    )
    # Only the image step saved a Content Piece.
    pieces = state["content_pieces"]
    assert [pieces[p]["kind"] for p in state["topics"]["topic-1"]["piece_ids"]] == ["image"]
    assert [s["agent"] for s in state["completed_steps"]] == ["linkedin_writer", "image_generator"]


# The quality loop

TOO_LONG = linkedin_draft(text=TEXT + "\n\n" + ("More to say. " * 30).strip())  # 1,732 characters
TOO_FEW_TAGS = linkedin_draft(hashtags=HASHTAGS[:5])


def test_a_failing_draft_is_revised_until_one_passes(model):
    model.reply = [TOO_LONG, TOO_FEW_TAGS, linkedin_draft()]
    [reply], state = run_plan(LINKEDIN, {"topic-1": topic()})

    calls = model.calls_to("linkedin_writer")
    assert len(calls) == 3
    revise = prompt_of(calls[1])
    assert TOO_LONG["text"] in revise  # the current draft, to change
    assert "length" in revise and "1,732 chars" in revise and "1300-1600" in revise
    assert "Automatic revisions: 2 of 3" in plain(reply)
    quality = revision_of(state)["quality_score"]
    assert (quality["passed"], quality["revisions_used"]) == (True, 2)
    assert revision_of(state)["text"] == POST


def test_the_loop_stops_at_the_cap_and_keeps_the_highest_scoring_attempt(model):
    # None passes; the over-long draft has the best review, so it scores highest.
    model.reply = [TOO_FEW_TAGS, TOO_LONG, TOO_FEW_TAGS, TOO_FEW_TAGS]
    model.verdict = [verdict(engagement=5), verdict(engagement=9), verdict(engagement=5)]
    [reply], state = run_plan(LINKEDIN, {"topic-1": topic()})

    assert len(model.calls_to("linkedin_writer")) == 4
    assert "Automatic revisions: 3 of 3" in plain(reply)
    revision = revision_of(state)
    assert revision["text"].startswith(TOO_LONG["text"])
    assert (revision["quality_score"]["passed"], revision["quality_score"]["revisions_used"]) == (
        False,
        3,
    )


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


def test_each_attempts_checks_and_quality_score_are_traced(model):
    model.reply = [TOO_LONG, linkedin_draft()]
    client = RecordingClient()
    with tracing_context(enabled=True, client=client):
        run_plan(LINKEDIN, {"topic-1": topic()})

    runs = sorted(client.runs.values(), key=lambda r: r["start_time"])
    traced = [(r.get("outputs") or {}) for r in runs if r.get("name") == "linkedin attempt"]
    assert [t["attempt"] for t in traced] == [1, 2]
    first, second = (t["quality_score"] for t in traced)
    assert first["checks"][0] == {
        "name": "length",
        "passed": False,
        "measured": "1,732 chars",
        "reason": "1732 characters, hashtags included; the target is 1300-1600",
    }
    assert (first["passed"], second["passed"]) == (False, True)


def test_a_draft_failing_a_feed_convention_is_revised_to_a_pass(model):
    markdown = linkedin_draft(text=TEXT.replace("Trust matters", "**Trust** matters"))
    model.reply = [markdown, linkedin_draft()]
    [reply], state = run_plan(LINKEDIN, {"topic-1": topic()})

    revise = prompt_of(model.calls_to("linkedin_writer")[1])
    assert "plain text (bold)" in revise
    assert "Automatic revisions: 1 of 3" in plain(reply)
    assert revision_of(state)["text"] == POST


def test_a_draft_with_a_figure_not_in_the_brief_is_revised_to_a_pass(model):
    ungrounded = linkedin_draft(text=TEXT.replace("30% faster", "45% faster"))
    model.reply = [ungrounded, linkedin_draft()]
    [reply], state = run_plan(LINKEDIN, {"topic-1": topic()})

    revise = prompt_of(model.calls_to("linkedin_writer")[1])
    assert "grounded figures (45% not in the Brief)" in revise
    assert "Automatic revisions: 1 of 3" in plain(reply)
    assert revision_of(state)["text"] == POST


def test_figures_are_checked_against_the_briefs_findings_and_statistics(model):
    # 950 is only in the Brief's statistics, and 900 only in its findings.
    model.reply = linkedin_draft(text=TEXT.replace("over 900", "950"))
    _, state = run_plan(LINKEDIN, {"topic-1": topic()})

    checks = revision_of(state)["quality_score"]["checks"]
    assert checks[-1]["name"] == "grounded figures" and checks[-1]["passed"]


# Templates and instructions


def test_a_template_named_in_the_request_is_used_recorded_and_shown(model):
    model.reply = linkedin_draft(template="industry insights")  # what the model says it used
    plan = [{"agent": "linkedin_writer", "instructions": None, "template": "tips"}]
    [reply], state = run_plan(plan, {"topic-1": topic()})

    assert "Template: tips" in prompt_of(model.calls_to("linkedin_writer")[0])
    assert "Template: tips\n" in plain(reply)
    assert revision_of(state)["template"] == "tips"


def test_a_revised_attempt_keeps_the_template_named_in_the_request(model):
    model.reply = [TOO_LONG, linkedin_draft(template="announcement")]
    plan = [{"agent": "linkedin_writer", "instructions": None, "template": "tips"}]
    _, state = run_plan(plan, {"topic-1": topic()})

    revision = revision_of(state)
    assert (revision["quality_score"]["revisions_used"], revision["template"]) == (1, "tips")


def test_without_a_template_the_writer_picks_one_records_it_and_says_so(model):
    model.reply = linkedin_draft(template="announcement")
    [reply], state = run_plan(LINKEDIN, {"topic-1": topic()})

    prompt = prompt_of(model.calls_to("linkedin_writer")[0]).lower()
    assert "pick the template that fits best" in prompt
    assert "Template: announcement (picked by the writer)" in plain(reply)
    assert revision_of(state)["template"] == "announcement"


STORY = "Last spring our clinic in Leeds piloted an AI triage tool, and I watched the queue shrink."


def test_a_personal_story_from_the_instructions_reaches_the_drafting_and_revise_calls(model):
    model.reply = [TOO_LONG, linkedin_draft(template="personal story")]
    plan = [{"agent": "linkedin_writer", "instructions": STORY, "template": "personal story"}]
    run_plan(plan, {"topic-1": topic()})

    draft, revise = model.calls_to("linkedin_writer")
    assert "Template: personal story" in prompt_of(draft)
    assert STORY in prompt_of(draft) and STORY in prompt_of(revise)


def test_a_personal_story_without_one_is_told_to_write_an_observation_and_invent_nothing(model):
    plan = [{"agent": "linkedin_writer", "instructions": None, "template": "personal story"}]
    run_plan(plan, {"topic-1": topic()})

    prompt = prompt_of(model.calls_to("linkedin_writer")[0]).lower()
    assert "instructions: none" in prompt
    assert "observation" in prompt and "something i keep seeing" in prompt
    assert "never invent first-person events" in prompt


def test_the_steps_instructions_reach_the_revise_call(model):
    model.reply = [TOO_LONG, linkedin_draft()]
    plan = [{"agent": "linkedin_writer", "instructions": "mention our webinar"}]
    run_plan(plan, {"topic-1": topic()})

    revise = model.calls_to("linkedin_writer")[1]
    assert "mention our webinar" in prompt_of(revise)


# Overrides: a length or hashtag count from the request


def check_named(state, name):
    [result] = [c for c in revision_of(state)["quality_score"]["checks"] if c["name"] == name]
    return result


def test_a_length_in_characters_replaces_the_default_range_as_the_most(model):
    # "keep it under 1,000 characters": 850-1000. POST's 1,341 characters are too many.
    plan = [
        {
            "agent": "linkedin_writer",
            "instructions": "keep it under 1,000 characters",
            "length": 1000,
        }
    ]
    [reply], state = run_plan(plan, {"topic-1": topic()})

    prompt = prompt_of(model.calls_to("linkedin_writer")[0])
    assert "Target length: 850-1000 characters" in prompt
    assert "1300-1600" not in prompt
    assert "length 1,341 chars ✗" in reply
    assert "850-1000" in check_named(state, "length")["reason"]
    assert check_named(state, "hashtags")["passed"]  # the other checks still apply


def test_a_hashtag_count_in_the_instructions_replaces_the_default_range(model):
    plan = [{"agent": "linkedin_writer", "instructions": "only 5 hashtags please"}]
    [reply], state = run_plan(plan, {"topic-1": topic()})

    prompt = prompt_of(model.calls_to("linkedin_writer")[0])
    assert "Hashtags: 5 hashtags" in prompt
    assert "8-12" not in prompt
    assert "hashtags 10 hashtags ✗" in reply
    result = check_named(state, "hashtags")
    assert "the target is 5" in result["reason"]
    assert check_named(state, "length")["passed"]


def test_a_hashtag_count_may_be_written_as_a_word(model):
    plan = [{"agent": "linkedin_writer", "instructions": "use just three hashtags"}]
    _, state = run_plan(plan, {"topic-1": topic()})

    assert "the target is 3" in check_named(state, "hashtags")["reason"]


# Refining a post: a refine step targets a post and starts from its latest Revision

HAND_EDITED = POST.replace(HOOK, "I rewrote this hook myself: AI reads scans twice.")
DRAFT_REVISION = {
    "text": POST,
    "template": "tips",
    "origin": "draft",
    "created_at": "2026-10-21T10:00:00+00:00",
}
HAND_EDIT = {"text": HAND_EDITED, "origin": "hand_edit", "created_at": "2026-10-22T10:00:00+00:00"}


def refine_post(instructions="make it punchier", revisions=(DRAFT_REVISION, HAND_EDIT), **options):
    """Run a refine step on the seeded post piece-1, by default with a hand edit, recording no
    template, as its latest Revision."""
    plan = [
        {"agent": "linkedin_writer", "target": "piece-1", "instructions": instructions, **options}
    ]
    post = {
        "kind": "linkedin_post",
        "topic_id": "topic-1",
        "title": "Your next scan may be read twice",
        "revisions": list(revisions),
    }
    seeded = topic() | {"piece_ids": ["piece-1"]}
    return run_plan(plan, {"topic-1": seeded}, pieces={"piece-1": post})


def test_refining_a_post_revises_its_latest_revision_even_a_hand_edit(model):
    refine_post()

    refine = prompt_of(model.calls_to("linkedin_writer")[0])
    assert HAND_EDITED in refine
    assert "make it punchier" in refine


def test_a_refined_post_is_added_as_a_new_refinement_revision_on_the_same_piece(model):
    model.reply = linkedin_draft(template="announcement")  # what the model says it used
    [reply], state = refine_post()

    assert list(state["content_pieces"]) == ["piece-1"]  # no new Content Piece
    draft, hand_edit, refined = state["content_pieces"]["piece-1"]["revisions"]
    assert (draft, hand_edit) == (DRAFT_REVISION, HAND_EDIT)  # earlier Revisions are unchanged
    assert datetime.fromisoformat(refined.pop("created_at")) > datetime.fromisoformat(
        HAND_EDIT["created_at"]
    )
    quality = refined.pop("quality_score")
    assert (quality["passed"], quality["revisions_used"]) == (True, 0)
    # The template carries over from the last Revision that recorded one.
    assert refined == {
        "text": POST,
        "template": "tips",
        "origin": "refinement",
        "brand_profile": DEFAULT_RECORD,
    }
    assert reply.startswith(f"```text\n{POST}\n```")
    after_post = plain(reply.split("```")[-1])
    assert "Characters: 1,341" in after_post and "Quality Score: 94/100" in after_post
    assert "Template: tips\n" in after_post


def test_a_refinement_that_fails_a_check_is_revised_until_one_passes(model):
    model.reply = [TOO_LONG, linkedin_draft()]
    [reply], state = refine_post()

    calls = model.calls_to("linkedin_writer")
    assert len(calls) == 2  # the refinement, then one automatic revision
    assert "1,732 chars" in prompt_of(calls[1])
    assert "Automatic revisions: 1 of 3" in plain(reply)
    *_, refined = state["content_pieces"]["piece-1"]["revisions"]
    assert (refined["text"], refined["quality_score"]["revisions_used"]) == (POST, 1)


def test_a_refinement_that_names_a_template_uses_it(model):
    _, state = refine_post("turn it into an announcement", template="announcement")

    assert "Template: announcement" in prompt_of(model.calls_to("linkedin_writer")[0])
    *_, refined = state["content_pieces"]["piece-1"]["revisions"]
    assert refined["template"] == "announcement"


def test_a_refinement_with_overrides_checks_against_them(model):
    _, state = refine_post("shorter, under 1,000 characters, only 5 hashtags", length=1000)

    refine = prompt_of(model.calls_to("linkedin_writer")[0])
    assert "850-1000 characters" in refine and "Hashtags: 5 hashtags" in refine
    *_, refined = state["content_pieces"]["piece-1"]["revisions"]
    length, tags = refined["quality_score"]["checks"][:2]
    assert "850-1000" in length["reason"] and "the target is 5" in tags["reason"]


def test_a_failed_refinement_posts_an_error_and_leaves_the_revisions_unchanged(model):
    model.reply = RuntimeError("every provider is down")
    [reply], state = refine_post()

    assert reply == (
        "I couldn't refine *Your next scan may be read twice*: the model didn't give a usable "
        "reply. Its latest Revision is unchanged, so ask again to retry."
    )
    assert state["content_pieces"]["piece-1"]["revisions"] == [DRAFT_REVISION, HAND_EDIT]


def test_refining_a_post_whose_topic_has_no_brief_says_so_and_changes_nothing(model):
    plan = [{"agent": "linkedin_writer", "target": "piece-1", "instructions": "shorter"}]
    seeded = topic() | {"brief": None, "piece_ids": ["piece-1"]}
    post = {
        "kind": "linkedin_post",
        "topic_id": "topic-1",
        "title": "AI now",
        "revisions": [HAND_EDIT],
    }
    [reply], state = run_plan(plan, {"topic-1": seeded}, pieces={"piece-1": post})

    assert reply == (
        "I can't refine *AI now* yet: *AI in healthcare* has no Research Brief to check it "
        "against. Ask me to research it again first."
    )
    assert model.calls == []
    assert state["content_pieces"]["piece-1"]["revisions"] == [HAND_EDIT]


# Repurposing: a post built on an existing Content Piece

BLOG_TEXT = "# AI in Healthcare\n\nAI reads scans 30% faster [2]."
BLOG_HAND_EDIT = BLOG_TEXT + "\n\nI added this closing line myself."
BLOG_PIECE = {
    "kind": "blog",
    "topic_id": "topic-1",
    "title": "AI in Healthcare",
    "revisions": [
        {"text": BLOG_TEXT, "origin": "draft", "created_at": "2026-10-21T10:00:00+00:00"},
        {"text": BLOG_HAND_EDIT, "origin": "hand_edit", "created_at": "2026-10-22T10:00:00+00:00"},
    ],
}


def repurpose(plan):
    seeded = topic() | {"piece_ids": ["piece-1"]}
    return run_plan(plan, {"topic-1": seeded}, pieces={"piece-1": BLOG_PIECE})


def test_a_post_built_on_a_blog_is_given_its_latest_revision_and_the_whole_brief(model):
    plan = [{"agent": "linkedin_writer", "instructions": None, "source_piece": "piece-1"}]
    repurpose(plan)

    prompt = prompt_of(model.calls_to("linkedin_writer")[0])
    assert BLOG_HAND_EDIT in prompt
    assert "AI in Healthcare" in prompt  # the blog's title
    assert "30%: faster scan reading (verified)" in prompt  # the Brief, as before
    assert "take every fact and figure from the research brief" in prompt.lower()


def test_a_post_built_on_a_blog_is_a_new_piece_and_leaves_the_blog_unchanged(model):
    plan = [{"agent": "linkedin_writer", "instructions": None, "source_piece": "piece-1"}]
    _, state = repurpose(plan)

    assert state["topics"]["topic-1"]["piece_ids"] == ["piece-1", "piece-2"]
    assert state["content_pieces"]["piece-1"] == BLOG_PIECE
    assert state["content_pieces"]["piece-2"]["kind"] == "linkedin_post"


def test_without_a_source_piece_the_post_is_drafted_from_the_brief_alone(model):
    _, state = repurpose(LINKEDIN)

    prompt = prompt_of(model.calls_to("linkedin_writer")[0])
    assert BLOG_TEXT not in prompt
    assert "build on" not in prompt.lower()


def test_a_long_opening_line_is_shortened_at_a_word_for_the_title(model):
    long_hook = "Hospitals that read every scan twice are finding the second reader is an AI system"
    model.reply = linkedin_draft(text=TEXT.replace(HOOK, long_hook))
    _, state = run_plan(LINKEDIN, {"topic-1": topic()})

    [piece_id] = state["topics"]["topic-1"]["piece_ids"]
    title = state["content_pieces"][piece_id]["title"]
    assert title == "Hospitals that read every scan twice are finding the second reader is an AI…"


def test_a_hashtag_range_in_the_instructions_replaces_the_default_range(model):
    plan = [{"agent": "linkedin_writer", "instructions": "use between 3 and 5 hashtags"}]
    _, state = run_plan(plan, {"topic-1": topic()})

    assert "Hashtags: 3-5 hashtags" in prompt_of(model.calls_to("linkedin_writer")[0])
    assert "the target is 3-5" in check_named(state, "hashtags")["reason"]


def test_no_hashtags_in_the_instructions_asks_for_a_post_without_any(model):
    model.reply = linkedin_draft(hashtags=[])
    plan = [{"agent": "linkedin_writer", "instructions": "no hashtags this time"}]
    _, state = run_plan(plan, {"topic-1": topic()})

    assert "Hashtags: 0 hashtags" in prompt_of(model.calls_to("linkedin_writer")[0])
    assert revision_of(state)["text"] == TEXT
    assert check_named(state, "hashtags")["passed"]
