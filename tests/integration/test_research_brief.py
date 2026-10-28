"""The Content Strategist builds the Research Brief through the compiled workflow.

Each test seeds a Topic with a Research Record and a plan, as the Query Handler would leave
them, then runs the workflow from the dispatcher. The model is faked at the LLM integration
module and returns a scripted draft Brief.
"""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langsmith import tracing_context

from contentblitz.integrations import llm
from contentblitz.workflow.graph import build_graph

GATHERED_AT = "2026-10-20T09:30:00+00:00"


def research_source(n, site, page_text=""):
    return {
        "title": f"Report {n} from {site}",
        "url": f"https://{site}/report-{n}",
        "site": site,
        "snippet": f"Snippet {n}.",
        "page_text": page_text,
        "query": "AI in healthcare",
        "provider": "SerpAPI",
        "retrieved_at": f"2026-10-20T09:2{n}:00+00:00",
    }


def research_record(*sites):
    return {
        "queries": ["AI in healthcare"],
        "providers": ["SerpAPI"],
        "gathered_at": GATHERED_AT,
        "sources": [research_source(n, site) for n, site in enumerate(sites, start=1)],
    }


HEALTHCARE = research_record("who.int", "nature.com", "fda.gov")
HEALTHCARE["sources"][1]["page_text"] = "Radiologists using AI read scans 30 percent faster."


def topic(name="AI in healthcare", record=HEALTHCARE):
    return {"name": name, "research": record, "brief": None, "piece_ids": []}


def draft(**overrides):
    """A draft Brief as the model would return it."""
    return {
        "summary": "AI is speeding up diagnosis. Regulators are catching up.",
        "key_findings": [
            {"statement": "AI reads scans faster than radiologists.", "source_numbers": [1, 2]},
            {"statement": "The FDA has cleared over 900 AI devices.", "source_numbers": [3]},
            {"statement": "Adoption is slowest in rural clinics.", "source_numbers": [2]},
            {"statement": "Most hospitals pilot AI before buying.", "source_numbers": [1]},
        ],
        "statistics": [
            {"figure": "30%", "description": "faster scan reading", "source_numbers": [2]},
            {
                "figure": "950",
                "description": "AI devices cleared by the FDA",
                "source_numbers": [3],
            },
        ],
        "angles": [
            {
                "title": "The radiologist's new colleague",
                "hook": "Your next scan may be read twice.",
            },
            {"title": "Regulation catches up", "hook": "950 approvals and counting."},
            {"title": "The rural gap", "hook": "Why the clinics that need AI most get it last."},
        ],
        "target_keyword": "AI in healthcare",
        "secondary_keywords": ["AI diagnosis", "medical AI regulation", "AI radiology"],
        **overrides,
    }


class FakeStrategist:
    """Stands in for the `content_strategist` model: returns the scripted draft Brief.

    Replies queued in `replies` are used first, one per call; an exception is raised.
    """

    def __init__(self, reply):
        self.reply = reply
        self.replies = []
        self.calls = []

    def __call__(self, agent, messages, schema):
        self.calls.append({"agent": agent, "messages": messages, "schema": schema})
        reply = self.replies.pop(0) if self.replies else self.reply
        if isinstance(reply, Exception):
            raise reply
        return schema.model_validate(reply)


@pytest.fixture
def model(monkeypatch):
    fake = FakeStrategist(draft())
    monkeypatch.setattr(llm, "complete_structured", fake)
    return fake


def config(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def run_plan(plan, topics, active="topic-1", graph=None, thread="t1"):
    """Run `plan` from the dispatcher, as if the Query Handler had just posted it.

    Returns the replies the plan produced and the state afterwards.
    """
    graph = graph or build_graph()
    before = len(graph.get_state(config(thread)).values.get("messages", []))
    seeded = {
        "messages": [HumanMessage(content="go")],
        "topics": topics,
        "active_topic_id": active,
        "plan": plan,
        "completed_steps": [],
        "abort_reason": None,
    }
    graph.update_state(config(thread), seeded, as_node="query_handler")
    result = graph.invoke(None, config(thread))
    return [m.content for m in result["messages"][before + 1 :]], result


RESEARCH_ONLY = [{"agent": "content_strategist", "instructions": None}]


def test_the_brief_is_saved_on_the_topic_with_sources_from_the_research_record(model):
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic()})

    brief = state["topics"]["topic-1"]["brief"]
    assert brief["summary"] == "AI is speeding up diagnosis. Regulators are catching up."
    assert brief["key_findings"][0] == {
        "statement": "AI reads scans faster than radiologists.",
        "source_numbers": [1, 2],
    }
    assert brief["statistics"][1] == {
        "figure": "950",
        "description": "AI devices cleared by the FDA",
        "source_numbers": [3],
        "verified": False,
    }
    assert brief["angles"][2] == {
        "title": "The rural gap",
        "hook": "Why the clinics that need AI most get it last.",
    }
    assert brief["target_keyword"] == "AI in healthcare"
    assert brief["secondary_keywords"] == ["AI diagnosis", "medical AI regulation", "AI radiology"]
    assert brief["sources"] == [
        {
            "number": 1,
            "title": "Report 1 from who.int",
            "url": "https://who.int/report-1",
            "site": "who.int",
            "retrieved_at": "2026-10-20T09:21:00+00:00",
        },
        {
            "number": 2,
            "title": "Report 2 from nature.com",
            "url": "https://nature.com/report-2",
            "site": "nature.com",
            "retrieved_at": "2026-10-20T09:22:00+00:00",
        },
        {
            "number": 3,
            "title": "Report 3 from fda.gov",
            "url": "https://fda.gov/report-3",
            "site": "fda.gov",
            "retrieved_at": "2026-10-20T09:23:00+00:00",
        },
    ]
    assert brief["research_gathered_at"] == GATHERED_AT
    assert brief["model"] == "openai/gpt-4o"
    assert datetime.fromisoformat(brief["created_at"]) > datetime.fromisoformat(GATHERED_AT)


def test_the_strategist_model_is_given_the_topic_instructions_and_numbered_sources(model):
    record = research_record("who.int", "nature.com")
    record["sources"][0]["page_text"] = "WHO says AI triage cut waiting times by 30%."
    plan = [{"agent": "content_strategist", "instructions": "focus on regulation"}]
    run_plan(plan, {"topic-1": topic(record=record)})

    [call] = model.calls
    assert call["agent"] == "content_strategist"
    prompt = "\n".join(m["content"] for m in call["messages"])
    for fact in [
        "AI in healthcare",
        "focus on regulation",
        "[1] Report 1 from who.int",
        "[2] Report 2 from nature.com",
        "Snippet 1.",
        "WHO says AI triage cut waiting times by 30%.",
    ]:
        assert fact in prompt


def test_citations_to_sources_that_dont_exist_are_removed_and_uncited_claims_dropped(model):
    model.reply = draft(
        key_findings=[
            {"statement": "AI reads scans faster.", "source_numbers": [2, 7, 0, 2]},
            {"statement": "Made up.", "source_numbers": [4, 9]},
            {"statement": "Never cited.", "source_numbers": []},
            {"statement": "The FDA has cleared AI devices.", "source_numbers": [3, -1]},
            {"statement": "Hospitals pilot AI first.", "source_numbers": [1]},
        ],
        statistics=[
            {"figure": "30%", "description": "faster reading", "source_numbers": [5, 2]},
            {"figure": "12", "description": "invented figure", "source_numbers": [4]},
        ],
    )
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic()})

    brief = state["topics"]["topic-1"]["brief"]
    assert brief["key_findings"] == [
        {"statement": "AI reads scans faster.", "source_numbers": [2]},
        {"statement": "The FDA has cleared AI devices.", "source_numbers": [3]},
        {"statement": "Hospitals pilot AI first.", "source_numbers": [1]},
    ]
    assert brief["statistics"] == [
        {"figure": "30%", "description": "faster reading", "source_numbers": [2], "verified": True},
    ]


REPORT = """\
# AI in healthcare

*Research gathered on 20 October 2026.*

## Summary

AI is speeding up diagnosis. Regulators are catching up.

## Key findings

- AI reads scans faster than radiologists. [[1]](https://who.int/report-1) \
[[2]](https://nature.com/report-2)
- The FDA has cleared over 900 AI devices. [[3]](https://fda.gov/report-3)
- Adoption is slowest in rural clinics. [[2]](https://nature.com/report-2)
- Most hospitals pilot AI before buying. [[1]](https://who.int/report-1)

## Statistics

| Figure | What it measures | Sources | Verified |
| --- | --- | --- | --- |
| 30% | faster scan reading | [[2]](https://nature.com/report-2) | Yes |
| 950 | AI devices cleared by the FDA | [[3]](https://fda.gov/report-3) | No |

*Verified: the number appears in a cited Source's text.*

## Content angles

- **The radiologist's new colleague**: Your next scan may be read twice.
- **Regulation catches up**: 950 approvals and counting.
- **The rural gap**: Why the clinics that need AI most get it last.

## Keywords

**Target keyword:** AI in healthcare

**Secondary keywords:** AI diagnosis, medical AI regulation, AI radiology

## Sources

1. [Report 1 from who.int](https://who.int/report-1), who.int, retrieved 20 October 2026
2. [Report 2 from nature.com](https://nature.com/report-2), nature.com, retrieved 20 October 2026
3. [Report 3 from fda.gov](https://fda.gov/report-3), fda.gov, retrieved 20 October 2026"""


def test_a_research_only_plan_posts_the_full_research_report(model):
    replies, _ = run_plan(RESEARCH_ONLY, {"topic-1": topic()})
    assert replies == [REPORT]


@pytest.mark.parametrize("writer", ["blog_writer", "linkedin_writer"])
def test_when_writers_follow_the_strategist_posts_a_one_line_summary(model, writer):
    model.reply = draft(
        statistics=[{"figure": "30%", "description": "faster reading", "source_numbers": [2]}]
    )
    plan = [*RESEARCH_ONLY, {"agent": writer, "instructions": None}]
    replies, state = run_plan(plan, {"topic-1": topic()})

    assert replies[0] == (
        "Research Brief ready: 4 findings, 1 statistic (1 verified), 3 Sources. "
        "Ask 'show me the research' to read it."
    )
    assert len(replies) == 2  # the writer ran after it
    assert state["topics"]["topic-1"]["brief"]["summary"]


def test_the_one_line_summary_counts_the_verified_statistics(model):
    # 30% is in Source 2's text; 950 is in no Source's text.
    plan = [*RESEARCH_ONLY, {"agent": "blog_writer", "instructions": None}]
    replies, _ = run_plan(plan, {"topic-1": topic()})

    assert replies[0].startswith("Research Brief ready: 4 findings, 2 statistics (1 verified), ")


def test_an_image_after_the_brief_still_gets_the_full_report(model):
    plan = [*RESEARCH_ONLY, {"agent": "image_generator", "instructions": None}]
    replies, _ = run_plan(plan, {"topic-1": topic()})
    assert replies == [REPORT, "Image Generation Agent: done (stub)."]


def test_a_report_with_no_statistics_says_so_instead_of_an_empty_table(model):
    model.reply = draft(statistics=[])
    [report] = run_plan(RESEARCH_ONLY, {"topic-1": topic()})[0]

    assert (
        "## Statistics\n\nNo statistics were found in the Sources.\n\n## Content angles" in report
    )


def test_table_cells_escape_pipes_so_the_statistics_table_keeps_its_shape(model):
    model.reply = draft(
        statistics=[{"figure": "1|2", "description": "odd | figure", "source_numbers": [1]}]
    )
    [report] = run_plan(RESEARCH_ONLY, {"topic-1": topic()})[0]

    assert "| 1\\|2 | odd \\| figure | [[1]](https://who.int/report-1) |" in report


def test_the_brief_never_takes_sources_from_the_model(model):
    model.reply = draft(sources=[{"number": 1, "title": "Fake", "url": "https://made.up/"}])
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic()})

    brief = state["topics"]["topic-1"]["brief"]
    assert [s["url"] for s in brief["sources"]] == [
        "https://who.int/report-1",
        "https://nature.com/report-2",
        "https://fda.gov/report-3",
    ]


def test_each_topic_gets_its_own_brief_from_its_own_research(model):
    finance = topic("AI in finance", research_record("ft.com", "ecb.europa.eu"))
    graph = build_graph()
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic(), "topic-2": finance}, graph=graph)
    healthcare_brief = state["topics"]["topic-1"]["brief"]
    assert state["topics"]["topic-2"]["brief"] is None

    model.reply = draft(summary="Banks use AI to catch fraud.")
    replies, state = run_plan(RESEARCH_ONLY, state["topics"], active="topic-2", graph=graph)

    assert replies[0].startswith("# AI in finance\n")
    assert state["topics"]["topic-1"]["brief"] == healthcare_brief
    finance_brief = state["topics"]["topic-2"]["brief"]
    assert finance_brief["summary"] == "Banks use AI to catch fraud."
    assert [s["site"] for s in finance_brief["sources"]] == ["ft.com", "ecb.europa.eu"]


def test_rebuilding_a_brief_replaces_the_topics_only_brief(model):
    graph = build_graph()
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic()}, graph=graph)

    model.reply = draft(summary="A newer summary.")
    _, state = run_plan(RESEARCH_ONLY, state["topics"], graph=graph)

    assert state["topics"]["topic-1"]["brief"]["summary"] == "A newer summary."


def test_with_no_research_record_the_strategist_stops_the_plan(model):
    plan = [*RESEARCH_ONLY, {"agent": "blog_writer", "instructions": None}]
    replies, state = run_plan(plan, {"topic-1": topic(record=None)})

    assert replies == [
        "There's no research on *AI in healthcare* to build a Research Brief from, "
        "so I stopped the plan. Ask me to research it first."
    ]
    assert state["topics"]["topic-1"]["brief"] is None
    assert model.calls == []


def test_with_no_active_topic_the_strategist_stops_the_plan(model):
    replies, _ = run_plan(RESEARCH_ONLY, {}, active=None)
    assert replies == [
        "There's no active Topic to build a Research Brief for, so I stopped the plan."
    ]


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


def test_the_briefs_counts_are_traced(model):
    model.reply = draft(
        key_findings=[
            *draft()["key_findings"],
            {"statement": "Made up.", "source_numbers": [9]},
        ],
    )
    client = RecordingClient()
    with tracing_context(enabled=True, client=client):
        run_plan(RESEARCH_ONLY, {"topic-1": topic()})

    outputs = [r.get("outputs") or {} for r in client.runs.values()]
    [traced] = [o for o in outputs if "findings" in o]
    assert traced == {
        "findings": 4,
        "statistics": 2,
        "verified_statistics": 1,
        "unverified_statistics": 1,
        "sources": 3,
        "dropped_findings": 1,
        "dropped_statistics": 0,
    }


def statistic(figure, *source_numbers):
    return {"figure": figure, "description": "a figure", "source_numbers": list(source_numbers)}


def test_a_statistic_is_verified_when_its_number_is_in_a_cited_sources_text(model):
    record = research_record("who.int", "nature.com", "fda.gov")
    record["sources"][1]["page_text"] = "Reading was 30% faster with AI."
    record["sources"][2]["page_text"] = "The FDA has cleared 950 devices."
    model.reply = draft(
        statistics=[
            statistic("30%", 1, 2),  # in Source 2, which it cites
            statistic("950", 1),  # only in Source 3, which it doesn't cite
            statistic("12", 1, 2, 3),  # in no Source at all
        ]
    )
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic(record=record)})

    statistics = state["topics"]["topic-1"]["brief"]["statistics"]
    assert [(s["figure"], s["verified"]) for s in statistics] == [
        ("30%", True),
        ("950", False),
        ("12", False),
    ]


@pytest.mark.parametrize(
    ("figure", "page_text", "verified"),
    [
        ("1,200", "Some 1200 hospitals use it.", True),
        ("1200", "Some 1,200 hospitals use it.", True),
        ("45%", "Usage rose 45 percent last year.", True),
        ("45 percent", "Usage rose 45% last year.", True),
        ("45 %", "Usage rose  45 %  last year.", True),
        ("$4.5 billion", "A market worth $4.5\n billion.", True),
        ("45%", "Usage rose 45-percent last year.", True),
        ("45-percent", "Usage rose 45% last year.", True),
        ("45%", "Usage rose 45percent last year.", True),
        ("45 Percent", "Usage rose 45 per cent last year.", True),
        ("1 200", "Some 1200 hospitals use it.", True),
        ("1,200", "Some 1 200 hospitals use it.", True),
        ("1,200", "Some 1\u00a0200 hospitals use it.", True),
        ("1\u00a0200\u00a0000", "Some 1,200,000 scans.", True),
        ("1 200", "Some 12 000 hospitals use it.", False),
        ("200", "Some 1 200 hospitals use it.", False),
        ("30%", "Usage rose 130% last year.", False),
        ("30%", "Usage rose 30.5% last year.", False),
        ("45%", "We asked 45 people.", False),
        ("1,200", "Some 12,000 hospitals use it.", False),
    ],
)
def test_matching_ignores_thousands_separators_percent_wording_and_spacing(
    model, figure, page_text, verified
):
    record = research_record("who.int")
    record["sources"][0]["page_text"] = page_text
    model.reply = draft(statistics=[statistic(figure, 1)])
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic(record=record)})

    [stat] = state["topics"]["topic-1"]["brief"]["statistics"]
    assert stat["verified"] is verified


@pytest.mark.parametrize(
    ("figure", "page_text", "verified"),
    [
        # A year isn't the number being claimed: 45% must be in the Source too.
        ("In 2024, 45% of clinics", "In 2024, 12% of clinics used AI.", False),
        ("In 2024, 45% of clinics", "Some 45% of clinics used AI last year.", True),
        ("45% of clinics (2024)", "In 2024, 12% of clinics used AI.", False),
        # Every number counts, not just the first.
        ("3 in 4 patients", "3 in 10 patients waited.", False),
        ("3 in 4 patients", "Now 3 in 4 patients are seen within a week.", True),
        ("1,200 of 5,000 hospitals", "1,200 hospitals use it.", False),
        # A year that is the only number is the claim itself.
        ("2019", "The first device was cleared in 2019.", True),
        ("2019", "The first device was cleared in 2021.", False),
        # A grouped number is not a year, even if its digits look like one.
        ("1,950 clinics in 2024", "There were 1,950 clinics in 2023.", True),
    ],
)
def test_every_significant_number_in_a_figure_must_be_in_a_cited_source(
    model, figure, page_text, verified
):
    record = research_record("who.int")
    record["sources"][0]["page_text"] = page_text
    model.reply = draft(statistics=[statistic(figure, 1)])
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic(record=record)})

    [stat] = state["topics"]["topic-1"]["brief"]["statistics"]
    assert stat["verified"] is verified


def test_a_number_in_a_cited_sources_snippet_verifies_it(model):
    record = research_record("who.int")
    record["sources"][0]["snippet"] = "Over 950 AI devices are cleared."
    model.reply = draft(statistics=[statistic("950", 1)])
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic(record=record)})

    assert state["topics"]["topic-1"]["brief"]["statistics"][0]["verified"] is True


THIN = draft(
    key_findings=[
        {"statement": "AI reads scans faster.", "source_numbers": [2]},
        {"statement": "Made up.", "source_numbers": [7]},
        {"statement": "Never cited.", "source_numbers": []},
    ]
)


def test_too_few_grounded_findings_retries_once_saying_what_was_dropped(model):
    model.replies = [THIN, draft()]
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic()})

    first, retry = model.calls
    assert retry["messages"][: len(first["messages"])] == first["messages"]
    feedback = retry["messages"][-1]["content"]
    for fact in ["Made up.", "Never cited.", "1 to 3"]:
        assert fact in feedback
    assert len(state["topics"]["topic-1"]["brief"]["key_findings"]) == 4


def test_a_draft_with_too_few_findings_and_none_dropped_is_asked_for_more(model):
    one = draft(key_findings=[{"statement": "AI reads scans faster.", "source_numbers": [2]}])
    model.replies = [one, draft()]
    run_plan(RESEARCH_ONLY, {"topic-1": topic()})

    feedback = model.calls[1]["messages"][-1]["content"]
    assert feedback == (
        "The Brief needs at least 2 key findings, each citing the Sources it comes from. "
        "The Sources are numbered 1 to 3."
    )


THIN_BRIEF = (
    "I couldn't build a Research Brief on *AI in healthcare*: too few findings could be traced "
    "to the Sources, so I stopped the plan. The research is kept, so ask again to retry the "
    "Brief without searching the web again."
)


def test_a_second_thin_draft_stops_the_plan_before_any_writer_runs(model):
    model.reply = THIN
    plan = [*RESEARCH_ONLY, {"agent": "blog_writer", "instructions": None}]
    replies, state = run_plan(plan, {"topic-1": topic()})

    assert replies == [THIN_BRIEF]
    assert len(model.calls) == 2
    assert state["topics"]["topic-1"] == topic()


NO_MODEL = (
    "I couldn't build a Research Brief on *AI in healthcare*: the model didn't give a usable "
    "reply, so I stopped the plan. The research is kept, so ask again to retry the Brief "
    "without searching the web again."
)


@pytest.mark.parametrize(
    "failure",
    [ConnectionError("every provider failed"), llm.StructuredOutputError("not a Brief")],
    ids=["every provider fails", "output never fits the schema"],
)
@pytest.mark.parametrize("first_replies", [[], [THIN]], ids=["first call", "retry"])
def test_a_model_failure_stops_the_plan_with_a_reason_instead_of_raising(
    model, failure, first_replies
):
    model.replies = [*first_replies, failure]
    plan = [*RESEARCH_ONLY, {"agent": "blog_writer", "instructions": None}]
    replies, state = run_plan(plan, {"topic-1": topic()})

    assert replies == [NO_MODEL]
    assert state["topics"]["topic-1"] == topic()


def test_a_brief_is_cut_to_the_most_findings_statistics_angles_and_keywords_kept(model):
    findings = [{"statement": f"Finding {n}.", "source_numbers": [1]} for n in range(1, 11)]
    model.reply = draft(
        key_findings=[{"statement": "Made up.", "source_numbers": [9]}, *findings],
        statistics=[statistic(str(n), 1) for n in range(1, 11)],
        angles=[{"title": f"Angle {n}", "hook": "A hook."} for n in range(1, 8)],
        secondary_keywords=[f"keyword {n}" for n in range(1, 9)],
    )
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic()})

    brief = state["topics"]["topic-1"]["brief"]
    assert [f["statement"] for f in brief["key_findings"]] == [f"Finding {n}." for n in range(1, 9)]
    assert [s["figure"] for s in brief["statistics"]] == [str(n) for n in range(1, 9)]
    assert [a["title"] for a in brief["angles"]] == [f"Angle {n}" for n in range(1, 6)]
    assert brief["secondary_keywords"] == [f"keyword {n}" for n in range(1, 7)]


def litellm_reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.parametrize(
    "too_few",
    [
        {"angles": draft()["angles"][:2]},
        {"secondary_keywords": ["AI diagnosis", "AI radiology"]},
    ],
    ids=["two angles", "two secondary keywords"],
)
def test_a_draft_with_too_few_angles_or_keywords_is_asked_for_again(monkeypatch, too_few):
    """Faked one level lower, at LiteLLM, so the integration module's schema retry runs."""
    replies = [json.dumps(draft(**too_few)), json.dumps(draft())]
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return litellm_reply(replies[len(calls) - 1])

    monkeypatch.setattr(llm.litellm, "completion", completion)
    _, state = run_plan(RESEARCH_ONLY, {"topic-1": topic()})

    assert len(calls) == 2
    assert "did not match the schema" in calls[1]["messages"][-1]["content"]
    brief = state["topics"]["topic-1"]["brief"]
    assert len(brief["angles"]) == 3
    assert len(brief["secondary_keywords"]) == 3
