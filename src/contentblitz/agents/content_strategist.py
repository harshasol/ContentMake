"""Content Strategist: turns the active Topic's Research Record into a Research Brief. See #6.

It makes one structured call to the `content_strategist` model for a draft Brief, then grounds
the draft in code: the Source list is copied from the Research Record, citations to Sources
that don't exist are removed, and findings or statistics left uncited are dropped.
"""

from datetime import UTC, datetime

from langsmith import traceable
from pydantic import BaseModel, Field

from contentblitz.core.config import get_model_config
from contentblitz.integrations import llm
from contentblitz.workflow.plan import WRITERS, current_step
from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.report import render_report, summary_line
from contentblitz.workflow.state import (
    ContentBlitzState,
    ResearchBrief,
    ResearchRecord,
    ResearchSource,
    Source,
)
from contentblitz.workflow.topics import active_topic

AGENT = "content_strategist"

SYSTEM_PROMPT = """\
You are the content strategist for a content marketer. Turn the numbered Sources below into a
Research Brief that every writer on the Topic will work from. Use only what the Sources say,
never your own knowledge, and cite every finding and statistic with the numbers of the Sources
it comes from. Let the marketer's instructions (a focus, a market or a time frame) shape the
findings and angles. Stay factual and brand-neutral: no brand voice, no calls to action."""

NO_TOPIC = "There's no active Topic to build a Research Brief for, so I stopped the plan."
NO_RESEARCH = (
    "There's no research on *{topic}* to build a Research Brief from, so I stopped the plan. "
    "Ask me to research it first."
)

RETRY_LATER = (
    "so I stopped the plan. The research is kept, so ask again to retry the Brief without "
    "searching the web again."
)
THIN_BRIEF = (
    "I couldn't build a Research Brief on *{topic}*: too few findings could be traced to the "
    "Sources, " + RETRY_LATER
)
NO_MODEL = (
    "I couldn't build a Research Brief on *{topic}*: the model didn't give a usable reply, "
    + RETRY_LATER
)

MIN_FINDINGS = 2

CITE = "The numbers of the Sources it comes from, as listed, e.g. [1, 3]."


class DraftFinding(BaseModel):
    statement: str = Field(description="One clear, factual statement.")
    source_numbers: list[int] = Field(description=CITE)


class DraftStatistic(BaseModel):
    figure: str = Field(description="The number as the Source gives it, e.g. '42%' or '1,200'.")
    description: str = Field(description="What the figure measures.")
    source_numbers: list[int] = Field(description=CITE)


class DraftAngle(BaseModel):
    title: str
    hook: str = Field(description="One sentence that makes a reader want to read on.")


class DraftBrief(BaseModel):
    summary: str = Field(description="2-4 sentences on what the research shows.")
    key_findings: list[DraftFinding] = Field(description="4-8 key findings.")
    statistics: list[DraftStatistic] = Field(
        default_factory=list, description="0-8 statistics or data points found in the Sources."
    )
    angles: list[DraftAngle] = Field(description="3-5 content angles, each with a hook.")
    target_keyword: str = Field(description="The search term a blog on this Topic should target.")
    secondary_keywords: list[str] = Field(description="3-6 related search terms.")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sources(record: ResearchRecord) -> list[Source]:
    """The Research Record's Sources in rank order, numbered from 1."""
    return [
        {
            "number": n,
            "title": s["title"],
            "url": s["url"],
            "site": s["site"],
            "retrieved_at": s["retrieved_at"],
        }
        for n, s in enumerate(record["sources"], start=1)
    ]


def _source_text(number: int, source: ResearchSource) -> str:
    text = source["page_text"] or "(page not read)"
    return (
        f"[{number}] {source['title']} ({source['site']})\n"
        f"Snippet: {source['snippet']}\nPage text: {text}"
    )


def _prompt(name: str, instructions: str | None, record: ResearchRecord) -> list[dict]:
    sources = "\n\n".join(_source_text(n, s) for n, s in enumerate(record["sources"], start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Topic: {name}\nInstructions: {instructions or 'none'}\n\n"
            f"Sources:\n\n{sources}",
        },
    ]


def _draft(messages: list[dict]) -> DraftBrief | None:
    """The model's draft Brief, or None if every provider failed or the reply never fit."""
    try:
        return llm.complete_structured(AGENT, messages, DraftBrief)
    except Exception:  # noqa: BLE001 - any model failure stops the plan with a reason (ADR 0001)
        return None


def _feedback(draft: DraftBrief, source_count: int) -> str:
    """Tells the model which findings grounding dropped, and why."""
    dropped = []
    for finding in draft.key_findings:
        if not finding.source_numbers:
            dropped.append(f"- {finding.statement} (it cites no Source)")
        elif not any(1 <= n <= source_count for n in finding.source_numbers):
            cited = ", ".join(str(n) for n in finding.source_numbers)
            dropped.append(f"- {finding.statement} (it cites [{cited}], which don't exist)")
    return (
        f"The Sources are numbered 1 to {source_count}. These findings were dropped because "
        "they cite no Source that exists:\n"
        + "\n".join(dropped)
        + f"\nWrite the Brief again with at least {MIN_FINDINGS} key findings, each citing the "
        "Sources it comes from."
    )


def _grounded[T: DraftFinding | DraftStatistic](claims: list[T], source_count: int) -> list[dict]:
    """The claims with citations only to real Sources; a claim left with none is dropped."""
    grounded = []
    for claim in claims:
        numbers = [n for n in dict.fromkeys(claim.source_numbers) if 1 <= n <= source_count]
        if numbers:
            grounded.append({**claim.model_dump(), "source_numbers": numbers})
    return grounded


def _brief(draft: DraftBrief, record: ResearchRecord) -> ResearchBrief:
    count = len(record["sources"])
    return {
        "summary": draft.summary,
        "key_findings": _grounded(draft.key_findings, count),
        "statistics": _grounded(draft.statistics, count),
        "angles": [a.model_dump() for a in draft.angles],
        "target_keyword": draft.target_keyword,
        "secondary_keywords": draft.secondary_keywords,
        "sources": _sources(record),
        "created_at": _now(),
        "model": get_model_config().for_agent(AGENT).primary,
        "research_gathered_at": record["gathered_at"],
    }


@traceable(name="research brief")
def _trace(draft: DraftBrief, brief: ResearchBrief) -> dict:
    """Puts the Brief's counts, and how many claims grounding dropped, in the step's trace."""
    findings, statistics = len(brief["key_findings"]), len(brief["statistics"])
    return {
        "findings": findings,
        "statistics": statistics,
        "sources": len(brief["sources"]),
        "dropped_findings": len(draft.key_findings) - findings,
        "dropped_statistics": len(draft.statistics) - statistics,
    }


def _writers_follow(state: ContentBlitzState) -> bool:
    """Whether a writer step comes after this one in the plan."""
    later = state.get("plan", [])[len(state.get("completed_steps", [])) + 1 :]
    return any(step["agent"] in WRITERS for step in later)


def content_strategist(state: ContentBlitzState) -> ContentBlitzState:
    topic = active_topic(state)
    if not topic:
        return {"abort_reason": NO_TOPIC}
    record = topic.get("research")
    if not record:
        return {"abort_reason": NO_RESEARCH.format(topic=topic["name"])}
    step = current_step(state) or {}
    messages = _prompt(topic["name"], step.get("instructions"), record)
    draft = _draft(messages)
    if draft is None:
        return {"abort_reason": NO_MODEL.format(topic=topic["name"])}
    brief = _brief(draft, record)
    _trace(draft, brief)
    if len(brief["key_findings"]) < MIN_FINDINGS:
        retry = [
            *messages,
            {"role": "assistant", "content": draft.model_dump_json()},
            {"role": "user", "content": _feedback(draft, len(record["sources"]))},
        ]
        draft = _draft(retry)
        if draft is None:
            return {"abort_reason": NO_MODEL.format(topic=topic["name"])}
        brief = _brief(draft, record)
        _trace(draft, brief)
        if len(brief["key_findings"]) < MIN_FINDINGS:
            return {"abort_reason": THIN_BRIEF.format(topic=topic["name"])}
    topics = dict(state["topics"])
    topics[state["active_topic_id"]] = {**topic, "brief": brief}
    if _writers_follow(state):
        shown = summary_line(brief)
    else:
        shown = render_report(topic["name"], brief)
    return {"messages": [progress_message(shown)], "topics": topics}
