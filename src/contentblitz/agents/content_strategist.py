"""Content Strategist: turns a Topic's Research Record into a Research Brief. See spec #6.

It makes one structured call to the `content_strategist` model for a draft Brief, then grounds
the draft in code: the Source list is copied from the Research Record, citations to Sources
that don't exist are removed, findings or statistics left uncited are dropped, and each
statistic is marked verified or not against the text of the Sources it cites.

If fewer than 2 findings survive, it retries once, telling the model what was dropped. If the
retry is still too thin, or the model fails, it stops the plan (spec #6, #17) and leaves the
Topic's Research Record as it was, so the next request rebuilds only the Brief.
"""

from datetime import UTC, datetime
from typing import NamedTuple

from langsmith import traceable
from pydantic import BaseModel, Field

from contentblitz.integrations import llm
from contentblitz.workflow.plan import current_step, writers_follow
from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.report import citation, render_report, summary_line
from contentblitz.workflow.state import (
    ContentBlitzState,
    ResearchBrief,
    ResearchRecord,
    ResearchSource,
    Source,
    Statistic,
)
from contentblitz.workflow.statistics import is_verified
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

# The fewest findings that may survive grounding (#17); the draft asks for 4-8.
MIN_FINDINGS = 2
# The most of each a Brief keeps; anything past them is cut after grounding, in the model's order.
MAX_FINDINGS, MAX_STATISTICS, MAX_ANGLES, MAX_SECONDARY_KEYWORDS = 8, 8, 5, 6

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
    angles: list[DraftAngle] = Field(
        min_length=3, description="3-5 content angles, each with a hook."
    )
    target_keyword: str = Field(description="The search term a blog on this Topic should target.")
    secondary_keywords: list[str] = Field(min_length=3, description="3-6 related search terms.")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _numbered(record: ResearchRecord) -> list[tuple[int, ResearchSource]]:
    """The Research Record's Sources with their numbers: their rank, counted from 1."""
    return list(enumerate(record["sources"], start=1))


def _real_citations(numbers: list[int], source_count: int) -> list[int]:
    """The cited numbers that name a Source, each once, in the order cited."""
    return [n for n in dict.fromkeys(numbers) if 1 <= n <= source_count]


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
        for n, s in _numbered(record)
    ]


def _source_text(number: int, source: ResearchSource) -> str:
    text = source["page_text"] or "(page not read)"
    return (
        f"[{number}] {source['title']} ({source['site']})\n"
        f"Snippet: {source['snippet']}\nPage text: {text}"
    )


def _prompt(name: str, instructions: str | None, record: ResearchRecord) -> list[dict]:
    sources = "\n\n".join(_source_text(n, s) for n, s in _numbered(record))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Topic: {name}\nInstructions: {instructions or 'none'}\n\n"
            f"Sources:\n\n{sources}",
        },
    ]


def _draft(messages: list[dict]) -> llm.Reply[DraftBrief] | None:
    """The model's draft Brief, or None if every provider failed or the reply never fit."""
    try:
        return llm.complete_structured_reply(AGENT, messages, DraftBrief)
    except Exception:  # noqa: BLE001 - any model failure stops the plan with a reason (#17)
        return None


def _feedback(draft: DraftBrief, source_count: int) -> str:
    """Tells the model how many findings the Brief needs, and which grounding dropped and why."""
    dropped = []
    for finding in draft.key_findings:
        if not finding.source_numbers:
            dropped.append(f"- {finding.statement} (it cites no Source)")
        elif not _real_citations(finding.source_numbers, source_count):
            cited = citation(finding.source_numbers)
            dropped.append(f"- {finding.statement} (it cites {cited}, which don't exist)")
    needed = (
        f"The Brief needs at least {MIN_FINDINGS} key findings, each citing the Sources it comes "
        f"from. The Sources are numbered 1 to {source_count}."
    )
    if not dropped:
        return needed
    return (
        f"{needed}\nThese findings were dropped because they cite no Source that exists:\n"
        + "\n".join(dropped)
    )


def _grounded[T: DraftFinding | DraftStatistic](claims: list[T], source_count: int) -> list[dict]:
    """The claims with citations only to real Sources; a claim left with none is dropped."""
    grounded = []
    for claim in claims:
        if numbers := _real_citations(claim.source_numbers, source_count):
            grounded.append({**claim.model_dump(), "source_numbers": numbers})
    return grounded


def _verified(statistics: list[dict], record: ResearchRecord) -> list[Statistic]:
    """Marks each statistic verified when its numbers are in the text of Sources it cites."""
    return [
        {
            **s,
            "verified": is_verified(
                s["figure"], [record["sources"][n - 1] for n in s["source_numbers"]]
            ),
        }
        for s in statistics
    ]


def _brief(draft: DraftBrief, record: ResearchRecord, model: str) -> ResearchBrief:
    count = len(record["sources"])
    return {
        "summary": draft.summary,
        "key_findings": _grounded(draft.key_findings, count)[:MAX_FINDINGS],
        "statistics": _verified(_grounded(draft.statistics, count)[:MAX_STATISTICS], record),
        "angles": [a.model_dump() for a in draft.angles[:MAX_ANGLES]],
        "target_keyword": draft.target_keyword,
        "secondary_keywords": draft.secondary_keywords[:MAX_SECONDARY_KEYWORDS],
        "sources": _sources(record),
        "created_at": _now(),
        "model": model,
        "research_gathered_at": record["gathered_at"],
    }


@traceable(name="research brief")
def _traced_counts(draft: DraftBrief, brief: ResearchBrief) -> dict:
    """The Brief's counts, and how many claims grounding dropped, put in the step's trace."""
    source_count = len(brief["sources"])

    def dropped(claims: list[DraftFinding] | list[DraftStatistic]) -> int:
        return sum(not _real_citations(c.source_numbers, source_count) for c in claims)

    statistics = len(brief["statistics"])
    verified = sum(s["verified"] for s in brief["statistics"])
    return {
        "findings": len(brief["key_findings"]),
        "statistics": statistics,
        "verified_statistics": verified,
        "unverified_statistics": statistics - verified,
        "sources": source_count,
        "dropped_findings": dropped(draft.key_findings),
        "dropped_statistics": dropped(draft.statistics),
    }


class _Attempt(NamedTuple):
    """One try at the Brief: the draft (None if the model failed), and a Brief or a failure."""

    draft: DraftBrief | None
    brief: ResearchBrief | None
    failure: str | None  # NO_MODEL or THIN_BRIEF, still to be given the Topic's name


def _attempt(messages: list[dict], record: ResearchRecord) -> _Attempt:
    """Drafts a Brief, grounds it, and traces its counts."""
    reply = _draft(messages)
    if reply is None:
        return _Attempt(None, None, NO_MODEL)
    brief = _brief(reply.value, record, reply.model)
    _traced_counts(reply.value, brief)
    if len(brief["key_findings"]) < MIN_FINDINGS:
        return _Attempt(reply.value, None, THIN_BRIEF)
    return _Attempt(reply.value, brief, None)


def _retry(messages: list[dict], thin_draft: DraftBrief, record: ResearchRecord) -> list[dict]:
    """The first request, the thin draft, and feedback on what it lacked."""
    return [
        *messages,
        {"role": "assistant", "content": thin_draft.model_dump_json()},
        {"role": "user", "content": _feedback(thin_draft, len(record["sources"]))},
    ]


def content_strategist(state: ContentBlitzState) -> ContentBlitzState:
    topic = active_topic(state)
    if not topic:
        return {"abort_reason": NO_TOPIC}
    record = topic.get("research")
    if not record:
        return {"abort_reason": NO_RESEARCH.format(topic=topic["name"])}
    step = current_step(state) or {}
    messages = _prompt(topic["name"], step.get("instructions"), record)
    attempt = _attempt(messages, record)
    if attempt.failure == THIN_BRIEF:
        attempt = _attempt(_retry(messages, attempt.draft, record), record)
    if attempt.brief is None:
        return {"abort_reason": attempt.failure.format(topic=topic["name"])}
    brief = attempt.brief
    topics = dict(state["topics"])
    topics[state["active_topic_id"]] = {**topic, "brief": brief}
    if writers_follow(state):
        chat_text = summary_line(brief)
    else:
        chat_text = render_report(topic["name"], brief)
    return {"messages": [progress_message(chat_text)], "topics": topics}
