"""Shared LangGraph state, saved per conversation (thread) by the checkpointer. See spec #4."""

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from contentblitz.workflow.formats import AgentName, PieceKind
from contentblitz.workflow.routing import Intent


class Source(TypedDict):
    """A Source as cited in a Research Brief: copied from the Research Record, never the model.

    Its number is its rank in the Research Record, from 1, and is the citation number
    everywhere: in the Brief, the research report and the writers' citations.
    """

    number: int
    title: str
    url: str
    site: str
    retrieved_at: str  # ISO 8601, UTC


class KeyFinding(TypedDict):
    statement: str
    source_numbers: list[int]  # at least one, each a Source in the Brief


class Statistic(TypedDict):
    figure: str
    description: str  # what the figure measures
    source_numbers: list[int]  # at least one, each a Source in the Brief


class ContentAngle(TypedDict):
    title: str
    hook: str


class ResearchBrief(TypedDict):
    """The Content Strategist's grounded summary of a Topic's Research Record. See spec #6."""

    summary: str
    key_findings: list[KeyFinding]
    statistics: list[Statistic]
    angles: list[ContentAngle]
    target_keyword: str
    secondary_keywords: list[str]
    sources: list[Source]
    created_at: str  # ISO 8601, UTC
    model: str  # the configured primary content_strategist model (a fallback isn't recorded)
    research_gathered_at: str  # the Research Record's gathered_at


class ResearchSource(TypedDict):
    """A Source as gathered by research, with its text; a Brief cites it as a `Source`."""

    title: str
    url: str
    site: str
    snippet: str
    page_text: str
    query: str
    provider: str
    retrieved_at: str  # ISO 8601, UTC


class ResearchRecord(TypedDict):
    """The Research Record: what the Deep Research Agent gathered, not yet a Research Brief."""

    queries: list[str]
    providers: list[str]
    gathered_at: str  # ISO 8601, UTC
    sources: list[ResearchSource]  # in rank order


class Topic(TypedDict):
    name: str
    brief: NotRequired[ResearchBrief | None]
    research: NotRequired[ResearchRecord | None]
    piece_ids: NotRequired[list[str]]


class Revision(TypedDict):
    """One saved version of a Content Piece's text; refinements start from the latest."""

    text: str


class ContentPiece(TypedDict):
    kind: PieceKind
    topic_id: str
    title: str
    revisions: list[Revision]


class PlanStep(TypedDict):
    agent: AgentName
    target: NotRequired[str | None]
    instructions: NotRequired[str | None]
    # On a research step: re-research from scratch, skipping any cached search results.
    fresh: NotRequired[bool]


class PendingTopicQuestion(TypedDict):
    """Asked when a request seems to be on a different Topic: the proposed name and held plan."""

    name: str
    plan: list[PlanStep]


class ContentBlitzState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    topics: dict[str, Topic]
    active_topic_id: str | None
    content_pieces: dict[str, ContentPiece]
    intent: Intent
    plan: list[PlanStep]
    completed_steps: list[PlanStep]
    # Set by an agent to stop the plan: the dispatcher posts it and runs no more steps.
    abort_reason: str | None
    pending_topic_question: PendingTopicQuestion | None
