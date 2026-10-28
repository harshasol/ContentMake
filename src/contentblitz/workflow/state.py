"""Shared LangGraph state, saved per conversation (thread) by the checkpointer. See spec #4."""

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from contentblitz.workflow.routing import Intent

AgentName = Literal[
    "research",
    "content_strategist",
    "blog_writer",
    "linkedin_writer",
    "image_generator",
]


class Source(TypedDict):
    title: str
    url: str


class ResearchBrief(TypedDict):
    """Filled by the Content Strategist (#6)."""

    summary: str
    key_findings: list[str]
    sources: list[Source]


class Topic(TypedDict):
    name: str
    brief: NotRequired[ResearchBrief | None]
    piece_ids: NotRequired[list[str]]


PieceKind = Literal["blog", "linkedin_post", "image"]


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


class PendingTopicQuestion(TypedDict):
    """Asked when a content request seems to be about a different subject from the active Topic."""

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
    pending_topic_question: PendingTopicQuestion | None
