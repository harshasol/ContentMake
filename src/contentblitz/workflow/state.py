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


class PlanStep(TypedDict):
    agent: AgentName
    target: NotRequired[str | None]  # Content Piece id, for refinements
    instructions: NotRequired[str | None]


class PendingTopicQuestion(TypedDict):
    """Asked when a content request seems to be about a different subject from the active Topic."""

    name: str  # the proposed Topic's name
    plan: list[PlanStep]  # the held plan, built against the active Topic


class ContentBlitzState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    topics: dict[str, Topic]
    active_topic_id: str | None
    intent: Intent
    plan: list[PlanStep]
    completed_steps: list[PlanStep]  # a prefix of `plan`, in the order the steps ran
    pending_topic_question: PendingTopicQuestion | None
