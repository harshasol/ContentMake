"""Shared LangGraph state, saved per conversation (thread) by the checkpointer. See spec #4."""

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from contentblitz.core.config import Aspect
from contentblitz.integrations.llm import ModelOutcome
from contentblitz.workflow.formats import AgentName, ImageUse, PieceKind
from contentblitz.workflow.quality_score import QualityScore
from contentblitz.workflow.routing import Intent, Template


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
    verified: bool  # its significant numbers all appear in the text of Sources it cites


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
    model: str  # the model that wrote the draft, as LiteLLM names it (a fallback, if one answered)
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


RevisionOrigin = Literal["draft", "refinement", "hand_edit", "restore"]
PLACEHOLDER_MODEL = "placeholder"


class ImageMetadata(TypedDict):
    """The fixed record an image Revision keeps about its image. See spec #21."""

    request: str  # the marketer's request, as the step's instructions or their message
    prompt: str  # the optimised prompt the image model was given
    alt_text: str
    use: ImageUse
    aspect: Aspect
    model: str  # the image model that made it, or PLACEHOLDER_MODEL
    size: str  # the saved image's "WIDTHxHEIGHT"
    created_at: str  # ISO 8601, UTC
    file_path: str  # the PNG under the data folder; old Revisions' files are kept
    provider_url: str | None  # the provider's link, if it gave one
    provider_url_temporary: bool  # provider links expire: the file is the copy to keep
    placeholder: bool
    fallbacks: list[ModelOutcome]  # each image model tried and failed before this one
    source_piece: NotRequired[str]  # the Content Piece it was made for ("an image for the blog")


class BrandProfileRecord(TypedDict):
    """The Brand Profile a Revision was made with, as it was then. See spec #22."""

    id: str
    name: str
    updated_at: str  # ISO 8601, UTC: which version of the profile


class Revision(TypedDict):
    """One saved version of a Content Piece's text; refinements start from the latest.

    Only the result a writer's quality loop keeps becomes a Revision, never its Attempts.
    """

    text: str  # Markdown for a blog, plain text for a LinkedIn post, the alt text for an image
    origin: RevisionOrigin
    created_at: str  # ISO 8601, UTC
    # The conversation's active Brand Profile when it was made; `add_piece` and `add_revision`
    # record it, so editing or switching a profile later never changes it.
    brand_profile: NotRequired[BrandProfileRecord]
    # A blog's SEO fields, and the template any writer used; the keyword and template carry over
    # to its refinements.
    meta_description: NotRequired[str]
    target_keyword: NotRequired[str]
    template: NotRequired[Template]
    # How it measured up when it was made: the score and each check's result. Text formats only.
    quality_score: NotRequired[QualityScore]
    # An image Revision's image: its file and how it was made.
    image: NotRequired[ImageMetadata]
    # A *restore* Revision: the number (from 1) of the Revision it copies.
    restored_from: NotRequired[int]


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
    # On a writer step, when the request sets them: they win over the Brief and the config. A
    # blog takes all three; a LinkedIn post takes a LinkedIn template and a length.
    target_keyword: NotRequired[str]
    template: NotRequired[Template]
    # A blog's in words, checked with a tolerance instead of the word range; a LinkedIn post's
    # in characters, the most it may have.
    length: NotRequired[int]
    # On a LinkedIn step: the Content Piece the post builds on ("turn that blog into a post").
    # On an image step: the Content Piece the image is for ("an image for the blog").
    source_piece: NotRequired[str]


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
    # A direct command from the sidebar that skips routing (#67); the Query Handler clears it.
    command: Literal["refresh_research"] | None
    # Conversation input from the sidebar: the active Brand Profile's id (None means Default).
    brand_profile_id: str | None
    # The active Brand Profile, loaded through the store at the start of each run as a
    # `BrandProfile.model_dump(mode="json")`. Read it with `active_brand_profile(state)`.
    brand_profile: dict
    # The ids of the deleted Brand Profiles the conversation has been told about, once each.
    missing_brand_profiles_noticed: list[str]
