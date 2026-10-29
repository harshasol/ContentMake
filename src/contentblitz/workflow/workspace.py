"""The workspace module: the one place the UI touches a conversation (#23, #59).

A `Workspace` works on one conversation (one thread of the compiled graph) through LangGraph's
state read and write. The Streamlit app only calls it and never reads graph state itself, so the
UI stays thin and every action can be tested against the real graph.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AnyMessage, HumanMessage

from contentblitz.core.config import ModelConfig, Settings, get_model_config, get_settings
from contentblitz.workflow.brand_profiles import DEFAULT_PROFILE_ID
from contentblitz.workflow.report import render_report
from contentblitz.workflow.state import (
    ContentBlitzState,
    ContentPiece,
    ImageMetadata,
    ResearchBrief,
    Revision,
)
from contentblitz.workflow.topics import active_topic, find_revision


@dataclass(frozen=True)
class ChatMessage:
    """One message of the conversation as the Chat tab shows it; `image` is set for an image
    message (the Revision it references, #21)."""

    role: Literal["user", "assistant"]
    text: str
    image: ImageMetadata | None = None


@dataclass(frozen=True)
class Research:
    """The active Topic's research as the Research tab shows it (#65): its Research Brief and the
    rendered report (`render_report`, #15)."""

    topic_name: str
    brief: ResearchBrief
    report: str


# The providers the Settings tab reports on, and the Settings field holding each one's key.
PROVIDER_KEYS = {
    "OpenAI": "openai_api_key",
    "Anthropic": "anthropic_api_key",
    "Gemini": "gemini_api_key",
    "SerpAPI": "serpapi_api_key",
    "Tavily": "tavily_api_key",
}


@dataclass(frozen=True)
class ConfigurationView:
    """What the Settings tab shows (#70). It holds whether each key is set, never a key value."""

    provider_keys: dict[str, bool]
    agents: dict[str, tuple[str, list[str]]]  # agent -> (model, fallbacks)
    search_providers: list[str]  # in fallback order
    image_models: list[str]  # in fallback order
    thresholds: dict[str, str]  # label -> the quality rule, as text


def _range(low: float, high: float) -> str:
    return f"{low}–{high}"


@dataclass(frozen=True)
class PieceRow:
    """One Content Piece as the dashboard lists it: its latest Revision number and Quality Score
    (None for an image, which has none)."""

    piece_id: str
    kind: str
    title: str
    revision: int
    score: int | None


@dataclass(frozen=True)
class TopicPieces:
    """The Content Pieces of one Topic, in the order they were made."""

    topic: str
    pieces: list[PieceRow]


@dataclass(frozen=True)
class PieceDetail:
    """A Content Piece with all its Revisions, oldest first (#61). The detail view renders one
    Revision of it, so viewing an older one (#62) is `revision(n)` instead of `latest`."""

    piece_id: str
    kind: str
    title: str
    topic: str
    revisions: list[Revision]

    @property
    def latest(self) -> Revision:
        return self.revisions[-1]

    def revision(self, number: int) -> Revision:
        """Revision `number`, counted from 1."""
        return self.revisions[number - 1]


def new_thread_id() -> str:
    return str(uuid.uuid4())


class Workspace:
    """One conversation. `graph` is the compiled workflow; `thread_id` picks the conversation."""

    def __init__(self, graph: Any, thread_id: str):
        self.graph = graph
        self.thread_id = thread_id

    @property
    def _config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def _state(self) -> ContentBlitzState:
        return self.graph.get_state(self._config).values  # {} for an unknown thread

    def messages(self) -> list[ChatMessage]:
        """The conversation's user and assistant messages, in order, read from the graph state."""
        state = self._state()
        return [
            ChatMessage("user" if m.type == "human" else "assistant", m.content, _image(m, state))
            for m in state.get("messages", [])
            if m.type in ("human", "ai")
        ]

    def research(self) -> Research | None:
        """The active Topic's Research Brief and report, or None when there is no active Topic or
        it has no Brief."""
        topic = active_topic(self._state())
        if not topic or not topic.get("brief"):
            return None
        return Research(topic["name"], topic["brief"], render_report(topic["name"], topic["brief"]))

    def configuration(
        self, settings: Settings | None = None, config: ModelConfig | None = None
    ) -> ConfigurationView:
        """The read-only description of the configuration (ADR 0002): which provider keys are
        set, each agent's model and fallbacks, the search order, the image models and the
        quality thresholds."""
        settings = settings or get_settings()
        config = config or get_model_config()
        quality, blog, post = config.quality, config.quality.blog, config.quality.linkedin
        return ConfigurationView(
            provider_keys={n: bool(getattr(settings, f)) for n, f in PROVIDER_KEYS.items()},
            agents={n: (a.primary, a.fallbacks) for n, a in config.agents.items()},
            search_providers=list(config.search.providers),
            image_models=list(config.image.models),
            thresholds={
                "Pass mark": f"{quality.pass_mark:g}",
                "Max automatic revisions": str(quality.max_revisions),
                "Blog words": _range(*blog.word_range),
                "Blog keyword density %": _range(*blog.density_range),
                "Blog meta description characters": _range(*blog.meta_length_range),
                "Blog max reading grade": f"{blog.max_grade:g}",
                "Blog min citations": str(blog.min_citations),
                "LinkedIn characters": _range(*post.char_range),
                "LinkedIn hashtags": _range(*post.hashtag_range),
                "LinkedIn max hook characters": str(post.max_hook_chars),
                "LinkedIn min paragraphs": str(post.min_paragraphs),
                "LinkedIn max paragraph characters": str(post.max_paragraph_chars),
            },
        )

    def dashboard(self, all_topics: bool) -> list[TopicPieces]:
        """The Content Pieces of the active Topic, or of every Topic, grouped by Topic (#60).

        Pieces come from each Topic's `piece_ids`; a Topic without pieces is left out, so an
        empty list means the tab shows its empty state.
        """
        state = self._state()
        topics = state.get("topics") or {}
        chosen = list(topics.values()) if all_topics else [active_topic(state)]
        pieces = state.get("content_pieces") or {}
        return [
            TopicPieces(topic["name"], [_row(i, pieces[i]) for i in topic["piece_ids"]])
            for topic in chosen
            if topic and topic.get("piece_ids")
        ]

    def piece(self, piece_id: str) -> PieceDetail | None:
        """A Content Piece with all its Revisions, or None when this conversation has no such
        piece (#61)."""
        state = self._state()
        piece = (state.get("content_pieces") or {}).get(piece_id)
        if not piece:
            return None
        topic = (state.get("topics") or {}).get(piece["topic_id"]) or {}
        return PieceDetail(
            piece_id, piece["kind"], piece["title"], topic.get("name", ""), piece["revisions"]
        )

    def send(self, text: str, brand_profile_id: str = DEFAULT_PROFILE_ID) -> None:
        """Send the marketer's message and run the workflow; read the replies with `messages()`."""
        self.graph.invoke(
            {"messages": [HumanMessage(content=text)], "brand_profile_id": brand_profile_id},
            config=self._config,
        )

    def has_active_topic(self) -> bool:
        """Whether the conversation has an active Topic (the Refresh research button needs one)."""
        return active_topic(self._state()) is not None

    def refresh_research(self, brand_profile_id: str = DEFAULT_PROFILE_ID) -> None:
        """Run the *refresh research* command on the active Topic: a direct command to the
        workflow, not a chat message. Without an active Topic the chat says so."""
        self.graph.invoke(
            {"command": "refresh_research", "brand_profile_id": brand_profile_id},
            config=self._config,
        )


def _image(message: AnyMessage, state: ContentBlitzState) -> ImageMetadata | None:
    reference = message.additional_kwargs.get("content_piece")
    if not reference:
        return None
    revision = find_revision(state, reference["piece_id"], reference["revision"])
    return revision.get("image") if revision else None


def _row(piece_id: str, piece: ContentPiece) -> PieceRow:
    latest = piece["revisions"][-1]
    quality = latest.get("quality_score")
    return PieceRow(
        piece_id,
        piece["kind"],
        piece["title"],
        len(piece["revisions"]),
        quality["score"] if quality else None,
    )
