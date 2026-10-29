"""The workspace module: the one place the UI touches a conversation (#23, #59).

A `Workspace` works on one conversation (one thread of the compiled graph) through LangGraph's
state read and write. The Streamlit app only calls it and never reads graph state itself, so the
UI stays thin and every action can be tested against the real graph.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AnyMessage, HumanMessage

from contentblitz.core import call_context
from contentblitz.core.config import ModelConfig, Settings, get_model_config, get_settings
from contentblitz.core.usage import (
    Breakdown,
    FallbackRow,
    RunRow,
    Totals,
    get_ledger,
)
from contentblitz.workflow.brand_profiles import DEFAULT_PROFILE_ID
from contentblitz.workflow.export import (
    REPORT_EXPORTS,
    Export,
    export_bundle,
    export_piece,
    export_report,
    formats,
)
from contentblitz.workflow.hand_edit import hand_edit_revision
from contentblitz.workflow.report import render_report
from contentblitz.workflow.state import (
    ContentBlitzState,
    ContentPiece,
    ImageMetadata,
    ResearchBrief,
    Revision,
)
from contentblitz.workflow.topics import active_topic, add_revision, find_revision


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
class UsageView:
    """What the Usage tab shows (#73), read from the usage ledger."""

    conversation: Totals
    today: Totals
    all_time: Totals
    cache_hit_rate: float | None
    by_agent: list[Breakdown]
    by_model: list[Breakdown]
    recent_runs: list[RunRow]
    fallback_events: list[FallbackRow]
    searches: list[Breakdown]

    @property
    def fallbacks(self) -> int:
        return len(self.fallback_events)


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

    def topics(self) -> list[tuple[str, str]]:
        """The conversation's Topics as (id, name), in the order they were made (#66)."""
        topics = self._state().get("topics") or {}
        return [(topic_id, topic["name"]) for topic_id, topic in topics.items()]

    def active_topic_id(self) -> str | None:
        return self._state().get("active_topic_id")

    def active_topic_name(self) -> str | None:
        topic = active_topic(self._state())
        return topic["name"] if topic else None

    def select_topic(self, topic_id: str | None) -> None:
        """Make a Topic active, or clear the active Topic with None ("New Topic"), so the next
        content request creates a Topic without the new-or-continue question (#66)."""
        if topic_id is not None and topic_id not in (self._state().get("topics") or {}):
            raise KeyError(topic_id)
        self.graph.update_state(
            self._config,
            {"active_topic_id": topic_id, "pending_topic_question": None},
            as_node="query_handler",
        )

    def piece(self, piece_id: str) -> PieceDetail | None:
        """A Content Piece with all its Revisions, or None when this conversation has no such
        piece (#61)."""
        return _piece_detail(self._state(), piece_id)

    def restore(self, piece_id: str, revision_number: int) -> None:
        """Restore Revision `revision_number` (from 1) of a piece: add a new latest Revision with
        origin *restore* copying its content and Quality Score, and naming it in `restored_from`.
        Earlier Revisions are never changed, so the next refinement starts from the restored one
        (#64). An unknown piece or Revision raises `KeyError`."""
        pieces = self._state().get("content_pieces") or {}
        if piece_id not in pieces:
            raise KeyError(piece_id)
        revisions = pieces[piece_id]["revisions"]
        if not 1 <= revision_number <= len(revisions):
            raise KeyError(revision_number)
        source = revisions[revision_number - 1]
        not_copied = ("origin", "created_at", "brand_profile")
        copied = {k: v for k, v in source.items() if k not in not_copied}
        restored: Revision = {
            **copied,
            "origin": "restore",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "restored_from": revision_number,
        }
        self.graph.update_state(
            self._config,
            add_revision(self._state(), piece_id, restored),
            as_node="query_handler",
        )

    def piece_exports(self, piece_id: str, revision_number: int | None = None) -> list[Export]:
        """Every export of Revision `revision_number` of a piece (the latest by default), built
        in memory (#68); `[]` for an unknown piece or Revision number, or an image whose file is
        missing."""
        piece = self.piece(piece_id)
        if piece is None:
            return []
        number = len(piece.revisions) if revision_number is None else revision_number
        if not 1 <= number <= len(piece.revisions):
            return []
        exports = [_export(piece, number, fmt) for fmt in formats(piece.kind)]
        return [] if None in exports else exports

    def research_exports(self) -> list[Export]:
        """The active Topic's research report as Markdown and HTML (#68); `[]` without research."""
        research = self.research()
        if research is None:
            return []
        return [export_report(research.topic_name, research.report, f) for f in REPORT_EXPORTS]

    def edit(self, piece_id: str, text: str, meta_description: str | None = None) -> None:
        """Save a hand edit of a blog (`text` and `meta_description`) or LinkedIn post (`text`)
        as a new *hand edit* Revision (#63). Its Quality Score is the rule checks re-run, with
        the reviewer's scores carried over as *not re-reviewed*. KeyError for an unknown piece,
        ValueError for an image."""
        state = self._state()
        if piece_id not in (state.get("content_pieces") or {}):
            raise KeyError(piece_id)
        revision = hand_edit_revision(state, piece_id, text, meta_description)
        self.graph.update_state(
            self._config, add_revision(state, piece_id, revision), as_node="query_handler"
        )

    def topic_bundle(self) -> Export | None:
        """The active Topic as one zip (#69): the latest Revision of every piece in its main
        format, plus the research report (Markdown) when there is research. A piece whose image
        file is missing is left out. None when nothing would go in, so the UI offers no bundle."""
        state = self._state()
        topic = active_topic(state)
        if not topic:
            return None
        exports: list[Export] = []
        for piece_id in topic.get("piece_ids", []):
            piece = _piece_detail(state, piece_id)
            if piece is None:
                continue
            export = _export(piece, len(piece.revisions), formats(piece.kind)[0])
            if export is not None:
                exports.append(export)
        research = self.research()
        if research is not None:
            exports.append(export_report(research.topic_name, research.report, REPORT_EXPORTS[0]))
        return export_bundle(topic["name"], exports) if exports else None

    def send(self, text: str, brand_profile_id: str = DEFAULT_PROFILE_ID) -> str:
        """Send the marketer's message and run the workflow; read the replies with `messages()`.
        Returns the run id, which `run_caption` and the usage ledger know the run by."""
        return self._run(
            {"messages": [HumanMessage(content=text)], "brand_profile_id": brand_profile_id},
            request=text,
        )

    def _run(self, inputs: dict, request: str) -> str:
        """One workflow invocation is one Run: every call it makes is tagged with its id, and the
        ledger keeps what was asked, for the Usage tab's recent runs."""
        with call_context.run(self.thread_id) as run_id:
            try:
                get_ledger().record_request(run_id, request)
            except OSError:
                pass  # the request is a label; a ledger that can't be written must not stop the run
            self.graph.invoke(inputs, config=self._config)
        return run_id

    def usage(self) -> UsageView:
        """The usage ledger's totals for this conversation, today and all time, and its tables."""
        ledger = get_ledger()
        return UsageView(
            conversation=ledger.totals(conversation_id=self.thread_id),
            today=ledger.totals(day=datetime.now(UTC).date()),
            all_time=ledger.totals(),
            cache_hit_rate=ledger.cache_hit_rate(),
            by_agent=ledger.by_agent(),
            by_model=ledger.by_model(),
            recent_runs=ledger.recent_runs(),
            fallback_events=ledger.fallback_events(),
            searches=ledger.searches_per_provider(),
        )

    def run_caption(self, run_id: str) -> str | None:
        """The run's cost and tokens as a one-line caption, read from the usage ledger. It is not
        a chat message, so it is never stored in the conversation or sent to a model."""
        return get_ledger().caption(run_id)

    def has_active_topic(self) -> bool:
        """Whether the conversation has an active Topic (the Refresh research button needs one)."""
        return active_topic(self._state()) is not None

    def refresh_research(self, brand_profile_id: str = DEFAULT_PROFILE_ID) -> str:
        """Run the *refresh research* command on the active Topic: a direct command to the
        workflow, not a chat message. Without an active Topic the chat says so. Returns the run
        id."""
        return self._run(
            {"command": "refresh_research", "brand_profile_id": brand_profile_id},
            request="Refresh research",
        )


def _piece_detail(state: ContentBlitzState, piece_id: str) -> PieceDetail | None:
    piece = (state.get("content_pieces") or {}).get(piece_id)
    if not piece:
        return None
    topic = (state.get("topics") or {}).get(piece["topic_id"]) or {}
    return PieceDetail(
        piece_id, piece["kind"], piece["title"], topic.get("name", ""), piece["revisions"]
    )


def _export(piece: PieceDetail, number: int, fmt: str) -> Export | None:
    """Revision `number` of the piece as `fmt`, or None for an image whose image data or file is
    missing (so there is nothing to export). Any other error is a bug and surfaces."""
    revision = piece.revision(number)
    if piece.kind == "image":
        image = revision.get("image")
        if not image or not Path(image["file_path"]).is_file():
            return None
    return export_piece(piece.topic, piece.kind, piece.title, revision, number, fmt)


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
