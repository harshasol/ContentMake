"""The workspace module: the one place the UI touches a conversation (#23, #59).

A `Workspace` works on one conversation (one thread of the compiled graph) through LangGraph's
state read and write. The Streamlit app only calls it and never reads graph state itself, so the
UI stays thin and every action can be tested against the real graph.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AnyMessage, HumanMessage

from contentblitz.workflow.brand_profiles import DEFAULT_PROFILE_ID
from contentblitz.workflow.state import ContentBlitzState, ContentPiece, ImageMetadata
from contentblitz.workflow.topics import active_topic, find_revision


@dataclass(frozen=True)
class ChatMessage:
    """One message of the conversation as the Chat tab shows it; `image` is set for an image
    message (the Revision it references, #21)."""

    role: Literal["user", "assistant"]
    text: str
    image: ImageMetadata | None = None


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

    def send(self, text: str, brand_profile_id: str = DEFAULT_PROFILE_ID) -> None:
        """Send the marketer's message and run the workflow; read the replies with `messages()`."""
        self.graph.invoke(
            {"messages": [HumanMessage(content=text)], "brand_profile_id": brand_profile_id},
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
