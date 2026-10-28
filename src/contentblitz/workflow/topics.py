"""Finding and creating Topics, and adding Content Pieces to them, in the workflow state."""

import re
from collections.abc import Iterable

from contentblitz.workflow.state import ContentBlitzState, ContentPiece, Topic

UNTITLED_TOPIC = "Untitled Topic"


def _next_id(prefix: str, ids: Iterable[str]) -> str:
    """`<prefix>-N` after the highest existing one, so a new id never replaces an existing one."""
    numbers = [int(m[1]) for i in ids if (m := re.fullmatch(rf"{prefix}-(\d+)", i))]
    return f"{prefix}-{max(numbers, default=0) + 1}"


def active_topic(state: ContentBlitzState) -> Topic | None:
    return (state.get("topics") or {}).get(state.get("active_topic_id") or "")


def create_topic(state: ContentBlitzState, name: str | None) -> ContentBlitzState:
    """The state update that adds a new Topic, with the next `topic-N` id, and makes it active."""
    topics = dict(state.get("topics") or {})
    topic_id = _next_id("topic", topics)
    topics[topic_id] = {"name": name or UNTITLED_TOPIC, "brief": None, "piece_ids": []}
    return {"topics": topics, "active_topic_id": topic_id}


def new_piece_id(state: ContentBlitzState) -> str:
    """The next `piece-N` id: a new Content Piece never replaces an existing one."""
    return _next_id("piece", state.get("content_pieces") or {})


def add_piece(state: ContentBlitzState, piece_id: str, piece: ContentPiece) -> ContentBlitzState:
    """The state update that saves a Content Piece and records it on its Topic's piece_ids."""
    topics = dict(state.get("topics") or {})
    topic = topics[piece["topic_id"]]
    topics[piece["topic_id"]] = {**topic, "piece_ids": [*topic.get("piece_ids", []), piece_id]}
    pieces = {**(state.get("content_pieces") or {}), piece_id: piece}
    return {"topics": topics, "content_pieces": pieces}
