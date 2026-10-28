"""Finding and creating Topics, and adding Content Pieces to them, in the workflow state."""

import re

from contentblitz.workflow.state import ContentBlitzState, ContentPiece, Topic

UNTITLED_TOPIC = "Untitled Topic"
_TOPIC_ID = re.compile(r"topic-(\d+)")


def active_topic(state: ContentBlitzState) -> Topic | None:
    return (state.get("topics") or {}).get(state.get("active_topic_id") or "")


def create_topic(state: ContentBlitzState, name: str | None) -> ContentBlitzState:
    """The state update that adds a new Topic and makes it active.

    Its id follows the highest existing `topic-N`, so it never replaces an existing Topic.
    """
    topics = dict(state.get("topics") or {})
    numbers = [int(m[1]) for tid in topics if (m := _TOPIC_ID.fullmatch(tid))]
    topic_id = f"topic-{max(numbers, default=0) + 1}"
    topics[topic_id] = {"name": name or UNTITLED_TOPIC, "brief": None, "piece_ids": []}
    return {"topics": topics, "active_topic_id": topic_id}


def add_piece(state: ContentBlitzState, piece_id: str, piece: ContentPiece) -> ContentBlitzState:
    """The state update that saves a Content Piece and records it on its Topic's piece_ids."""
    topics = dict(state.get("topics") or {})
    topic = topics[piece["topic_id"]]
    topics[piece["topic_id"]] = {**topic, "piece_ids": [*topic.get("piece_ids", []), piece_id]}
    pieces = {**(state.get("content_pieces") or {}), piece_id: piece}
    return {"topics": topics, "content_pieces": pieces}
