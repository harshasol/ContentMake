"""Resolves which Content Piece a refine request targets, and plans its one refine step."""

from contentblitz.workflow.formats import PIECE_FORMATS
from contentblitz.workflow.routing import RoutingDecision
from contentblitz.workflow.state import ContentBlitzState, PlanStep


def pieces_context(state: ContentBlitzState) -> str:
    """The Content Pieces summary the routing call resolves "it" and "the LinkedIn post" from."""
    pieces = state.get("content_pieces") or {}
    if not pieces:
        return "Content Pieces: none yet."
    topics = state.get("topics") or {}
    lines = [
        f"- {pid}: {p['kind']} on {topics.get(p['topic_id'], {}).get('name', p['topic_id'])}, "
        f"titled {p['title']}"
        for pid, p in pieces.items()
    ]
    return "Content Pieces:\n" + "\n".join(lines)


def resolve_target(state: ContentBlitzState, decision: RoutingDecision) -> list[str]:
    """Ids of the pieces the request could mean: one is the target, several need a question.

    A named piece wins, on any Topic. Otherwise: pieces of the implied kind (any kind if none
    is implied) on the active Topic.
    """
    pieces = state.get("content_pieces") or {}
    if decision.refine_target in pieces:
        return [decision.refine_target]
    kinds = [f for f in decision.formats if f in PIECE_FORMATS] or list(PIECE_FORMATS)
    topic_id = state.get("active_topic_id")
    return [pid for pid, p in pieces.items() if p["kind"] in kinds and p["topic_id"] == topic_id]


def refine_step(state: ContentBlitzState, piece_id: str, instructions: str | None) -> PlanStep:
    kind = state["content_pieces"][piece_id]["kind"]
    return {"agent": PIECE_FORMATS[kind].agent, "target": piece_id, "instructions": instructions}


def _label(state: ContentBlitzState, piece_id: str) -> str:
    piece = state["content_pieces"][piece_id]
    return f"*{piece['title']}* ({PIECE_FORMATS[piece['kind']].label})"


def describe_refine(state: ContentBlitzState, piece_id: str) -> str:
    return f"Refining {_label(state, piece_id)}"


def which_piece_question(state: ContentBlitzState, candidates: list[str]) -> str:
    options = "\n".join(f"- {_label(state, pid)}" for pid in candidates)
    return f"Which piece would you like me to refine?\n{options}"
