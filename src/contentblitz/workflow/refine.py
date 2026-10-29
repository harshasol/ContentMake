"""Resolves which Content Piece a refine request targets, and plans its one refine step."""

from contentblitz.workflow.formats import PIECE_FORMATS
from contentblitz.workflow.plan import writer_options
from contentblitz.workflow.routing import RoutingDecision
from contentblitz.workflow.state import ContentBlitzState, PlanStep
from contentblitz.workflow.topics import active_topic


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
    return [pid for pid in active_topic_pieces(state) if pieces[pid]["kind"] in kinds]


def resolve_source(state: ContentBlitzState, decision: RoutingDecision) -> RoutingDecision:
    """The decision with its source piece kept only if it names an existing Content Piece."""
    if decision.source_piece in (state.get("content_pieces") or {}):
        return decision
    return decision.model_copy(update={"source_piece": None})


def refine_step(state: ContentBlitzState, piece_id: str, decision: RoutingDecision) -> PlanStep:
    """The one step that refines `piece_id`. It also gets the request's options for the piece's
    format, so a refinement can change, say, its template."""
    kind = state["content_pieces"][piece_id]["kind"]
    step: PlanStep = {
        "agent": PIECE_FORMATS[kind].agent,
        "target": piece_id,
        "instructions": decision.instructions,
    }
    return step | writer_options(decision, kind)


def _label(state: ContentBlitzState, piece_id: str) -> str:
    piece = state["content_pieces"][piece_id]
    return f"*{piece['title']}* ({PIECE_FORMATS[piece['kind']].label})"


def describe_refine(state: ContentBlitzState, piece_id: str) -> str:
    return f"Refining {_label(state, piece_id)}"


def active_topic_pieces(state: ContentBlitzState) -> list[str]:
    topic = active_topic(state)
    return list(topic.get("piece_ids", [])) if topic else []


def no_piece_of_kind(state: ContentBlitzState, decision: RoutingDecision) -> str:
    """Says the active Topic has no piece of the requested kind, and lists the ones it has."""
    kinds = " or ".join(PIECE_FORMATS[f].label for f in decision.formats if f in PIECE_FORMATS)
    topic = active_topic(state)
    options = "\n".join(f"- {_label(state, pid)}" for pid in active_topic_pieces(state))
    return f"There's no {kinds} on *{topic['name']}* to refine. It has:\n{options}"


def which_piece_question(state: ContentBlitzState, candidates: list[str]) -> str:
    options = "\n".join(f"- {_label(state, pid)}" for pid in candidates)
    return f"Which piece would you like me to refine?\n{options}"
