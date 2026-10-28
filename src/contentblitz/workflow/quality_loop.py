"""The quality loop: the subgraph a writer step runs in, one per format. See spec #19.

For now it is draft → finalise. The checks (#29, #30), the reviewer (#31) and the revise loop
(#32) join it as nodes between them, so the plan dispatcher never needs rewiring: it runs the
writer's step, and the step runs its format's loop.

Each draft the loop makes is an attempt. Attempts are not Revisions: finalise saves only the
one it keeps.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.state import ContentBlitzState


class Attempt(TypedDict):
    """One draft inside the loop, as the format shapes it."""

    draft: dict


class LoopState(ContentBlitzState, total=False):
    attempts: list[Attempt]  # in the order they were made
    result: ContentBlitzState  # the step's update to the plan, set by finalise


class LoopResult(TypedDict):
    result: ContentBlitzState


@dataclass(frozen=True)
class QualityFormat:
    """What makes one format's loop: how it drafts, and how it saves what the loop keeps."""

    name: str
    # The first draft, or None when every provider failed.
    draft: Callable[[ContentBlitzState], dict | None]
    # The step's update for the kept draft: the saved Content Piece and the chat message.
    finalise: Callable[[ContentBlitzState, dict], ContentBlitzState]
    # The chat message when no draft could be made. The plan carries on.
    failure: Callable[[ContentBlitzState], str]


def _draft_node(fmt: QualityFormat):
    def draft(state: LoopState) -> LoopState:
        made = fmt.draft(state)
        return {"attempts": [] if made is None else [{"draft": made}]}

    return draft


def _finalise_node(fmt: QualityFormat):
    def finalise(state: LoopState) -> LoopState:
        attempts = state.get("attempts") or []
        if not attempts:
            return {"result": {"messages": [progress_message(fmt.failure(state))]}}
        return {"result": fmt.finalise(state, attempts[-1]["draft"])}

    return finalise


def quality_loop(fmt: QualityFormat) -> Callable[[ContentBlitzState], ContentBlitzState]:
    """The writer step for `fmt`: runs the format's loop subgraph and returns its update."""
    builder = StateGraph(LoopState, input_schema=ContentBlitzState, output_schema=LoopResult)
    builder.add_node("draft", _draft_node(fmt))
    builder.add_node("finalise", _finalise_node(fmt))
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "finalise")
    builder.add_edge("finalise", END)
    loop = builder.compile(name=f"{fmt.name} quality loop")

    def run(state: ContentBlitzState) -> ContentBlitzState:
        return loop.invoke(state)["result"]

    return run
