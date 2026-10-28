"""The quality loop: the subgraph a writer step runs in, one per format. See spec #19.

For now it is draft → check → finalise. The reviewer (#31) and the revise loop (#32) join it
as nodes between them, so the plan dispatcher never needs rewiring: it runs the writer's step,
and the step runs its format's loop.

Each draft the loop makes is an attempt, and check scores it with the format's content checks.
Attempts are not Revisions: finalise saves only the one it keeps.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from contentblitz.workflow.content_checks import CheckResult
from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.quality_score import QualityScore, quality_score
from contentblitz.workflow.state import ContentBlitzState


class Attempt(TypedDict):
    """One draft inside the loop, as the format shapes it, and how it measured up."""

    draft: dict
    quality_score: NotRequired[QualityScore]  # set by check


class LoopState(ContentBlitzState, total=False):
    attempts: list[Attempt]  # in the order they were made
    result: ContentBlitzState  # the step's update to the plan, set by finalise


class LoopResult(TypedDict):
    result: ContentBlitzState


@dataclass(frozen=True)
class QualityFormat:
    """What makes one format's loop: how it drafts, checks, and saves what the loop keeps."""

    name: str
    # The first draft, or None when every provider failed.
    draft: Callable[[ContentBlitzState], dict | None]
    # The content checks' results for a draft, in the order the chat lists them.
    checks: Callable[[ContentBlitzState, dict], list[CheckResult]]
    # The step's update for the kept attempt: the saved Content Piece and the chat message.
    finalise: Callable[[ContentBlitzState, Attempt], ContentBlitzState]
    # The chat message when no draft could be made. The plan carries on.
    failure: Callable[[ContentBlitzState], str]


def _draft_node(fmt: QualityFormat):
    def draft(state: LoopState) -> LoopState:
        made = fmt.draft(state)
        return {"attempts": [] if made is None else [{"draft": made}]}

    return draft


def _check_node(fmt: QualityFormat):
    def check(state: LoopState) -> LoopState:
        """Scores the latest attempt with the format's content checks."""
        attempts = state.get("attempts") or []
        if not attempts:
            return {}
        *earlier, latest = attempts
        checks = fmt.checks(state, latest["draft"])
        return {"attempts": [*earlier, {**latest, "quality_score": quality_score(checks)}]}

    return check


def _finalise_node(fmt: QualityFormat):
    def finalise(state: LoopState) -> LoopState:
        attempts = state.get("attempts") or []
        if not attempts:
            return {"result": {"messages": [progress_message(fmt.failure(state))]}}
        return {"result": fmt.finalise(state, attempts[-1])}

    return finalise


def quality_loop(fmt: QualityFormat) -> Callable[[ContentBlitzState], ContentBlitzState]:
    """The writer step for `fmt`: runs the format's loop subgraph and returns its update."""
    builder = StateGraph(LoopState, input_schema=ContentBlitzState, output_schema=LoopResult)
    builder.add_node("draft", _draft_node(fmt))
    builder.add_node("check", _check_node(fmt))
    builder.add_node("finalise", _finalise_node(fmt))
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "check")
    builder.add_edge("check", "finalise")
    builder.add_edge("finalise", END)
    loop = builder.compile(name=f"{fmt.name} quality loop")

    def run(state: ContentBlitzState) -> ContentBlitzState:
        return loop.invoke(state)["result"]

    return run
