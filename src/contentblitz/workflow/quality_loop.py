"""The quality loop: the subgraph a writer step runs in, one per format. See spec #19.

For now it is draft → check → judge → finalise. The revise loop (#32) joins it between judge
and finalise, so the plan dispatcher never needs rewiring: it runs the writer's step, and the
step runs its format's loop.

Each draft the loop makes is an attempt. Check runs the format's content checks on it, and
judge asks the AI reviewer for its verdict and gives the attempt its Quality Score. Attempts
are not Revisions: finalise saves only the one it keeps.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from contentblitz.core.config import get_model_config
from contentblitz.workflow.content_checks import CheckResult
from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.quality_score import QualityScore, quality_score
from contentblitz.workflow.reviewer import Review
from contentblitz.workflow.state import ContentBlitzState


class Attempt(TypedDict):
    """One draft inside the loop, as the format shapes it, and how it measured up."""

    draft: dict
    checks: NotRequired[list[CheckResult]]  # set by check
    quality_score: NotRequired[QualityScore]  # set by judge


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
    # The AI reviewer's verdict on a draft, with the format's judging criteria, or None when
    # the reviewer is unavailable.
    judge: Callable[[ContentBlitzState, dict], Review | None]
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
        """Runs the format's content checks on the latest attempt."""
        attempts = state.get("attempts") or []
        if not attempts:
            return {}
        *earlier, latest = attempts
        return {"attempts": [*earlier, {**latest, "checks": fmt.checks(state, latest["draft"])}]}

    return check


def _judge_node(fmt: QualityFormat):
    def judge(state: LoopState) -> LoopState:
        """Asks the reviewer for its verdict on the latest attempt, and gives it its score."""
        attempts = state.get("attempts") or []
        if not attempts:
            return {}
        *earlier, latest = attempts
        review = fmt.judge(state, latest["draft"])
        pass_mark = get_model_config().quality.pass_mark
        scored = quality_score(latest["checks"], review, pass_mark)
        return {"attempts": [*earlier, {**latest, "quality_score": scored}]}

    return judge


def _finalise_node(fmt: QualityFormat):
    def finalise(state: LoopState) -> LoopState:
        attempts = state.get("attempts") or []
        if not attempts:
            return {"result": {"messages": [progress_message(fmt.failure(state))]}}
        kept = attempts[-1]
        quality = {**kept["quality_score"], "revisions_used": len(attempts) - 1}
        return {"result": fmt.finalise(state, {**kept, "quality_score": quality})}

    return finalise


def quality_loop(fmt: QualityFormat) -> Callable[[ContentBlitzState], ContentBlitzState]:
    """The writer step for `fmt`: runs the format's loop subgraph and returns its update."""
    builder = StateGraph(LoopState, input_schema=ContentBlitzState, output_schema=LoopResult)
    builder.add_node("draft", _draft_node(fmt))
    builder.add_node("check", _check_node(fmt))
    builder.add_node("judge", _judge_node(fmt))
    builder.add_node("finalise", _finalise_node(fmt))
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "check")
    builder.add_edge("check", "judge")
    builder.add_edge("judge", "finalise")
    builder.add_edge("finalise", END)
    loop = builder.compile(name=f"{fmt.name} quality loop")

    def run(state: ContentBlitzState) -> ContentBlitzState:
        return loop.invoke(state)["result"]

    return run
