"""The quality loop: the subgraph a writer step runs in, one per format. See spec #19.

The loop is draft → check → judge → (revise → check → judge)… → finalise. The plan dispatcher
runs the writer's step, and the step runs its format's loop.

Each draft the loop makes is an Attempt. Check runs the format's content checks on it, and
judge asks the AI reviewer for its verdict and gives the Attempt its Quality Score. An Attempt
that doesn't pass is revised: the writer changes the current draft, given the failed checks and
the reviewer's suggestions, at most `max_revisions` times. The loop stops at the first Attempt
that passes; if none does, or a revise call fails, it keeps the best one so far. Attempts are
not Revisions: finalise saves only the one it keeps, with how many revisions were used.

The loop brings in the active Brand Profile (spec #22), so a format gets it for free: check
hands the format its words to avoid, and judge gives the reviewer the profile to score brand
voice against.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from contentblitz.agents.quality_judge import review
from contentblitz.core.config import get_model_config
from contentblitz.workflow.active_brand_profile import active_brand_profile
from contentblitz.workflow.content_checks import CheckResult
from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.quality_score import (
    QualityScore,
    Review,
    comparable_points,
    quality_score,
)
from contentblitz.workflow.state import ContentBlitzState

SUGGESTIONS_SHOWN = 3


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
    # The current draft changed to answer the feedback (see `feedback`), or None when every
    # provider failed.
    revise: Callable[[ContentBlitzState, dict, str], dict | None]
    # The content checks' results for a draft, in the order the chat lists them, given the
    # active Brand Profile's words to avoid.
    checks: Callable[[ContentBlitzState, dict, list[str]], list[CheckResult]]
    # What the AI reviewer reads of a draft.
    reviewed_text: Callable[[dict], str]
    # What the reviewer judges engagement on, e.g. "its hook, flow, clarity and call to action".
    judging_criteria: str
    # The step's update for the kept Attempt: the saved Content Piece and the chat message.
    finalise: Callable[[ContentBlitzState, Attempt], ContentBlitzState]
    # The chat message when no draft could be made. The plan carries on.
    failure: Callable[[ContentBlitzState], str]


def feedback(attempt: Attempt) -> str:
    """What the writer is asked to fix in an Attempt: each failed check with its measured
    value and target, and the reviewer's suggestions."""
    quality = attempt["quality_score"]
    failed = [
        f"- {c['name']} ({c['measured']}): {c['reason']}"
        for c in quality["checks"]
        if not c["passed"]
    ]
    suggestions = [f"- {s}" for s in (quality["review"] or {}).get("suggestions", [])]
    parts = []
    if failed:
        parts.append("These checks failed:\n" + "\n".join(failed))
    if suggestions:
        parts.append("The reviewer suggests:\n" + "\n".join(suggestions))
    return "\n\n".join(parts)


def _review_text(verdict: Review | None) -> str:
    if verdict is None:
        return "*Engagement: reviewer unavailable*"
    suggestions = "\n".join(f"- {s}" for s in verdict["suggestions"][:SUGGESTIONS_SHOWN])
    scores = (
        f"**Engagement:** {verdict['engagement']}/10\n\n"
        f"**Brand voice:** {verdict['brand_voice']}/10"
    )
    return f"{scores}\n\n**Suggestions:**\n{suggestions}" if suggestions else scores


def quality_text(quality: QualityScore) -> str:
    """The Quality Score block every format's chat message ends with: the score, each check
    with its measured value, the revisions used, and the reviewer's verdict."""
    checks = "\n".join(
        f"- {c['name']} {c['measured']} {'✓' if c['passed'] else '✗'}" for c in quality["checks"]
    )
    max_revisions = get_model_config().quality.max_revisions
    return (
        f"**Quality Score:** {quality['score']}/100\n\n{checks}\n\n"
        f"**Automatic revisions:** {quality['revisions_used']} of {max_revisions}\n\n"
        f"{_review_text(quality['review'])}"
    )


def _draft_node(fmt: QualityFormat):
    def draft(state: LoopState) -> LoopState:
        made = fmt.draft(state)
        return {"attempts": [] if made is None else [{"draft": made}]}

    return draft


def _check_node(fmt: QualityFormat):
    def check(state: LoopState) -> LoopState:
        """Runs the format's content checks on the latest Attempt."""
        attempts = state.get("attempts") or []
        if not attempts:
            return {}
        *earlier, latest = attempts
        avoid = active_brand_profile(state).words_to_avoid
        checks = fmt.checks(state, latest["draft"], avoid)
        return {"attempts": [*earlier, {**latest, "checks": checks}]}

    return check


def _judge_node(fmt: QualityFormat):
    @traceable(name=f"{fmt.name} attempt")
    def trace(attempt: int, quality: QualityScore, brand_profile_id: str) -> dict:
        """Puts an Attempt's check results, reviewer scores and Quality Score in the trace,
        with the active Brand Profile's id and the brand-voice score (spec #22)."""
        return {
            "attempt": attempt,
            "quality_score": quality,
            "brand_profile_id": brand_profile_id,
            "brand_voice": (quality["review"] or {}).get("brand_voice"),
        }

    def judge(state: LoopState) -> LoopState:
        """Asks the reviewer for its verdict on the latest Attempt, and gives it its score."""
        attempts = state.get("attempts") or []
        if not attempts:
            return {}
        *earlier, latest = attempts
        profile = active_brand_profile(state)
        verdict = review(fmt.reviewed_text(latest["draft"]), fmt.judging_criteria, profile)
        pass_mark = get_model_config().quality.pass_mark
        scored = quality_score(latest["checks"], verdict, pass_mark)
        trace(len(attempts), scored, profile.id)
        return {"attempts": [*earlier, {**latest, "quality_score": scored}]}

    return judge


def _after_judge(state: LoopState) -> str:
    """Finalise once the latest Attempt passes or the revisions are used up; else revise."""
    attempts = state.get("attempts") or []
    if not attempts or attempts[-1]["quality_score"]["passed"]:
        return "finalise"
    revisions_used = len(attempts) - 1
    return "finalise" if revisions_used >= get_model_config().quality.max_revisions else "revise"


def _revise_node(fmt: QualityFormat):
    def revise(state: LoopState) -> LoopState:
        """Asks the writer to revise the latest Attempt. A failed call adds no Attempt."""
        attempts = state["attempts"]
        made = fmt.revise(state, attempts[-1]["draft"], feedback(attempts[-1]))
        return {} if made is None else {"attempts": [*attempts, {"draft": made}]}

    return revise


def _after_revise(state: LoopState) -> str:
    """Check the new Attempt, or finalise when the revise call made none."""
    return "finalise" if "checks" in state["attempts"][-1] else "check"


def _best(attempts: list[Attempt]) -> Attempt:
    """The Attempt to keep: one that passed, else the highest-scoring, the earliest on a tie.

    Scores are compared with `comparable_points`, so an Attempt the reviewer missed doesn't win
    on its check points scaled to 100.
    """
    return max(
        attempts,
        key=lambda a: (a["quality_score"]["passed"], comparable_points(a["quality_score"])),
    )


def _finalise_node(fmt: QualityFormat):
    def finalise(state: LoopState) -> LoopState:
        attempts = state.get("attempts") or []
        if not attempts:
            return {"result": {"messages": [progress_message(fmt.failure(state))]}}
        kept = _best(attempts)
        quality = {**kept["quality_score"], "revisions_used": len(attempts) - 1}
        return {"result": fmt.finalise(state, {**kept, "quality_score": quality})}

    return finalise


def quality_loop(fmt: QualityFormat) -> Callable[[ContentBlitzState], ContentBlitzState]:
    """The writer step for `fmt`: runs the format's loop subgraph and returns its update."""
    builder = StateGraph(LoopState, input_schema=ContentBlitzState, output_schema=LoopResult)
    builder.add_node("draft", _draft_node(fmt))
    builder.add_node("check", _check_node(fmt))
    builder.add_node("judge", _judge_node(fmt))
    builder.add_node("revise", _revise_node(fmt))
    builder.add_node("finalise", _finalise_node(fmt))
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "check")
    builder.add_edge("check", "judge")
    builder.add_conditional_edges("judge", _after_judge, ["revise", "finalise"])
    builder.add_conditional_edges("revise", _after_revise, ["check", "finalise"])
    builder.add_edge("finalise", END)
    loop = builder.compile(name=f"{fmt.name} quality loop")

    def run(state: ContentBlitzState) -> ContentBlitzState:
        return loop.invoke(state)["result"]

    return run
