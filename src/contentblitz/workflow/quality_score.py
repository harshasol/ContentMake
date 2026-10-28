"""The Quality Score: how a draft measures up, from 0 to 100. See spec #19.

The checks are worth 70 points, shared equally between the checks that apply. The reviewer is
worth the other 30: the average of its scores (1-10), scaled to 30. A draft passes when every
check passes and the reviewer's average is at least the pass mark.

When the reviewer is unavailable, the check points alone are scaled to 0-100, and the draft
doesn't pass: a pass needs the reviewer's average, so the quality loop keeps revising.
"""

from typing import NotRequired, TypedDict

from contentblitz.workflow.content_checks import CheckResult

CHECK_POINTS = 70
REVIEWER_POINTS = 30
MAX_SUGGESTIONS = 5


class Review(TypedDict):
    """The AI reviewer's verdict on a draft (see agents/quality_judge.py)."""

    engagement: int  # 1-10
    brand_voice: NotRequired[int]  # 1-10, only when a Brand Profile was given
    suggestions: list[str]  # at most MAX_SUGGESTIONS, most important first


class QualityScore(TypedDict):
    score: int  # 0-100
    passed: bool
    checks: list[CheckResult]  # in the order the chat lists them
    review: Review | None  # None when the reviewer was unavailable
    reviewer_unavailable: bool  # the reviewer gave no verdict, so the draft couldn't pass
    # How many automatic revisions the quality loop made, set on the Attempt it keeps.
    revisions_used: NotRequired[int]


def review_average(review: Review) -> float:
    """The average of the reviewer's scores: engagement, and brand voice when it was scored."""
    scores = [review["engagement"], *([review["brand_voice"]] if "brand_voice" in review else [])]
    return sum(scores) / len(scores)


def _check_points(checks: list[CheckResult]) -> float:
    """The checks' share of CHECK_POINTS, split equally between the checks that apply."""
    return CHECK_POINTS * sum(c["passed"] for c in checks) / len(checks)


def _reviewer_points(review: Review) -> float:
    """The reviewer's share of REVIEWER_POINTS: its average, scaled from 10."""
    return review_average(review) / 10 * REVIEWER_POINTS


def quality_score(
    checks: list[CheckResult], review: Review | None, pass_mark: float
) -> QualityScore:
    """The score for a draft's check results and the reviewer's verdict, if it gave one."""
    check_points = _check_points(checks)
    if review is None:
        score = check_points * 100 / CHECK_POINTS
        passed = False
    else:
        score = check_points + _reviewer_points(review)
        passed = all(c["passed"] for c in checks) and review_average(review) >= pass_mark
    return {
        "score": round(score),
        "passed": passed,
        "checks": checks,
        "review": review,
        "reviewer_unavailable": review is None,
    }


def comparable_points(quality: QualityScore) -> float:
    """The points that rank one Attempt against another, from 0 to 100.

    The score itself, except that an unreviewed Attempt gets no reviewer points rather than
    its check points scaled to 100, so it doesn't outrank a reviewed Attempt on the reviewer
    failing alone. When every Attempt is unreviewed, they rank as their scores do.
    """
    points = _check_points(quality["checks"])
    if quality["review"] is not None:
        points += _reviewer_points(quality["review"])
    return points
