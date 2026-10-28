"""The Quality Score: how a draft measures up, from 0 to 100. See spec #19.

The checks are worth 70 points, shared equally between the checks that apply. The reviewer
(#31) will be worth the other 30. Without a reviewer's scores, the check points alone are
scaled to 0-100: the same score the reviewer keeps for "reviewer unavailable".
"""

from typing import TypedDict

from contentblitz.workflow.content_checks import CheckResult

CHECK_POINTS = 70


class QualityScore(TypedDict):
    score: int  # 0-100
    checks: list[CheckResult]  # in the order the chat lists them


def quality_score(checks: list[CheckResult]) -> QualityScore:
    """The score for a draft's check results, from the checks alone."""
    check_points = CHECK_POINTS * sum(c["passed"] for c in checks) / len(checks)
    return {"score": round(check_points * 100 / CHECK_POINTS), "checks": checks}
