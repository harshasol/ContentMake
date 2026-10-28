"""The AI reviewer: rates a draft for engagement and suggests improvements. See spec #19.

One structured call to the `quality_judge` model, so a cheaper model can judge while a
stronger one writes. Each format gives its own judging criteria. Given a Brand Profile, the
reviewer also scores brand voice; without one it doesn't, so #22 can add brand-voice scoring
without changing the loop.
"""

from typing import NotRequired, TypedDict

from pydantic import BaseModel, Field

from contentblitz.integrations import llm

AGENT = "quality_judge"
MAX_SUGGESTIONS = 5

SYSTEM_PROMPT = """\
You are an editor reviewing a draft for a content marketer. Score how engaging it is from 1
to 10, judging {criteria}. Then give up to {max_suggestions} concrete suggestions that would
improve it most, each one a specific change, most important first."""

BRAND_PROMPT = """
Also score from 1 to 10 how well it follows this Brand Profile's voice:
{profile}"""


class Review(TypedDict):
    """The reviewer's verdict on a draft."""

    engagement: int  # 1-10
    brand_voice: NotRequired[int]  # 1-10, only when a Brand Profile was given
    suggestions: list[str]  # at most MAX_SUGGESTIONS, most important first


class EngagementVerdict(BaseModel):
    engagement: int = Field(ge=1, le=10, description="How engaging the draft is, 1-10.")
    suggestions: list[str] = Field(
        description=f"Up to {MAX_SUGGESTIONS} concrete improvements, most important first."
    )


class BrandVerdict(EngagementVerdict):
    brand_voice: int = Field(
        ge=1, le=10, description="How well the draft follows the Brand Profile's voice, 1-10."
    )


def _profile_text(profile: dict) -> str:
    return "\n".join(f"- {field}: {value}" for field, value in profile.items())


def review(text: str, criteria: str, brand_profile: dict | None = None) -> Review | None:
    """The reviewer's verdict on `text`, or None if the call failed: the reviewer is
    unavailable and the draft is scored on its checks alone."""
    system = SYSTEM_PROMPT.format(criteria=criteria, max_suggestions=MAX_SUGGESTIONS)
    schema = EngagementVerdict
    if brand_profile is not None:
        system += BRAND_PROMPT.format(profile=_profile_text(brand_profile))
        schema = BrandVerdict
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    try:
        verdict = llm.complete_structured_reply(AGENT, messages, schema).value
    except Exception:  # noqa: BLE001 - a failed review falls back to the checks alone
        return None
    found = verdict.model_dump()
    return {**found, "suggestions": found["suggestions"][:MAX_SUGGESTIONS]}
