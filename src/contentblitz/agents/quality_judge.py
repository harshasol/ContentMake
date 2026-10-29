"""The AI reviewer: scores a draft for engagement and brand voice, and suggests improvements.
See specs #19 and #22.

One structured call to the `quality_judge` model, so a cheaper model can judge while a
stronger one writes. Each format gives its own judging criteria, and the quality loop gives the
active Brand Profile (Default included), whose voice comes from `brand_voice.voice_text`, so
brand voice is part of every Quality Score. Its verdict is a `Review`, which the Quality Score
holds.
"""

from pydantic import BaseModel, Field

from contentblitz.integrations import llm
from contentblitz.workflow.brand_profiles import BrandProfile
from contentblitz.workflow.brand_voice import voice_text
from contentblitz.workflow.quality_score import MAX_SUGGESTIONS, Review

AGENT = "quality_judge"

SYSTEM_PROMPT = """\
You are an editor reviewing a draft for a content marketer. Score how engaging it is from 1
to 10, judging {criteria}. Then give up to {max_suggestions} concrete suggestions that would
improve it most, each one a specific change, most important first.
Also score from 1 to 10 how well it follows this Brand Profile's voice: its audience, tone,
do's and don'ts and words to avoid. Where the draft strays from the brand voice, include
brand-voice suggestions among the others.

{profile}"""


class Verdict(BaseModel):
    engagement: int = Field(ge=1, le=10, description="How engaging the draft is, 1-10.")
    brand_voice: int = Field(
        ge=1, le=10, description="How well the draft follows the Brand Profile's voice, 1-10."
    )
    suggestions: list[str] = Field(
        description=f"Up to {MAX_SUGGESTIONS} concrete improvements, most important first."
    )


def review(text: str, criteria: str, brand_profile: BrandProfile) -> Review | None:
    """The reviewer's verdict on `text`, judged on `criteria` and `brand_profile`'s voice, or
    None if the call failed: the reviewer is unavailable and the draft is scored on its checks
    alone."""
    system = SYSTEM_PROMPT.format(
        criteria=criteria, max_suggestions=MAX_SUGGESTIONS, profile=voice_text(brand_profile)
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    try:
        verdict = llm.complete_structured_reply(AGENT, messages, Verdict).value
    except Exception:  # noqa: BLE001 - a failed review falls back to the checks alone
        return None
    found = verdict.model_dump()
    return {**found, "suggestions": found["suggestions"][:MAX_SUGGESTIONS]}
