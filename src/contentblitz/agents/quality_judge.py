"""The AI reviewer: scores a draft for engagement and suggests improvements. See spec #19.

One structured call to the `quality_judge` model, so a cheaper model can judge while a
stronger one writes. Each format gives its own judging criteria. Given a Brand Profile, the
reviewer also scores brand voice; without one it doesn't. The writers always give it the
active Brand Profile (spec #22, Default included) through `judged_profile`, so brand voice is
part of every Quality Score. Its verdict is a `Review`, which the Quality Score holds.
"""

from pydantic import BaseModel, Field

from contentblitz.integrations import llm
from contentblitz.workflow.brand_profiles import BrandProfile
from contentblitz.workflow.quality_score import MAX_SUGGESTIONS, Review

AGENT = "quality_judge"

SYSTEM_PROMPT = """\
You are an editor reviewing a draft for a content marketer. Score how engaging it is from 1
to 10, judging {criteria}. Then give up to {max_suggestions} concrete suggestions that would
improve it most, each one a specific change, most important first."""

BRAND_PROMPT = """
Also score from 1 to 10 how well it follows this Brand Profile's voice: its audience, tone,
do's and don'ts and words to avoid. Where the draft strays from the brand voice, include
brand-voice suggestions among the others.
{profile}"""

# The Brand Profile fields the reviewer judges brand voice against. The writing sample is left
# out to save tokens; the learned voice description stands in for it.
JUDGED_FIELDS = {
    "name": "name",
    "audience": "audience",
    "tone": "tone",
    "dos": "do",
    "donts": "don't",
    "words_to_avoid": "words to avoid",
    "voice_description": "voice",
}


class EngagementVerdict(BaseModel):
    engagement: int = Field(ge=1, le=10, description="How engaging the draft is, 1-10.")
    suggestions: list[str] = Field(
        description=f"Up to {MAX_SUGGESTIONS} concrete improvements, most important first."
    )


class BrandVerdict(EngagementVerdict):
    brand_voice: int = Field(
        ge=1, le=10, description="How well the draft follows the Brand Profile's voice, 1-10."
    )


def judged_profile(profile: BrandProfile) -> dict:
    """What the reviewer is given of a Brand Profile: the fields it judges brand voice against
    that are filled in, by the name the prompt uses."""
    found = profile.model_dump(include=set(JUDGED_FIELDS))
    return {label: found[field] for field, label in JUDGED_FIELDS.items() if found[field]}


def _profile_text(profile: dict) -> str:
    def text(value):
        return "; ".join(value) if isinstance(value, list) else value

    return "\n".join(f"- {field}: {text(value)}" for field, value in profile.items())


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
