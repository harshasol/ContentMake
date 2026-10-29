"""A Brand Profile's voice as prompt text: the one place that says which fields make it up.

The writers' drafting and revising prompts (`brand_voice(state)`) and the AI reviewer's
brand-voice score (`voice_text(profile)`) both give it: the profile's name, audience, tone
descriptors, do's and don'ts, words to avoid and, when it has one, its learned voice
description. The writing sample is never sent: the learned description stands in for it, which
saves tokens. Research, the Content Strategist and the Query Handler stay brand-neutral. See
spec #22.
"""

from contentblitz.workflow.active_brand_profile import active_brand_profile
from contentblitz.workflow.brand_profiles import BrandProfile
from contentblitz.workflow.state import ContentBlitzState


def _listed(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- none"


def voice_text(profile: BrandProfile) -> str:
    """`profile`'s voice, for a prompt."""
    parts = [
        f"Brand Profile: {profile.name}",
        f"Write for: {profile.audience or 'a general audience'}",
        f"Tone: {', '.join(profile.tone)}",
        f"Do:\n{_listed(profile.dos)}",
        f"Don't:\n{_listed(profile.donts)}",
        f"Never use these words or phrases: {', '.join(profile.words_to_avoid) or 'none'}",
    ]
    if profile.voice_description:
        parts.append(f"The brand's voice: {profile.voice_description}")
    return "\n".join(parts)


def brand_voice(state: ContentBlitzState) -> str:
    """The active Brand Profile's voice, for a writer's prompt."""
    return voice_text(active_brand_profile(state))
