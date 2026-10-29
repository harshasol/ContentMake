"""The active Brand Profile's voice, as the writers' drafting and revising prompts give it.

The SEO Blog Writer and the LinkedIn Writer get the profile's audience, tone descriptors, do's
and don'ts, words to avoid and, when it has one, its learned voice description. The writing
sample is never sent: the learned description stands in for it, which saves tokens. Research,
the Content Strategist and the Query Handler stay brand-neutral. See spec #22.
"""

from contentblitz.workflow.active_brand_profile import active_brand_profile
from contentblitz.workflow.state import ContentBlitzState


def _listed(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- none"


def brand_voice(state: ContentBlitzState) -> str:
    """The active Brand Profile's voice, for a writer's prompt."""
    profile = active_brand_profile(state)
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
