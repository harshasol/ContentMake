"""Content Strategist: turns raw research into a structured Research Brief."""

from contentblitz.workflow.progress import stub_progress
from contentblitz.workflow.state import ContentBlitzState


def content_strategist(state: ContentBlitzState) -> ContentBlitzState:
    return stub_progress("Content Strategist", state)
