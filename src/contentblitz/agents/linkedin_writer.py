"""LinkedIn Post Writer: writes LinkedIn Content Pieces from a Research Brief."""

from contentblitz.workflow.progress import stub_progress
from contentblitz.workflow.state import ContentBlitzState


def linkedin_writer(state: ContentBlitzState) -> ContentBlitzState:
    return stub_progress("LinkedIn Post Writer", state)
