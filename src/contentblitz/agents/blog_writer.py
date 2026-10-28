"""SEO Blog Writer: writes search-optimised blog Content Pieces from a Research Brief."""

from contentblitz.workflow.progress import stub_progress
from contentblitz.workflow.state import ContentBlitzState


def blog_writer(state: ContentBlitzState) -> ContentBlitzState:
    return stub_progress("SEO Blog Writer", state)
