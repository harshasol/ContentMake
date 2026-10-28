"""Deep Research Agent: searches the web and gathers Sources for a Topic."""

from contentblitz.workflow.progress import stub_progress
from contentblitz.workflow.state import ContentBlitzState


def research(state: ContentBlitzState) -> ContentBlitzState:
    return stub_progress("Deep Research Agent", state)
