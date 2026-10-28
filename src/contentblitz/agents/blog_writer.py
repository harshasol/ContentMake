"""SEO Blog Writer: writes search-optimised blog Content Pieces from a Research Brief."""

from langchain_core.messages import AIMessage

from contentblitz.workflow.state import ContentBlitzState


def blog_writer(state: ContentBlitzState) -> ContentBlitzState:
    return {"messages": [AIMessage(content="SEO Blog Writer: done (stub).")]}
