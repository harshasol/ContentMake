"""Content Strategist: turns raw research into a structured Research Brief."""

from langchain_core.messages import AIMessage

from contentblitz.workflow.state import ContentBlitzState


def content_strategist(state: ContentBlitzState) -> ContentBlitzState:
    return {"messages": [AIMessage(content="Content Strategist is not implemented yet.")]}
