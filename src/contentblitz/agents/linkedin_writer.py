"""LinkedIn Post Writer: writes LinkedIn Content Pieces from a Research Brief."""

from langchain_core.messages import AIMessage

from contentblitz.workflow.state import ContentBlitzState


def linkedin_writer(state: ContentBlitzState) -> ContentBlitzState:
    return {"messages": [AIMessage(content="LinkedIn Post Writer is not implemented yet.")]}
