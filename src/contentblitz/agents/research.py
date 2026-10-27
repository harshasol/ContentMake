"""Deep Research Agent: searches the web and gathers Sources for a Topic."""

from langchain_core.messages import AIMessage

from contentblitz.workflow.state import ContentBlitzState


def research(state: ContentBlitzState) -> ContentBlitzState:
    return {"messages": [AIMessage(content="Deep Research Agent is not implemented yet.")]}
