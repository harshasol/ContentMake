"""The LangGraph workflow: Query Handler first, then whichever agent it routes to."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from contentblitz.agents.blog_writer import blog_writer
from contentblitz.agents.content_strategist import content_strategist
from contentblitz.agents.image_generator import image_generator
from contentblitz.agents.linkedin_writer import linkedin_writer
from contentblitz.agents.query_handler import query_handler
from contentblitz.agents.research import research
from contentblitz.workflow.state import ContentBlitzState

AGENT_NODES = {
    "research": research,
    "content_strategist": content_strategist,
    "blog_writer": blog_writer,
    "linkedin_writer": linkedin_writer,
    "image_generator": image_generator,
}


def _next(state: ContentBlitzState) -> str:
    route = state.get("route", "respond")
    return END if route == "respond" else route


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(ContentBlitzState)
    builder.add_node("query_handler", query_handler)
    for name, node in AGENT_NODES.items():
        builder.add_node(name, node)
        builder.add_edge(name, END)
    builder.add_edge(START, "query_handler")
    builder.add_conditional_edges("query_handler", _next, [*AGENT_NODES, END])
    return builder.compile(checkpointer=checkpointer or MemorySaver())
