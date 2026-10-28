"""The LangGraph workflow: Query Handler → dispatcher → agent → dispatcher → … → end.

An agent stops the plan by returning `{"abort_reason": reason}`: the reason is posted in the
chat and the remaining steps don't run. ADR 0001: research comes before writing, so a failed
research step stops the plan.
"""

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from contentblitz.agents.blog_writer import blog_writer
from contentblitz.agents.content_strategist import content_strategist
from contentblitz.agents.image_generator import image_generator
from contentblitz.agents.linkedin_writer import linkedin_writer
from contentblitz.agents.query_handler import query_handler
from contentblitz.agents.research import research
from contentblitz.workflow.plan import current_step
from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.state import ContentBlitzState

AgentNode = Callable[[ContentBlitzState], ContentBlitzState]

AGENT_NODES: dict[str, AgentNode] = {
    "research": research,
    "content_strategist": content_strategist,
    "blog_writer": blog_writer,
    "linkedin_writer": linkedin_writer,
    "image_generator": image_generator,
}


def dispatcher(state: ContentBlitzState) -> ContentBlitzState:
    """A no-op hub: its conditional edge sends the run to the next plan step's agent, or ends."""
    return {}


def _next_step(state: ContentBlitzState) -> str:
    if state.get("abort_reason"):
        return END
    step = current_step(state)
    return step["agent"] if step else END


def _as_step(agent: AgentNode) -> AgentNode:
    """Run `agent` for the current step, then record that step as completed.

    If the agent aborts the plan, its reason is posted as a chat message.
    """

    def run(state: ContentBlitzState) -> ContentBlitzState:
        step = current_step(state)
        update = agent(state)
        if reason := update.get("abort_reason"):
            update = {**update, "messages": [*update.get("messages", []), progress_message(reason)]}
        return {**update, "completed_steps": [*state.get("completed_steps", []), step]}

    return run


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(ContentBlitzState)
    builder.add_node("query_handler", query_handler)
    builder.add_node("dispatcher", dispatcher)
    for name, agent in AGENT_NODES.items():
        builder.add_node(name, _as_step(agent))
        builder.add_edge(name, "dispatcher")
    builder.add_edge(START, "query_handler")
    builder.add_edge("query_handler", "dispatcher")
    builder.add_conditional_edges("dispatcher", _next_step, [*AGENT_NODES, END])
    return builder.compile(checkpointer=checkpointer or MemorySaver())
