"""Progress messages the agents post while a plan runs."""

from langchain_core.messages import AIMessage

from contentblitz.workflow.plan import current_step
from contentblitz.workflow.state import ContentBlitzState


def stub_progress(agent_label: str, state: ContentBlitzState) -> ContentBlitzState:
    """What a stub agent posts for its step: the refine target and instructions, if any."""
    step = current_step(state) or {}
    if not step.get("target"):
        text = f"{agent_label}: done (stub)."
    else:
        text = f'{agent_label}: refining {step["target"]} with "{step.get("instructions")}" (stub).'
    return {"messages": [AIMessage(content=text)]}
