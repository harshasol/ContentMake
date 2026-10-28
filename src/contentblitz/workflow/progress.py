"""Progress messages: the plan and each agent's report while it runs. Routing history skips them."""

from langchain_core.messages import AIMessage, AnyMessage

from contentblitz.workflow.plan import current_step
from contentblitz.workflow.state import ContentBlitzState

PROGRESS = "progress"


def progress_message(text: str) -> AIMessage:
    return AIMessage(content=text, name=PROGRESS)


def is_progress(message: AnyMessage) -> bool:
    return message.type == "ai" and message.name == PROGRESS


def stub_progress(agent_label: str, state: ContentBlitzState) -> ContentBlitzState:
    """What a stub agent posts for its step: the refine target and instructions, if any."""
    step = current_step(state) or {}
    if not step.get("target"):
        text = f"{agent_label}: done (stub)."
    else:
        text = f'{agent_label}: refining {step["target"]} with "{step.get("instructions")}" (stub).'
    return {"messages": [progress_message(text)]}
