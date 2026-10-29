"""Progress messages: the plan and each agent's report while it runs. Routing history skips them."""

from langchain_core.messages import AIMessage, AnyMessage

PROGRESS = "progress"


def progress_message(text: str, metadata: dict | None = None) -> AIMessage:
    """A progress message; `metadata` goes in its additional_kwargs (an image message's
    `content_piece` reference, which the Chat tab renders the image from)."""
    return AIMessage(content=text, name=PROGRESS, additional_kwargs=metadata or {})


def is_progress(message: AnyMessage) -> bool:
    return message.type == "ai" and message.name == PROGRESS
