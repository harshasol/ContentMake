"""Image Generation Agent: builds an image prompt from the Brand Profile, then generates it."""

from langchain_core.messages import AIMessage

from contentblitz.workflow.refine import stub_progress
from contentblitz.workflow.state import ContentBlitzState


def image_generator(state: ContentBlitzState) -> ContentBlitzState:
    return {"messages": [AIMessage(content=stub_progress("Image Generation Agent", state))]}
