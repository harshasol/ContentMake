"""Image Generation Agent: builds an image prompt from the Brand Profile, then generates it."""

from contentblitz.workflow.progress import stub_progress
from contentblitz.workflow.state import ContentBlitzState


def image_generator(state: ContentBlitzState) -> ContentBlitzState:
    return stub_progress("Image Generation Agent", state)
