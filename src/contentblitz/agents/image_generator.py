"""Image Generation Agent: turns the marketer's request into an image prompt, generates the
image, and saves it as a file and as a Content Piece. See spec #21.

For a new image step:

1. **Prompt optimisation.** One structured call to the `image_generator` model, given the step's
   instructions (or the marketer's message), the Topic, the Research Brief's summary and angles
   when there is one, and a visual style (the default from config unless one is passed in). It
   returns the optimised prompt, the intended use, any shape the marketer asked for, and alt
   text. The prompt always ends with the no-text rule.
2. **Size.** The shape the marketer asked for, otherwise the intended use's: landscape for a blog
   header, square for LinkedIn or general use. The image model maps it to a size from config.
3. **Generation** through the LLM integration module (ADR 0002), with the first image model.
4. **Storage.** The image is saved as a PNG in the Topic's images folder under the data folder,
   and as an *image* Content Piece with a *draft* Revision holding its metadata. Images have no
   Quality Score.

A failed prompt optimisation or generation posts an error, and the plan carries on.
A refine step that targets an image is still a stub.
"""

from datetime import UTC, datetime

from langsmith import traceable
from pydantic import BaseModel, Field

from contentblitz.core.config import Aspect, get_model_config
from contentblitz.integrations import llm
from contentblitz.workflow.image_files import image_path, save_png
from contentblitz.workflow.plan import current_step
from contentblitz.workflow.progress import progress_message, stub_progress
from contentblitz.workflow.state import (
    ContentBlitzState,
    ContentPiece,
    ImageMetadata,
    ImageUse,
    Revision,
    RevisionOrigin,
    Topic,
)
from contentblitz.workflow.topics import add_piece, new_piece_id

AGENT = "image_generator"

NO_TEXT_RULE = "Do not draw any text, letters, words, numbers or logos anywhere in the image."

USE_ASPECT: dict[ImageUse, Aspect] = {
    "blog_header": "landscape",
    "linkedin": "square",
    "general": "square",
}

SYSTEM_PROMPT = """\
You write prompts for an image generation model on behalf of a content marketer. Turn the
marketer's short request into one detailed image prompt: the subject, the composition, the
setting, the lighting, the colours and the mood, in the visual style you are given. Make it fit
the Topic, and the Research Brief when there is one. Never ask for text, captions, labels, signs
or logos in the image.

Also say what the image is for: a blog header, a LinkedIn post, or general use. Give the shape
(landscape, square or portrait) only if the marketer asked for one, and leave it empty
otherwise. Write a short alt text that describes the image for someone who can't see it."""

TITLE_CHARS = 80

NO_IMAGE = (
    "I couldn't make the image for *{topic}*: the model didn't give a usable reply. "
    "The rest of the plan carries on, so ask again to retry the image."
)


class ImagePrompt(BaseModel):
    prompt: str = Field(description="The detailed image prompt.")
    use: ImageUse = Field(description="What the image is for.")
    aspect: Aspect | None = Field(
        None, description="The shape the marketer asked for, or null when they didn't ask."
    )
    alt_text: str = Field(description="A short description of the image, under 125 characters.")


def _brief_text(topic: Topic) -> str:
    brief = topic.get("brief")
    if not brief:
        return "none"
    angles = "\n".join(f"- {a['title']}: {a['hook']}" for a in brief["angles"])
    return f"{brief['summary']}\n\nContent angles:\n{angles or 'none'}"


def _with_no_text_rule(prompt: str) -> str:
    prompt = prompt.strip()
    return prompt if prompt.endswith(NO_TEXT_RULE) else f"{prompt} {NO_TEXT_RULE}"


def optimise_prompt(request: str, topic: Topic, visual_style: str | None = None) -> ImagePrompt:
    """The marketer's request as a detailed image prompt, ending with the no-text rule.

    `visual_style` is the Brand Profile's when there is one; otherwise the default from config.
    Raises when every provider fails or the reply never fits.
    """
    style = visual_style or get_model_config().image.default_style
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Request: {request}\n"
            f"Topic: {topic['name']}\n"
            f"Visual style: {style}\n\n"
            f"Research Brief: {_brief_text(topic)}",
        },
    ]
    optimised = llm.complete_structured_reply(AGENT, messages, ImagePrompt).value
    return optimised.model_copy(update={"prompt": _with_no_text_rule(optimised.prompt)})


def aspect_of(optimised: ImagePrompt) -> Aspect:
    """The shape the marketer asked for, otherwise the intended use's."""
    return optimised.aspect or USE_ASPECT[optimised.use]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def image_revision(
    topic_id: str,
    piece_id: str,
    revision_number: int,
    request: str,
    optimised: ImagePrompt,
    image: llm.GeneratedImage,
    origin: RevisionOrigin,
) -> Revision:
    """Saves the image as Revision `revision_number` (from 1) of `piece_id`'s file, and returns
    that Revision with its image metadata."""
    path = image_path(topic_id, piece_id, revision_number)
    width, height = save_png(image.data, path)
    created_at = _now()
    metadata: ImageMetadata = {
        "request": request,
        "prompt": optimised.prompt,
        "alt_text": optimised.alt_text,
        "use": optimised.use,
        "aspect": aspect_of(optimised),
        "model": image.model,
        "size": f"{width}x{height}",
        "created_at": created_at,
        "file_path": str(path),
        "provider_url": image.provider_url,
        "provider_url_temporary": image.provider_url is not None,
        "placeholder": False,
        "fallbacks": [],
    }
    return {
        "text": optimised.alt_text,
        "origin": origin,
        "created_at": created_at,
        "image": metadata,
    }


def image_message(piece_id: str, revision_number: int, metadata: ImageMetadata):
    """The chat message saying the image is ready; its metadata references the Revision."""
    size = metadata["size"].replace("x", "×")
    text = (
        f"Image ready: *{metadata['alt_text']}*\n\n"
        f"**Model:** {metadata['model']}\n\n"
        f"**Size:** {size}"
    )
    reference = {"content_piece": {"piece_id": piece_id, "revision": revision_number}}
    return progress_message(text, reference)


@traceable(name="image")
def _trace(request: str, optimised_prompt: str, model: str) -> dict:
    """Puts the request, the optimised prompt and the model in the image step's trace."""
    return {"request": request, "optimised_prompt": optimised_prompt, "model": model}


def _request(state: ContentBlitzState) -> str:
    """The step's instructions, or the marketer's latest message without any."""
    step = current_step(state) or {}
    if step.get("instructions"):
        return step["instructions"]
    human = [m for m in state.get("messages", []) if m.type == "human"]
    return str(human[-1].content) if human else ""


def _title(alt_text: str) -> str:
    alt_text = alt_text.strip() or "Image"
    if len(alt_text) <= TITLE_CHARS:
        return alt_text
    return alt_text[: TITLE_CHARS - 1].rsplit(" ", 1)[0] + "…"


def _new_image(state: ContentBlitzState) -> ContentBlitzState:
    topic_id = state["active_topic_id"]
    topic = state["topics"][topic_id]
    request = _request(state)
    try:
        optimised = optimise_prompt(request, topic)
        image = llm.generate_image(optimised.prompt, aspect_of(optimised))
        piece_id = new_piece_id(state)
        revision = image_revision(topic_id, piece_id, 1, request, optimised, image, "draft")
    except Exception:  # noqa: BLE001 - a failed image posts an error; the plan carries on
        return {"messages": [progress_message(NO_IMAGE.format(topic=topic["name"]))]}
    _trace(request, optimised.prompt, image.model)
    piece: ContentPiece = {
        "kind": "image",
        "topic_id": topic_id,
        "title": _title(optimised.alt_text),
        "revisions": [revision],
    }
    message = image_message(piece_id, 1, revision["image"])
    return {**add_piece(state, piece_id, piece), "messages": [message]}


def image_generator(state: ContentBlitzState) -> ContentBlitzState:
    if (current_step(state) or {}).get("target"):
        return stub_progress("Image Generation Agent", state)  # refining an image: #45
    return _new_image(state)
