"""Image Generation Agent: turns the marketer's request into an image prompt, generates the
image, and saves it as a file and as a Content Piece. See spec #21.

For a new image step:

1. **Prompt optimisation.** One structured call to the `image_generator` model, given the step's
   instructions (or the marketer's message), the Topic, the Research Brief's summary and angles
   when there is one, the active Brand Profile's visual style (colours, mood, medium and notes),
   and the title and kind of the Content Piece the image is for, when the step names a source
   piece. It returns the optimised prompt, the intended use, any shape the marketer asked for,
   and alt text. The prompt always ends with the no-text rule. An image for a blog is a blog
   header and one for a LinkedIn post a LinkedIn visual, whatever use the model gave.
2. **Size.** The shape the marketer asked for, otherwise the intended use's: landscape for a blog
   header, square for LinkedIn or general use. The image model maps it to a size from config.
3. **Generation** through the LLM integration module (ADR 0002), which tries the image models
   in config order and rejects broken results (not a PNG, JPEG or WebP, under 512 px, or
   nearly one colour).
4. **Placeholder.** When every model fails, or no image model key is configured, a placeholder
   card is drawn in the requested shape, in the active Brand Profile's primary colour (its
   first colour) or, when the profile has no colours, the configured placeholder colour. It is
   saved like any other image, with the model *placeholder* and the placeholder flag set, and
   the chat says so.
5. **Storage.** The image is saved as a PNG in the Topic's images folder under the data folder,
   and as an *image* Content Piece with a *draft* Revision holding its metadata. Images have no
   Quality Score. An image for a Content Piece records that piece's id in its metadata; the
   piece itself is unchanged.

A refine step that targets an image re-runs prompt optimisation from the latest Revision's
optimised prompt plus the instructions, keeps that Revision's shape unless the instructions ask
for another, and saves the result as a new file and a *refinement* Revision on the same Content
Piece. Earlier Revisions and their files are never changed.

A refined image made for a Content Piece gives the optimiser that piece again, so its title,
kind and use stay the same.

New and refined images share one path, so both fall back to a placeholder. When prompt
optimisation fails, the image is drawn from a plain prompt instead: the request (or, for a
refinement, the latest prompt plus the instructions) and the no-text rule, in the shape the
request names, otherwise the source piece's or the latest Revision's, otherwise square, with the
request (or the latest Revision's) as alt text. The models tried and failed, with each outcome,
are kept on the Revision; the trace has every model tried, including the one used, and any
optimiser failure.
"""

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import NamedTuple, get_args

from langchain_core.messages import AIMessage
from langsmith import traceable
from pydantic import BaseModel, Field

from contentblitz.core.config import Aspect, get_model_config
from contentblitz.integrations import llm
from contentblitz.workflow.active_brand_profile import active_brand_profile
from contentblitz.workflow.brand_profiles import VisualStyle
from contentblitz.workflow.formats import PIECE_FORMATS, ImageUse
from contentblitz.workflow.image_files import (
    PLACEHOLDER_LABEL,
    image_path,
    placeholder_png,
    save_png,
    size_label,
)
from contentblitz.workflow.plan import current_step
from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.state import (
    PLACEHOLDER_MODEL,
    ContentBlitzState,
    ContentPiece,
    ImageMetadata,
    Revision,
    RevisionOrigin,
    Topic,
)
from contentblitz.workflow.topics import add_piece, add_revision, new_piece_id, piece_topic

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
    "I couldn't make the image for *{topic}*. "
    "The rest of the plan carries on, so ask again to retry the image."
)

NO_REFINED_IMAGE = (
    "I couldn't make a new version of *{title}*. The image is unchanged, so ask again to retry."
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


def _use_for(piece: ContentPiece | None) -> ImageUse | None:
    """How an image for `piece` is used, from its kind: a blog header or a LinkedIn visual."""
    return PIECE_FORMATS[piece["kind"]].image_use if piece else None


def _for_piece(piece: ContentPiece | None) -> str:
    """The line naming the Content Piece the image is for, or nothing without one."""
    if not piece:
        return ""
    return f"For the {PIECE_FORMATS[piece['kind']].label}: {piece['title']}\n"


def optimise_prompt(
    request: str,
    topic: Topic,
    visual_style: str | None = None,
    piece: ContentPiece | None = None,
) -> ImagePrompt:
    """The marketer's request as a detailed image prompt, ending with the no-text rule.

    `visual_style` is the Brand Profile's when there is one; otherwise the default from config.
    `piece` is the Content Piece the image is for, when the request names one: its title and
    kind go into the prompt, and its kind sets the intended use. Raises when every provider
    fails or the reply never fits.
    """
    style = visual_style or get_model_config().image.default_style
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Request: {request}\n"
            f"Topic: {topic['name']}\n"
            f"{_for_piece(piece)}"
            f"Visual style: {style}\n\n"
            f"Research Brief: {_brief_text(topic)}",
        },
    ]
    optimised = llm.complete_structured_reply(AGENT, messages, ImagePrompt).value
    update: dict = {"prompt": _with_no_text_rule(optimised.prompt)}
    if use := _use_for(piece):
        update["use"] = use
    return optimised.model_copy(update=update)


def aspect_of(optimised: ImagePrompt) -> Aspect:
    """The shape the marketer asked for, otherwise the intended use's."""
    return optimised.aspect or USE_ASPECT[optimised.use]


def asked_aspect(request: str) -> Aspect | None:
    """The shape a request names in so many words ("a square image"), if any."""
    words = re.findall(r"[a-z]+", request.lower())
    return next((aspect for aspect in get_args(Aspect) if aspect in words), None)


def describe_visual_style(style: VisualStyle) -> str:
    """A Brand Profile's visual style as the optimiser reads it: its colours (primary first),
    mood, medium and notes, leaving out any that are empty."""
    parts = [
        f"colours {', '.join(style.colours)} (primary {style.colours[0]})" if style.colours else "",
        f"mood: {style.mood}" if style.mood else "",
        f"medium: {style.medium}",
        f"notes: {style.notes}" if style.notes else "",
    ]
    return "; ".join(part for part in parts if part)


def make_image(
    prompt: str, aspect: Aspect, topic_name: str, colour: str | None = None
) -> llm.GeneratedImage:
    """The image for `prompt`, from the first image model that gives a usable one, or a
    placeholder card in `aspect`'s shape when none does, in `colour` ("#RRGGBB") or the
    configured placeholder colour. Never raises for a failed model."""
    try:
        return llm.generate_image(prompt, aspect)
    except llm.NoUsableImage as error:
        fallbacks = error.fallbacks
    config = get_model_config().image
    width, height = config.size_for(config.primary, aspect).split("x")
    colour = colour or config.placeholder_colour
    data = placeholder_png((int(width), int(height)), colour, topic_name)
    return llm.GeneratedImage(data, PLACEHOLDER_MODEL, None, fallbacks)


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
        "placeholder": image.model == PLACEHOLDER_MODEL,
        "fallbacks": list(image.fallbacks),
    }
    return {
        "text": optimised.alt_text,
        "origin": origin,
        "created_at": created_at,
        "image": metadata,
    }


def image_message(piece_id: str, revision_number: int, metadata: ImageMetadata) -> AIMessage:
    """The chat message saying the image is ready; its metadata references the Revision."""
    heading = (
        f"{PLACEHOLDER_LABEL} for *{metadata['alt_text']}*: no image model gave a usable image, "
        "so this is a placeholder card. Ask again to retry the image."
        if metadata["placeholder"]
        else f"Image ready: *{metadata['alt_text']}*"
    )
    text = f"{heading}\n\n**Model:** {metadata['model']}\n\n**Size:** {size_label(metadata)}"
    reference = {"content_piece": {"piece_id": piece_id, "revision": revision_number}}
    return progress_message(text, reference)


@traceable(name="image")
def _trace(
    request: str,
    optimised_prompt: str,
    optimiser_failure: str | None,
    image: llm.GeneratedImage,
) -> dict:
    """Puts the request, the optimised prompt (or the plain one used when optimisation failed,
    and why it failed), each model tried with its outcome, and the placeholder flag in the image
    step's trace."""
    return {
        "request": request,
        "optimised_prompt": optimised_prompt,
        "optimiser_failure": optimiser_failure,
        "models_tried": image.models_tried(),
        "placeholder": image.model == PLACEHOLDER_MODEL,
    }


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


class _ImageJob(NamedTuple):
    """What a new image and a refined one differ in; `_make_image` does the rest."""

    topic_id: str
    piece_id: str
    revision_number: int
    origin: RevisionOrigin
    request: str  # recorded on the Revision and in the trace
    optimiser_request: str
    source_id: str | None  # the Content Piece the image is for, recorded on the Revision
    source: ContentPiece | None  # that piece, when it is on the Topic, for the optimiser
    fallback: ImagePrompt  # drawn when prompt optimisation fails
    kept_aspect: Aspect | None  # the shape kept unless the optimiser gives another
    failed: str  # the chat message when no image is saved
    save: Callable[[Revision], ContentBlitzState]  # the state update that keeps the Revision


def _new_image(state: ContentBlitzState) -> _ImageJob:
    topic_id = state["active_topic_id"]
    topic = state["topics"][topic_id]
    request = _request(state)
    source_id = (current_step(state) or {}).get("source_piece")
    source = state.get("content_pieces", {}).get(source_id) if source_id else None
    piece_id = new_piece_id(state)

    def save(revision: Revision) -> ContentBlitzState:
        piece: ContentPiece = {
            "kind": "image",
            "topic_id": topic_id,
            "title": _title(revision["text"]),
            "revisions": [revision],
        }
        return add_piece(state, piece_id, piece)

    return _ImageJob(
        topic_id=topic_id,
        piece_id=piece_id,
        revision_number=1,
        origin="draft",
        request=request,
        optimiser_request=request,
        source_id=source_id,
        source=source,
        fallback=ImagePrompt(
            prompt=_with_no_text_rule(request),
            use=_use_for(source) or "general",
            aspect=asked_aspect(request),
            alt_text=request.strip(),
        ),
        kept_aspect=None,
        failed=NO_IMAGE.format(topic=topic["name"]),
        save=save,
    )


def _refine_request(latest_prompt: str, instructions: str) -> str:
    return (
        f"Change this image as the marketer asks: {instructions}\n"
        f"Keep everything else from its current prompt: {latest_prompt}"
    )


def _refine_image(state: ContentBlitzState, piece_id: str) -> _ImageJob:
    """A new image from the latest Revision's prompt plus the instructions, saved as a new file
    and a *refinement* Revision. Earlier Revisions and their files are left as they are."""
    piece = state["content_pieces"][piece_id]
    topic = piece_topic(state, piece_id)
    latest = piece["revisions"][-1]["image"]
    instructions = _request(state)
    # Still the image for the same piece, which the optimiser is given again when it is there.
    source_id = latest.get("source_piece")
    source = state["content_pieces"].get(source_id) if source_id in topic["piece_ids"] else None
    latest_prompt = latest["prompt"].removesuffix(NO_TEXT_RULE).strip()
    return _ImageJob(
        topic_id=piece["topic_id"],
        piece_id=piece_id,
        revision_number=len(piece["revisions"]) + 1,
        origin="refinement",
        request=instructions,
        optimiser_request=_refine_request(latest["prompt"], instructions),
        source_id=source_id,
        source=source,
        fallback=ImagePrompt(
            prompt=_with_no_text_rule(f"{latest_prompt} {instructions}"),
            use=latest["use"],
            aspect=asked_aspect(instructions),
            alt_text=latest["alt_text"],
        ),
        kept_aspect=latest["aspect"],
        failed=NO_REFINED_IMAGE.format(title=piece["title"]),
        save=lambda revision: add_revision(state, piece_id, revision),
    )


def _make_image(state: ContentBlitzState, job: _ImageJob) -> ContentBlitzState:
    """Optimises the prompt in the active Brand Profile's visual style (or falls back to the
    job's plain one), makes the image (or a placeholder in the profile's primary colour), saves
    it as the job's Revision, and posts the image message."""
    topic = state["topics"][job.topic_id]
    style = active_brand_profile(state).visual_style
    try:
        optimised = optimise_prompt(
            job.optimiser_request, topic, describe_visual_style(style), job.source
        )
        optimiser_failure = None
    except Exception as error:  # noqa: BLE001 - a plain prompt still makes an image
        optimised, optimiser_failure = job.fallback, f"{type(error).__name__}: {error}"
    if job.kept_aspect and not optimised.aspect:
        optimised = optimised.model_copy(update={"aspect": job.kept_aspect})
    try:
        primary = style.colours[0] if style.colours else None
        image = make_image(optimised.prompt, aspect_of(optimised), topic["name"], primary)
        revision = image_revision(
            job.topic_id,
            job.piece_id,
            job.revision_number,
            job.request,
            optimised,
            image,
            job.origin,
        )
    except Exception:  # noqa: BLE001 - a failed image posts an error; the plan carries on
        return {"messages": [progress_message(job.failed)]}
    _trace(job.request, optimised.prompt, optimiser_failure, image)
    if job.source_id:
        revision["image"]["source_piece"] = job.source_id
    message = image_message(job.piece_id, job.revision_number, revision["image"])
    return {**job.save(revision), "messages": [message]}


def image_generator(state: ContentBlitzState) -> ContentBlitzState:
    """Makes a new image, or refines the image a refine step targets."""
    target = (current_step(state) or {}).get("target")
    return _make_image(state, _refine_image(state, target) if target else _new_image(state))
