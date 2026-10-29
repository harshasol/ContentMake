"""The Image Generation Agent makes an image and saves it through the compiled workflow.

Each test seeds a Topic and a plan with an image step, as the Query Handler would leave them,
then runs the workflow from the dispatcher. Prompt optimisation and image generation are faked
at the LLM integration module: the fake image model returns bytes made with Pillow at the size
it was asked for. The data folder is a temporary directory.
"""

import io
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langsmith import tracing_context
from PIL import Image

from contentblitz.agents.image_generator import make_image, optimise_prompt
from contentblitz.integrations import llm
from contentblitz.workflow.graph import build_graph

REAL_GENERATE_IMAGE = llm.generate_image
SIZES = {"landscape": (1536, 1024), "square": (1024, 1024), "portrait": (1024, 1536)}
REQUEST = "an image of a robot doctor reading a scan"
OPTIMISED = (
    "A friendly robot doctor studies a glowing chest X-ray in a bright, modern hospital room, "
    "soft daylight, teal and white palette."
)
ALT_TEXT = "A robot doctor studying a chest X-ray"


def image_prompt(**overrides):
    """A prompt-optimisation reply as the `image_generator` model would return it."""
    return {
        "prompt": OPTIMISED,
        "use": "general",
        "aspect": None,
        "alt_text": ALT_TEXT,
        **overrides,
    }


def png(size, blank=False):
    buffer = io.BytesIO()
    image = Image.new("RGB", size, (20, 120, 140))
    if not blank:
        image.paste((240, 200, 60), (0, 0, size[0] // 2, size[1] // 2))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeModels:
    """Stands in for the prompt optimiser and the image model.

    `reply` is the optimiser's reply, or a list of them, one per call, the last repeating.
    The image model returns a Pillow-made PNG of the aspect's size, as gpt-image-1 would.
    """

    def __init__(self):
        self.reply = image_prompt()
        self.prompt_calls = []
        self.image_calls = []
        self.provider_url = None

    def complete(self, agent, messages, schema):
        self.prompt_calls.append({"agent": agent, "messages": messages})
        reply = self.reply
        if isinstance(reply, list):
            reply = reply[min(len(self.prompt_calls), len(reply)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return llm.Reply(schema.model_validate(reply), "openai/gpt-4o-mini")

    def generate(self, prompt, aspect, config=None, client=None):
        self.image_calls.append({"prompt": prompt, "aspect": aspect})
        return llm.GeneratedImage(png(SIZES[aspect]), "openai/gpt-image-1", self.provider_url)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTENTBLITZ_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def models(monkeypatch, data_dir):
    fake = FakeModels()
    monkeypatch.setattr(llm, "complete_structured_reply", fake.complete)
    monkeypatch.setattr(llm, "generate_image", fake.generate)
    return fake


BRIEF = {
    "summary": "AI is speeding up diagnosis in radiology.",
    "key_findings": [],
    "statistics": [],
    "angles": [
        {"title": "The radiologist's new colleague", "hook": "Your next scan may be read twice."}
    ],
    "target_keyword": "AI in radiology",
    "secondary_keywords": [],
    "sources": [],
    "created_at": "2026-10-20T10:00:00+00:00",
    "model": "openai/gpt-4o",
    "research_gathered_at": "2026-10-20T09:30:00+00:00",
}


def topic(name="AI in radiology", brief=None):
    return {"name": name, "research": None, "brief": brief, "piece_ids": []}


def config(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def run_plan(plan, topics=None, graph=None, request=REQUEST):
    """Run `plan` from the dispatcher, as if the Query Handler had just posted it. Returns the
    messages the plan produced and the state afterwards."""
    graph = graph or build_graph()
    before = len(graph.get_state(config()).values.get("messages", []))
    seeded = {
        "messages": [HumanMessage(content=request)],
        "plan": plan,
        "completed_steps": [],
        "abort_reason": None,
    }
    if topics is not None:
        seeded |= {"topics": topics, "active_topic_id": "topic-1"}
    graph.update_state(config(), seeded, as_node="query_handler")
    result = graph.invoke(None, config())
    return result["messages"][before + 1 :], result


def image_step(instructions=REQUEST):
    return [{"agent": "image_generator", "instructions": instructions}]


def prompt_of(call):
    return "\n".join(m["content"] for m in call["messages"])


class RecordingClient:
    """A LangSmith client that keeps the runs it is sent, so a test can read the trace."""

    def __init__(self):
        self.runs = {}

    def create_run(self, **run):
        self.runs[run["id"]] = run

    def update_run(self, run_id, **update):
        self.runs.setdefault(run_id, {}).update(update)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def traced_image_step(run):
    """The outputs the image step put in its trace while `run()` ran."""
    client = RecordingClient()
    with tracing_context(enabled=True, client=client):
        run()
    [traced] = [r.get("outputs") for r in client.runs.values() if r.get("name") == "image"]
    return traced


def only_image(state):
    """The only image Content Piece on the Topic, its id, and its only Revision."""
    [piece_id] = state["topics"]["topic-1"]["piece_ids"]
    piece = state["content_pieces"][piece_id]
    [revision] = piece["revisions"]
    return piece_id, piece, revision


def test_an_image_step_saves_a_png_under_the_topics_images_folder(models, data_dir):
    _, state = run_plan(image_step(), {"topic-1": topic()})

    piece_id, _, revision = only_image(state)
    path = Path(revision["image"]["file_path"])
    assert path == data_dir / "images" / "topic-1" / f"{piece_id}-r1.png"
    with Image.open(path) as saved:
        assert (saved.format, saved.size) == ("PNG", (1024, 1024))


def test_the_image_is_saved_as_an_image_content_piece_with_a_draft_revision(models, data_dir):
    _, state = run_plan(image_step(), {"topic-1": topic()})

    piece_id, piece, revision = only_image(state)
    assert (piece["kind"], piece["topic_id"], piece["title"]) == ("image", "topic-1", ALT_TEXT)
    assert revision["origin"] == "draft"
    assert "quality_score" not in revision
    image = revision["image"]
    created_at = image.pop("created_at")
    assert datetime.fromisoformat(created_at) == datetime.fromisoformat(revision["created_at"])
    assert image == {
        "request": REQUEST,
        "prompt": f"{OPTIMISED} {NO_TEXT_RULE}",
        "alt_text": ALT_TEXT,
        "use": "general",
        "aspect": "square",
        "model": "openai/gpt-image-1",
        "size": "1024x1024",
        "file_path": str(data_dir / "images" / "topic-1" / f"{piece_id}-r1.png"),
        "provider_url": None,
        "provider_url_temporary": False,
        "placeholder": False,
        "fallbacks": [],
    }


def test_a_provider_url_is_kept_and_marked_temporary(models):
    models.provider_url = "https://images.example.com/abc.png"
    _, state = run_plan(image_step(), {"topic-1": topic()})

    image = only_image(state)[2]["image"]
    assert image["provider_url"] == "https://images.example.com/abc.png"
    assert image["provider_url_temporary"] is True


NO_TEXT_RULE = "Do not draw any text, letters, words, numbers or logos anywhere in the image."


def test_the_optimised_prompt_ends_with_the_no_text_rule_and_is_what_the_model_draws(models):
    run_plan(image_step(), {"topic-1": topic()})

    [call] = models.image_calls
    assert call["prompt"] == f"{OPTIMISED} {NO_TEXT_RULE}"


def test_the_no_text_rule_is_not_repeated_when_the_optimiser_already_ends_with_it(models):
    models.reply = image_prompt(prompt=f"{OPTIMISED} {NO_TEXT_RULE}")
    run_plan(image_step(), {"topic-1": topic()})

    assert models.image_calls[0]["prompt"] == f"{OPTIMISED} {NO_TEXT_RULE}"


def test_the_chat_says_the_image_is_ready_and_references_the_content_piece(models):
    [message], state = run_plan(image_step(), {"topic-1": topic()})

    piece_id = only_image(state)[0]
    assert message.content == (
        f"Image ready: *{ALT_TEXT}*\n\n**Model:** openai/gpt-image-1\n\n**Size:** 1024×1024"
    )
    assert message.additional_kwargs["content_piece"] == {"piece_id": piece_id, "revision": 1}


@pytest.mark.parametrize(
    ("use", "aspect", "size"),
    [("blog_header", "landscape", (1536, 1024)), ("linkedin", "square", (1024, 1024))],
)
def test_the_shape_follows_the_intended_use(models, use, aspect, size):
    models.reply = image_prompt(use=use)
    _, state = run_plan(image_step(), {"topic-1": topic()})

    image = only_image(state)[2]["image"]
    assert (image["use"], image["aspect"]) == (use, aspect)
    with Image.open(image["file_path"]) as saved:
        assert saved.size == size


def test_an_explicitly_requested_shape_wins_over_the_intended_use(models):
    models.reply = image_prompt(use="blog_header", aspect="portrait")
    _, state = run_plan(image_step("a portrait visual for the blog"), {"topic-1": topic()})

    image = only_image(state)[2]["image"]
    assert image["aspect"] == "portrait"
    with Image.open(image["file_path"]) as saved:
        assert saved.size == (1024, 1536)


def test_the_optimiser_is_given_the_instructions_topic_brief_and_default_style(models):
    run_plan(image_step(), {"topic-1": topic(brief=BRIEF)})

    [call] = models.prompt_calls
    assert call["agent"] == "image_generator"
    prompt = prompt_of(call)
    for fact in [
        REQUEST,
        "AI in radiology",
        "AI is speeding up diagnosis in radiology.",
        "The radiologist's new colleague: Your next scan may be read twice.",
        "clean, modern editorial illustration",  # the default style from config
    ]:
        assert fact in prompt, fact


def test_the_optimiser_needs_no_brief(models):
    run_plan(image_step(), {"topic-1": topic(brief=None)})

    prompt = prompt_of(models.prompt_calls[0])
    assert "AI in radiology" in prompt
    assert "Research Brief: none" in prompt


def test_the_optimiser_uses_a_visual_style_it_is_given_instead_of_the_default(models):
    # The Brand Profile (#22) passes its visual style here; the workflow doesn't yet.
    optimise_prompt(REQUEST, topic(), visual_style="bold neon cyberpunk photography")

    prompt = prompt_of(models.prompt_calls[0])
    assert "bold neon cyberpunk photography" in prompt
    assert "clean, modern editorial illustration" not in prompt


def test_a_second_image_on_the_same_topic_is_a_separate_content_piece(models, data_dir):
    graph = build_graph()
    run_plan(image_step(), {"topic-1": topic()}, graph=graph)
    models.reply = image_prompt(alt_text="A robot nurse at a bedside")
    _, state = run_plan(image_step("another of a robot nurse"), graph=graph)

    piece_ids = state["topics"]["topic-1"]["piece_ids"]
    assert piece_ids == ["piece-1", "piece-2"]
    titles = [state["content_pieces"][p]["title"] for p in piece_ids]
    assert titles == [ALT_TEXT, "A robot nurse at a bedside"]
    files = sorted(p.name for p in (data_dir / "images" / "topic-1").iterdir())
    assert files == ["piece-1-r1.png", "piece-2-r1.png"]


def test_the_request_falls_back_to_the_marketers_message_without_instructions(models):
    _, state = run_plan(image_step(instructions=None), {"topic-1": topic()})

    assert only_image(state)[2]["image"]["request"] == REQUEST
    assert REQUEST in prompt_of(models.prompt_calls[0])


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("every provider is down"), llm.StructuredOutputError("not a prompt")],
)
def test_a_failed_prompt_optimisation_draws_the_request_and_the_plan_carries_on(models, failure):
    plan = [*image_step(), *image_step()]
    models.reply = [failure, image_prompt()]
    messages, state = run_plan(plan, {"topic-1": topic()})

    first, second = (state["content_pieces"][p] for p in state["topics"]["topic-1"]["piece_ids"])
    image = first["revisions"][0]["image"]
    assert models.image_calls[0] == {"prompt": f"{REQUEST} {NO_TEXT_RULE}", "aspect": "square"}
    assert (image["prompt"], image["alt_text"], image["use"]) == (
        f"{REQUEST} {NO_TEXT_RULE}",
        REQUEST,
        "general",
    )
    assert messages[0].content.startswith(f"Image ready: *{REQUEST}*")
    assert second["revisions"][0]["image"]["prompt"] == f"{OPTIMISED} {NO_TEXT_RULE}"
    assert not state["abort_reason"]


def test_an_image_that_cant_be_saved_posts_an_error_and_the_plan_carries_on(
    models, data_dir, monkeypatch
):
    unwritable = data_dir / "not-a-folder"
    unwritable.write_text("")
    monkeypatch.setenv("CONTENTBLITZ_DATA_DIR", str(unwritable))
    messages, state = run_plan([*image_step(), *image_step()], {"topic-1": topic()})

    assert [m.content for m in messages] == [
        "I couldn't make the image for *AI in radiology*. "
        "The rest of the plan carries on, so ask again to retry the image."
    ] * 2
    assert state["topics"]["topic-1"]["piece_ids"] == []
    assert not state["abort_reason"]


def test_without_a_model_key_a_failed_optimisation_still_saves_a_placeholder(
    image_models, data_dir, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    image_models.optimiser.reply = RuntimeError("no API key")
    [message], state = run_plan(image_step(), {"topic-1": topic()})

    _, image = assert_placeholder(state, data_dir, (1024, 1024))
    assert image["prompt"] == f"{REQUEST} {NO_TEXT_RULE}"
    assert "placeholder" in message.content.lower()
    assert not state["abort_reason"]


@pytest.mark.parametrize(
    ("instructions", "kind", "size"),
    [
        ("a portrait image of a robot doctor", None, (1024, 1536)),
        (REQUEST, "blog", (1536, 1024)),
        ("a square image for the blog", "blog", (1024, 1024)),
    ],
    ids=["explicit", "source-piece", "explicit-wins"],
)
def test_a_failed_optimisation_takes_its_shape_from_the_request_or_the_piece(
    models, instructions, kind, size
):
    models.reply = RuntimeError("down")
    if kind:
        _, state = run_for_piece(kind, instructions)
        image = state["content_pieces"]["piece-2"]["revisions"][0]["image"]
    else:
        _, state = run_plan(image_step(instructions), {"topic-1": topic()})
        image = only_image(state)[2]["image"]
    with Image.open(image["file_path"]) as saved:
        assert saved.size == size


def test_request_optimised_prompt_and_each_model_tried_are_traced(models):
    traced = traced_image_step(lambda: run_plan(image_step(), {"topic-1": topic()}))

    assert traced == {
        "request": REQUEST,
        "optimised_prompt": f"{OPTIMISED} {NO_TEXT_RULE}",
        "optimiser_failure": None,
        "models_tried": [{"model": "openai/gpt-image-1", "outcome": "used"}],
        "placeholder": False,
    }


def test_a_failed_prompt_optimisation_is_traced(models):
    models.reply = RuntimeError("down")
    traced = traced_image_step(lambda: run_plan(image_step(), {"topic-1": topic()}))

    assert traced["optimiser_failure"] == "RuntimeError: down"
    assert traced["optimised_prompt"] == f"{REQUEST} {NO_TEXT_RULE}"


# --- Model fallback, result checks and the placeholder (#44) ---

GPT_IMAGE, DALL_E = "openai/gpt-image-1", "openai/dall-e-3"
DEFAULT_COLOUR = (0x4A, 0x55, 0x68)  # the placeholder colour shipped in config


class ScriptedImageModels:
    """Each image model's results, one per call, the last repeating: image bytes, or an error
    to raise. The real fallback and checks in `llm.generate_image` run on top."""

    def __init__(self):
        self.results = {GPT_IMAGE: [], DALL_E: []}
        self.calls = []

    def generate_with(self, model, prompt, aspect, config=None, client=None):
        self.calls.append(model)
        results = self.results[model]
        result = results[min(self.calls.count(model), len(results)) - 1]
        if isinstance(result, Exception):
            raise result
        return llm.GeneratedImage(result, model)


@pytest.fixture
def image_models(models, monkeypatch):
    """The optimiser stays faked; image generation runs for real down to each model's call."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "generate_image", REAL_GENERATE_IMAGE)
    scripted = ScriptedImageModels()
    scripted.optimiser = models
    monkeypatch.setattr(llm, "generate_image_with", scripted.generate_with)
    return scripted


def test_when_the_first_model_fails_the_second_is_tried_and_its_image_saved(image_models):
    image_models.results = {
        GPT_IMAGE: [RuntimeError("quota exceeded")],
        DALL_E: [png((1024, 1024))],
    }
    [message], state = run_plan(image_step(), {"topic-1": topic()})

    image = only_image(state)[2]["image"]
    assert image_models.calls == [GPT_IMAGE, DALL_E]
    assert (image["model"], image["placeholder"]) == (DALL_E, False)
    assert image["fallbacks"] == [
        {"model": "openai/gpt-image-1", "outcome": "RuntimeError: quota exceeded"}
    ]
    with Image.open(image["file_path"]) as saved:
        assert saved.size == (1024, 1024)
    assert "**Model:** openai/dall-e-3" in message.content


@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        (png((1024, 1024), blank=True), "nearly a single colour"),
        (png((256, 256)), "256×256 is smaller than 512 px"),
        (b"<html>quota exceeded</html>", "not an image"),
    ],
    ids=["blank", "tiny", "undecodable"],
)
def test_a_broken_result_is_rejected_and_the_next_model_tried(image_models, result, outcome):
    image_models.results = {GPT_IMAGE: [result], DALL_E: [png((1024, 1024))]}
    _, state = run_plan(image_step(), {"topic-1": topic()})

    image = only_image(state)[2]["image"]
    assert image["model"] == DALL_E
    assert image["fallbacks"] == [
        {"model": "openai/gpt-image-1", "outcome": f"failed check: {outcome}"}
    ]


def assert_placeholder(state, data_dir, size):
    piece_id, piece, revision = only_image(state)
    image = revision["image"]
    assert (piece["kind"], revision["origin"]) == ("image", "draft")
    assert (image["model"], image["placeholder"]) == ("placeholder", True)
    assert image["file_path"] == str(data_dir / "images" / "topic-1" / f"{piece_id}-r1.png")
    assert image["size"] == f"{size[0]}x{size[1]}"
    with Image.open(image["file_path"]) as saved:
        assert (saved.format, saved.size) == ("PNG", size)
        assert saved.convert("RGB").getpixel((5, 5)) == DEFAULT_COLOUR
    return piece_id, image


@pytest.mark.parametrize(
    ("use", "size"), [("general", (1024, 1024)), ("blog_header", (1536, 1024))]
)
def test_when_every_model_fails_a_placeholder_in_the_requested_shape_is_saved(
    image_models, data_dir, use, size
):
    image_models.optimiser.reply = image_prompt(use=use)
    image_models.results = {
        GPT_IMAGE: [RuntimeError("quota exceeded")],
        DALL_E: [png((1024, 1024), blank=True)],
    }
    [message], state = run_plan(image_step(), {"topic-1": topic()})

    piece_id, image = assert_placeholder(state, data_dir, size)
    assert image["fallbacks"] == [
        {"model": "openai/gpt-image-1", "outcome": "RuntimeError: quota exceeded"},
        {"model": "openai/dall-e-3", "outcome": "failed check: nearly a single colour"},
    ]
    assert "placeholder" in message.content.lower()
    assert message.additional_kwargs["content_piece"] == {"piece_id": piece_id, "revision": 1}


def test_without_an_image_model_key_a_placeholder_is_saved_and_no_model_called(
    image_models, data_dir, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    [message], state = run_plan(image_step(), {"topic-1": topic()})

    _, image = assert_placeholder(state, data_dir, (1024, 1024))
    assert image_models.calls == []
    assert image["fallbacks"] == [
        {"model": "openai/gpt-image-1", "outcome": "no API key configured"},
        {"model": "openai/dall-e-3", "outcome": "no API key configured"},
    ]
    assert "placeholder" in message.content.lower()


def test_the_plan_carries_on_after_a_placeholder(image_models):
    image_models.results = {
        GPT_IMAGE: [RuntimeError("down"), png((1024, 1024))],
        DALL_E: [RuntimeError("down")],
    }
    messages, state = run_plan([*image_step(), *image_step()], {"topic-1": topic()})

    first, second = (state["content_pieces"][p] for p in state["topics"]["topic-1"]["piece_ids"])
    assert first["revisions"][0]["image"]["placeholder"] is True
    assert second["revisions"][0]["image"]["model"] == GPT_IMAGE
    assert len(messages) == 2
    assert not state["abort_reason"]


def test_the_placeholder_uses_the_visual_styles_primary_colour(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    style = "bold flat illustration, primary colour #FF6600, accents #003366"

    image = make_image("a robot doctor", "portrait", "AI in radiology", visual_style=style)

    assert (image.model, image.fallbacks[0]) == (
        "placeholder",
        {"model": "openai/gpt-image-1", "outcome": "no API key configured"},
    )
    with Image.open(io.BytesIO(image.data)) as card:
        assert card.size == (1024, 1536)
        assert card.convert("RGB").getpixel((5, 5)) == (0xFF, 0x66, 0x00)


def test_the_placeholder_uses_the_configured_colour_when_the_style_has_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")

    image = make_image("a robot doctor", "square", "AI in radiology", visual_style="soft pastels")

    with Image.open(io.BytesIO(image.data)) as card:
        assert card.convert("RGB").getpixel((5, 5)) == DEFAULT_COLOUR


def test_each_model_tried_and_the_placeholder_flag_are_traced(image_models):
    image_models.results = {GPT_IMAGE: [RuntimeError("down")], DALL_E: [RuntimeError("down")]}
    traced = traced_image_step(lambda: run_plan(image_step(), {"topic-1": topic()}))

    assert traced["placeholder"] is True
    assert traced["models_tried"] == [
        {"model": "openai/gpt-image-1", "outcome": "RuntimeError: down"},
        {"model": "openai/dall-e-3", "outcome": "RuntimeError: down"},
        {"model": "placeholder", "outcome": "used"},
    ]


def test_the_model_that_made_the_image_is_traced_after_those_that_failed(image_models):
    image_models.results = {GPT_IMAGE: [RuntimeError("down")], DALL_E: [png((1024, 1024))]}
    traced = traced_image_step(lambda: run_plan(image_step(), {"topic-1": topic()}))

    assert traced["models_tried"] == [
        {"model": "openai/gpt-image-1", "outcome": "RuntimeError: down"},
        {"model": "openai/dall-e-3", "outcome": "used"},
    ]


# Refining an image (#45)

LATEST_PROMPT = f"A robot doctor in a dim, moody hospital corridor. {NO_TEXT_RULE}"
BRIGHTER = "A robot doctor in a bright, sunlit hospital corridor, airy and cheerful."


def seeded_image(data_dir, aspect="landscape", revisions=1):
    """A Topic with an image Content Piece whose Revisions each have a file on disk, as earlier
    image steps would have left them. Returns the topics and the content pieces."""
    folder = data_dir / "images" / "topic-1"
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for number in range(1, revisions + 1):
        path = folder / f"piece-1-r{number}.png"
        path.write_bytes(png(SIZES[aspect]))
        latest = number == revisions
        image = {
            "request": "an image of a robot doctor",
            "prompt": LATEST_PROMPT if latest else f"Version {number}. {NO_TEXT_RULE}",
            "alt_text": "A robot doctor in a corridor",
            "use": "blog_header",
            "aspect": aspect,
            "model": "openai/gpt-image-1",
            "size": "x".join(str(n) for n in SIZES[aspect]),
            "created_at": "2026-10-20T10:00:00+00:00",
            "file_path": str(path),
            "provider_url": None,
            "provider_url_temporary": False,
            "placeholder": False,
            "fallbacks": [],
        }
        saved.append(
            {
                "text": image["alt_text"],
                "origin": "draft" if number == 1 else "refinement",
                "created_at": image["created_at"],
                "image": image,
            }
        )
    piece = {
        "kind": "image",
        "topic_id": "topic-1",
        "title": "A robot doctor in a corridor",
        "revisions": saved,
    }
    topics = {"topic-1": {**topic(), "piece_ids": ["piece-1"]}}
    return topics, {"piece-1": piece}


def run_refine(instructions, topics, pieces):
    graph = build_graph()
    graph.update_state(config(), {"content_pieces": pieces}, as_node="query_handler")
    step = {"agent": "image_generator", "target": "piece-1", "instructions": instructions}
    return run_plan([step], topics, graph=graph, request=instructions)


def test_refining_starts_from_the_latest_prompt_plus_the_instructions(models, data_dir):
    topics, pieces = seeded_image(data_dir, revisions=2)
    run_refine("make it brighter", topics, pieces)

    [call] = models.prompt_calls
    assert call["agent"] == "image_generator"
    prompt = prompt_of(call)
    assert LATEST_PROMPT in prompt
    assert "make it brighter" in prompt
    assert "Version 1." not in prompt


def test_a_refined_image_is_a_new_refinement_revision_and_file_on_the_same_piece(models, data_dir):
    models.reply = image_prompt(prompt=BRIGHTER, use="blog_header")
    topics, pieces = seeded_image(data_dir)
    old = pieces["piece-1"]["revisions"][0]
    old_bytes = Path(old["image"]["file_path"]).read_bytes()

    _, state = run_refine("make it brighter", topics, pieces)

    assert state["topics"]["topic-1"]["piece_ids"] == ["piece-1"]
    piece = state["content_pieces"]["piece-1"]
    assert piece["title"] == "A robot doctor in a corridor"
    first, second = piece["revisions"]
    assert first == old
    assert Path(first["image"]["file_path"]).read_bytes() == old_bytes
    assert second["origin"] == "refinement"
    image = second["image"]
    path = data_dir / "images" / "topic-1" / "piece-1-r2.png"
    assert image["file_path"] == str(path)
    assert (image["request"], image["prompt"]) == ("make it brighter", f"{BRIGHTER} {NO_TEXT_RULE}")
    assert models.image_calls == [{"prompt": f"{BRIGHTER} {NO_TEXT_RULE}", "aspect": "landscape"}]
    with Image.open(path) as saved:
        assert (saved.format, saved.size) == ("PNG", (1536, 1024))


def test_a_refined_image_keeps_the_latest_revisions_shape(models, data_dir):
    # The optimiser now calls it a LinkedIn image, but the marketer didn't ask for a new shape.
    models.reply = image_prompt(prompt=BRIGHTER, use="linkedin")
    topics, pieces = seeded_image(data_dir, aspect="portrait")

    _, state = run_refine("make it brighter", topics, pieces)

    image = state["content_pieces"]["piece-1"]["revisions"][-1]["image"]
    assert image["aspect"] == "portrait"
    with Image.open(image["file_path"]) as saved:
        assert saved.size == (1024, 1536)


def test_a_refined_image_keeps_the_piece_it_was_made_for(models, data_dir):
    topics, pieces = seeded_image(data_dir)
    pieces["piece-1"]["revisions"][0]["image"]["source_piece"] = "piece-9"

    _, state = run_refine("make it brighter", topics, pieces)

    assert state["content_pieces"]["piece-1"]["revisions"][-1]["image"]["source_piece"] == (
        "piece-9"
    )


def test_refining_an_image_for_a_piece_gives_the_optimiser_that_pieces_title_and_kind(
    models, data_dir
):
    models.reply = image_prompt(prompt=BRIGHTER, use="general")
    topics, pieces = seeded_image(data_dir)
    pieces["piece-1"]["revisions"][0]["image"]["source_piece"] = "piece-2"
    pieces["piece-2"] = {
        "kind": "linkedin_post",
        "topic_id": "topic-1",
        "title": SOURCES["linkedin_post"]["title"],
        "revisions": [{"text": SOURCES["linkedin_post"]["text"], "origin": "draft"}],
    }
    topics["topic-1"]["piece_ids"].append("piece-2")

    _, state = run_refine("make it brighter", topics, pieces)

    prompt = prompt_of(models.prompt_calls[0])
    assert f"For the LinkedIn post: {SOURCES['linkedin_post']['title']}" in prompt
    assert state["content_pieces"]["piece-1"]["revisions"][-1]["image"]["use"] == "linkedin"


def test_a_refined_image_made_for_no_piece_records_none(models, data_dir):
    topics, pieces = seeded_image(data_dir)

    _, state = run_refine("make it brighter", topics, pieces)

    assert "source_piece" not in state["content_pieces"]["piece-1"]["revisions"][-1]["image"]


def test_instructions_that_ask_for_a_new_shape_change_it(models, data_dir):
    models.reply = image_prompt(prompt=BRIGHTER, use="blog_header", aspect="square")
    topics, pieces = seeded_image(data_dir, aspect="landscape")

    _, state = run_refine("make it square", topics, pieces)

    image = state["content_pieces"]["piece-1"]["revisions"][-1]["image"]
    assert image["aspect"] == "square"
    with Image.open(image["file_path"]) as saved:
        assert saved.size == (1024, 1024)


def test_the_chat_shows_the_refined_image_as_for_a_new_one(models, data_dir):
    models.reply = image_prompt(prompt=BRIGHTER, alt_text="A robot doctor in a sunlit corridor")
    topics, pieces = seeded_image(data_dir, revisions=2)

    [message], _ = run_refine("make it brighter", topics, pieces)

    assert message.content == (
        "Image ready: *A robot doctor in a sunlit corridor*\n\n"
        "**Model:** openai/gpt-image-1\n\n**Size:** 1536×1024"
    )
    assert message.additional_kwargs["content_piece"] == {"piece_id": "piece-1", "revision": 3}


def test_a_failed_refinement_optimisation_draws_the_latest_prompt_plus_the_instructions(
    models, data_dir
):
    models.reply = RuntimeError("every provider is down")
    topics, pieces = seeded_image(data_dir, aspect="portrait")

    [message], state = run_refine("make it brighter", topics, pieces)

    first, second = state["content_pieces"]["piece-1"]["revisions"]
    assert first == pieces["piece-1"]["revisions"][0]
    image = second["image"]
    latest = LATEST_PROMPT.removesuffix(f" {NO_TEXT_RULE}")
    assert image["prompt"] == f"{latest} make it brighter {NO_TEXT_RULE}"
    assert (image["aspect"], image["use"]) == ("portrait", "blog_header")
    assert image["alt_text"] == "A robot doctor in a corridor"
    assert message.additional_kwargs["content_piece"] == {"piece_id": "piece-1", "revision": 2}
    assert not state["abort_reason"]


def test_a_refinement_falls_back_through_the_models_too(image_models, data_dir):
    image_models.optimiser.reply = image_prompt(prompt=BRIGHTER)
    image_models.results = {GPT_IMAGE: [RuntimeError("quota exceeded")], DALL_E: [png((512, 512))]}
    topics, pieces = seeded_image(data_dir)

    _, state = run_refine("make it brighter", topics, pieces)

    image = state["content_pieces"]["piece-1"]["revisions"][-1]["image"]
    assert (image["model"], image["placeholder"]) == (DALL_E, False)
    assert image["fallbacks"] == [
        {"model": "openai/gpt-image-1", "outcome": "RuntimeError: quota exceeded"}
    ]


def test_a_refinement_with_no_usable_image_is_a_placeholder_revision(
    image_models, data_dir, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    image_models.optimiser.reply = image_prompt(prompt=BRIGHTER)
    topics, pieces = seeded_image(data_dir)

    [message], state = run_refine("make it brighter", topics, pieces)

    first, second = state["content_pieces"]["piece-1"]["revisions"]
    assert first == pieces["piece-1"]["revisions"][0]
    image = second["image"]
    assert (second["origin"], image["model"], image["placeholder"]) == (
        "refinement",
        "placeholder",
        True,
    )
    with Image.open(image["file_path"]) as saved:
        assert saved.size == (1536, 1024)  # the landscape shape carries over
    assert "placeholder" in message.content.lower()
    assert message.additional_kwargs["content_piece"] == {"piece_id": "piece-1", "revision": 2}


SOURCES = {
    "blog": {"title": "How AI reads your X-ray", "text": "# How AI reads your X-ray\n\nBody."},
    "linkedin_post": {"title": "Radiology's new colleague", "text": "AI reads scans too."},
}
SOURCE_LABELS = {"blog": "blog", "linkedin_post": "LinkedIn post"}


def run_for_piece(kind, instructions=REQUEST):
    """Run an image step made for a seeded `kind` Content Piece ("an image for the blog")."""
    source = {
        "kind": kind,
        "topic_id": "topic-1",
        "title": SOURCES[kind]["title"],
        "revisions": [{"text": SOURCES[kind]["text"], "origin": "draft"}],
    }
    graph = build_graph()
    graph.update_state(config(), {"content_pieces": {"piece-1": source}}, as_node="query_handler")
    seeded = {**topic(), "piece_ids": ["piece-1"]}
    step = {"agent": "image_generator", "instructions": instructions, "source_piece": "piece-1"}
    return run_plan([step], {"topic-1": seeded}, graph=graph)


@pytest.mark.parametrize("kind", ["blog", "linkedin_post"])
def test_the_optimiser_is_given_the_source_pieces_title_and_kind(models, kind):
    run_for_piece(kind)

    prompt = prompt_of(models.prompt_calls[0])
    assert f"For the {SOURCE_LABELS[kind]}: {SOURCES[kind]['title']}" in prompt


@pytest.mark.parametrize(
    ("kind", "use", "size"),
    [("blog", "blog_header", (1536, 1024)), ("linkedin_post", "linkedin", (1024, 1024))],
)
def test_an_image_for_a_piece_takes_its_shape_from_the_pieces_kind(models, kind, use, size):
    models.reply = image_prompt(use="general")
    _, state = run_for_piece(kind)

    image = state["content_pieces"]["piece-2"]["revisions"][0]["image"]
    assert image["use"] == use
    with Image.open(image["file_path"]) as saved:
        assert saved.size == size


def test_a_shape_the_marketer_names_wins_over_the_pieces_kind(models):
    models.reply = image_prompt(use="blog_header", aspect="portrait")
    _, state = run_for_piece("blog", "a portrait image for the blog")

    image = state["content_pieces"]["piece-2"]["revisions"][0]["image"]
    assert image["aspect"] == "portrait"
    with Image.open(image["file_path"]) as saved:
        assert saved.size == (1024, 1536)


def test_an_image_for_a_piece_records_it_and_leaves_the_piece_unchanged(models):
    _, state = run_for_piece("blog")

    assert state["topics"]["topic-1"]["piece_ids"] == ["piece-1", "piece-2"]
    blog, image = state["content_pieces"]["piece-1"], state["content_pieces"]["piece-2"]
    assert image["kind"] == "image"
    assert image["revisions"][0]["image"]["source_piece"] == "piece-1"
    assert blog == {
        "kind": "blog",
        "topic_id": "topic-1",
        "title": SOURCES["blog"]["title"],
        "revisions": [{"text": SOURCES["blog"]["text"], "origin": "draft"}],
    }


def test_an_image_for_no_piece_records_none(models):
    _, state = run_plan(image_step(), {"topic-1": topic()})

    assert "source_piece" not in only_image(state)[2]["image"]
