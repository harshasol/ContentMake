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
from PIL import Image

from contentblitz.agents.image_generator import optimise_prompt
from contentblitz.integrations import llm
from contentblitz.workflow.graph import build_graph

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


def png(size):
    buffer = io.BytesIO()
    image = Image.new("RGB", size, (20, 120, 140))
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
def test_a_failed_prompt_optimisation_posts_an_error_and_the_plan_carries_on(models, failure):
    plan = [*image_step(), *image_step()]
    models.reply = [failure, image_prompt()]
    messages, state = run_plan(plan, {"topic-1": topic()})

    assert messages[0].content == (
        "I couldn't make the image for *AI in radiology*: the model didn't give a usable reply. "
        "The rest of the plan carries on, so ask again to retry the image."
    )
    assert messages[1].content.startswith("Image ready:")
    assert len(state["topics"]["topic-1"]["piece_ids"]) == 1
    assert not state["abort_reason"]


def test_request_optimised_prompt_and_model_are_traced(models):
    from langsmith import tracing_context

    class RecordingClient:
        def __init__(self):
            self.runs = {}

        def create_run(self, **run):
            self.runs[run["id"]] = run

        def update_run(self, run_id, **update):
            self.runs.setdefault(run_id, {}).update(update)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    client = RecordingClient()
    with tracing_context(enabled=True, client=client):
        run_plan(image_step(), {"topic-1": topic()})

    [traced] = [r.get("outputs") for r in client.runs.values() if r.get("name") == "image"]
    assert traced == {
        "request": REQUEST,
        "optimised_prompt": f"{OPTIMISED} {NO_TEXT_RULE}",
        "model": "openai/gpt-image-1",
    }
