"""The Brand Profiles tab (#56), driven through Streamlit's AppTest."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from contentblitz.integrations import llm
from contentblitz.workflow import graph
from contentblitz.workflow.active_brand_profile import load_brand_profile
from contentblitz.workflow.brand_profiles import BrandProfileStore
from contentblitz.workflow.brand_voice import brand_voice

APP = str(Path(__file__).parents[2] / "src/contentblitz/web_app/streamlit_app.py")

TEXT_INPUTS = {"name", "audience", "colours", "mood"}
TEXT_AREAS = {
    "tone",
    "dos",
    "donts",
    "words_to_avoid",
    "writing_sample",
    "voice_description",
    "notes",
}


@pytest.fixture
def seen(monkeypatch):
    """The (brand_profile_id, profile name) each run was given."""
    runs = []

    def recording_handler(state):
        runs.append((state.get("brand_profile_id"), state["brand_profile"]["name"]))
        return {"messages": [AIMessage(content="ok")]}

    st.cache_resource.clear()
    monkeypatch.setattr(graph, "query_handler", recording_handler)
    return runs


def start():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    return at


def click(at, key):
    at.button(key=key).click().run()
    assert not at.exception
    return at


def field(at, name):
    if name == "medium":
        return at.selectbox(key="bp_medium")
    if name in TEXT_INPUTS:
        return at.text_input(key=f"bp_{name}")
    assert name in TEXT_AREAS
    return at.text_area(key=f"bp_{name}")


def fill(at, **values):
    for name, value in values.items():
        field(at, name).set_value(value)
    return at


def test_a_profile_created_in_the_tab_can_be_chosen_and_reaches_the_next_run(seen):
    at = click(start(), "bp_new")
    fill(
        at,
        name="Acme Health",
        audience="Hospital CIOs",
        tone="confident\nno jargon",
        dos="use British spelling",
        donts="make medical claims",
        words_to_avoid="revolutionary\ngame-changer",
        writing_sample="We build calm software.",
        colours="#0A7CFF, #FFFFFF",
        mood="calm",
        medium="photo",
        notes="soft light",
    )
    click(at, "bp_save")

    [saved] = BrandProfileStore().list()[1:]
    assert saved.fields() == {
        "name": "Acme Health",
        "audience": "Hospital CIOs",
        "tone": ["confident", "no jargon"],
        "dos": ["use British spelling"],
        "donts": ["make medical claims"],
        "words_to_avoid": ["revolutionary", "game-changer"],
        "writing_sample": "We build calm software.",
        "voice_description": None,
        "visual_style": {
            "colours": ["#0A7CFF", "#FFFFFF"],
            "mood": "calm",
            "medium": "photo",
            "notes": "soft light",
        },
    }
    picker = at.sidebar.selectbox[0]
    assert picker.options == ["Default", "Acme Health"]

    picker.set_value("acme-health").run()
    at.chat_input[0].set_value("hello").run()

    assert seen == [("acme-health", "Acme Health")]


def shown_after(at, name):
    """The message of the error shown right after a form field, if any."""
    [tab] = [t for t in at.tabs if t.label == "Brand Profiles"]
    elements = list(tab.children.values())
    position = next(i for i, e in enumerate(elements) if getattr(e, "key", None) == f"bp_{name}")
    following = elements[position + 1] if position + 1 < len(elements) else None
    return following.value if following is not None and following.type == "error" else None


@pytest.mark.parametrize(
    ("values", "at_field", "message"),
    [
        ({"name": ""}, "name", "is required"),
        ({"colours": "#0A7CFF, blue"}, "colours", "must be a #RRGGBB hex colour, like #0A7CFF"),
        ({"tone": ""}, "tone", "needs at least 1"),
        ({"tone": "a\nb\nc\nd\ne\nf\ng"}, "tone", "can have at most 6"),
        ({"mood": "x" * 201}, "mood", "must be at most 200 characters"),
        ({"name": "default"}, "name", "a Brand Profile called 'Default' exists"),
    ],
)
def test_a_validation_error_is_shown_next_to_its_field_and_nothing_is_saved(
    seen, values, at_field, message
):
    at = click(start(), "bp_new")
    fill(at, **{"name": "Acme Health", "tone": "confident", **values})
    click(at, "bp_save")

    assert shown_after(at, at_field) == message
    assert [p.name for p in BrandProfileStore().list()] == ["Default"]
    assert at.sidebar.selectbox[0].options == ["Default"]


def tab_text(at):
    [tab] = [t for t in at.tabs if t.label == "Brand Profiles"]
    return [e.value for e in tab.children.values() if e.type == "markdown"]


def test_the_tab_lists_every_profile_default_first_with_audience_and_updated_at(seen):
    store = BrandProfileStore(clock=lambda: datetime(2026, 11, 3, 9, 30, tzinfo=UTC))
    store.create({"name": "Zen Yoga", "audience": "Busy parents", "tone": ["calm"]})
    store.create({"name": "Acme Health", "audience": "Hospital CIOs", "tone": ["confident"]})

    text = "\n".join(tab_text(start()))

    assert text.index("Default") < text.index("Acme Health") < text.index("Zen Yoga")
    assert "Hospital CIOs" in text
    assert "Busy parents" in text
    assert "A general business audience" in text
    assert "2026-11-03 09:30 UTC" in text


def test_default_is_read_only_with_only_a_duplicate_button(seen):
    profile = BrandProfileStore().create({"name": "Acme Health", "tone": ["confident"]})
    at = start()

    keys = {b.key for b in at.button}
    assert {"bp_duplicate_default", f"bp_edit_{profile.id}", f"bp_delete_{profile.id}"} <= keys
    assert not {"bp_edit_default", "bp_delete_default"} & keys


def test_editing_a_profile_saves_the_change_and_refreshes_updated_at(seen):
    BrandProfileStore(clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)).create(
        {"name": "Acme Health", "audience": "Hospital CIOs", "tone": ["confident"]}
    )
    at = click(start(), "bp_edit_acme-health")
    assert field(at, "name").value == "Acme Health"
    assert field(at, "tone").value == "confident"

    fill(at, name="Acme Care", audience="Clinic managers")
    click(at, "bp_save")

    [saved] = BrandProfileStore().list()[1:]
    assert (saved.id, saved.name, saved.audience) == ("acme-health", "Acme Care", "Clinic managers")
    assert saved.updated_at > datetime(2026, 1, 1, tzinfo=UTC)
    assert at.sidebar.selectbox[0].options == ["Default", "Acme Care"]


def test_editing_keeps_every_saved_field_the_marketer_did_not_change(seen):
    store = BrandProfileStore()
    original = store.create(
        {
            "name": "Acme Health",
            "tone": ["confident"],
            "words_to_avoid": ["revolutionary"],
            "voice_description": "Short, plain sentences.",
            "visual_style": {"colours": ["#0A7CFF"], "medium": "3D", "mood": "calm"},
        }
    )
    at = click(start(), "bp_edit_acme-health")
    fill(at, audience="Clinic managers")
    click(at, "bp_save")

    assert store.get("acme-health").fields() == {
        **original.fields(),
        "audience": "Clinic managers",
    }


@pytest.mark.parametrize("source", ["default", "acme-health"])
def test_duplicating_a_profile_opens_the_form_prefilled_under_a_new_name(seen, source):
    BrandProfileStore().create({"name": "Acme Health", "tone": ["confident", "warm"]})
    original = BrandProfileStore().get(source)
    at = click(start(), f"bp_duplicate_{source}")

    assert field(at, "name").value == f"{original.name} copy"
    assert field(at, "tone").value == "\n".join(original.tone)
    assert len(BrandProfileStore().list()) == 2  # nothing saved until Save

    click(at, "bp_save")

    copy = BrandProfileStore().list()[1:]
    names = [p.name for p in copy]
    assert f"{original.name} copy" in names
    [made] = [p for p in copy if p.name == f"{original.name} copy"]
    assert {**made.fields(), "name": original.name} == original.fields()


def test_deleting_a_profile_asks_for_confirmation_first(seen):
    BrandProfileStore().create({"name": "Acme Health", "tone": ["confident"]})
    at = click(start(), "bp_delete_acme-health")
    assert [p.name for p in BrandProfileStore().list()] == ["Default", "Acme Health"]

    click(at, "bp_cancel_delete")
    assert [p.name for p in BrandProfileStore().list()] == ["Default", "Acme Health"]

    click(at, "bp_delete_acme-health")
    click(at, "bp_confirm_delete")
    assert [p.name for p in BrandProfileStore().list()] == ["Default"]
    assert at.sidebar.selectbox[0].options == ["Default"]


def test_deleting_the_active_profile_falls_back_to_default(seen):
    BrandProfileStore().create({"name": "Acme Health", "tone": ["confident"]})
    at = start()
    at.sidebar.selectbox[0].set_value("acme-health").run()

    click(at, "bp_delete_acme-health")
    click(at, "bp_confirm_delete")
    at.chat_input[0].set_value("hello").run()

    assert at.sidebar.selectbox[0].value == "default"
    assert seen == [("acme-health", "Default")]


SAMPLE = "We keep it short. We say what we mean, and we never hide behind jargon. " * 6
LEARNED = {
    "voice_description": "Short, plain sentences in the first person plural. Dry humour.",
    "tone": ["direct", "plain-spoken", "dry"],
}


@pytest.fixture
def voice_calls(monkeypatch):
    """The (agent, last message) of each structured call; the model returns LEARNED."""
    calls = []

    def fake(agent, messages, schema, config=None):
        calls.append((agent, messages[-1]["content"]))
        return schema.model_validate(LEARNED)

    monkeypatch.setattr(llm, "complete_structured", fake)
    return calls


def test_learn_voice_fills_the_form_without_saving_and_the_writers_get_it_once_saved(
    seen, voice_calls
):
    at = click(start(), "bp_new")
    fill(at, name="Acme Health", tone="confident", writing_sample=SAMPLE)
    click(at, "bp_learn_voice")

    assert [agent for agent, _ in voice_calls] == ["brand_voice"]
    assert SAMPLE.strip() in voice_calls[0][1]
    assert field(at, "voice_description").value == LEARNED["voice_description"]
    assert field(at, "tone").value == "direct\nplain-spoken\ndry"
    assert [p.name for p in BrandProfileStore().list()] == ["Default"]

    fill(at, tone="direct\nplain-spoken\ndry\nwarm")  # the marketer edits before saving
    click(at, "bp_save")

    saved = BrandProfileStore().get("acme-health")
    assert saved.voice_description == LEARNED["voice_description"]
    assert saved.tone == ["direct", "plain-spoken", "dry", "warm"]
    state = load_brand_profile({"brand_profile_id": "acme-health"})
    assert f"The brand's voice: {LEARNED['voice_description']}" in brand_voice(state)


@pytest.mark.parametrize(
    ("sample", "message"),
    [("x" * 299, "too short"), ("x" * 10_001, "too long")],
    ids=["299 chars", "10001 chars"],
)
def test_a_sample_of_the_wrong_length_gets_a_clear_message_and_no_call(
    seen, voice_calls, sample, message
):
    at = click(start(), "bp_new")
    fill(at, name="Acme Health", tone="confident", writing_sample=sample)
    click(at, "bp_learn_voice")

    shown = shown_after(at, "learn_voice")
    assert message in shown and "300" in shown and "10,000" in shown
    assert voice_calls == []
    assert field(at, "tone").value == "confident"
    assert field(at, "voice_description").value == ""


def test_a_failed_learn_voice_call_shows_an_error_and_leaves_the_form_unchanged(seen, monkeypatch):
    def failing(*args, **kwargs):
        raise llm.StructuredOutputError("bad reply")

    monkeypatch.setattr(llm, "complete_structured", failing)
    store = BrandProfileStore()
    store.create({"name": "Acme Health", "tone": ["confident"], "voice_description": "Old."})
    at = click(start(), "bp_edit_acme-health")
    fill(at, writing_sample=SAMPLE)
    click(at, "bp_learn_voice")

    assert "couldn't learn the voice" in shown_after(at, "learn_voice")
    assert field(at, "tone").value == "confident"
    assert field(at, "voice_description").value == "Old."
    assert store.get("acme-health").writing_sample is None
