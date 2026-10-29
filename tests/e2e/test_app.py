from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage
from PIL import Image
from streamlit.testing.v1 import AppTest

from contentblitz.agents import query_handler as qh
from contentblitz.core.config import Settings
from contentblitz.workflow import graph
from contentblitz.workflow.brand_profiles import BrandProfileStore

APP = str(Path(__file__).parents[2] / "src/contentblitz/web_app/streamlit_app.py")


def chat_messages(at):
    return [(m.name, m.markdown[0].value) for m in at.chat_message]


def test_chat_round_trip(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key=None))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Chat",
        "Content Dashboard",
        "Research",
        "Brand Profiles",
        "Usage",
    ]

    at.chat_input[0].set_value("hello").run()
    assert at.chat_message[-1].markdown[0].value.endswith("hello")


def test_chat_shows_every_assistant_message_from_a_run_in_order(monkeypatch):
    def two_step_handler(state):
        n = len(state["messages"])
        return {"messages": [AIMessage(content=f"plan {n}"), AIMessage(content=f"step {n}")]}

    st.cache_resource.clear()
    monkeypatch.setattr(graph, "query_handler", two_step_handler)
    at = AppTest.from_file(APP, default_timeout=30).run()

    at.chat_input[0].set_value("first").run()
    at.chat_input[0].set_value("second").run()

    assert chat_messages(at) == [
        ("user", "first"),
        ("assistant", "plan 1"),
        ("assistant", "step 1"),
        ("user", "second"),
        ("assistant", "plan 4"),
        ("assistant", "step 4"),
    ]


def test_the_brand_profile_picker_passes_the_chosen_profile_into_the_next_run(monkeypatch):
    store = BrandProfileStore()
    store.create({"name": "Zen Yoga", "tone": ["calm"]})
    store.create({"name": "Acme Health", "tone": ["confident"]})
    seen = []

    def recording_handler(state):
        seen.append((state.get("brand_profile_id"), state["brand_profile"]["name"]))
        return {"messages": [AIMessage(content="ok")]}

    st.cache_resource.clear()
    monkeypatch.setattr(graph, "query_handler", recording_handler)
    at = AppTest.from_file(APP, default_timeout=30).run()

    picker = at.sidebar.selectbox[0]
    assert picker.label == "Brand Profile"
    assert picker.options == ["Default", "Acme Health", "Zen Yoga"]
    assert picker.value == "default"  # a new conversation starts with Default
    at.chat_input[0].set_value("first").run()

    at.sidebar.selectbox[0].set_value("acme-health").run()
    assert "Acme Health" in at.sidebar.markdown[0].value  # the active profile is shown
    at.chat_input[0].set_value("second").run()

    assert seen == [("default", "Default"), ("acme-health", "Acme Health")]


def test_a_deleted_active_profile_gets_the_workflows_notice(monkeypatch):
    store = BrandProfileStore()
    profile = store.create({"name": "Acme Health", "tone": ["confident"]})
    monkeypatch.setattr(graph, "query_handler", lambda s: {"messages": [AIMessage("ok")]})
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.selectbox[0].set_value(profile.id).run()

    store.delete(profile.id)
    at.chat_input[0].set_value("hello").run()

    assert at.sidebar.selectbox[0].value == "default"
    # No run loaded it before it was deleted, so the notice can only name its id.
    notice = "Brand Profile *acme-health* no longer exists, so this conversation uses Default."
    assert chat_messages(at)[1:] == [("assistant", notice), ("assistant", "ok")]


def _image_run(monkeypatch, tmp_path, *, placeholder=False, write_file=True):
    """Fakes a run whose last message references Revision 1 of an image Content Piece."""
    path = tmp_path / "piece-1-r1.png"
    if write_file:
        Image.new("RGB", (1536, 1024), "teal").save(path, format="PNG")
    metadata = {
        "request": "an image for my blog",
        "prompt": "A calm radiology suite at dawn. Do not draw any text.",
        "alt_text": "A calm radiology suite",
        "use": "blog_header",
        "aspect": "landscape",
        "model": "placeholder" if placeholder else "gpt-image-1",
        "size": "1536x1024",
        "created_at": "2026-11-01T09:00:00+00:00",
        "file_path": str(path),
        "provider_url": "https://provider.example/expired.png",
        "provider_url_temporary": True,
        "placeholder": placeholder,
        "fallbacks": [],
    }
    revision = {
        "text": "A calm radiology suite",
        "origin": "draft",
        "created_at": metadata["created_at"],
        "image": metadata,
    }

    def image_handler(state):
        piece = {
            "kind": "image",
            "topic_id": "topic-1",
            "title": "A calm radiology suite",
            "revisions": [revision],
        }
        return {
            "content_pieces": {"piece-1": piece},
            "messages": [
                AIMessage(content="Plan: image"),
                AIMessage(
                    content="Image ready: *A calm radiology suite*",
                    additional_kwargs={"content_piece": {"piece_id": "piece-1", "revision": 1}},
                ),
            ],
        }

    st.cache_resource.clear()
    monkeypatch.setattr(graph, "query_handler", image_handler)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("an image for my blog").run()
    assert not at.exception
    return at


def test_chat_shows_an_image_content_piece_from_its_file(monkeypatch, tmp_path):
    at = _image_run(monkeypatch, tmp_path)

    message = at.chat_message[-1]
    assert message.markdown[0].value == "Image ready: *A calm radiology suite*"
    assert len(message.image) == 1
    [url] = message.image[0].value
    assert "provider.example" not in url  # served from the saved file; provider links expire
    assert message.image[0].captions == ["gpt-image-1 · 1536×1024"]
    # other messages render as before: text only
    assert at.chat_message[-2].markdown[0].value == "Plan: image"
    assert not at.chat_message[-2].image


def test_chat_keeps_showing_the_image_on_later_runs(monkeypatch, tmp_path):
    at = _image_run(monkeypatch, tmp_path)

    at.chat_input[0].set_value("thanks").run()

    image_messages = [m for m in at.chat_message if m.image]
    assert len(image_messages) == 2  # one per run; history replays the first
    assert image_messages[0].image[0].captions == ["gpt-image-1 · 1536×1024"]


def test_chat_shows_the_optimised_prompt_in_a_collapsible_section(monkeypatch, tmp_path):
    at = _image_run(monkeypatch, tmp_path)

    [expander] = at.chat_message[-1].expander
    assert expander.label == "Optimised prompt"
    assert expander.code[0].value == "A calm radiology suite at dawn. Do not draw any text."


def test_chat_captions_a_placeholder_image(monkeypatch, tmp_path):
    at = _image_run(monkeypatch, tmp_path, placeholder=True)

    assert at.chat_message[-1].image[0].captions == ["Placeholder image · 1536×1024"]


def test_chat_shows_a_notice_when_the_image_file_is_missing(monkeypatch, tmp_path):
    at = _image_run(monkeypatch, tmp_path, write_file=False)

    message = at.chat_message[-1]
    assert not message.image
    assert "missing" in message.warning[0].value
