"""The sidebar Topic selector (#66), driven through Streamlit's AppTest."""

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from contentblitz.workflow import graph

APP = str(Path(__file__).parents[2] / "src/contentblitz/web_app/streamlit_app.py")
REVISION = {"origin": "draft", "created_at": "2026-10-29T00:00:00+00:00", "text": "x"}


def _brief(summary):
    return {
        "summary": summary,
        "key_findings": [],
        "statistics": [],
        "angles": [],
        "target_keyword": "k",
        "secondary_keywords": [],
        "sources": [],
        "created_at": "2026-10-28T09:05:00+00:00",
        "model": "gpt-4o",
        "research_gathered_at": "2026-10-28T09:00:00+00:00",
    }


def _app(monkeypatch):
    def handler(state):
        return {
            "topics": {
                "topic-1": {"name": "AI", "brief": _brief("About AI."), "piece_ids": ["piece-1"]},
                "topic-2": {"name": "Yoga", "brief": _brief("About yoga."), "piece_ids": []},
            },
            "active_topic_id": "topic-1",
            "content_pieces": {
                "piece-1": {
                    "kind": "blog",
                    "topic_id": "topic-1",
                    "title": "AI in Health",
                    "revisions": [REVISION],
                }
            },
        }

    st.cache_resource.clear()
    monkeypatch.setattr(graph, "query_handler", handler)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("go").run()
    return at


def _tab(at, label):
    return next(t for t in at.tabs if t.label == label)


def test_the_selector_lists_the_topics_plus_new_topic_and_shows_the_active_one(monkeypatch):
    at = _app(monkeypatch)

    selector = at.sidebar.selectbox(key="topic_selector")

    assert selector.options == ["New Topic", "AI", "Yoga"]
    assert selector.value == "topic-1"
    assert not selector.disabled


def test_changing_the_selection_changes_the_dashboard_and_research_tab(monkeypatch):
    at = _app(monkeypatch)
    assert "About AI." in "\n".join(m.value for m in _tab(at, "Research").markdown)
    assert [h.value for h in _tab(at, "Content Dashboard").subheader] == ["AI"]

    at.sidebar.selectbox(key="topic_selector").select("topic-2").run()

    assert "About yoga." in "\n".join(m.value for m in _tab(at, "Research").markdown)
    assert not _tab(at, "Content Dashboard").dataframe  # Yoga has no pieces


def test_choosing_new_topic_clears_the_active_topic(monkeypatch):
    at = _app(monkeypatch)

    at.sidebar.selectbox(key="topic_selector").select("New Topic").run()

    assert not at.exception
    assert at.sidebar.selectbox(key="topic_selector").value == "New Topic"
    assert any("No research yet" in i.value for i in _tab(at, "Research").info)
