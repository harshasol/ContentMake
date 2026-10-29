"""The Content Dashboard tab lists Content Pieces, filtered by Topic (#60)."""

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from contentblitz.agents import query_handler as qh
from contentblitz.core.config import Settings
from contentblitz.workflow import graph

APP = str(Path(__file__).parents[2] / "src/contentblitz/web_app/streamlit_app.py")
REVISION = {"origin": "draft", "created_at": "2026-10-29T00:00:00+00:00", "text": "x"}
SCORE = {"score": 88, "passed": True, "checks": [], "review": None, "reviewer_unavailable": False}


def test_an_empty_dashboard_names_the_next_step(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key=None))
    at = AppTest.from_file(APP, default_timeout=30).run()

    text = " ".join(i.value for i in at.tabs[1].info)

    assert "No Content Pieces yet. Ask for a blog in the Chat tab" in text


def test_the_dashboard_lists_a_blog_written_in_a_scripted_run(monkeypatch):
    def writes_a_blog(state):
        return {
            "topics": {
                "topic-1": {"name": "AI", "piece_ids": ["piece-1"]},
                "topic-2": {"name": "Yoga", "piece_ids": []},
            },
            "active_topic_id": "topic-1",
            "content_pieces": {
                "piece-1": {
                    "kind": "blog",
                    "topic_id": "topic-1",
                    "title": "AI in Health",
                    "revisions": [{**REVISION, "quality_score": SCORE}],
                }
            },
        }

    st.cache_resource.clear()
    monkeypatch.setattr(graph, "query_handler", writes_a_blog)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("write a blog").run()

    [table] = at.tabs[1].dataframe
    row = table.value.iloc[0]
    assert "AI in Health" in row["Title"]
    assert (row["Revision"], row["Quality Score"]) == (1, "88")

    at.tabs[1].radio[0].set_value("All Topics").run()
    assert [h.value for h in at.tabs[1].subheader] == ["AI"]  # Yoga has no pieces
