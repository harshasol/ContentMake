"""Hand-editing a piece in the Content Dashboard adds a Revision (#63)."""

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from contentblitz.workflow import graph

APP = str(Path(__file__).parents[2] / "src/contentblitz/web_app/streamlit_app.py")
REVIEW = {"engagement": 7, "brand_voice": 5, "suggestions": []}
SCORE = {
    "score": 80,
    "passed": True,
    "checks": [],
    "review": REVIEW,
    "reviewer_unavailable": False,
}
BRIEF = {
    "summary": "s",
    "key_findings": [],
    "statistics": [],
    "angles": [],
    "target_keyword": "ai",
    "secondary_keywords": [],
    "sources": [],
    "created_at": "2026-10-28T09:05:00+00:00",
    "model": "gpt-4o",
    "research_gathered_at": "2026-10-28T09:00:00+00:00",
}


def _revision(**extra):
    return {
        "origin": "draft",
        "created_at": "2026-10-29T00:00:00+00:00",
        "text": "Original text",
        "quality_score": SCORE,
        **extra,
    }


def _opened(monkeypatch, kind="blog", **extra):
    piece = {
        "kind": kind,
        "topic_id": "topic-1",
        "title": "My piece",
        "revisions": [_revision(**extra)],
    }

    def scripted(state):
        return {
            "topics": {"topic-1": {"name": "AI", "brief": BRIEF, "piece_ids": ["piece-1"]}},
            "active_topic_id": "topic-1",
            "content_pieces": {"piece-1": piece},
        }

    st.cache_resource.clear()
    monkeypatch.setattr(graph, "query_handler", scripted)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("go").run()
    at.tabs[1].selectbox[0].select("piece-1").run()
    return at


def _button(tab, label):
    return next(b for b in tab.button if b.label == label)


def test_editing_and_saving_a_blog_adds_a_revision_with_the_edit(monkeypatch):
    at = _opened(monkeypatch, meta_description="Old meta")

    _button(at.tabs[1], "Edit").click().run()
    at.tabs[1].text_area[0].set_value("Edited **text**")
    at.tabs[1].text_input[0].set_value("Edited meta")
    _button(at.tabs[1], "Save").click().run()

    tab = at.tabs[1]
    assert "Revision 2 of 2" in " ".join(c.value for c in tab.caption)
    assert "Edited **text**" in " ".join(m.value for m in tab.markdown)
    assert "Meta description: Edited meta" in " ".join(c.value for c in tab.caption)
    assert "not re-reviewed" in " ".join(m.value for m in tab.markdown)
    assert "Edit" in [b.label for b in tab.button]  # back out of the form
    assert "Save" not in [b.label for b in tab.button]


def test_cancelling_an_edit_adds_no_revision(monkeypatch):
    at = _opened(monkeypatch)

    _button(at.tabs[1], "Edit").click().run()
    at.tabs[1].text_area[0].set_value("Changed my mind")
    _button(at.tabs[1], "Cancel").click().run()

    assert "Revision 1 of 1" in " ".join(c.value for c in at.tabs[1].caption)
    assert "Original text" in " ".join(m.value for m in at.tabs[1].markdown)


def test_a_linkedin_post_is_edited_as_plain_text_without_a_meta_description(monkeypatch):
    at = _opened(monkeypatch, kind="linkedin_post")

    _button(at.tabs[1], "Edit").click().run()
    assert len(at.tabs[1].text_input) == 0
    at.tabs[1].text_area[0].set_value("My own words\n\n#ai")
    _button(at.tabs[1], "Save").click().run()

    assert "My own words\n\n#ai" in [t.value for t in at.tabs[1].text]
    assert "Revision 2 of 2" in " ".join(c.value for c in at.tabs[1].caption)


def test_an_image_offers_no_edit(monkeypatch):
    at = _opened(monkeypatch, kind="image")

    assert "Edit" not in [b.label for b in at.tabs[1].button]
