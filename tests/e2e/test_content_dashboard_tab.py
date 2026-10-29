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


def _scripted(monkeypatch, pieces):
    def writes_pieces(state):
        return {
            "topics": {"topic-1": {"name": "AI", "piece_ids": list(pieces)}},
            "active_topic_id": "topic-1",
            "content_pieces": pieces,
        }

    st.cache_resource.clear()
    monkeypatch.setattr(graph, "query_handler", writes_pieces)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("go").run()
    return at


CHECKS = [
    {"name": "keyword density", "passed": True, "measured": "1.4%", "reason": "1-2% wanted"},
    {"name": "length", "passed": False, "measured": "90 words", "reason": "Add 410 words"},
]
BLOG_SCORE = {
    "score": 62,
    "passed": False,
    "checks": CHECKS,
    "review": {"engagement": 7, "brand_voice": 5, "suggestions": ["Open with a story"]},
    "reviewer_unavailable": False,
}


def _blog(**extra):
    revision = {
        **REVISION,
        "text": "# Heading\n\nBody **bold**",
        "meta_description": "A short meta",
        "target_keyword": "ai health",
        "template": "how-to",
        "quality_score": BLOG_SCORE,
        **extra,
    }
    return {"kind": "blog", "topic_id": "topic-1", "title": "AI in Health", "revisions": [revision]}


def _markdown(tab) -> str:
    return " ".join(m.value for m in tab.markdown)


def test_opening_a_blog_shows_it_rendered_with_its_seo_fields_and_score_breakdown(monkeypatch):
    at = _scripted(monkeypatch, {"piece-1": _blog()})

    at.tabs[1].selectbox[0].select("piece-1").run()

    tab = at.tabs[1]
    assert "Body **bold**" in _markdown(tab)
    text = " ".join(c.value for c in tab.caption)
    assert "A short meta" in text and "ai health" in text and "how-to" in text
    checks = [d.value for d in tab.dataframe if "Check" in d.value.columns][0]
    assert list(checks["Check"]) == ["keyword density", "length"]
    assert list(checks["Result"]) == ["Pass", "Fail"]
    assert list(checks["Measured"]) == ["1.4%", "90 words"]
    assert list(checks["Suggestion"]) == ["1-2% wanted", "Add 410 words"]
    assert "Open with a story" in _markdown(tab)
    assert "Engagement 7/10" in _markdown(tab)


def test_an_unavailable_reviewer_is_said_so(monkeypatch):
    blog = _blog(quality_score={**BLOG_SCORE, "review": None, "reviewer_unavailable": True})
    at = _scripted(monkeypatch, {"piece-1": blog})

    at.tabs[1].selectbox[0].select("piece-1").run()

    assert "AI reviewer was unavailable" in _markdown(at.tabs[1])


def test_opening_a_linkedin_post_shows_plain_text_with_line_breaks_and_a_count(monkeypatch):
    text = "Hook line\n\nSecond para #ai"
    post = {
        "kind": "linkedin_post",
        "topic_id": "topic-1",
        "title": "Post",
        "revisions": [{**REVISION, "text": text, "quality_score": BLOG_SCORE}],
    }
    at = _scripted(monkeypatch, {"piece-1": post})

    at.tabs[1].selectbox[0].select("piece-1").run()

    assert text in [t.value for t in at.tabs[1].text]
    assert f"{len(text)} characters" in " ".join(c.value for c in at.tabs[1].caption)


def _image(path):
    meta = {
        "request": "r",
        "prompt": "an optimised prompt",
        "alt_text": "A blue square",
        "use": "general",
        "aspect": "square",
        "model": "gpt-image-1",
        "size": "8x8",
        "created_at": REVISION["created_at"],
        "file_path": str(path),
        "provider_url": None,
        "provider_url_temporary": False,
        "placeholder": False,
        "fallbacks": [],
    }
    revision = {**REVISION, "text": "A blue square", "image": meta}
    return {"kind": "image", "topic_id": "topic-1", "title": "Header", "revisions": [revision]}


def test_opening_an_image_shows_it_with_alt_text_caption_and_prompt(monkeypatch, tmp_path):
    from PIL import Image

    path = tmp_path / "i.png"
    Image.new("RGB", (8, 8), "#336699").save(path)
    at = _scripted(monkeypatch, {"piece-1": _image(path)})

    at.tabs[1].selectbox[0].select("piece-1").run()

    tab = at.tabs[1]
    [image] = tab.get("image")
    assert image.captions == ["gpt-image-1 · 8×8"]
    assert "A blue square" in " ".join(c.value for c in tab.caption)
    assert [c.value for c in tab.code] == ["an optimised prompt"]
    assert not tab.warning


def test_an_image_whose_file_is_missing_shows_a_warning(monkeypatch, tmp_path):
    at = _scripted(monkeypatch, {"piece-1": _image(tmp_path / "gone.png")})

    at.tabs[1].selectbox[0].select("piece-1").run()

    assert "image file" in at.tabs[1].warning[0].value


def _refined_blog():
    first = _blog()["revisions"][0]
    brand = {"id": "default", "name": "Acme", "updated_at": "2026-10-28T00:00:00+00:00"}
    second = {
        **first,
        "origin": "refinement",
        "text": "Second version",
        "brand_profile": brand,
        "quality_score": {**BLOG_SCORE, "score": 80},
    }
    return {**_blog(), "revisions": [{**first, "text": "First version"}, second]}


def test_the_history_lists_every_revision_and_view_opens_an_older_one(monkeypatch):
    at = _scripted(monkeypatch, {"piece-1": _refined_blog()})
    at.tabs[1].selectbox[0].select("piece-1").run()

    tab = at.tabs[1]
    [history] = [d.value for d in tab.dataframe if "Origin" in d.value.columns]
    assert list(history["Revision"]) == [1, 2]
    assert list(history["Origin"]) == ["Draft", "Refinement"]
    assert list(history["Brand Profile"]) == ["—", "Acme"]
    assert list(history["Quality Score"]) == ["62", "80"]
    assert "Second version" in _markdown(tab)
    assert not [i for i in tab.info if "older Revision" in i.value]

    at.tabs[1].button(key="view_revision_1").click().run()

    tab = at.tabs[1]
    assert "First version" in _markdown(tab)
    assert "Second version" not in _markdown(tab)
    assert any("older Revision 1" in i.value for i in tab.info)
    assert [d for d in tab.dataframe if "Check" in d.value.columns]  # its score, too

    at.tabs[1].button(key="view_revision_2").click().run()
    assert "Second version" in _markdown(at.tabs[1])


def test_viewing_an_older_revision_does_not_change_the_state(monkeypatch):
    at = _scripted(monkeypatch, {"piece-1": _refined_blog()})
    at.tabs[1].selectbox[0].select("piece-1").run()
    at.tabs[1].button(key="view_revision_1").click().run()

    [table] = [d.value for d in at.tabs[1].dataframe if "Kind" in d.value.columns]
    assert table.iloc[0]["Revision"] == 2


def test_restoring_revision_1_adds_a_revision_showing_its_content(monkeypatch):
    at = _scripted(monkeypatch, {"piece-1": _refined_blog()})
    at.tabs[1].selectbox[0].select("piece-1").run()
    assert [b.key for b in at.tabs[1].button if "restore" in b.key] == ["restore_revision_1"]

    at.tabs[1].button(key="restore_revision_1").click().run()

    tab = at.tabs[1]
    [history] = [d.value for d in tab.dataframe if "Origin" in d.value.columns]
    assert list(history["Revision"]) == [1, 2, 3]
    assert list(history["Origin"]) == ["Draft", "Refinement", "Restore"]
    assert "First version" in _markdown(tab)  # the new latest shows the restored content
    assert not [i for i in tab.info if "older Revision" in i.value]


def test_export_buttons_appear_on_an_opened_piece(monkeypatch):
    at = _scripted(monkeypatch, {"piece-1": _blog()})
    assert [b.proto.label for b in at.tabs[1].get("download_button")] == ["Download Topic bundle"]

    at.tabs[1].selectbox[0].select("piece-1").run()

    labels = [b.proto.label for b in at.tabs[1].get("download_button")]
    assert labels == ["Download Topic bundle", "Download Markdown", "Download HTML"]


def test_the_topic_bundle_button_appears_only_when_the_topic_has_pieces(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key=None))
    empty = AppTest.from_file(APP, default_timeout=30).run()
    assert not empty.tabs[1].get("download_button")

    at = _scripted(monkeypatch, {"piece-1": _refined_blog()})

    labels = [b.proto.label for b in at.tabs[1].get("download_button")]
    assert labels == ["Download Topic bundle"]
