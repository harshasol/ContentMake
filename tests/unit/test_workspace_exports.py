"""Workspace exports of the viewed Revision and of the research report (#68)."""

from types import SimpleNamespace

from contentblitz.workflow.workspace import Workspace

BRIEF = {
    "summary": "S",
    "key_findings": [],
    "statistics": [],
    "angles": [],
    "sources": [],
    "target_keyword": "k",
    "secondary_keywords": [],
    "created_at": "2026-10-28T09:05:00+00:00",
    "model": "m",
    "research_gathered_at": "2026-10-28T09:00:00+00:00",
}


def _workspace(values):
    graph = SimpleNamespace(get_state=lambda config: SimpleNamespace(values=values))
    return Workspace(graph, "t")


def _revision(text):
    return {"text": text, "origin": "draft", "created_at": "2026-10-29T10:00:00+00:00"}


def _state():
    piece = {
        "kind": "linkedin_post",
        "topic_id": "topic-1",
        "title": "Post",
        "revisions": [_revision("first"), _revision("second")],
    }
    topic = {"name": "AI", "piece_ids": ["p1"], "brief": BRIEF}
    return {
        "topics": {"topic-1": topic},
        "active_topic_id": "topic-1",
        "content_pieces": {"p1": piece},
    }


def test_exports_default_to_the_latest_revision_and_can_target_an_older_one():
    workspace = _workspace(_state())

    latest = workspace.piece_exports("p1")
    older = workspace.piece_exports("p1", 1)

    assert [e.file_name for e in latest] == [
        "ai-linkedin-post-post-r2.txt",
        "ai-linkedin-post-post-r2.md",
    ]
    assert latest[0].data == b"second"
    assert older[0].file_name == "ai-linkedin-post-post-r1.txt"
    assert older[0].data == b"first"


def test_an_image_whose_file_is_missing_offers_no_export():
    state = _state()
    state["content_pieces"]["p1"] = {
        "kind": "image",
        "topic_id": "topic-1",
        "title": "Hero",
        "revisions": [{**_revision("alt"), "image": {"file_path": "/nope/x.png"}}],
    }

    assert _workspace(state).piece_exports("p1") == []


def test_unknown_piece_has_no_exports():
    assert _workspace(_state()).piece_exports("nope") == []


def test_research_exports_markdown_and_html():
    exports = _workspace(_state()).research_exports()

    assert [e.file_name for e in exports] == ["ai-research-report.md", "ai-research-report.html"]
    assert b"AI" in exports[0].data


def test_no_research_means_no_exports():
    assert _workspace({}).research_exports() == []
