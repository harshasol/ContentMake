"""Workspace Topic bundle: one zip of the latest Revision of every piece and the report (#69)."""

import io
import zipfile
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


def _revision(text, **extra):
    return {"text": text, "origin": "draft", "created_at": "2026-10-29T10:00:00+00:00", **extra}


def _state(tmp_path, brief=BRIEF):
    png = tmp_path / "hero.png"
    png.write_bytes(b"PNGDATA")
    pieces = {
        "p1": {
            "kind": "blog",
            "topic_id": "topic-1",
            "title": "My Blog",
            "revisions": [_revision("old body"), _revision("new body")],
        },
        "p2": {
            "kind": "linkedin_post",
            "topic_id": "topic-1",
            "title": "Post",
            "revisions": [_revision("a post")],
        },
        "p3": {
            "kind": "image",
            "topic_id": "topic-1",
            "title": "Hero",
            "revisions": [_revision("alt", image={"file_path": str(png)})],
        },
    }
    topic = {"name": "AI Health", "piece_ids": list(pieces), "brief": brief}
    return {
        "topics": {"topic-1": topic},
        "active_topic_id": "topic-1",
        "content_pieces": pieces,
    }


def _zip(export):
    return zipfile.ZipFile(io.BytesIO(export.data))


def test_the_bundle_holds_the_latest_revision_of_every_piece_and_the_report(tmp_path):
    bundle = _workspace(_state(tmp_path)).topic_bundle()

    assert bundle.file_name == "ai-health-bundle.zip"
    assert bundle.mime == "application/zip"
    archive = _zip(bundle)
    assert sorted(archive.namelist()) == [
        "ai-health-blog-my-blog-r2.md",
        "ai-health-image-hero-r1.png",
        "ai-health-linkedin-post-post-r1.txt",
        "ai-health-research-report.md",
    ]
    assert b"new body" in archive.read("ai-health-blog-my-blog-r2.md")
    assert b"old body" not in archive.read("ai-health-blog-my-blog-r2.md")
    assert archive.read("ai-health-image-hero-r1.png") == b"PNGDATA"
    assert archive.read("ai-health-linkedin-post-post-r1.txt") == b"a post"


def test_a_topic_without_research_bundles_only_its_pieces(tmp_path):
    state = _state(tmp_path, brief=None)

    names = _zip(_workspace(state).topic_bundle()).namelist()

    assert "ai-health-research-report.md" not in names
    assert len(names) == 3


def test_a_piece_whose_image_file_is_missing_is_left_out(tmp_path):
    state = _state(tmp_path)
    state["content_pieces"]["p3"]["revisions"][0]["image"]["file_path"] = "/nope/x.png"

    names = _zip(_workspace(state).topic_bundle()).namelist()

    assert not any(n.endswith(".png") for n in names)
    assert len(names) == 3


def test_no_pieces_and_no_research_offers_no_bundle(tmp_path):
    state = _state(tmp_path, brief=None)
    state["topics"]["topic-1"]["piece_ids"] = []

    assert _workspace(state).topic_bundle() is None
    assert _workspace({}).topic_bundle() is None


def test_research_alone_makes_a_bundle(tmp_path):
    state = _state(tmp_path)
    state["topics"]["topic-1"]["piece_ids"] = []

    assert _zip(_workspace(state).topic_bundle()).namelist() == ["ai-health-research-report.md"]
