"""Hand-editing a Content Piece through the workspace (#63): a new `hand_edit` Revision whose
Quality Score comes from the rule checks only. No model is called."""

import pytest

from contentblitz.integrations import llm
from contentblitz.workflow.brand_profiles import DEFAULT_PROFILE, BrandProfile
from contentblitz.workflow.graph import build_graph
from contentblitz.workflow.workspace import Workspace

CONFIG = {"configurable": {"thread_id": "t1"}}
REVIEW = {"engagement": 8, "brand_voice": 6, "suggestions": ["Open with a story"]}
BRIEF = {
    "summary": "s",
    "key_findings": [{"statement": "Reads scans faster.", "source_numbers": [1]}],
    "statistics": [{"figure": "30%", "description": "faster", "source_numbers": [1]}],
    "angles": [],
    "target_keyword": "ai",
    "secondary_keywords": [],
    "sources": [{"number": 1, "title": "t", "url": "https://x.io", "site": "x.io"}],
}
OLD_SCORE = {
    "score": 90,
    "passed": True,
    "checks": [{"name": "length", "passed": True, "measured": "9", "reason": "ok"}],
    "review": REVIEW,
    "reviewer_unavailable": False,
    "revisions_used": 1,
}
BLOG = {
    "text": "# Title\n\nOld body [1].",
    "meta_description": "Old meta.",
    "target_keyword": "ai",
    "template": "how-to",
    "origin": "draft",
    "created_at": "2026-10-21T10:00:00+00:00",
    "quality_score": OLD_SCORE,
}
POST = {
    "text": "Old post.\n\n#ai",
    "origin": "draft",
    "created_at": "2026-10-21T10:00:00+00:00",
    "quality_score": OLD_SCORE,
}
IMAGE = {"text": "alt", "origin": "draft", "created_at": "2026-10-21T10:00:00+00:00"}


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a hand edit must not call a model")

    monkeypatch.setattr(llm, "complete_structured", refuse)
    monkeypatch.setattr(llm, "complete_structured_reply", refuse)


def workspace(profile: BrandProfile | None = None):
    graph = build_graph()
    pieces = {
        "p1": {"kind": "blog", "topic_id": "topic-1", "title": "B", "revisions": [BLOG]},
        "p2": {"kind": "linkedin_post", "topic_id": "topic-1", "title": "P", "revisions": [POST]},
        "p3": {"kind": "image", "topic_id": "topic-1", "title": "I", "revisions": [IMAGE]},
    }
    state = {
        "topics": {"topic-1": {"name": "AI", "brief": BRIEF, "piece_ids": list(pieces)}},
        "active_topic_id": "topic-1",
        "content_pieces": pieces,
    }
    if profile:
        state["brand_profile"] = profile.model_dump(mode="json")
    graph.update_state(CONFIG, state, as_node="query_handler")
    return Workspace(graph, "t1")


def check(revision, name):
    return next(c for c in revision["quality_score"]["checks"] if c["name"] == name)


def test_saving_a_blog_edit_adds_a_hand_edit_revision_and_keeps_the_earlier_ones():
    ws = workspace()

    ws.edit("p1", "# New\n\nEdited body [1].", "Edited meta.")

    piece = ws.piece("p1")
    assert len(piece.revisions) == 2
    assert piece.revision(1) == BLOG
    latest = piece.latest
    assert (latest["origin"], latest["text"], latest["meta_description"]) == (
        "hand_edit",
        "# New\n\nEdited body [1].",
        "Edited meta.",
    )
    assert (latest["target_keyword"], latest["template"]) == ("ai", "how-to")
    assert latest["brand_profile"]["id"] == "default"


def test_the_hand_edit_records_the_active_brand_profile():
    profile = DEFAULT_PROFILE.model_copy(
        update={"id": "acme", "name": "Acme", "words_to_avoid": ["synergy"]}
    )
    ws = workspace(profile)

    ws.edit("p2", "New post.\n\n#ai")

    assert ws.piece("p2").latest["brand_profile"]["id"] == "acme"


def test_the_rule_checks_run_again_on_the_edited_blog_including_words_to_avoid():
    profile = DEFAULT_PROFILE.model_copy(
        update={"id": "acme", "name": "Acme", "words_to_avoid": ["synergy"]}
    )
    ws = workspace(profile)

    ws.edit("p1", "# T\n\nPure synergy [1].", "Meta.")

    latest = ws.piece("p1").latest
    assert not check(latest, "words to avoid")["passed"]
    assert latest["quality_score"]["checks"] != OLD_SCORE["checks"]


def test_a_linkedin_edit_is_rechecked_with_the_linkedin_rules():
    ws = workspace()

    ws.edit("p2", "A post with no tags at all.")

    latest = ws.piece("p2").latest
    assert latest["origin"] == "hand_edit"
    assert "meta_description" not in latest
    assert not check(latest, "hashtags")["passed"]


def test_the_reviewers_scores_are_carried_over_and_marked_not_re_reviewed():
    ws = workspace()

    ws.edit("p2", "Edited.\n\n#ai")

    quality = ws.piece("p2").latest["quality_score"]
    assert quality["review"] == REVIEW
    assert quality["not_re_reviewed"] is True
    assert "revisions_used" not in quality  # no revision loop ran


def test_an_image_cannot_be_hand_edited():
    ws = workspace()

    with pytest.raises(ValueError):
        ws.edit("p3", "new alt")

    assert len(ws.piece("p3").revisions) == 1


def test_an_unknown_piece_is_rejected():
    with pytest.raises(KeyError):
        workspace().edit("nope", "x")


def test_a_blog_edit_is_checked_against_the_length_the_edited_revision_was(monkeypatch):
    ws = workspace()
    ws.graph.update_state(
        CONFIG,
        {
            "content_pieces": {
                "p1": {
                    "kind": "blog",
                    "topic_id": "topic-1",
                    "title": "B",
                    "revisions": [{**BLOG, "check_targets": {"length": 40}}],
                }
            }
        },
        as_node="query_handler",
    )
    body = "# T\n\n" + "word " * 40 + "[1]."

    ws.edit("p1", body, "Meta.")

    latest = ws.piece("p1").latest
    assert latest["check_targets"] == {"length": 40}
    assert check(latest, "length")["passed"]


def test_a_linkedin_edit_is_checked_against_the_hashtags_the_edited_revision_was():
    ws = workspace()
    post = {**POST, "check_targets": {"hashtags": [0, 0]}}
    ws.graph.update_state(
        CONFIG,
        {
            "content_pieces": {
                "p2": {
                    "kind": "linkedin_post",
                    "topic_id": "topic-1",
                    "title": "P",
                    "revisions": [post],
                }
            }
        },
        as_node="query_handler",
    )

    ws.edit("p2", "A post with no tags at all.")

    latest = ws.piece("p2").latest
    assert latest["check_targets"] == {"hashtags": [0, 0]}
    assert check(latest, "hashtags")["passed"]


def test_a_revision_without_targets_is_checked_with_the_defaults():
    ws = workspace()

    ws.edit("p2", "Edited.\n\n#ai")

    assert "check_targets" not in ws.piece("p2").latest


def test_a_restore_carries_the_check_targets_over():
    ws = workspace()
    ws.graph.update_state(
        CONFIG,
        {
            "content_pieces": {
                "p1": {
                    "kind": "blog",
                    "topic_id": "topic-1",
                    "title": "B",
                    "revisions": [{**BLOG, "check_targets": {"length": 40}}],
                }
            }
        },
        as_node="query_handler",
    )
    ws.edit("p1", "# T\n\nBody [1].", "Meta.")

    ws.restore("p1", 1)

    assert ws.piece("p1").latest["check_targets"] == {"length": 40}
