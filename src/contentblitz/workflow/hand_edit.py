"""A hand edit: the Revision a marketer's own edit of a text piece becomes (spec #23, #63).

Its Quality Score is the format's rule checks run again, against the length and hashtag targets
the edited Revision was checked with (`check_targets`; the defaults for one that has none). The
AI reviewer is not asked, so its scores are carried over from the Revision edited and marked
`not_re_reviewed`.
"""

from datetime import UTC, datetime

from contentblitz.core.config import get_model_config
from contentblitz.workflow.active_brand_profile import active_brand_profile
from contentblitz.workflow.content_checks import Draft, Post, check_blog, check_linkedin
from contentblitz.workflow.quality_score import quality_score
from contentblitz.workflow.state import CheckTargets, ContentBlitzState, ResearchBrief, Revision
from contentblitz.workflow.statistics import brief_facts

# What the checks see when a Topic has no Brief (fresh research clears it): no Sources or facts
# to match against.
NO_BRIEF: ResearchBrief = {
    "summary": "",
    "key_findings": [],
    "statistics": [],
    "angles": [],
    "target_keyword": "",
    "secondary_keywords": [],
    "sources": [],
    "created_at": "",
    "model": "",
    "research_gathered_at": "",
}


def _tags(targets: CheckTargets) -> tuple[int, int] | None:
    return tuple(targets["hashtags"]) if "hashtags" in targets else None


def hand_edit_revision(
    state: ContentBlitzState, piece_id: str, text: str, meta_description: str | None
) -> Revision:
    """The `hand_edit` Revision of a blog or LinkedIn post: the edited text (and a blog's meta
    description), the keyword and template carried over, and the re-checked Quality Score.

    Raises KeyError for an unknown piece and ValueError for an image, which can't be edited.
    """
    piece = state["content_pieces"][piece_id]
    if piece["kind"] == "image":
        raise ValueError("An image can't be hand-edited.")
    latest = piece["revisions"][-1]
    brief = state["topics"][piece["topic_id"]].get("brief") or NO_BRIEF
    targets = latest.get("check_targets") or {}
    avoid = active_brand_profile(state).words_to_avoid
    rules = get_model_config().quality
    revision: Revision = {
        "text": text,
        "origin": "hand_edit",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if piece["kind"] == "blog":
        meta = latest.get("meta_description", "") if meta_description is None else meta_description
        draft = Draft(
            body=text,
            meta_description=meta,
            keyword=latest.get("target_keyword") or brief["target_keyword"],
            source_numbers=[s["number"] for s in brief["sources"]],
        )
        checks = check_blog(draft, rules.blog, targets.get("length"), avoid)
        revision["meta_description"] = meta
        for key in ("target_keyword", "template"):
            if key in latest:
                revision[key] = latest[key]
    else:
        post = Post(text=text, brief_facts=brief_facts(brief))
        checks = check_linkedin(post, rules.linkedin, targets.get("length"), _tags(targets), avoid)
    if targets:
        revision["check_targets"] = targets
    review = (latest.get("quality_score") or {}).get("review")
    revision["quality_score"] = {
        **quality_score(checks, review, rules.pass_mark),
        "not_re_reviewed": True,
    }
    return revision
