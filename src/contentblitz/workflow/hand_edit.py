"""A hand edit: the Revision a marketer's own edit of a text piece becomes (spec #23, #63).

Its Quality Score is the format's rule checks run again; the AI reviewer is not asked, so its
scores are carried over from the Revision edited and marked `not_re_reviewed`.
"""

from datetime import UTC, datetime

from contentblitz.core.config import get_model_config
from contentblitz.workflow.active_brand_profile import active_brand_profile
from contentblitz.workflow.content_checks import Draft, Post, check_blog, check_linkedin
from contentblitz.workflow.quality_score import quality_score
from contentblitz.workflow.state import ContentBlitzState, ResearchBrief, Revision


def brief_facts(brief: ResearchBrief) -> str:
    """The Brief's findings and statistics, which every figure in a post must come from."""
    findings = [f["statement"] for f in brief["key_findings"]]
    statistics = [f"{s['figure']}: {s['description']}" for s in brief["statistics"]]
    return "\n".join([*findings, *statistics])


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
    # A Topic can have no Brief (fresh research clears it): the checks then have no Sources or
    # facts to match against.
    brief = state["topics"][piece["topic_id"]].get("brief") or {
        "target_keyword": "",
        "sources": [],
        "key_findings": [],
        "statistics": [],
    }
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
        checks = check_blog(draft, rules.blog, avoid=avoid)
        revision["meta_description"] = meta
        for key in ("target_keyword", "template"):
            if key in latest:
                revision[key] = latest[key]
    else:
        post = Post(text=text, brief_facts=brief_facts(brief))
        checks = check_linkedin(post, rules.linkedin, avoid=avoid)
    review = (latest.get("quality_score") or {}).get("review")
    revision["quality_score"] = {
        **quality_score(checks, review, rules.pass_mark),
        "not_re_reviewed": True,
    }
    return revision
