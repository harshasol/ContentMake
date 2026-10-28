"""SEO Blog Writer: writes a blog from the active Topic's Research Brief. See spec #19.

The blog step runs the blog's quality loop. Its draft is one structured call to the
`blog_writer` model, given the whole Brief, its target and secondary keywords, the four
template guides, the target word range and the step's instructions. It returns the template
it used, the title, the meta description and the Markdown body, citing the Brief's Sources
as `[n]`. The loop checks the draft against the blog's SEO rules, and the AI reviewer rates its
hook, flow, clarity and call to action. A draft that doesn't pass is revised: the same call,
given the current draft and what to fix. The kept Attempt is posted and saved with its
Quality Score and how many revisions were used.

A refine step that targets a blog runs the same loop, but its first Attempt changes the blog's
latest Revision (which may be a hand edit) as the marketer asks, instead of drafting from
scratch. Its keyword and template carry over unless the request sets new ones, and the kept
Attempt is added to the blog as a new Revision. The refinement call is the loop's first Attempt,
so it doesn't count as one of the automatic revisions.

If every provider fails, the step posts an error and the plan carries on: other pieces don't
depend on the blog, unlike research and the Brief.
"""

from dataclasses import replace
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from contentblitz.core.config import get_model_config
from contentblitz.integrations import llm
from contentblitz.workflow.content_checks import CheckResult, Draft, check_blog, word_range
from contentblitz.workflow.plan import current_step
from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.quality_loop import Attempt, QualityFormat, quality_loop
from contentblitz.workflow.quality_score import QualityScore
from contentblitz.workflow.report import citation
from contentblitz.workflow.reviewer import Review, review
from contentblitz.workflow.routing import Template
from contentblitz.workflow.state import (
    ContentBlitzState,
    ContentPiece,
    ResearchBrief,
    Revision,
    RevisionOrigin,
    Topic,
)
from contentblitz.workflow.topics import active_topic, add_piece, add_revision, new_piece_id

AGENT = "blog_writer"

TEMPLATES: dict[str, str] = {
    "how-to": "the problem, then the steps to solve it, then tips",
    "listicle": "numbered items, one H2 each",
    "comparison": "the options, the criteria they are compared on, then a verdict",
    "thought leadership": "a thesis, the evidence for it, then its implications",
}

SYSTEM_PROMPT = """\
You are an SEO blog writer for a content marketer. Write one blog post from the Research Brief
below, using its findings and statistics and nothing from your own knowledge.
Prefer verified statistics; use an unverified one only when you must.
Cite every claim with the number of the Source it comes from, as [n], and end with a "Sources"
section listing every Source you cited, with its link. Write Markdown with exactly one H1
title, then H2 sections with H3s only under them. Optimise for the target keyword and work the
secondary keywords in naturally. Follow the structure of the template you are given, or pick
the template that fits best when you aren't given one.
Follow the marketer's instructions."""

REVISE_PROMPT = """\
Revise your blog post above to fix what follows. Change only what is needed to fix it, and keep
the rest of the post as it is.

{feedback}"""

REFINE_PROMPT = """\
Revise your blog post above as the marketer asks below. Change only what is needed for that, and
keep the rest of the post as it is.

{instructions}"""

JUDGING_CRITERIA = "its hook, flow, clarity and call to action"
SUGGESTIONS_SHOWN = 3

NO_MODEL = (
    "I couldn't write the blog on *{topic}*: the model didn't give a usable reply. "
    "The rest of the plan carries on, so ask again to retry the blog."
)
NO_REFINEMENT = (
    "I couldn't refine *{title}*: the model didn't give a usable reply. "
    "Its latest Revision is unchanged, so ask again to retry."
)
NO_BRIEF = (
    "I can't refine *{title}* yet: *{topic}* has no Research Brief to check it against. "
    "Ask me to research it again first."
)


class BlogDraft(BaseModel):
    template: Template = Field(description="The template whose structure the post follows.")
    title: str = Field(description="The post's title, the same as its H1.")
    meta_description: str = Field(
        description="The search-result snippet: 150-160 characters, with the target keyword."
    )
    body: str = Field(
        description="The full post in Markdown: one H1, H2/H3 sections, [n] citations, and a "
        "closing Sources section."
    )


def _brief_text(brief: ResearchBrief) -> str:
    findings = "\n".join(
        f"- {f['statement']} {citation(f['source_numbers'])}" for f in brief["key_findings"]
    )
    statistics = "\n".join(
        f"- {s['figure']}: {s['description']} {citation(s['source_numbers'])} "
        f"({'verified' if s['verified'] else 'unverified'})"
        for s in brief["statistics"]
    )
    angles = "\n".join(f"- {a['title']}: {a['hook']}" for a in brief["angles"])
    sources = "\n".join(
        f"[{s['number']}] {s['title']} ({s['site']}): {s['url']}" for s in brief["sources"]
    )
    return (
        f"Summary: {brief['summary']}\n\n"
        f"Key findings:\n{findings or 'none'}\n\n"
        f"Statistics:\n{statistics or 'none'}\n\n"
        f"Content angles:\n{angles or 'none'}\n\n"
        f"Sources:\n{sources}"
    )


def _target(state: ContentBlitzState) -> str | None:
    """The blog a refine step targets, or None when the step writes a new blog."""
    return (current_step(state) or {}).get("target")


def _topic(state: ContentBlitzState) -> Topic:
    """The Topic the blog is on: the refined blog's, or the active Topic for a new blog."""
    piece_id = _target(state)
    if piece_id:
        return state["topics"][state["content_pieces"][piece_id]["topic_id"]]
    return active_topic(state)


def _carried_over(state: ContentBlitzState, field: str) -> str | None:
    """For a refinement, the field from the refined blog's latest Revision that records it."""
    piece_id = _target(state)
    revisions = state["content_pieces"][piece_id]["revisions"] if piece_id else []
    return next((r[field] for r in reversed(revisions) if r.get(field)), None)


def _keyword(state: ContentBlitzState) -> str:
    """The target keyword: the request's when it sets one, otherwise the refined blog's,
    otherwise the Brief's suggestion."""
    step = current_step(state) or {}
    return (
        step.get("target_keyword")
        or _carried_over(state, "target_keyword")
        or _topic(state)["brief"]["target_keyword"]
    )


def _template(state: ContentBlitzState) -> str | None:
    """The template: the request's when it names one, otherwise the refined blog's. None lets
    the writer pick."""
    return (current_step(state) or {}).get("template") or _carried_over(state, "template")


def _length_target(state: ContentBlitzState) -> str:
    """The length the drafting call aims for: the request's, or the configured word range."""
    words = (current_step(state) or {}).get("length")
    low, high = word_range(get_model_config().quality.blog, words)
    return f"{words} words ({low}-{high})" if words else f"{low}-{high} words"


def _prompt(state: ContentBlitzState) -> list[dict]:
    topic = _topic(state)
    brief = topic["brief"]
    step = current_step(state) or {}
    templates = "\n".join(f"- {name}: {guide}" for name, guide in TEMPLATES.items())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Topic: {topic['name']}\n"
            f"Target keyword: {_keyword(state)}\n"
            f"Secondary keywords: {', '.join(brief['secondary_keywords']) or 'none'}\n"
            f"Target length: {_length_target(state)}\n"
            f"Template: {_template(state) or 'none given, pick the template that fits best'}\n"
            f"Instructions: {step.get('instructions') or 'none'}\n\n"
            f"Templates:\n{templates}\n\n"
            f"Research Brief:\n\n{_brief_text(brief)}",
        },
    ]


def _write_draft(state: ContentBlitzState, messages: list[dict]) -> dict | None:
    """The model's blog draft, or None if every provider failed or the reply never fit."""
    try:
        reply = llm.complete_structured_reply(AGENT, messages, BlogDraft)
    except Exception:  # noqa: BLE001 - a failed blog posts an error; the plan carries on
        return None
    draft = reply.value.model_dump()
    # The template the request named, or the refined blog's, is the one recorded, whatever the
    # model says it used.
    return {**draft, "template": _template(state) or draft["template"]}


def _draft(state: ContentBlitzState) -> dict | None:
    return _write_draft(state, _prompt(state))


def _draft_text(draft: dict) -> str:
    """A draft as the writer reads it back."""
    return (
        f"Template: {draft['template']}\nTitle: {draft['title']}\n"
        f"Meta description: {draft['meta_description']}\n\n{draft['body']}"
    )


def _change(state: ContentBlitzState, draft: dict, request: str) -> dict | None:
    """`draft` changed as `request` asks: the same call, given the draft to change."""
    return _write_draft(
        state,
        [
            *_prompt(state),
            {"role": "assistant", "content": _draft_text(draft)},
            {"role": "user", "content": request},
        ],
    )


def _revise(state: ContentBlitzState, draft: dict, feedback: str) -> dict | None:
    """The current draft revised to fix what the feedback names, keeping the rest."""
    return _change(state, draft, REVISE_PROMPT.format(feedback=feedback))


def _latest_revision(state: ContentBlitzState) -> dict:
    """The refined blog's latest Revision, which may be a hand edit, as a draft."""
    piece = state["content_pieces"][_target(state)]
    latest = piece["revisions"][-1]
    return {
        "template": _template(state) or "none recorded",
        "title": piece["title"],
        "meta_description": latest.get("meta_description", ""),
        "body": latest["text"],
    }


def _refine(state: ContentBlitzState) -> dict | None:
    """The first Attempt of a refinement: the latest Revision changed as the marketer asks."""
    instructions = (current_step(state) or {}).get("instructions") or ""
    return _change(state, _latest_revision(state), REFINE_PROMPT.format(instructions=instructions))


def _checks(state: ContentBlitzState, draft: dict) -> list[CheckResult]:
    brief = _topic(state)["brief"]
    checked = Draft(
        body=draft["body"],
        meta_description=draft["meta_description"],
        keyword=_keyword(state),
        source_numbers=[s["number"] for s in brief["sources"]],
    )
    words = (current_step(state) or {}).get("length")
    return check_blog(checked, get_model_config().quality.blog, words)


def _judge(state: ContentBlitzState, draft: dict) -> Review | None:
    text = f"Meta description: {draft['meta_description']}\n\nBlog post:\n\n{draft['body']}"
    return review(text, JUDGING_CRITERIA)


def _review_text(found: Review | None) -> str:
    if found is None:
        return "*Engagement: reviewer unavailable*"
    suggestions = "\n".join(f"- {s}" for s in found["suggestions"][:SUGGESTIONS_SHOWN])
    rating = f"**Engagement:** {found['engagement']}/10"
    return f"{rating}\n\n**Suggestions:**\n{suggestions}" if suggestions else rating


def _quality_text(quality: QualityScore) -> str:
    checks = "\n".join(
        f"- {c['name']} {c['measured']} {'✓' if c['passed'] else '✗'}" for c in quality["checks"]
    )
    max_revisions = get_model_config().quality.max_revisions
    return (
        f"**Quality Score:** {quality['score']}/100\n\n{checks}\n\n"
        f"**Automatic revisions:** {quality['revisions_used']} of {max_revisions}\n\n"
        f"{_review_text(quality['review'])}"
    )


def _chat_text(draft: dict, quality: QualityScore) -> str:
    return (
        f"{draft['body']}\n\n---\n\n"
        f"**Meta description:** {draft['meta_description']}\n\n"
        f"**Template:** {draft['template']}\n\n"
        f"{_quality_text(quality)}"
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _revision(state: ContentBlitzState, attempt: Attempt, origin: RevisionOrigin) -> Revision:
    draft = attempt["draft"]
    return {
        "text": draft["body"],
        "meta_description": draft["meta_description"],
        "target_keyword": _keyword(state),
        "template": draft["template"],
        "origin": origin,
        "created_at": _now(),
        "quality_score": attempt["quality_score"],
    }


def _posted(attempt: Attempt) -> ContentBlitzState:
    return {"messages": [progress_message(_chat_text(attempt["draft"], attempt["quality_score"]))]}


def _finalise(state: ContentBlitzState, attempt: Attempt) -> ContentBlitzState:
    """Saves the kept Attempt as a new blog Content Piece with its first Revision, and posts it."""
    piece: ContentPiece = {
        "kind": "blog",
        "topic_id": state["active_topic_id"],
        "title": attempt["draft"]["title"],
        "revisions": [_revision(state, attempt, "draft")],
    }
    return {**add_piece(state, new_piece_id(state), piece), **_posted(attempt)}


def _finalise_refinement(state: ContentBlitzState, attempt: Attempt) -> ContentBlitzState:
    """Adds the kept Attempt to the refined blog as a new Revision, and posts it.

    The blog keeps its title, so the marketer can still refer to it by the name they know.
    """
    revision = _revision(state, attempt, "refinement")
    return {**add_revision(state, _target(state), revision), **_posted(attempt)}


def _failure(state: ContentBlitzState) -> str:
    return NO_MODEL.format(topic=active_topic(state)["name"])


def _refine_failure(state: ContentBlitzState) -> str:
    return NO_REFINEMENT.format(title=state["content_pieces"][_target(state)]["title"])


BLOG = QualityFormat(
    name="blog",
    draft=_draft,
    revise=_revise,
    checks=_checks,
    judge=_judge,
    finalise=_finalise,
    failure=_failure,
)
# A refinement runs the same loop, starting from the refined blog's latest Revision.
BLOG_REFINEMENT = replace(
    BLOG,
    name="blog refinement",
    draft=_refine,
    finalise=_finalise_refinement,
    failure=_refine_failure,
)
_write = quality_loop(BLOG)
_refine_blog = quality_loop(BLOG_REFINEMENT)


def blog_writer(state: ContentBlitzState) -> ContentBlitzState:
    """Writes a new blog, or refines the blog a refine step targets."""
    if not _target(state):
        return _write(state)  # the plan builder made sure the Topic has a Brief (ADR 0001)
    topic = _topic(state)
    if not topic.get("brief"):
        title = state["content_pieces"][_target(state)]["title"]
        return {"messages": [progress_message(NO_BRIEF.format(title=title, topic=topic["name"]))]}
    return _refine_blog(state)
