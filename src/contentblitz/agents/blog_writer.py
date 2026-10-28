"""SEO Blog Writer: writes a blog from the active Topic's Research Brief. See spec #19.

The blog step runs the blog's quality loop. Its draft is one structured call to the
`blog_writer` model, given the whole Brief, its target and secondary keywords, the four
template guides, the target word range and the step's instructions. It returns the template
it used, the title, the meta description and the Markdown body, citing the Brief's Sources
as `[n]`. The loop checks the draft against the blog's SEO rules, and the kept attempt is
posted and saved with its Quality Score.

If every provider fails, the step posts an error and the plan carries on: other pieces don't
depend on the blog, unlike research and the Brief.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from contentblitz.core.config import get_model_config
from contentblitz.integrations import llm
from contentblitz.workflow.content_checks import CheckResult, Draft, check_blog
from contentblitz.workflow.plan import current_step
from contentblitz.workflow.progress import progress_message, stub_progress
from contentblitz.workflow.quality_loop import Attempt, QualityFormat, quality_loop
from contentblitz.workflow.quality_score import QualityScore
from contentblitz.workflow.report import citation
from contentblitz.workflow.state import ContentBlitzState, ContentPiece, ResearchBrief, Revision
from contentblitz.workflow.topics import active_topic, add_piece, new_piece_id

AGENT = "blog_writer"

Template = Literal["how-to", "listicle", "comparison", "thought leadership"]

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
secondary keywords in naturally. Pick the template that fits best and follow its structure.
Follow the marketer's instructions."""

NO_MODEL = (
    "I couldn't write the blog on *{topic}*: the model didn't give a usable reply. "
    "The rest of the plan carries on, so ask again to retry the blog."
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


def _prompt(state: ContentBlitzState) -> list[dict]:
    topic = active_topic(state)
    brief = topic["brief"]
    step = current_step(state) or {}
    low, high = get_model_config().quality.blog.word_range
    templates = "\n".join(f"- {name}: {guide}" for name, guide in TEMPLATES.items())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Topic: {topic['name']}\n"
            f"Target keyword: {brief['target_keyword']}\n"
            f"Secondary keywords: {', '.join(brief['secondary_keywords']) or 'none'}\n"
            f"Target length: {low}-{high} words\n"
            f"Instructions: {step.get('instructions') or 'none'}\n\n"
            f"Templates:\n{templates}\n\n"
            f"Research Brief:\n\n{_brief_text(brief)}",
        },
    ]


def _draft(state: ContentBlitzState) -> dict | None:
    """The model's blog draft, or None if every provider failed or the reply never fit."""
    try:
        reply = llm.complete_structured_reply(AGENT, _prompt(state), BlogDraft)
    except Exception:  # noqa: BLE001 - a failed blog posts an error; the plan carries on
        return None
    return reply.value.model_dump()


def _checks(state: ContentBlitzState, draft: dict) -> list[CheckResult]:
    brief = active_topic(state)["brief"]
    checked = Draft(
        body=draft["body"],
        meta_description=draft["meta_description"],
        keyword=brief["target_keyword"],
        source_numbers=[s["number"] for s in brief["sources"]],
    )
    return check_blog(checked, get_model_config().quality.blog)


def _quality_text(quality: QualityScore) -> str:
    checks = "\n".join(
        f"- {c['name']} {c['measured']} {'✓' if c['passed'] else '✗'}" for c in quality["checks"]
    )
    return f"**Quality Score:** {quality['score']}/100\n\n{checks}"


def _chat_text(draft: dict, quality: QualityScore) -> str:
    return (
        f"{draft['body']}\n\n---\n\n"
        f"**Meta description:** {draft['meta_description']}\n\n"
        f"**Template:** {draft['template']}\n\n"
        f"{_quality_text(quality)}"
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _finalise(state: ContentBlitzState, attempt: Attempt) -> ContentBlitzState:
    """Saves the kept attempt as a new blog Content Piece with its first Revision, and posts it."""
    draft, quality = attempt["draft"], attempt["quality_score"]
    revision: Revision = {
        "text": draft["body"],
        "meta_description": draft["meta_description"],
        "target_keyword": active_topic(state)["brief"]["target_keyword"],
        "template": draft["template"],
        "origin": "draft",
        "created_at": _now(),
        "quality_score": quality,
    }
    piece: ContentPiece = {
        "kind": "blog",
        "topic_id": state["active_topic_id"],
        "title": draft["title"],
        "revisions": [revision],
    }
    saved = add_piece(state, new_piece_id(state), piece)
    return {**saved, "messages": [progress_message(_chat_text(draft, quality))]}


def _failure(state: ContentBlitzState) -> str:
    return NO_MODEL.format(topic=active_topic(state)["name"])


BLOG = QualityFormat(
    name="blog", draft=_draft, checks=_checks, finalise=_finalise, failure=_failure
)
_write = quality_loop(BLOG)


def blog_writer(state: ContentBlitzState) -> ContentBlitzState:
    if (current_step(state) or {}).get("target"):
        return stub_progress("SEO Blog Writer", state)  # refining a blog arrives in #34
    return _write(state)
