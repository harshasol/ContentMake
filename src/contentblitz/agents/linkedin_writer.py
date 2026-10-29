"""LinkedIn Writer: writes a LinkedIn post from the active Topic's Research Brief. See spec #20.

The LinkedIn step runs the LinkedIn quality loop from #19. Its draft is one structured call to
the `linkedin_writer` model, given the whole Brief, its keywords, the four template guides,
the character and hashtag ranges, and the step's instructions. It returns the template it used,
the post's plain text and its hashtags; the post is the text, a blank line, then the hashtags on
one line. The loop checks the post against LinkedIn's feed conventions, and the AI reviewer
scores its hook, readability on a phone, tone and call to action. A draft that doesn't pass is
revised: the same call, given the current post and what to fix. The kept Attempt is posted,
ready to copy, and saved with its Quality Score.

If every provider fails, the step posts an error and the plan carries on.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from contentblitz.agents.quality_judge import review
from contentblitz.core.config import get_model_config
from contentblitz.integrations import llm
from contentblitz.workflow.content_checks import (
    CheckResult,
    Post,
    char_range,
    check_linkedin,
    hashtag_range,
)
from contentblitz.workflow.plan import current_step
from contentblitz.workflow.progress import progress_message, stub_progress
from contentblitz.workflow.quality_loop import Attempt, QualityFormat, quality_loop, quality_text
from contentblitz.workflow.quality_score import QualityScore, Review
from contentblitz.workflow.routing import LinkedInTemplate
from contentblitz.workflow.state import (
    ContentBlitzState,
    ContentPiece,
    PlanStep,
    ResearchBrief,
    Revision,
    RevisionOrigin,
    Topic,
)
from contentblitz.workflow.topics import active_topic, add_piece, new_piece_id

AGENT = "linkedin_writer"

TEMPLATES: dict[str, str] = {
    "industry insights": "a striking statistic or claim, why it matters, a takeaway",
    "personal story": "a narrative arc, told from a story the marketer gives in the "
    'instructions; without one, written as an observation ("Something I keep seeing...")',
    "tips": "a hook, then 3-5 numbered tips as short paragraphs",
    "announcement": "the news, the benefit, a call to action",
}

SYSTEM_PROMPT = """\
You are a LinkedIn writer for a content marketer. Write one LinkedIn post from the Research
Brief below, using its findings and statistics and nothing from your own knowledge: every figure
in the post must come from the Brief. Prefer verified statistics; use an unverified one only
when you must. Write plain text with short paragraphs separated by blank lines, and give the
hashtags separately; draw the hashtags from the keywords where possible.
Follow the structure of the template you are given, or pick the template that fits best when you
aren't given one. Follow the marketer's instructions."""

REVISE_PROMPT = """\
Revise your LinkedIn post above to fix what follows. Change only what is needed to fix it, and
keep the rest of the post as it is.

{feedback}"""

JUDGING_CRITERIA = (
    "the strength of its hook, its readability on a phone, a professional tone with some "
    "personality, and how compelling its call to action is"
)
TITLE_CHARS = 80

NO_MODEL = (
    "I couldn't write the LinkedIn post on *{topic}*: the model didn't give a usable reply. "
    "The rest of the plan carries on, so ask again to retry the post."
)


class LinkedInDraft(BaseModel):
    template: LinkedInTemplate = Field(description="The template whose structure the post follows.")
    text: str = Field(
        description="The post in plain text, without its hashtags: short paragraphs separated "
        "by blank lines, no Markdown."
    )
    hashtags: list[str] = Field(description="The post's hashtags, each a # and one word.")


def _brief_text(brief: ResearchBrief) -> str:
    findings = "\n".join(f"- {f['statement']}" for f in brief["key_findings"])
    statistics = "\n".join(
        f"- {s['figure']}: {s['description']} "
        f"({'verified' if s['verified'] else 'unverified'})"
        for s in brief["statistics"]
    )
    angles = "\n".join(f"- {a['title']}: {a['hook']}" for a in brief["angles"])
    keywords = ", ".join([brief["target_keyword"], *brief["secondary_keywords"]])
    return (
        f"Summary: {brief['summary']}\n\n"
        f"Key findings:\n{findings or 'none'}\n\n"
        f"Statistics:\n{statistics or 'none'}\n\n"
        f"Content angles:\n{angles or 'none'}\n\n"
        f"Keywords: {keywords}"
    )


def _brief_facts(brief: ResearchBrief) -> str:
    """The Brief's findings and statistics, which every figure in the post must come from."""
    findings = [f["statement"] for f in brief["key_findings"]]
    statistics = [f"{s['figure']}: {s['description']}" for s in brief["statistics"]]
    return "\n".join([*findings, *statistics])


def _step(state: ContentBlitzState) -> PlanStep:
    """The LinkedIn step being run, or an empty step outside a plan."""
    return current_step(state) or {}


def _topic(state: ContentBlitzState) -> Topic:
    return active_topic(state)


def _range_text(low: int, high: int) -> str:
    return f"{low}" if low == high else f"{low}-{high}"


def _prompt(state: ContentBlitzState) -> list[dict]:
    topic = _topic(state)
    rules = get_model_config().quality.linkedin
    templates = "\n".join(f"- {name}: {guide}" for name, guide in TEMPLATES.items())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Topic: {topic['name']}\n"
            f"Target length: {_range_text(*char_range(rules))} characters, hashtags included\n"
            f"Hashtags: {_range_text(*hashtag_range(rules))} hashtags\n"
            "Template: none given, pick the template that fits best\n"
            f"Instructions: {_step(state).get('instructions') or 'none'}\n\n"
            f"Templates:\n{templates}\n\n"
            f"Research Brief:\n\n{_brief_text(topic['brief'])}",
        },
    ]


def _post(text: str, hashtags: list[str]) -> str:
    """The post: its text, a blank line, then its hashtags on one line, each with a `#`."""
    tags = " ".join(t if t.startswith("#") else f"#{t}" for t in (h.strip() for h in hashtags))
    return f"{text.strip()}\n\n{tags}" if tags else text.strip()


def _write_draft(messages: list[dict]) -> dict | None:
    """The model's post, or None if every provider failed or the reply never fit."""
    try:
        reply = llm.complete_structured_reply(AGENT, messages, LinkedInDraft)
    except Exception:  # noqa: BLE001 - a failed post posts an error; the plan carries on
        return None
    draft = reply.value
    return {"template": draft.template, "post": _post(draft.text, draft.hashtags)}


def _first_draft(state: ContentBlitzState) -> dict | None:
    return _write_draft(_prompt(state))


def _draft_text(draft: dict) -> str:
    """A draft as the writer reads it back."""
    return f"Template: {draft['template']}\n\n{draft['post']}"


def _revise(state: ContentBlitzState, draft: dict, feedback: str) -> dict | None:
    """The current draft revised to fix what the feedback names, keeping the rest."""
    return _write_draft(
        [
            *_prompt(state),
            {"role": "assistant", "content": _draft_text(draft)},
            {"role": "user", "content": REVISE_PROMPT.format(feedback=feedback)},
        ]
    )


def _checks(state: ContentBlitzState, draft: dict) -> list[CheckResult]:
    post = Post(text=draft["post"], brief_facts=_brief_facts(_topic(state)["brief"]))
    return check_linkedin(post, get_model_config().quality.linkedin)


def _judge(state: ContentBlitzState, draft: dict) -> Review | None:
    return review(f"LinkedIn post:\n\n{draft['post']}", JUDGING_CRITERIA)


def _fenced(post: str) -> str:
    """The post in a code block, which keeps its line breaks and can be copied as it is."""
    fence = "```"
    while fence in post:
        fence += "`"
    return f"{fence}text\n{post}\n{fence}"


def _chat_text(draft: dict, quality: QualityScore) -> str:
    return (
        f"{_fenced(draft['post'])}\n\n"
        f"**Characters:** {len(draft['post']):,}\n\n"
        f"**Template:** {draft['template']}\n\n"
        f"{quality_text(quality)}"
    )


def _title(post: str) -> str:
    """The post's opening line, which the marketer knows it by, shortened if it is long."""
    hook = post.strip().splitlines()[0] if post.strip() else "LinkedIn post"
    if len(hook) <= TITLE_CHARS:
        return hook
    return hook[: TITLE_CHARS - 1].rsplit(" ", 1)[0] + "…"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _revision(attempt: Attempt, origin: RevisionOrigin) -> Revision:
    draft = attempt["draft"]
    return {
        "text": draft["post"],
        "template": draft["template"],
        "origin": origin,
        "created_at": _now(),
        "quality_score": attempt["quality_score"],
    }


def _posted(attempt: Attempt) -> ContentBlitzState:
    return {"messages": [progress_message(_chat_text(attempt["draft"], attempt["quality_score"]))]}


def _finalise(state: ContentBlitzState, attempt: Attempt) -> ContentBlitzState:
    """Saves the kept Attempt as a new LinkedIn Content Piece with its first Revision, and
    posts it."""
    piece: ContentPiece = {
        "kind": "linkedin_post",
        "topic_id": state["active_topic_id"],
        "title": _title(attempt["draft"]["post"]),
        "revisions": [_revision(attempt, "draft")],
    }
    return {**add_piece(state, new_piece_id(state), piece), **_posted(attempt)}


def _failure(state: ContentBlitzState) -> str:
    return NO_MODEL.format(topic=active_topic(state)["name"])


LINKEDIN = QualityFormat(
    name="linkedin",
    draft=_first_draft,
    revise=_revise,
    checks=_checks,
    judge=_judge,
    finalise=_finalise,
    failure=_failure,
)
_write_new_post = quality_loop(LINKEDIN)


def linkedin_writer(state: ContentBlitzState) -> ContentBlitzState:
    """Writes a new LinkedIn post. The plan builder made sure the Topic has a Brief (ADR 0001)."""
    if _step(state).get("target"):
        return stub_progress("LinkedIn Post Writer", state)  # refining comes in LinkedIn 5
    return _write_new_post(state)
