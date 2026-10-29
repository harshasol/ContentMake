"""LinkedIn Writer: writes a LinkedIn post from the active Topic's Research Brief. See spec #20.

The LinkedIn step runs the LinkedIn quality loop from #19. Its draft is one structured call to
the `linkedin_writer` model, given the whole Brief, its keywords, the four template guides,
the character and hashtag ranges, and the step's instructions. A post built on another Content
Piece ("turn that blog into a LinkedIn post") is also given that piece's latest Revision as
background, but still takes its facts from the Brief. It returns the template it used,
the post's plain text and its hashtags; the post is the text, a blank line, then the hashtags on
one line. The loop checks the post against LinkedIn's feed conventions, and the AI reviewer
scores its hook, readability on a phone, tone and call to action. A draft that doesn't pass is
revised: the same call, given the current post and what to fix. The kept Attempt is posted,
ready to copy, and saved with its Quality Score.

A length the request sets (in characters) is the most the post may have, and a hashtag count in
the instructions ("only 5 hashtags") is the number it must have; either replaces the configured
range for that run, in the drafting call and the checks.

A refine step that targets a post runs the same loop, but its first Attempt changes the post's
latest Revision (which may be a hand edit) as the marketer asks. Its template carries over
unless the request sets a new one, and the kept Attempt is added as a new Revision.

If every provider fails, the step posts an error and the plan carries on.
"""

import re
from dataclasses import replace
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from contentblitz.core.config import get_model_config
from contentblitz.integrations import llm
from contentblitz.workflow.brand_voice import brand_voice
from contentblitz.workflow.content_checks import (
    CheckResult,
    Post,
    char_range,
    check_linkedin,
    hashtag_range,
    range_text,
)
from contentblitz.workflow.formats import PIECE_FORMATS, LinkedInTemplate
from contentblitz.workflow.hand_edit import brief_facts
from contentblitz.workflow.plan import current_step
from contentblitz.workflow.progress import progress_message
from contentblitz.workflow.quality_loop import Attempt, QualityFormat, quality_loop, quality_text
from contentblitz.workflow.quality_score import QualityScore
from contentblitz.workflow.routing import Template
from contentblitz.workflow.state import (
    ContentBlitzState,
    ContentPiece,
    PlanStep,
    ResearchBrief,
    Revision,
    RevisionOrigin,
    Topic,
)
from contentblitz.workflow.topics import (
    active_topic,
    add_piece,
    add_revision,
    new_piece_id,
    piece_title,
    piece_topic,
)

AGENT = "linkedin_writer"

TEMPLATES: dict[str, str] = {
    "industry insights": "a striking statistic or claim, why it matters, a takeaway",
    "personal story": "a narrative arc, told from a story the marketer gives in the "
    'instructions; without one, written as an observation ("Something I keep seeing...") that '
    "never invents first-person events",
    "tips": "a hook, then 3-5 numbered tips as short paragraphs",
    "announcement": "the news, the benefit, a call to action",
}

SYSTEM_PROMPT = """\
You are a LinkedIn writer for a content marketer. Write one LinkedIn post from the Research
Brief below, using its findings and statistics and nothing from your own knowledge: every figure
in the post must come from the Brief. Prefer verified statistics; use an unverified one only
when you must. Open with one short line that grabs attention before LinkedIn's "...see more"
cut-off: a bold claim, a question or a surprising statistic. Then write short paragraphs
separated by blank lines, easy to scan on a phone, and close with a call to action: a question
or an invitation to comment or share. Write plain text: no Markdown, and no [n] citations.
Give the hashtags separately, and draw the hashtags from the keywords where possible.
Follow the structure of the template you are given, or pick the template that fits best when you
aren't given one. Never invent first-person events from the marketer's life: tell a personal
story only from one they give you in the instructions, and otherwise write it as an
observation. Follow the marketer's instructions."""

REVISE_PROMPT = """\
Revise your LinkedIn post above to fix what follows. Change only what is needed to fix it, and
keep the rest of the post as it is.

{feedback}"""

REFINE_PROMPT = """\
Revise your LinkedIn post above as the marketer asks below. Change only what is needed for that,
and keep the rest of the post as it is.

{instructions}"""

JUDGING_CRITERIA = (
    "the strength of its hook, its readability on a phone, a professional tone with some "
    "personality, and how compelling its call to action is"
)
TITLE_CHARS = 80

NO_MODEL = (
    "I couldn't write the LinkedIn post on *{topic}*: the model didn't give a usable reply. "
    "The rest of the plan carries on, so ask again to retry the post."
)
NO_REFINEMENT = (
    "I couldn't refine *{title}*: the model didn't give a usable reply. "
    "Its latest Revision is unchanged, so ask again to retry."
)
NO_BRIEF = (
    "I can't refine *{title}* yet: *{topic}* has no Research Brief to check it "
    "against. Ask me to research it again first."
)

SOURCE_PROMPT = """\
Background: the marketer's {label} *{title}*, which this post should build on.
Take every fact and figure from the Research Brief, not from it.

{text}"""

_NUMBER_WORDS = "no one two three four five six seven eight nine ten eleven twelve".split()
_COUNT = rf"(\d+|{'|'.join(_NUMBER_WORDS)})"
# "between 6 and 10 hashtags", "6-10 hashtags", "6 to 10 hashtags"
_HASHTAG_RANGE = re.compile(rf"\b{_COUNT}\s*(?:-|–|to|and)\s*{_COUNT}\s+hashtags?\b", re.I)
# "only 5 hashtags", "three hashtags", "no hashtags"
_HASHTAG_COUNT = re.compile(rf"\b{_COUNT}\s+hashtags?\b", re.I)


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
        f"- {s['figure']}: {s['description']} ({'verified' if s['verified'] else 'unverified'})"
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


def _step(state: ContentBlitzState) -> PlanStep:
    """The LinkedIn step being run, or an empty step outside a plan."""
    return current_step(state) or {}


def _target(state: ContentBlitzState) -> str | None:
    """The post a refine step targets, or None when the step writes a new post."""
    return _step(state).get("target")


def _topic(state: ContentBlitzState) -> Topic:
    """The Topic the post is on: the refined post's, or the active Topic for a new post."""
    piece_id = _target(state)
    return piece_topic(state, piece_id) if piece_id else active_topic(state)


def _carried_over_template(state: ContentBlitzState) -> Template | None:
    """For a refinement, the template of the refined post's latest Revision that records one."""
    piece_id = _target(state)
    revisions = state["content_pieces"][piece_id]["revisions"] if piece_id else []
    return next((r["template"] for r in reversed(revisions) if r.get("template")), None)


def _template(state: ContentBlitzState) -> Template | None:
    """The template: the request's when it names one, otherwise the refined post's. None lets
    the writer pick."""
    return _step(state).get("template") or _carried_over_template(state)


def _count(word: str) -> int:
    word = word.lower()
    return int(word) if word.isdigit() else _NUMBER_WORDS.index(word)  # "no" is 0


def _hashtag_range(state: ContentBlitzState) -> tuple[int, int] | None:
    """The hashtag range the instructions ask for ("between 6 and 10 hashtags", "only 5
    hashtags", "no hashtags"), if any."""
    instructions = _step(state).get("instructions") or ""
    if found := _HASHTAG_RANGE.search(instructions):
        return _count(found[1]), _count(found[2])
    if found := _HASHTAG_COUNT.search(instructions):
        return _count(found[1]), _count(found[1])
    return None


def _ranges(state: ContentBlitzState) -> tuple[tuple[int, int], tuple[int, int]]:
    """The character and hashtag ranges for this run: the request's, or the configured ones."""
    rules = get_model_config().quality.linkedin
    chars = char_range(rules, _step(state).get("length"))
    return chars, hashtag_range(rules, _hashtag_range(state))


def _source_text(state: ContentBlitzState) -> str:
    """The piece the post builds on, as background, or nothing when it builds on none."""
    piece_id = _step(state).get("source_piece")
    if not piece_id:
        return ""
    piece = state["content_pieces"][piece_id]
    label = PIECE_FORMATS[piece["kind"]].label
    background = SOURCE_PROMPT.format(
        label=label, title=piece["title"], text=piece["revisions"][-1]["text"]
    )
    return f"\n\n{background}"


def _prompt(state: ContentBlitzState) -> list[dict]:
    topic = _topic(state)
    rules = get_model_config().quality.linkedin
    chars, tags = _ranges(state)
    templates = "\n".join(f"- {name}: {guide}" for name, guide in TEMPLATES.items())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Topic: {topic['name']}\n"
            f"Target length: {range_text(*chars)} characters, hashtags included\n"
            f"Hashtags: {range_text(*tags)} hashtags\n"
            f"Opening line: at most {rules.max_hook_chars} characters\n"
            f"Paragraphs: at least {rules.min_paragraphs}, each at most "
            f"{rules.max_paragraph_chars} characters\n"
            f"Template: {_template(state) or 'none given, pick the template that fits best'}\n"
            f"Instructions: {_step(state).get('instructions') or 'none'}\n\n"
            f"Templates:\n{templates}\n\n"
            f"{brand_voice(state)}\n\n"
            f"Research Brief:\n\n{_brief_text(topic['brief'])}{_source_text(state)}",
        },
    ]


def _post(text: str, hashtags: list[str]) -> str:
    """The post: its text, a blank line, then its hashtags on one line, each with a `#`."""
    tags = " ".join(t if t.startswith("#") else f"#{t}" for t in (h.strip() for h in hashtags))
    return f"{text.strip()}\n\n{tags}" if tags else text.strip()


def _write_draft(state: ContentBlitzState, messages: list[dict]) -> dict | None:
    """The model's post, or None if every provider failed or the reply never fit."""
    try:
        reply = llm.complete_structured_reply(AGENT, messages, LinkedInDraft)
    except Exception:  # noqa: BLE001 - a failed post posts an error; the plan carries on
        return None
    draft = reply.value
    # The template the request named is the one recorded, whatever the model says it used.
    template = _template(state) or draft.template
    return {"template": template, "post": _post(draft.text, draft.hashtags)}


def _first_draft(state: ContentBlitzState) -> dict | None:
    return _write_draft(state, _prompt(state))


def _draft_text(draft: dict) -> str:
    """A draft as the writer reads it back."""
    return f"Template: {draft['template']}\n\n{draft['post']}"


def _redraft(state: ContentBlitzState, draft: dict, request: str) -> dict | None:
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
    return _redraft(state, draft, REVISE_PROMPT.format(feedback=feedback))


def _refined_latest_revision(state: ContentBlitzState) -> dict | None:
    """The first Attempt of a refinement: the latest Revision, which may be a hand edit,
    changed as the marketer asks."""
    latest = state["content_pieces"][_target(state)]["revisions"][-1]
    draft = {"template": _template(state) or "none recorded", "post": latest["text"]}
    instructions = _step(state).get("instructions") or ""
    return _redraft(state, draft, REFINE_PROMPT.format(instructions=instructions))


def _checks(state: ContentBlitzState, draft: dict, avoid: list[str]) -> list[CheckResult]:
    post = Post(text=draft["post"], brief_facts=brief_facts(_topic(state)["brief"]))
    rules = get_model_config().quality.linkedin
    return check_linkedin(post, rules, _step(state).get("length"), _hashtag_range(state), avoid)


def _reviewed_text(draft: dict) -> str:
    return f"LinkedIn post:\n\n{draft['post']}"


def _fenced(post: str) -> str:
    """The post in a code block, which keeps its line breaks and can be copied as it is."""
    fence = "```"
    while fence in post:
        fence += "`"
    return f"{fence}text\n{post}\n{fence}"


def _chat_text(state: ContentBlitzState, draft: dict, quality: QualityScore) -> str:
    picked = "" if _template(state) else " (picked by the writer)"
    return (
        f"{_fenced(draft['post'])}\n\n"
        f"**Characters:** {len(draft['post']):,}\n\n"
        f"**Template:** {draft['template']}{picked}\n\n"
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


def _posted(state: ContentBlitzState, attempt: Attempt) -> ContentBlitzState:
    text = _chat_text(state, attempt["draft"], attempt["quality_score"])
    return {"messages": [progress_message(text)]}


def _finalise(state: ContentBlitzState, attempt: Attempt) -> ContentBlitzState:
    """Saves the kept Attempt as a new LinkedIn Content Piece with its first Revision, and
    posts it."""
    piece: ContentPiece = {
        "kind": "linkedin_post",
        "topic_id": state["active_topic_id"],
        "title": _title(attempt["draft"]["post"]),
        "revisions": [_revision(attempt, "draft")],
    }
    return {**add_piece(state, new_piece_id(state), piece), **_posted(state, attempt)}


def _finalise_refinement(state: ContentBlitzState, attempt: Attempt) -> ContentBlitzState:
    """Adds the kept Attempt to the refined post as a new Revision, and posts it. The post keeps
    its title, so the marketer can still refer to it by the name they know."""
    revision = _revision(attempt, "refinement")
    return {**add_revision(state, _target(state), revision), **_posted(state, attempt)}


def _failure(state: ContentBlitzState) -> str:
    return NO_MODEL.format(topic=active_topic(state)["name"])


def _refine_failure(state: ContentBlitzState) -> str:
    return NO_REFINEMENT.format(title=piece_title(state, _target(state)))


LINKEDIN = QualityFormat(
    name="linkedin",
    draft=_first_draft,
    revise=_revise,
    checks=_checks,
    reviewed_text=_reviewed_text,
    judging_criteria=JUDGING_CRITERIA,
    finalise=_finalise,
    failure=_failure,
)
# A refinement runs the same loop, starting from the refined post's latest Revision.
LINKEDIN_REFINEMENT = replace(
    LINKEDIN,
    name="linkedin refinement",
    draft=_refined_latest_revision,
    finalise=_finalise_refinement,
    failure=_refine_failure,
)
_write_new_post = quality_loop(LINKEDIN)
_refine_existing_post = quality_loop(LINKEDIN_REFINEMENT)


def linkedin_writer(state: ContentBlitzState) -> ContentBlitzState:
    """Writes a new LinkedIn post, or refines the post a refine step targets."""
    if not _target(state):
        # The plan builder made sure the Topic has a Brief (ADR 0001).
        return _write_new_post(state)
    topic = _topic(state)
    if not topic.get("brief"):
        title = piece_title(state, _target(state))
        return {"messages": [progress_message(NO_BRIEF.format(title=title, topic=topic["name"]))]}
    return _refine_existing_post(state)
