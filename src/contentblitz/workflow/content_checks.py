"""Content checks: deterministic measurements of a draft against its format's rules. See spec #19.

Each check returns a `CheckResult`: pass or fail, the measured value as the chat shows it,
and a short reason that names the target, which is also the revise loop's feedback.
A blog's measurements leave out the Sources section and Markdown syntax. A LinkedIn post
(spec #20) is plain text: its paragraphs are separated by blank lines, and it ends with its
hashtags on one line.

A format's rules come from its config section, and `check_<format>` runs its checks in the
order the chat lists them.
"""

import re
from dataclasses import dataclass
from typing import TypedDict

import textstat

from contentblitz.core.config import BlogQuality, LinkedInQuality
from contentblitz.workflow.statistics import figures, is_grounded


@dataclass(frozen=True)
class Draft:
    """What the checks measure: a draft's text and what it should match."""

    body: str  # Markdown
    meta_description: str
    keyword: str  # the target keyword phrase
    source_numbers: list[int]  # the Brief's Source numbers, which `[n]` citations must match


@dataclass(frozen=True)
class Post:
    """What the LinkedIn checks measure: the whole post, and the Brief it must stay true to."""

    text: str  # the body, a blank line, then the hashtags on one line
    brief_facts: str  # the Brief's findings and statistics, which every figure must come from


class CheckResult(TypedDict):
    name: str  # as the chat shows it, e.g. "keyword density"
    passed: bool
    measured: str  # e.g. "1.4%", "171 chars"
    reason: str  # what was measured against what target


_WORD = re.compile(r"\w+(?:['’]\w+)*")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SOURCES_HEADING = re.compile(r"sources", re.IGNORECASE)
_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")  # [n], [n, m], [n-m]; not links
_INLINE_CODE = re.compile(r"`[^`]*`")
_ENTRY = re.compile(r"^\s*(?:[-*+]\s+)?\[?(\d+)[.)\]:]")  # "1. ...", "[1] ...", "- [1] ..."


def _lines(body: str) -> list[tuple[str, bool]]:
    """Each line of the body, with whether it is inside a fenced code block."""
    lines, in_code = [], False
    for line in body.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
            lines.append((line, True))
        else:
            lines.append((line, in_code))
    return lines


def _split_sources(body: str) -> tuple[list[tuple[str, bool]], list[str]]:
    """`_lines` outside the Sources section, and the section's lines after its heading.

    The Sources section runs up to the next heading at its level.
    """
    kept: list[tuple[str, bool]] = []
    sources: list[str] = []
    sources_level = 0
    for line, in_code in _lines(body):
        heading = None if in_code else _HEADING.match(line)
        if heading:
            level = len(heading[1])
            if sources_level and level <= sources_level:
                sources_level = 0
            if not sources_level and _SOURCES_HEADING.fullmatch(heading[2].strip(" *_:")):
                sources_level = level
                continue
        if sources_level:
            sources.append(line)
        else:
            kept.append((line, in_code))
    return kept, sources


def _without_sources(body: str) -> list[tuple[str, bool]]:
    """`_lines`, leaving out the Sources section."""
    return _split_sources(body)[0]


def _headings(body: str) -> list[tuple[int, str]]:
    """The body's headings outside code and the Sources section, as (level, text)."""
    found = []
    for line, in_code in _without_sources(body):
        heading = None if in_code else _HEADING.match(line)
        if heading:
            found.append((len(heading[1]), heading[2]))
    return found


def _prose(body: str) -> str:
    """The body's text as a reader sees it: no Sources section, no Markdown syntax."""
    text = "\n".join(line for line, _ in _without_sources(body))
    text = re.sub(r"^\s*(?:```|~~~).*$", "", text, flags=re.MULTILINE)  # code fences
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links keep their text
    text = re.sub(r"\[\d+(?:\s*[,–-]\s*\d+)*\]", "", text)  # [n] citations
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags
    text = re.sub(r"^\s{0,3}(?:#{1,6}|>|[-*+]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"[*_`~]+", "", text)  # emphasis and code marks


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _occurrences(words: list[str], phrase: list[str]) -> int:
    """Non-overlapping occurrences of `phrase` in `words`, on word boundaries."""
    count, i = 0, 0
    while phrase and i + len(phrase) <= len(words):
        if words[i : i + len(phrase)] == phrase:
            count, i = count + 1, i + len(phrase)
        else:
            i += 1
    return count


def _has_keyword(text: str, keyword: str) -> bool:
    return _occurrences(_words(text), _words(keyword)) > 0


def _range(low: float, high: float) -> str:
    return f"{low:g}-{high:g}"


def keyword_density(draft: Draft, density_range: tuple[float, float]) -> CheckResult:
    """Occurrences of the keyword phrase x its words / all words, as a % within the range."""
    words = _words(_prose(draft.body))
    phrase = _words(draft.keyword)
    density = 100 * _occurrences(words, phrase) * len(phrase) / len(words) if words else 0.0
    low, high = density_range
    measured = f"{density:.1f}%"
    return {
        "name": "keyword density",
        "passed": low <= density <= high,
        "measured": measured,
        "reason": f'"{draft.keyword}" is {measured} of {len(words)} words; '
        f"the target is {_range(low, high)}%",
    }


def meta_description(draft: Draft, length_range: tuple[int, int]) -> CheckResult:
    """The meta description's length is within the range, and it contains the keyword."""
    meta = draft.meta_description.strip()
    low, high = length_range
    problems = []
    if not low <= len(meta) <= high:
        problems.append(f"it is {len(meta)} characters; the target is {_range(low, high)}")
    if not _has_keyword(meta, draft.keyword):
        problems.append(f'it doesn\'t contain "{draft.keyword}"')
    return {
        "name": "meta description",
        "passed": not problems,
        "measured": f"{len(meta)} chars",
        "reason": "; ".join(problems)
        or f'{len(meta)} characters, within {_range(low, high)}, with "{draft.keyword}"',
    }


def headings(draft: Draft, min_h2s: int = 3) -> CheckResult:
    """One H1 with the keyword; at least `min_h2s` H2s, one with the keyword; no skipped levels.

    The Sources heading doesn't count.
    """
    found = _headings(draft.body)
    h1s = [text for level, text in found if level == 1]
    h2s = [text for level, text in found if level == 2]
    keyword = draft.keyword
    problems = []
    if len(h1s) != 1:
        problems.append(f"there must be exactly one H1, not {len(h1s)}")
    elif not _has_keyword(h1s[0], keyword):
        problems.append(f'the H1 doesn\'t contain "{keyword}"')
    if len(h2s) < min_h2s:
        problems.append(f"there must be at least {min_h2s} H2s, not {len(h2s)}")
    if h2s and not any(_has_keyword(text, keyword) for text in h2s):
        problems.append(f'no H2 contains "{keyword}"')
    levels = [level for level, _ in found]
    first_h2 = levels.index(2) if 2 in levels else len(levels)
    if any(level >= 3 for level in levels[:first_h2]):
        problems.append("there is an H3 before the first H2")
    skips = [
        (before, after)
        for before, after in zip([1, *levels], levels, strict=False)
        if after > before + 1
    ]
    if skips:
        before, after = skips[0]
        problems.append(f"an H{after} skips a level after an H{before}")
    plural = "" if len(h1s) == 1 else "s"
    return {
        "name": "headings",
        "passed": not problems,
        "measured": f"{len(h1s)} H1{plural}, {len(h2s)} H2s",
        "reason": "; ".join(problems)
        or f'one H1 and {len(h2s)} H2s, in order, with "{keyword}" in the H1 and an H2',
    }


def readability(draft: Draft, max_grade: float) -> CheckResult:
    """The prose's Flesch-Kincaid grade level is at most `max_grade`.

    A line that doesn't end a sentence, such as a heading or a list item, reads as one. If the
    grade can't be computed (say textstat's cmudict is missing), the check fails and says so,
    rather than stopping the writer's step.
    """
    lines = [line.strip() for line in _prose(draft.body).splitlines() if line.strip()]
    text = " ".join(line if line[-1] in ".!?" else f"{line}." for line in lines)
    try:
        grade = textstat.flesch_kincaid_grade(text)
    except Exception as error:  # noqa: BLE001 - an unmeasured grade fails the check, not the step
        return {
            "name": "readability",
            "passed": False,
            "measured": "unavailable",
            "reason": f"the reading grade couldn't be measured ({type(error).__name__}); "
            f"the target is at most grade {max_grade:g}",
        }
    measured = f"grade {grade:.1f}"
    passed = grade <= max_grade
    reason = f"it reads at {measured}; the target is at most grade {max_grade:g}"
    return {
        "name": "readability",
        "passed": passed,
        "measured": measured,
        "reason": reason if passed else f"{reason}: use shorter sentences and simpler words",
    }


def length(draft: Draft, word_range: tuple[int, int]) -> CheckResult:
    """The prose's word count is within the range."""
    count = len(_words(_prose(draft.body)))
    low, high = word_range
    return {
        "name": "length",
        "passed": low <= count <= high,
        "measured": f"{count:,} words",
        "reason": f"{count} words; the target is {_range(low, high)}",
    }


def _cited(body: str) -> set[int]:
    """The Source numbers the body cites as `[n]`, `[n, m]` or `[n-m]`, outside the Sources
    section, code and link text."""
    numbers: set[int] = set()
    for line, in_code in _without_sources(body):
        if in_code:
            continue
        for group in _CITATION.findall(_INLINE_CODE.sub("", line)):
            for part in group.split(","):
                first, _, last = re.sub(r"\s", "", part).replace("–", "-").partition("-")
                numbers.update(range(int(first), int(last or first) + 1))
    return numbers


def _listed(body: str) -> set[int]:
    """The Source numbers the Sources section lists, as `1. ...` or `[1] ...` entries."""
    return {int(entry[1]) for line in _split_sources(body)[1] if (entry := _ENTRY.match(line))}


def _numbers(numbers: set[int]) -> str:
    return ", ".join(f"[{n}]" for n in sorted(numbers))


def citations(draft: Draft, min_citations: int) -> CheckResult:
    """Every `[n]` is a Source in the Brief, at least `min_citations` distinct Sources are
    cited, and the Sources section lists exactly the cited Sources."""
    cited, listed = _cited(draft.body), _listed(draft.body)
    invalid = cited - set(draft.source_numbers)
    valid = cited - invalid
    count = f"{len(valid)} Source{'' if len(valid) == 1 else 's'}"
    problems = []
    if invalid:
        problems.append(f"it cites {_numbers(invalid)}, not among the Brief's Sources")
    if len(valid) < min_citations:
        problems.append(f"it cites {count}; the target is at least {min_citations}")
    if listed - valid:
        problems.append(f"the Sources section lists uncited {_numbers(listed - valid)}")
    if valid - listed:
        problems.append(f"the Sources section doesn't list {_numbers(valid - listed)}")
    return {
        "name": "citations",
        "passed": not problems,
        "measured": f"{count} cited",
        "reason": "; ".join(problems)
        or f"{count} from the Brief cited, at least {min_citations}, all listed under Sources",
    }


LENGTH_TOLERANCE = 0.15  # either side of a length the request sets


def word_range(rules: BlogQuality, words: int | None = None) -> tuple[int, int]:
    """The word range a blog is written to: `words` ±15% when the request sets a length,
    otherwise the configured range."""
    if words is None:
        return rules.word_range
    return round(words * (1 - LENGTH_TOLERANCE)), round(words * (1 + LENGTH_TOLERANCE))


def check_blog(draft: Draft, rules: BlogQuality, words: int | None = None) -> list[CheckResult]:
    """Every blog check's result, in the order the chat lists them.

    `words` is the length the request set, if any; it replaces the configured word range.
    """
    return [
        keyword_density(draft, rules.density_range),
        meta_description(draft, rules.meta_length_range),
        headings(draft),
        readability(draft, rules.max_grade),
        length(draft, word_range(rules, words)),
        citations(draft, rules.min_citations),
    ]


# LinkedIn post checks (spec #20)

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_HASHTAG = re.compile(r"#[^\W_]+")  # letters and digits only
_BODY_HASHTAG = re.compile(r"(?<![\w&#])#[^\s#]+")


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_BREAK.split(text.strip()) if p.strip()]


def _split_hashtags(text: str) -> tuple[list[str], list[str]]:
    """The post's body paragraphs, and the tokens of its closing hashtag line: its last
    paragraph, when that starts with `#`."""
    paragraphs = _paragraphs(text)
    if paragraphs and paragraphs[-1].startswith("#"):
        return paragraphs[:-1], paragraphs[-1].split()
    return paragraphs, []


def post_length(post: Post, char_range: tuple[int, int]) -> CheckResult:
    """The whole post's character count, hashtags included, is within the range."""
    count = len(post.text.strip())
    low, high = char_range
    return {
        "name": "length",
        "passed": low <= count <= high,
        "measured": f"{count:,} chars",
        "reason": f"{count} characters, hashtags included; the target is {_range(low, high)}",
    }


def hashtags(post: Post, hashtag_range: tuple[int, int]) -> CheckResult:
    """The closing line's hashtags are within the range, unique in any case and well formed
    (`#` and letters or digits only), and there are none in the body."""
    body, tags = _split_hashtags(post.text)
    low, high = hashtag_range
    target = f"{low}" if low == high else _range(low, high)
    seen: set[str] = set()
    duplicates = []
    for tag in tags:
        if tag.lower() in seen:
            duplicates.append(tag)
        seen.add(tag.lower())
    malformed = [tag for tag in tags if not _HASHTAG.fullmatch(tag)]
    misplaced = [tag for paragraph in body for tag in _BODY_HASHTAG.findall(paragraph)]
    problems = []
    if not low <= len(tags) <= high:
        problems.append(f"there are {len(tags)} hashtags; the target is {target}")
    if duplicates:
        problems.append(f"{', '.join(duplicates)} duplicate earlier hashtags")
    if malformed:
        problems.append(
            f"{', '.join(malformed)} must be # followed by letters and digits only"
        )
    if misplaced:
        problems.append(
            f"{', '.join(misplaced)} in the body must move to the closing hashtag line"
        )
    return {
        "name": "hashtags",
        "passed": not problems,
        "measured": f"{len(tags)} hashtag{'' if len(tags) == 1 else 's'}",
        "reason": "; ".join(problems)
        or f"{len(tags)} unique hashtags, within {target}, all in the closing line",
    }


_URL = re.compile(r"(?:https?://|www\.)\S+")


def hook(post: Post, max_chars: int) -> CheckResult:
    """The first line, which shows before LinkedIn's "...see more", is non-empty, at most
    `max_chars` long, and isn't a hashtag or a link."""
    text = post.text.strip()
    first = text.splitlines()[0].strip() if text else ""
    problems = []
    if not first:
        problems.append("the first line is empty")
    if len(first) > max_chars:
        problems.append(f"the first line is {len(first)} characters; the most is {max_chars}")
    if first.startswith("#"):
        problems.append("the first line is a hashtag")
    if _URL.fullmatch(first):
        problems.append("the first line is a link")
    return {
        "name": "hook",
        "passed": not problems,
        "measured": f"{len(first)} chars",
        "reason": "; ".join(problems)
        or f"a {len(first)}-character opening line, at most {max_chars}, before the cut-off",
    }


def paragraphs(post: Post, min_paragraphs: int, max_chars: int) -> CheckResult:
    """At least `min_paragraphs` paragraphs separated by blank lines before the hashtags, none
    longer than `max_chars`."""
    body, _ = _split_hashtags(post.text)
    long = [len(p) for p in body if len(p) > max_chars]
    problems = []
    if len(body) < min_paragraphs:
        problems.append(
            f"there are {len(body)} paragraphs; the target is at least {min_paragraphs}, "
            "separated by blank lines"
        )
    if long:
        sizes = ", ".join(f"{n} characters" for n in long)
        problems.append(f"paragraphs of {sizes} are longer than the most, {max_chars}")
    count = f"{len(body)} paragraph{'' if len(body) == 1 else 's'}"
    return {
        "name": "paragraphs",
        "passed": not problems,
        "measured": count,
        "reason": "; ".join(problems)
        or f"{count}, at least {min_paragraphs}, none longer than {max_chars} characters",
    }


_CLOSING_QUESTION = re.compile(r"\?\W*$")


def call_to_action(post: Post, phrases: list[str]) -> CheckResult:
    """The last paragraph before the hashtags ends with a question or has a call-to-action
    phrase."""
    body, _ = _split_hashtags(post.text)
    last = body[-1] if body else ""
    found = next(
        (p for p in phrases if re.search(rf"\b{re.escape(p)}s?\b", last, re.IGNORECASE)), None
    )
    if _CLOSING_QUESTION.search(last):
        measured = "question"
    elif found:
        measured = f'"{found}"'
    else:
        measured = "none"
    passed = measured != "none"
    return {
        "name": "call to action",
        "passed": passed,
        "measured": measured,
        "reason": f"the closing paragraph invites a response ({measured})"
        if passed
        else "the closing paragraph must end with a question or invite readers to comment, "
        "share or follow",
    }


_MARKDOWN = {
    "heading": re.compile(r"^\s{0,3}#{1,6}\s", re.MULTILINE),
    "bold": re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__"),
    "italic": re.compile(
        r"(?<![\w*])\*(?![\s*])[^*\n]+?(?<!\s)\*(?![\w*])"
        r"|(?<![\w_])_(?![\s_])[^_\n]+?(?<!\s)_(?![\w_])"
    ),
    "link": re.compile(r"\[[^\]\n]+\]\([^)\s]+\)"),
    "citation": _CITATION,
}


def plain_text(post: Post) -> CheckResult:
    """No Markdown headings, bold or italic markers or link syntax, which LinkedIn shows as
    typed, and no `[n]` citations, which LinkedIn posts don't carry."""
    body, _ = _split_hashtags(post.text)
    text = "\n\n".join(body)
    found = [kind for kind, pattern in _MARKDOWN.items() if pattern.search(text)]
    return {
        "name": "plain text",
        "passed": not found,
        "measured": ", ".join(found) or "no Markdown",
        "reason": f"remove the {', '.join(found)}: LinkedIn shows Markdown as typed, and posts "
        "carry no [n] citations"
        if found
        else "plain text, with no Markdown or citations",
    }


def grounded_figures(post: Post) -> CheckResult:
    """Every figure in the post before its hashtags appears in the Brief's findings or
    statistics, matched as #16 matches a statistic to its Sources."""
    body, _ = _split_hashtags(post.text)
    found = figures("\n\n".join(body))
    ungrounded = list(dict.fromkeys(f for f in found if not is_grounded(f, post.brief_facts)))
    if ungrounded:
        listed = ", ".join(ungrounded)
        measured = f"{listed} not in the Brief"
        reason = (
            f"{listed} not in the Brief's findings or statistics: use its figures or drop these"
        )
    elif found:
        measured = f"{len(found)} figure{'' if len(found) == 1 else 's'}, all in the Brief"
        reason = f"every figure ({', '.join(found)}) is in the Brief"
    else:
        measured = reason = "no figures"
    return {
        "name": "grounded figures",
        "passed": not ungrounded,
        "measured": measured,
        "reason": reason,
    }


def char_range(rules: LinkedInQuality, chars: int | None = None) -> tuple[int, int]:
    """The character range a post is written to: when the request sets a length, it is the
    most, with 15% below it; otherwise the configured range."""
    if chars is None:
        return rules.char_range
    return round(chars * (1 - LENGTH_TOLERANCE)), chars


def hashtag_range(rules: LinkedInQuality, count: int | None = None) -> tuple[int, int]:
    """The hashtag range: exactly the count the request sets, otherwise the configured range."""
    return rules.hashtag_range if count is None else (count, count)


def check_linkedin(
    post: Post, rules: LinkedInQuality, chars: int | None = None, tags: int | None = None
) -> list[CheckResult]:
    """Every LinkedIn check's result, in the order the chat lists them.

    `chars` and `tags` are the length and hashtag count the request set, if any; they replace
    the configured ranges.
    """
    return [
        post_length(post, char_range(rules, chars)),
        hashtags(post, hashtag_range(rules, tags)),
        hook(post, rules.max_hook_chars),
        paragraphs(post, rules.min_paragraphs, rules.max_paragraph_chars),
        call_to_action(post, rules.cta_phrases),
        plain_text(post),
        grounded_figures(post),
    ]
