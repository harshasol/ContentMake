"""Content checks: deterministic measurements of a draft against its format's rules. See #19.

Each check returns a `CheckResult`: pass or fail, the measured value as the chat shows it,
and a short reason that names the target, which is also the revise loop's feedback.
Measurements leave out the Sources section and Markdown syntax.

A format's rules come from its config section, and `check_<format>` runs its checks in the
order the chat lists them.
"""

import re
from dataclasses import dataclass
from typing import TypedDict

import textstat

from contentblitz.core.config import BlogQuality


@dataclass(frozen=True)
class Draft:
    """What the checks measure: a draft's text and what it should match."""

    body: str  # Markdown
    meta_description: str
    keyword: str  # the target keyword phrase
    source_numbers: list[int]  # the Brief's Source numbers, which `[n]` citations must match


class CheckResult(TypedDict):
    name: str  # as the chat shows it, e.g. "keyword density"
    passed: bool
    measured: str  # e.g. "1.4%", "171 chars"
    reason: str  # what was measured against what target


_WORD = re.compile(r"\w+(?:['’]\w+)*")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SOURCES_HEADING = re.compile(r"sources", re.IGNORECASE)


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


def _without_sources(body: str) -> list[tuple[str, bool]]:
    """`_lines`, leaving out the Sources section up to the next heading at its level."""
    kept: list[tuple[str, bool]] = []
    sources_level = 0
    for line, in_code in _lines(body):
        heading = None if in_code else _HEADING.match(line)
        if heading:
            level = len(heading[1])
            if sources_level and level <= sources_level:
                sources_level = 0
            if not sources_level and _SOURCES_HEADING.fullmatch(heading[2].strip(" *_:")):
                sources_level = level
        if not sources_level:
            kept.append((line, in_code))
    return kept


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

    A line that doesn't end a sentence, such as a heading or a list item, reads as one.
    """
    lines = [line.strip() for line in _prose(draft.body).splitlines() if line.strip()]
    text = " ".join(line if line[-1] in ".!?" else f"{line}." for line in lines)
    grade = textstat.flesch_kincaid_grade(text)
    measured = f"grade {grade:.1f}"
    return {
        "name": "readability",
        "passed": grade <= max_grade,
        "measured": measured,
        "reason": f"Flesch-Kincaid {measured}; the target is at most grade {max_grade:g}",
    }


def check_blog(draft: Draft, rules: BlogQuality) -> list[CheckResult]:
    """Every blog check's result, in the order the chat lists them."""
    return [
        keyword_density(draft, rules.density_range),
        meta_description(draft, rules.meta_length_range),
        headings(draft),
        readability(draft, rules.max_grade),
    ]
