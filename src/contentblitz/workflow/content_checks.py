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
