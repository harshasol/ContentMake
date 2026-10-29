"""Checking a Research Brief's statistics against the text of the Sources they cite. See spec #6.

A statistic is verified when every significant number in its figure appears in the page text
or snippet of a Source it cites. A year standing alone (1900-2099) isn't significant, unless it
is the only number in the figure: in "In 2024, 45% of clinics" only 45% is checked, and
"3 in 4 patients" needs both 3 and 4. Matching ignores thousands separators, `%` versus
"percent", and spacing; it never asks a model.

The same matching tells whether a LinkedIn post's figures come from its Brief (spec #20).
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # state imports the content checks, which import this module
    from contentblitz.workflow.state import ResearchBrief, ResearchSource

# A number, with any thousands separators (a comma or a space between groups of three digits)
# kept so it can be told apart from a year.
_NUMBER = re.compile(r"(?<![\d.])(?:\d{1,3}(?:[, ]\d{3})+(?!\d)|\d+)(?:\.\d+)?%?")
_PERCENT = re.compile(r"(?<=\d)[\s-]*per\s*cent\b", re.IGNORECASE)
_YEAR = re.compile(r"(?:19|20)\d{2}")
_CURRENCY = re.compile(r"[$€£¥]\s?$")


def _normalised(text: str) -> str:
    """`45 percent`, `45-percent`, `45percent` and `45 %` become `45%`; any run of space,
    no-break space included, becomes one space."""
    text = _PERCENT.sub("%", text)
    text = re.sub(r"(?<=\d)\s+%", "%", text)
    return re.sub(r"\s+", " ", text)


def _numbers(text: str) -> list[tuple[str, str]]:
    """Each number in the text, as written and without thousands separators."""
    return [(n, re.sub("[, ]", "", n)) for n in _NUMBER.findall(_normalised(text))]


def _significant(figure: str) -> list[str]:
    """The figure's numbers, leaving out years unless a year is all there is."""
    numbers = _numbers(figure)
    significant = [plain for written, plain in numbers if not _YEAR.fullmatch(written)]
    return significant or [plain for _, plain in numbers]


def _found(number: str, text_numbers: set[str]) -> bool:
    """`45` is found in `45` or `45%`; `45%` only in `45%`."""
    return number in text_numbers or f"{number}%" in text_numbers


def is_verified(figure: str, sources: list["ResearchSource"]) -> bool:
    """Whether every significant number in the figure appears in one of these Sources."""
    wanted = _significant(figure)
    if not wanted:
        return False
    found = [{plain for _, plain in _numbers(f"{s['snippet']}\n{s['page_text']}")} for s in sources]
    return all(any(_found(n, numbers) for numbers in found) for n in wanted)


def figures(text: str) -> list[str]:
    """The figures in the text, as written: percentages, currency amounts, and other numbers
    of 2 or more digits, leaving out years (1900-2099)."""
    text = _normalised(text)
    found = []
    for match in _NUMBER.finditer(text):
        written = match[0]
        currency = _CURRENCY.search(text[: match.start()])
        digits = sum(c.isdigit() for c in written)
        if written.endswith("%") or currency or (digits >= 2 and not _YEAR.fullmatch(written)):
            found.append(f"{currency[0].strip()}{written}" if currency else written)
    return found


def is_grounded(figure: str, text: str) -> bool:
    """Whether the figure's number appears in the text, matched as `is_verified` matches."""
    plain = re.sub(r"[$€£¥, ]", "", figure)
    return _found(plain, {p for _, p in _numbers(text)})


def brief_facts(brief: "ResearchBrief") -> str:
    """The Brief's findings and statistics, which every figure in a post must come from."""
    findings = [f["statement"] for f in brief["key_findings"]]
    statistics = [f"{s['figure']}: {s['description']}" for s in brief["statistics"]]
    return "\n".join([*findings, *statistics])
