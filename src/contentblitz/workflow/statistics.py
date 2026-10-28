"""Checking a Research Brief's statistics against the text of the Sources they cite. See spec #6.

A statistic is verified when every significant number in its figure appears in the page text
or snippet of a Source it cites. A year standing alone (1900-2099) isn't significant, unless it
is the only number in the figure: in "In 2024, 45% of clinics" only 45% is checked, and
"3 in 4 patients" needs both 3 and 4. Matching ignores thousands separators, `%` versus
"percent", and spacing; it never asks a model.
"""

import re

from contentblitz.workflow.state import ResearchSource

# A number, with any thousands separators kept so they can be told apart from a year.
_NUMBER = re.compile(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?%?")
_PERCENT = re.compile(r"\s*\b(?:percent|per\s+cent)\b", re.IGNORECASE)
_YEAR = re.compile(r"(?:19|20)\d{2}")


def _normalised(text: str) -> str:
    """`45 percent` and `45 %` become `45%`, and runs of space one."""
    text = _PERCENT.sub("%", text)
    text = re.sub(r"(?<=\d)\s+%", "%", text)
    return re.sub(r"\s+", " ", text)


def _numbers(text: str) -> list[tuple[str, str]]:
    """Each number in the text, as written and without thousands separators."""
    return [(n, n.replace(",", "")) for n in _NUMBER.findall(_normalised(text))]


def _significant(figure: str) -> list[str]:
    """The figure's numbers, leaving out years unless a year is all there is."""
    numbers = _numbers(figure)
    significant = [plain for written, plain in numbers if not _YEAR.fullmatch(written)]
    return significant or [plain for _, plain in numbers]


def _found(number: str, text_numbers: set[str]) -> bool:
    """`45` is found in `45` or `45%`; `45%` only in `45%`."""
    return number in text_numbers or f"{number}%" in text_numbers


def is_verified(figure: str, sources: list[ResearchSource]) -> bool:
    """Whether every significant number in the figure appears in one of these Sources."""
    wanted = _significant(figure)
    if not wanted:
        return False
    found = [{plain for _, plain in _numbers(f"{s['snippet']}\n{s['page_text']}")} for s in sources]
    return all(any(_found(n, numbers) for numbers in found) for n in wanted)
