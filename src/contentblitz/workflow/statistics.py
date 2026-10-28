"""Checking a Research Brief's statistics against the text of the Sources they cite. See #16.

A statistic is verified when its main number (the first number in its figure) appears in the
page text or snippet of a Source it cites. Matching ignores thousands separators, `%` versus
"percent", and spacing; it never asks a model.
"""

import re

from contentblitz.workflow.state import ResearchSource

_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:\s*%)?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_PERCENT = re.compile(r"\s*\b(?:percent|per\s+cent)\b", re.IGNORECASE)


def _normalised(text: str) -> str:
    """`1,200` becomes `1200`, `45 percent` and `45 %` become `45%`, and runs of space one."""
    text = _THOUSANDS.sub("", text)
    text = _PERCENT.sub("%", text)
    text = re.sub(r"(?<=\d)\s+%", "%", text)
    return re.sub(r"\s+", " ", text)


def _main_number(figure: str) -> str | None:
    found = _NUMBER.search(_normalised(figure))
    return found.group() if found else None


def is_verified(figure: str, sources: list[ResearchSource]) -> bool:
    """Whether the figure's main number appears in any of these Sources' text or snippet."""
    number = _main_number(figure)
    if number is None:
        return False
    # Not part of a longer number: 30% isn't found in 130% or 30.5%, 1200 isn't in 12000.
    pattern = re.compile(rf"(?<![\d.]){re.escape(number)}(?!\d|\.\d)")
    return any(pattern.search(_normalised(f"{s['snippet']}\n{s['page_text']}")) for s in sources)
