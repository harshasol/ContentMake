"""The content checks measure a blog draft against its format's rules, deterministically.

Each sample is hand-written Markdown, so every measured value below was counted by hand.
"""

import pytest

from contentblitz.core.config import BlogQuality
from contentblitz.workflow.content_checks import Draft, check_blog

KEYWORD = "AI tools"


def filler(n):
    """`n` words that are not the keyword."""
    return " ".join(["word"] * n)


def check(name, body, meta="", keyword=KEYWORD, rules=None):
    """The result of the check called `name` on a blog draft."""
    draft = Draft(body=body, meta_description=meta, keyword=keyword, source_numbers=[1, 2, 3])
    results = {r["name"]: r for r in check_blog(draft, rules or BlogQuality())}
    return results[name]


# Keyword density


def test_keyword_density_counts_the_phrase_in_any_case_times_its_words_over_all_words():
    # 2 occurrences x 2 words / 200 words = 2.0%
    body = f"AI tools {filler(98)} ai TOOLS {filler(98)}"
    result = check("keyword density", body)
    assert result["passed"]
    assert result["measured"] == "2.0%"
    assert "1-2%" in result["reason"]


def test_keyword_density_only_matches_the_phrase_on_word_boundaries():
    # "AI toolset" and "maI tools" are not the phrase: 1 match x 2 words / 200 words = 1.0%
    body = f"AI tools {filler(96)} AI toolset maI tools {filler(98)}"
    result = check("keyword density", body)
    assert result["measured"] == "1.0%"
    assert result["passed"]


def test_keyword_density_outside_the_range_fails_and_says_the_target():
    # 1 x 2 / 400 words = 0.5%
    result = check("keyword density", f"AI tools {filler(398)}")
    assert not result["passed"]
    assert result["measured"] == "0.5%"
    assert "1-2%" in result["reason"]

    # 3 x 2 / 100 words = 6.0%
    stuffed = check("keyword density", f"AI tools AI tools AI tools {filler(94)}")
    assert not stuffed["passed"]
    assert stuffed["measured"] == "6.0%"


def test_keyword_density_range_comes_from_the_rules():
    # 1 x 2 / 400 words = 0.5%
    rules = BlogQuality(density_range=(0.5, 1.0))
    assert check("keyword density", f"AI tools {filler(398)}", rules=rules)["passed"]


def test_markdown_syntax_and_the_sources_section_are_left_out_of_the_counts():
    # Counted words: "AI tools" (2) in the H1, "AI tools here" (3), "Lead" (1), then 194 filler
    # words in a list, a link's text and emphasis = 200 words, 2 matches = 2.0%. Citations,
    # URLs, heading and list markers, and the whole Sources section aren't words.
    body = (
        "# AI tools\n\n"
        "**AI** _tools_ here [1, 2].\n\n"
        "## Lead\n\n"
        f"- {filler(90)}\n"
        f"- [{filler(4)}](https://example.com/a-b-c) {filler(100)} [3]\n\n"
        "## Sources\n\n"
        "1. [AI tools report](https://example.com/ai-tools)\n"
        "2. [AI tools survey](https://example.com/ai-tools-survey)\n"
    )
    result = check("keyword density", body)
    assert result["measured"] == "2.0%"


# Meta description


def meta_of(length, keyword=KEYWORD):
    """A meta description of exactly `length` characters that starts with `keyword`."""
    text = f"{keyword} help marketers write faster"
    return (text + " " + "x" * length)[:length]


@pytest.mark.parametrize(
    ("length", "passed"), [(149, False), (150, True), (160, True), (161, False)]
)
def test_meta_description_length_must_be_within_the_range(length, passed):
    meta = meta_of(length)
    assert len(meta) == length
    result = check("meta description", "", meta=meta)
    assert result["passed"] is passed
    assert result["measured"] == f"{length} chars"
    assert "150-160" in result["reason"]


def test_meta_description_must_contain_the_keyword_in_any_case():
    assert check("meta description", "", meta=meta_of(155, "ai TOOLS"))["passed"]

    result = check("meta description", "", meta=meta_of(155, "AI toolset"))
    assert not result["passed"]
    assert result["measured"] == "155 chars"
    assert "AI tools" in result["reason"]


def test_meta_length_range_comes_from_the_rules():
    rules = BlogQuality(meta_length_range=(100, 120))
    assert check("meta description", "", meta=meta_of(110), rules=rules)["passed"]


# Headings

GOOD_HEADINGS = """\
# Choosing AI tools

Intro.

## Why AI tools matter

### For small teams

Text.

## How to pick one

## What it costs

### Licences

## Sources

1. [Report](https://example.com)
"""


def test_one_h1_and_three_h2s_with_the_keyword_in_order_pass():
    result = check("headings", GOOD_HEADINGS)
    assert result["passed"], result["reason"]
    # The Sources heading is left out of the count
    assert result["measured"] == "1 H1, 3 H2s"


@pytest.mark.parametrize(
    ("body", "measured", "problem"),
    [
        ("## Why AI tools matter\n## B\n## C\n", "0 H1s, 3 H2s", "exactly one H1"),
        (
            "# AI tools\n# Also AI tools\n## AI tools now\n## B\n## C\n",
            "2 H1s, 3 H2s",
            "exactly one H1",
        ),
        ("# Choosing a stack\n## AI tools now\n## B\n## C\n", "1 H1, 3 H2s", "H1 doesn't contain"),
        ("# AI tools\n## AI tools now\n## B\n", "1 H1, 2 H2s", "at least 3 H2s"),
        ("# AI tools\n## A\n## B\n## C\n", "1 H1, 3 H2s", "no H2 contains"),
        ("# AI tools\n### Early\n## AI tools now\n## B\n## C\n", "1 H1, 3 H2s", "H3 before"),
        ("# AI tools\n## AI tools now\n#### Deep\n## B\n## C\n", "1 H1, 3 H2s", "skips"),
    ],
    ids=[
        "no H1",
        "two H1s",
        "no keyword in H1",
        "two H2s",
        "no keyword in H2s",
        "early H3",
        "skipped level",
    ],
)
def test_heading_structure_violations_fail_and_say_why(body, measured, problem):
    result = check("headings", body)
    assert not result["passed"]
    assert result["measured"] == measured
    assert problem.lower() in result["reason"].lower()


def test_headings_inside_code_blocks_are_not_headings():
    body = GOOD_HEADINGS.replace("Text.", "```\n# not a heading\n```")
    assert check("headings", body)["passed"]


# Readability

EASY = "The cat sat on the mat. The dog ran to the park."
HARD = (
    "Organizational transformation necessitates comprehensive reconsideration of "
    "institutional methodologies, particularly regarding interdepartmental communication "
    "infrastructure."
)


def test_readability_is_the_flesch_kincaid_grade_at_or_below_the_maximum():
    # 12 one-syllable words in 2 sentences: 0.39 x 6 + 11.8 x 1 - 15.59 = -1.45
    result = check("readability", EASY)
    assert result["passed"]
    assert result["measured"] == "grade -1.4"
    assert "10" in result["reason"]


def test_readability_above_the_maximum_fails_and_says_the_target():
    result = check("readability", HARD)
    assert not result["passed"]
    assert result["measured"] == "grade 42.1"
    assert "10" in result["reason"]


def test_readability_maximum_comes_from_the_rules():
    assert not check("readability", EASY, rules=BlogQuality(max_grade=-2))["passed"]


def test_readability_reads_headings_as_sentences_and_leaves_out_the_sources_section():
    # "The cat and the dog." and "The cat sat on the mat.": 11 one-syllable words in 2
    # sentences, 0.39 x 5.5 + 11.8 - 15.59 = -1.645. The Sources section isn't read.
    body = (
        "# The cat and the dog\n\nThe cat sat on the mat.\n\n"
        f"## Sources\n\n1. [{HARD}](https://example.com)\n"
    )
    assert check("readability", body)["measured"] == "grade -1.6"


# Length


@pytest.mark.parametrize(
    ("words", "passed"), [(899, False), (900, True), (1800, True), (1801, False)]
)
def test_length_must_be_within_the_word_range(words, passed):
    result = check("length", filler(words))
    assert result["passed"] is passed
    assert result["measured"] == f"{words:,} words"
    assert "900-1800" in result["reason"]


def test_length_leaves_out_markdown_and_the_sources_section():
    # "Title" (1) + 896 filler words in a list + a link's text (3) = 900 words. Citations,
    # URLs, heading and list markers, and the whole Sources section aren't words.
    body = (
        "# Title\n\n"
        f"- **{filler(896)}** [1, 2]\n"
        f"- [{filler(3)}](https://example.com/a-b-c) [3]\n\n"
        "## Sources\n\n"
        "1. [AI tools report](https://example.com/ai-tools)\n"
    )
    result = check("length", body)
    assert result["measured"] == "900 words"
    assert result["passed"]


def test_word_range_comes_from_the_rules():
    assert check("length", filler(600), rules=BlogQuality(word_range=(500, 700)))["passed"]


# Citations


def sources(*numbers):
    """A Sources section listing `numbers`."""
    entries = "".join(f"{n}. [Report {n}](https://example.com/{n})\n" for n in numbers)
    return f"\n## Sources\n\n{entries}"


def test_citations_of_three_brief_sources_all_listed_pass():
    body = "AI tools help [1]. Teams agree [2, 3]. Again [1].\n" + sources(1, 2, 3)
    result = check("citations", body)
    assert result["passed"], result["reason"]
    assert result["measured"] == "3 Sources cited"


def test_a_citation_that_is_not_a_brief_source_fails():
    body = "AI tools help [1]. Teams agree [2, 3]. Also [7].\n" + sources(1, 2, 3)
    result = check("citations", body)
    assert not result["passed"]
    assert result["measured"] == "3 Sources cited"
    assert "[7]" in result["reason"]


def test_too_few_distinct_sources_fail_and_say_the_minimum():
    body = "AI tools help [1]. Teams agree [2]. Again [1, 2].\n" + sources(1, 2)
    result = check("citations", body)
    assert not result["passed"]
    assert result["measured"] == "2 Sources cited"
    assert "at least 3" in result["reason"]


def test_minimum_distinct_citations_comes_from_the_rules():
    body = "AI tools help [1]. Teams agree [2].\n" + sources(1, 2)
    assert check("citations", body, rules=BlogQuality(min_citations=2))["passed"]


def test_a_sources_section_listing_an_uncited_source_fails():
    body = "AI tools help [1]. Teams agree [2, 3].\n" + sources(1, 2, 3, 4)
    result = check("citations", body, rules=BlogQuality(min_citations=3))
    assert not result["passed"]
    assert "[4]" in result["reason"]
    assert "uncited" in result["reason"]


def test_a_sources_section_missing_a_cited_source_fails():
    body = "AI tools help [1]. Teams agree [2, 3].\n" + sources(1, 3)
    result = check("citations", body)
    assert not result["passed"]
    assert "[2]" in result["reason"]
    assert "doesn't list" in result["reason"]


def test_a_blog_without_a_sources_section_fails():
    result = check("citations", "AI tools help [1]. Teams agree [2, 3].\n")
    assert not result["passed"]
    assert "doesn't list [1], [2], [3]" in result["reason"]


def test_citation_ranges_and_bracketed_source_lists_are_read():
    # [1-3] cites 1, 2 and 3; the Sources section may number its entries as "[n]".
    body = "AI tools help [1-3].\n\n## Sources\n\n[1] Report 1\n[2] Report 2\n[3] Report 3\n"
    assert check("citations", body)["passed"]


def test_brackets_in_code_and_link_text_are_not_citations():
    body = "AI tools help [1, 2, 3]. See [2](https://example.com).\n```\nx = a[9]\n```\n"
    assert check("citations", body + sources(1, 2, 3))["passed"]
