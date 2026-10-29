"""The content checks measure a LinkedIn post against LinkedIn's feed conventions.

Each sample is a hand-written plain-text post, so every measured value below was counted by
hand. A post is its body, a blank line, then its hashtags on one line.
"""

import pytest

from contentblitz.core.config import LinkedInQuality
from contentblitz.workflow.content_checks import Post, check_linkedin

TAGS = "#AI #Health #MedTech #Radiology #Diagnosis #Hospitals #FutureOfWork #Innovation"  # 8
HOOK = "AI now reads scans faster than ever."  # 36 chars
BODY = (
    f"{HOOK}\n\n"
    "Hospitals are adopting it quickly.\n\n"
    "Radiologists get a second pair of eyes.\n\n"
    "What would you do with the time it saves?"
)


FILLER = "Care teams adapt fast. " * 20


def padded(total):
    """A post exactly `total` characters long that passes every check: BODY and TAGS, with
    paragraphs of filler of at most 200 characters before the closing question."""
    *paragraphs, question = BODY.split("\n\n")
    extra = total - len(f"{BODY}\n\n{TAGS}")
    while extra > 0:
        size = min(extra - 2, 200) if extra - 2 <= 200 or extra - 202 >= 3 else 150
        paragraphs.append(FILLER[:size].strip().ljust(size, "."))
        extra -= size + 2
    text = "\n\n".join([*paragraphs, question, TAGS])
    assert len(text) == total
    return text


def check(name, text, facts="", rules=None, chars=None, hashtags=None):
    """The result of the LinkedIn check called `name` on a post."""
    post = Post(text=text, brief_facts=facts)
    results = check_linkedin(post, rules or LinkedInQuality(), chars, hashtags)
    return {r["name"]: r for r in results}[name]


def test_the_checks_are_listed_in_the_order_the_chat_shows_them():
    results = check_linkedin(Post(text=padded(1400), brief_facts=""), LinkedInQuality())
    assert [r["name"] for r in results] == [
        "length",
        "hashtags",
    ]
    assert all(set(r) == {"name", "passed", "measured", "reason"} for r in results)


# Length


@pytest.mark.parametrize(
    ("total", "passed"), [(1299, False), (1300, True), (1600, True), (1601, False)]
)
def test_length_counts_every_character_including_hashtags(total, passed):
    result = check("length", padded(total))
    assert result["passed"] is passed
    assert result["measured"] == f"{total:,} chars"
    assert "1300-1600" in result["reason"]


def test_length_range_comes_from_the_rules():
    rules = LinkedInQuality(char_range=(1000, 1200))
    assert not check("length", padded(1300), rules=rules)["passed"]


def test_a_requested_length_replaces_the_range_with_the_length_as_the_most():
    # "keep it under 1,000 characters": 850-1000
    result = check("length", padded(1300), chars=1000)
    assert not result["passed"]
    assert "850-1000" in result["reason"]


# Hashtags


def with_tags(tags, body=BODY):
    return f"{body}\n\n{tags}"


def test_eight_unique_well_formed_hashtags_at_the_end_pass():
    result = check("hashtags", with_tags(TAGS))
    assert result["passed"]
    assert result["measured"] == "8 hashtags"


@pytest.mark.parametrize(("count", "passed"), [(7, False), (8, True), (12, True), (13, False)])
def test_the_hashtag_count_must_be_within_the_range(count, passed):
    tags = " ".join(f"#Tag{n}" for n in range(count))
    result = check("hashtags", with_tags(tags))
    assert result["passed"] is passed
    assert result["measured"] == f"{count} hashtags"
    if not passed:
        assert "8-12" in result["reason"]


def test_hashtags_differing_only_in_case_are_duplicates():
    result = check("hashtags", with_tags(f"{TAGS} #ai"))
    assert not result["passed"]
    assert "#ai" in result["reason"] and "duplicate" in result["reason"]


@pytest.mark.parametrize("bad", ["#AI-health", "#AI_health", "#", "health", "#AI!"])
def test_malformed_hashtags_fail(bad):
    result = check("hashtags", with_tags(f"{TAGS} {bad}"))
    assert not result["passed"]
    assert bad in result["reason"]


def test_a_hashtag_in_the_body_is_misplaced():
    body = BODY.replace("Hospitals are", "#Hospitals are")
    result = check("hashtags", with_tags(TAGS.replace(" #Hospitals", " #Care"), body))
    assert not result["passed"]
    assert "#Hospitals" in result["reason"] and "body" in result["reason"]


def test_a_post_without_a_closing_hashtag_line_has_no_hashtags():
    result = check("hashtags", BODY)
    assert not result["passed"]
    assert result["measured"] == "0 hashtags"


def test_hashtag_range_comes_from_the_rules_or_the_request():
    rules = LinkedInQuality(hashtag_range=(3, 5))
    assert not check("hashtags", with_tags(TAGS), rules=rules)["passed"]
    five = with_tags("#AI #Health #MedTech #Radiology #Diagnosis")
    assert check("hashtags", five, hashtags=5)["passed"]
    result = check("hashtags", with_tags(TAGS), hashtags=5)
    assert not result["passed"]
    assert "5" in result["reason"]
