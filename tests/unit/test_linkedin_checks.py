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
        "hook",
        "paragraphs",
        "call to action",
        "plain text",
        "grounded figures",
    ]
    assert all(set(r) == {"name", "passed", "measured", "reason"} for r in results)


def test_words_to_avoid_is_checked_last_only_when_the_profile_has_some():
    post = Post(text="A revolutionary idea.\n\nWhat do you think?\n\n#Disruptive", brief_facts="")
    assert "words to avoid" not in [r["name"] for r in check_linkedin(post, LinkedInQuality())]
    avoid = ["Revolutionary", "disruptive", "synergy"]
    [*_, last] = check_linkedin(post, LinkedInQuality(), avoid=avoid)
    assert (last["name"], last["passed"]) == ("words to avoid", False)
    assert last["measured"] == "Revolutionary, disruptive"  # hashtags are checked too


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
    assert check("hashtags", five, hashtags=(5, 5))["passed"]
    result = check("hashtags", with_tags(TAGS), hashtags=(5, 5))
    assert not result["passed"]
    assert "5" in result["reason"]


def test_a_padded_post_passes_every_check():
    results = check_linkedin(Post(text=padded(1450), brief_facts=""), LinkedInQuality())
    assert [r["name"] for r in results if not r["passed"]] == []


# Hook


def test_a_short_first_line_is_a_hook():
    result = check("hook", with_tags(TAGS))
    assert result["passed"]
    assert result["measured"] == "36 chars"


@pytest.mark.parametrize(("size", "passed"), [(150, True), (151, False)])
def test_the_first_line_must_be_at_most_150_characters(size, passed):
    hook = ("Long hook " * 20)[: size - 1].strip().ljust(size - 1, "x") + "."
    result = check("hook", with_tags(TAGS, BODY.replace(HOOK, hook)))
    assert result["passed"] is passed
    assert result["measured"] == f"{size} chars"
    if not passed:
        assert "150" in result["reason"]


def test_the_hook_is_the_first_line_even_when_its_paragraph_goes_on():
    body = BODY.replace(HOOK, f"{HOOK}\n{'More detail follows here. ' * 8}")
    assert check("hook", with_tags(TAGS, body))["passed"]


@pytest.mark.parametrize(
    "first_line", ["#AI is changing radiology", "https://example.com/ai-in-healthcare"]
)
def test_a_first_line_that_is_a_hashtag_or_a_link_is_not_a_hook(first_line):
    result = check("hook", with_tags(TAGS, BODY.replace(HOOK, first_line)))
    assert not result["passed"]


def test_hook_length_comes_from_the_rules():
    assert not check("hook", with_tags(TAGS), rules=LinkedInQuality(max_hook_chars=30))["passed"]


# Paragraphs


def test_four_short_paragraphs_pass():
    result = check("paragraphs", with_tags(TAGS))
    assert result["passed"]
    assert result["measured"] == "4 paragraphs"


def test_a_wall_of_text_fails():
    wall = " ".join(BODY.split("\n\n")) + " " + "Care teams adapt fast. " * 50
    result = check("paragraphs", with_tags(TAGS, wall.strip()))
    assert not result["passed"]
    assert result["measured"] == "1 paragraph"
    assert "at least 4" in result["reason"] and "300" in result["reason"]


def test_too_few_paragraphs_fail():
    body = "\n\n".join(BODY.split("\n\n")[1:])
    result = check("paragraphs", with_tags(TAGS, body))
    assert not result["passed"]
    assert result["measured"] == "3 paragraphs"


@pytest.mark.parametrize(("size", "passed"), [(300, True), (301, False)])
def test_no_paragraph_may_be_longer_than_300_characters(size, passed):
    long = FILLER[: size - 1].strip().ljust(size - 1, "x") + "."
    body = BODY.replace("Hospitals are adopting it quickly.", long)
    result = check("paragraphs", with_tags(TAGS, body))
    assert result["passed"] is passed
    if not passed:
        assert "301" in result["reason"]


def test_paragraph_limits_come_from_the_rules():
    rules = LinkedInQuality(min_paragraphs=5)
    assert not check("paragraphs", with_tags(TAGS), rules=rules)["passed"]
    rules = LinkedInQuality(max_paragraph_chars=40)
    assert not check("paragraphs", with_tags(TAGS), rules=rules)["passed"]


# Call to action


def ending(last):
    return with_tags(TAGS, BODY.replace("What would you do with the time it saves?", last))


def test_a_closing_question_is_a_call_to_action():
    result = check("call to action", ending("What would you do with the time it saves? 👇"))
    assert result["passed"]
    assert result["measured"] == "question"


@pytest.mark.parametrize(
    ("last", "phrase"),
    [
        ("Share this with a colleague who reads scans.", "share"),
        ("Let me know how your team uses AI.", "let me know"),
        ("Follow me for more on medical AI.", "follow"),
        ("Drop a comment below.", "comment"),
    ],
)
def test_a_closing_invitation_is_a_call_to_action(last, phrase):
    result = check("call to action", ending(last))
    assert result["passed"]
    assert result["measured"] == f'"{phrase}"'


def test_an_ending_with_no_question_or_invitation_fails():
    result = check("call to action", ending("AI is here to stay."))
    assert not result["passed"]
    assert result["measured"] == "none"
    assert "question" in result["reason"]


def test_a_question_earlier_in_the_post_is_not_the_call_to_action():
    body = BODY.replace("Hospitals are adopting it quickly.", "Are hospitals ready?")
    result = check("call to action", with_tags(TAGS, body.replace(
        "What would you do with the time it saves?", "AI is here to stay.")))
    assert not result["passed"]


def test_call_to_action_phrases_come_from_the_rules():
    rules = LinkedInQuality(cta_phrases=["register"])
    assert check("call to action", ending("Register for the webinar."), rules=rules)["passed"]
    assert not check("call to action", ending("Share this post."), rules=rules)["passed"]


# Plain text


def test_a_plain_text_post_passes():
    result = check("plain text", with_tags(TAGS))
    assert result["passed"]
    assert result["measured"] == "no Markdown"


@pytest.mark.parametrize(
    ("markdown", "kind"),
    [
        ("## Why it matters", "heading"),
        ("Hospitals are **adopting** it quickly.", "bold"),
        ("Hospitals are __adopting__ it quickly.", "bold"),
        ("Hospitals are *adopting* it quickly.", "italic"),
        ("Hospitals are _adopting_ it quickly.", "italic"),
        ("Hospitals are [adopting it](https://example.com) quickly.", "link"),
        ("Hospitals are adopting it quickly [2].", "citation"),
    ],
)
def test_markdown_and_citations_fail_plain_text(markdown, kind):
    body = BODY.replace("Hospitals are adopting it quickly.", markdown)
    result = check("plain text", with_tags(TAGS, body))
    assert not result["passed"]
    assert kind in result["measured"]


def test_plain_text_allows_snake_case_asterisk_footnotes_and_bare_links():
    body = BODY.replace(
        "Hospitals are adopting it quickly.",
        "Hospitals use model_v2 at 2*3 sites. Read more at https://example.com/a_b_c.",
    )
    assert check("plain text", with_tags(TAGS, body))["passed"]


# Grounded figures

FACTS = (
    "AI reads scans 30% faster than radiologists.\n"
    "The FDA has cleared 1,250 AI devices.\n"
    "$4.5 billion: spent on medical AI in 2024\n"
    "7 percent: fewer missed findings"
)


def figures(sentence, facts=FACTS):
    body = BODY.replace("Hospitals are adopting it quickly.", sentence)
    return check("grounded figures", with_tags(TAGS, body), facts=facts)


def test_figures_from_the_brief_pass():
    result = figures("Scans are read 30% faster, and 1,250 devices are cleared.")
    assert result["passed"]
    assert result["measured"] == "2 figures, all in the Brief"


def test_a_figure_not_in_the_brief_fails_and_is_named():
    result = figures("Scans are read 45% faster, across 1,250 devices and 80 hospitals.")
    assert not result["passed"]
    assert result["measured"] == "45%, 80 not in the Brief"
    assert "45%" in result["reason"] and "80" in result["reason"]


@pytest.mark.parametrize(
    "sentence",
    [
        "The FDA has cleared 1250 AI devices.",  # no thousands separator
        "Scans are read 30 percent faster.",  # percent spelled out
        "Spending reached $4.5 billion.",  # a currency amount
        "Missed findings fell 7%.",  # a single-digit percentage matches "7 percent"
    ],
)
def test_differently_formatted_forms_of_a_brief_figure_are_grounded(sentence):
    assert figures(sentence)["passed"]


@pytest.mark.parametrize(
    ("sentence", "ungrounded"),
    [
        ("Spending reached $9 billion.", "$9"),  # a currency amount, even of one digit
        ("Missed findings fell 3%.", "3%"),  # a percentage, even of one digit
        ("It took 12 months.", "12"),  # two digits or more
        ("Accuracy rose 2.5 points.", "2.5"),
    ],
)
def test_percentages_currency_and_numbers_of_two_or_more_digits_are_figures(sentence, ungrounded):
    result = figures(sentence)
    assert not result["passed"]
    assert result["measured"] == f"{ungrounded} not in the Brief"


def test_years_and_single_digits_are_not_figures():
    result = figures("Since 2019, and again in 2025, 3 hospitals in 1 region joined.")
    assert result["passed"]
    assert result["measured"] == "no figures"


def test_a_brief_percentage_does_not_ground_the_bare_number():
    # "30%" in the post needs a 30% in the Brief; a bare 30 doesn't ground it.
    assert not figures("Scans are read 30% faster.", facts="30 hospitals use AI.")["passed"]
    assert figures("30 hospitals use AI.", facts="AI is 30% faster.")["passed"]


def test_the_hashtag_line_is_not_searched_for_figures():
    text = with_tags(f"{TAGS} #Health2030 #Top50", BODY)
    assert check("grounded figures", text, facts=FACTS)["passed"]


def test_a_requested_hashtag_range_or_none_replaces_the_configured_range():
    assert check("hashtags", with_tags(TAGS), hashtags=(6, 10))["passed"]
    result = check("hashtags", with_tags(TAGS), hashtags=(0, 0))
    assert not result["passed"]
    assert "the target is 0" in result["reason"]
    assert check("hashtags", BODY, hashtags=(0, 0))["passed"]
