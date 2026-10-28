"""The AI reviewer scores brand voice only when it is given a Brand Profile.

No Brand Profile reaches the workflow until #22, so this is tested at the reviewer itself. The
model is faked at the LLM integration module and fills in whichever verdict it is asked for.
"""

import pytest

from contentblitz.integrations import llm
from contentblitz.workflow.reviewer import review

PROFILE = {"name": "MedTech Weekly", "tone": "warm, plain-spoken", "avoid": "jargon"}


@pytest.fixture
def calls(monkeypatch):
    calls = []

    def fake(agent, messages, schema):
        calls.append({"agent": agent, "messages": messages})
        verdict = {"engagement": 8, "brand_voice": 6, "suggestions": [f"fix {n}" for n in range(7)]}
        fields = {k: v for k, v in verdict.items() if k in schema.model_fields}
        return llm.Reply(schema.model_validate(fields), "openai/gpt-4o-mini")

    monkeypatch.setattr(llm, "complete_structured_reply", fake)
    return calls


def test_without_a_brand_profile_there_is_no_brand_voice_score(calls):
    found = review("A draft.", "its hook")

    assert "brand_voice" not in found
    assert found["engagement"] == 8
    assert [call["agent"] for call in calls] == ["quality_judge"]


def test_with_a_brand_profile_the_reviewer_scores_brand_voice_against_it(calls):
    found = review("A draft.", "its hook", brand_profile=PROFILE)

    assert (found["engagement"], found["brand_voice"]) == (8, 6)
    prompt = "\n".join(m["content"] for m in calls[0]["messages"])
    for value in PROFILE.values():
        assert value in prompt, value


def test_the_reviewer_keeps_at_most_5_suggestions(calls):
    assert review("A draft.", "its hook")["suggestions"] == [f"fix {n}" for n in range(5)]
