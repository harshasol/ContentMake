"""The AI reviewer scores engagement and brand voice against the Brand Profile it is given.

The quality loop always gives it the active Brand Profile (#22, tested through the workflow);
here the reviewer is tested on its own. The model is faked at the LLM integration module.
"""

import pytest

from contentblitz.agents.quality_judge import review
from contentblitz.integrations import llm
from contentblitz.workflow.brand_profiles import DEFAULT_PROFILE

PROFILE = DEFAULT_PROFILE.model_copy(
    update={
        "name": "MedTech Weekly",
        "tone": ["warm", "plain-spoken"],
        "words_to_avoid": ["jargon"],
        "writing_sample": "We keep hospitals running, quietly and well.",
    }
)


@pytest.fixture
def calls(monkeypatch):
    calls = []

    def fake(agent, messages, schema):
        calls.append({"agent": agent, "messages": messages})
        verdict = {"engagement": 8, "brand_voice": 6, "suggestions": [f"fix {n}" for n in range(7)]}
        return llm.Reply(schema.model_validate(verdict), "openai/gpt-4o-mini")

    monkeypatch.setattr(llm, "complete_structured_reply", fake)
    return calls


def test_the_reviewer_scores_brand_voice_against_the_profiles_voice(calls):
    found = review("A draft.", "its hook", brand_profile=PROFILE)

    assert (found["engagement"], found["brand_voice"]) == (8, 6)
    assert [call["agent"] for call in calls] == ["quality_judge"]
    prompt = "\n".join(m["content"] for m in calls[0]["messages"])
    for fact in ["MedTech Weekly", "warm", "plain-spoken", "jargon"]:
        assert fact in prompt, fact
    assert "quietly and well" not in prompt  # the writing sample is not sent


def test_the_reviewer_keeps_at_most_5_suggestions(calls):
    found = review("A draft.", "its hook", brand_profile=PROFILE)

    assert found["suggestions"] == [f"fix {n}" for n in range(5)]
