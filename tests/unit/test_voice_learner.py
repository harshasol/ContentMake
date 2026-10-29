"""Learn voice (#57): one structured call to the `brand_voice` model, faked at the LLM module."""

import pytest
from pydantic import ValidationError

from contentblitz.agents.voice_learner import LearnedVoice, LearnVoiceError, learn_voice
from contentblitz.integrations import llm

SAMPLE = "We keep it short. We say what we mean. " * 10  # 390 characters

LEARNED = {
    "voice_description": "Short, punchy sentences in the first person plural, plain words.",
    "tone": ["direct", "plain-spoken", "warm"],
}


@pytest.fixture
def calls(monkeypatch):
    calls = []

    def fake(agent, messages, schema, config=None):
        calls.append({"agent": agent, "messages": messages, "schema": schema})
        return schema.model_validate(LEARNED)

    monkeypatch.setattr(llm, "complete_structured", fake)
    return calls


def test_learning_a_voice_makes_one_structured_call_to_brand_voice_with_the_sample(calls):
    learned = learn_voice(SAMPLE)

    assert learned == LearnedVoice(**LEARNED)
    [call] = calls
    assert call["agent"] == "brand_voice"
    assert call["schema"] is LearnedVoice
    assert SAMPLE.strip() in call["messages"][-1]["content"]


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        ("x" * 299, "too short"),
        ("   " + "x" * 299 + "   ", "too short"),
        ("x" * 10_001, "too long"),
    ],
)
def test_a_sample_outside_300_to_10000_characters_is_refused_without_a_call(
    calls, sample, message
):
    with pytest.raises(LearnVoiceError, match=message) as refused:
        learn_voice(sample)

    assert "300" in str(refused.value) and "10,000" in str(refused.value)
    assert calls == []


@pytest.mark.parametrize("sample", ["x" * 300, "x" * 10_000])
def test_samples_at_the_limits_are_accepted(calls, sample):
    learn_voice(sample)

    assert len(calls) == 1


def test_a_failed_call_raises_a_learn_voice_error(monkeypatch):
    def failing(*args, **kwargs):
        raise llm.StructuredOutputError("bad reply")

    monkeypatch.setattr(llm, "complete_structured", failing)

    with pytest.raises(LearnVoiceError, match="couldn't learn"):
        learn_voice(SAMPLE)


@pytest.mark.parametrize(
    "fields",
    [
        {**LEARNED, "voice_description": "word " * 120},
        {**LEARNED, "tone": ["direct", "warm"]},
        {**LEARNED, "tone": ["a", "b", "c", "d", "e", "f", "g"]},
        {**LEARNED, "tone": ["direct", "warm", "x" * 41]},
    ],
)
def test_the_reply_must_be_under_120_words_with_3_to_6_short_tone_descriptors(fields):
    with pytest.raises(ValidationError):
        LearnedVoice.model_validate(fields)
