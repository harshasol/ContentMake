"""Learn voice: describe a brand's voice from a sample of its writing. See spec #22, ticket #57.

One structured call to the `brand_voice` model. It returns a voice description (under 120
words) and 3-6 tone descriptors, which the Brand Profiles tab puts into the form for the
marketer to review before saving. A sample outside 300-10,000 characters is refused unsent.
"""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, field_validator

from contentblitz.integrations import llm

AGENT = "brand_voice"
MIN_SAMPLE_CHARS = 300
MAX_SAMPLE_CHARS = 10_000
MAX_DESCRIPTION_WORDS = 120
SAMPLE_LIMITS = (
    f"Paste {MIN_SAMPLE_CHARS}–{MAX_SAMPLE_CHARS:,} characters of the brand's writing, "
    "such as a few paragraphs from a blog post or newsletter."
)

SYSTEM_PROMPT = f"""\
You are a brand editor. Describe the voice of the writing sample the marketer gives you, so a
writer could match it. In under {MAX_DESCRIPTION_WORDS} words, cover sentence length and rhythm,
formality, point of view, vocabulary, humour and signature phrases. Then suggest 3 to 6 short
tone descriptors (a word or two each, like "confident" or "no jargon")."""

ToneDescriptor = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)
]


class LearnedVoice(BaseModel):
    voice_description: str = Field(
        description=f"The brand's voice, in under {MAX_DESCRIPTION_WORDS} words."
    )
    tone: list[ToneDescriptor] = Field(
        min_length=3, max_length=6, description="3 to 6 short tone descriptors."
    )

    @field_validator("voice_description")
    @classmethod
    def _under_word_limit(cls, value: str) -> str:
        value = value.strip()
        if len(value.split()) >= MAX_DESCRIPTION_WORDS:
            raise ValueError(f"must be under {MAX_DESCRIPTION_WORDS} words")
        return value


class LearnVoiceError(ValueError):
    """Learn voice couldn't run: the sample was the wrong length, or the call failed. The
    message is for the marketer."""


def learn_voice(sample: str) -> LearnedVoice:
    """The voice of `sample`. Raises LearnVoiceError, with a message for the marketer."""
    sample = sample.strip()
    if len(sample) < MIN_SAMPLE_CHARS:
        raise LearnVoiceError(f"That sample is too short to learn from. {SAMPLE_LIMITS}")
    if len(sample) > MAX_SAMPLE_CHARS:
        raise LearnVoiceError(f"That sample is too long. {SAMPLE_LIMITS}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Writing sample:\n\n{sample}"},
    ]
    try:
        return llm.complete_structured(AGENT, messages, LearnedVoice)
    except Exception as error:  # noqa: BLE001 - any failure leaves the form as it was
        raise LearnVoiceError(
            "ContentBlitz couldn't learn the voice just now. Try again in a moment."
        ) from error
