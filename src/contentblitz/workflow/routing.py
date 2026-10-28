"""The routing decision the Query Handler's model returns for each message. See spec #4."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Intent = Literal["new_content", "refine", "chat", "unclear"]
# "research" alone means a research-only request.
Format = Literal["research", "blog", "linkedin_post", "image"]
TopicAnswer = Literal["new", "continue"]


def _drop_defaults(schema: dict) -> None:
    # OpenAI strict mode requires every field and rejects "default"; pydantic still applies them.
    for prop in schema.get("properties", {}).values():
        prop.pop("default", None)


class RoutingDecision(BaseModel):
    model_config = ConfigDict(json_schema_extra=_drop_defaults)

    intent: Intent
    subject: str | None = Field(None, description="What the message is about, if anything.")
    same_subject: bool = Field(False, description="Whether the subject matches the active Topic.")
    formats: list[Format] = Field(
        default_factory=list,
        description="Formats in the order asked; ['research'] means research only.",
    )
    fresh_research: bool = Field(False, description="Whether fresh research was asked for.")
    refine_target: str | None = Field(None, description="Content Piece id to refine.")
    instructions: str | None = Field(None, description="Instructions to pass on to the writers.")
    reply: str | None = Field(None, description="The answer, for chat only.")
    clarifying_question: str | None = Field(
        None, description="One short question with suggested requests, for unclear only."
    )
    topic_answer: TopicAnswer | None = Field(
        None, description="The answer to a pending Topic question, if one is pending."
    )
