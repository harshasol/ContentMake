"""The routing decision the Query Handler's model returns for each message. See spec #4."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from contentblitz.workflow.formats import Format

Intent = Literal["new_content", "refine", "chat", "unclear"]
TopicAnswer = Literal["new", "continue"]
BlogTemplate = Literal["how-to", "listicle", "comparison", "thought leadership"]
LinkedInTemplate = Literal["industry insights", "personal story", "tips", "announcement"]
Template = Literal[BlogTemplate, LinkedInTemplate]
LengthUnit = Literal["words", "characters"]


def _drop_defaults(schema: dict) -> None:
    # OpenAI strict mode requires every field and rejects "default"; pydantic still applies them.
    for prop in schema.get("properties", {}).values():
        prop.pop("default", None)


class RoutingDecision(BaseModel):
    model_config = ConfigDict(json_schema_extra=_drop_defaults)

    intent: Intent
    topic_name: str | None = Field(None, description="The Topic the message is about, if any.")
    same_topic: bool = Field(False, description="Whether the request is on the active Topic.")
    formats: list[Format] = Field(
        default_factory=list,
        description="Formats in the order asked; ['research'] means research only.",
    )
    fresh_research: bool = Field(False, description="Whether fresh research was asked for.")
    refine_target: str | None = Field(None, description="Content Piece id to refine.")
    source_piece: str | None = Field(
        None,
        description="For new_content: the id of the Content Piece the new one builds on, only "
        "if the request points at one ('turn that blog into a LinkedIn post').",
    )
    instructions: str | None = Field(None, description="Instructions to pass on to the writers.")
    target_keyword: str | None = Field(
        None, description="The blog's target keyword, only if the user set one."
    )
    template: Template | None = Field(
        None, description="The blog's or LinkedIn post's template, only if the user chose one."
    )
    length: int | None = Field(
        None,
        description="The length the user asked for, only if they asked for one: in words for a "
        "blog, in characters for a LinkedIn post.",
    )
    length_unit: LengthUnit | None = Field(
        None, description="The unit of length: 'characters' when the user gave characters."
    )
    reply: str | None = Field(None, description="The answer, for chat only.")
    show_research_report: bool = Field(
        False, description="For chat only: the user asked to see the research report."
    )
    clarifying_question: str | None = Field(
        None, description="One short question with suggested requests, for unclear only."
    )
    topic_answer: TopicAnswer | None = Field(
        None, description="The answer to a pending Topic question, if one is pending."
    )
