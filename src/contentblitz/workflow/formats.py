"""The formats a user can ask for: each one's agent, label, Content Piece kind, and what a
request may set for its writer."""

from typing import Literal, NamedTuple, get_args

AgentName = Literal[
    "research",
    "content_strategist",
    "blog_writer",
    "linkedin_writer",
    "image_generator",
]


BlogTemplate = Literal["how-to", "listicle", "comparison", "thought leadership"]
LinkedInTemplate = Literal["industry insights", "personal story", "tips", "announcement"]
LengthUnit = Literal["words", "characters"]


class FormatSpec(NamedTuple):
    agent: AgentName
    label: str
    kind: str | None  # the Content Piece kind it makes; None for research
    # The routing decision's fields the request may set for its writer, copied onto its step.
    options: tuple[str, ...] = ()
    templates: tuple[str, ...] = ()  # the templates its writer follows
    length_unit: LengthUnit | None = None  # the unit a length for it is asked in


FORMATS: dict[str, FormatSpec] = {
    "research": FormatSpec("research", "research", None),
    "blog": FormatSpec(
        "blog_writer",
        "blog",
        "blog",
        options=("target_keyword", "length"),
        templates=get_args(BlogTemplate),
        length_unit="words",
    ),
    "linkedin_post": FormatSpec(
        "linkedin_writer",
        "LinkedIn post",
        "linkedin_post",
        options=("length", "source_piece"),
        templates=get_args(LinkedInTemplate),
        length_unit="characters",
    ),
    "image": FormatSpec("image_generator", "image", "image"),
}
PIECE_FORMATS = {kind: spec for spec in FORMATS.values() if (kind := spec.kind)}

# "research" alone means a research-only request.
Format = Literal[tuple(FORMATS)]
PieceKind = Literal[tuple(PIECE_FORMATS)]


def format_of(agent: AgentName) -> str | None:
    """The format an agent makes, or None for the Content Strategist."""
    return next((f for f, spec in FORMATS.items() if spec.agent == agent), None)


def step_label(agent: AgentName) -> str:
    fmt = format_of(agent)
    return FORMATS[fmt].label if fmt else "brief"
