"""The formats a user can ask for: each one's agent, label and Content Piece kind."""

from typing import Literal, NamedTuple

AgentName = Literal[
    "research",
    "content_strategist",
    "blog_writer",
    "linkedin_writer",
    "image_generator",
]


class FormatSpec(NamedTuple):
    agent: AgentName
    label: str
    kind: str | None  # the Content Piece kind it makes; None for research


FORMATS: dict[str, FormatSpec] = {
    "research": FormatSpec("research", "research", None),
    "blog": FormatSpec("blog_writer", "blog", "blog"),
    "linkedin_post": FormatSpec("linkedin_writer", "LinkedIn post", "linkedin_post"),
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
