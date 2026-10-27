"""Shared LangGraph state. Grows as agents are implemented (Topics, Brief, Content Pieces)."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal[
    "respond",
    "research",
    "content_strategist",
    "blog_writer",
    "linkedin_writer",
    "image_generator",
]


class ContentBlitzState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    route: Route
