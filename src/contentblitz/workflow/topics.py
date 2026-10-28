"""Finding and creating Topics in the workflow state."""

from contentblitz.workflow.state import ContentBlitzState, Topic

UNTITLED_TOPIC = "Untitled Topic"


def active_topic(state: ContentBlitzState) -> Topic | None:
    return (state.get("topics") or {}).get(state.get("active_topic_id") or "")


def create_topic(state: ContentBlitzState, name: str | None) -> ContentBlitzState:
    """The state update that adds a new Topic and makes it active."""
    topics = dict(state.get("topics") or {})
    topic_id = f"topic-{len(topics) + 1}"
    topics[topic_id] = {"name": name or UNTITLED_TOPIC, "brief": None}
    return {"topics": topics, "active_topic_id": topic_id}
