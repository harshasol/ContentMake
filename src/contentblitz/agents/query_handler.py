"""Query Handler: classifies each message and routes it to the agent that should handle it.

Skeleton behaviour: every message is answered directly. Plan-based routing comes next.
"""

from langchain_core.messages import AIMessage

from contentblitz.core.config import get_settings
from contentblitz.integrations import llm
from contentblitz.workflow.state import ContentBlitzState

SYSTEM_PROMPT = (
    "You are ContentBlitz, a content marketing assistant that researches topics and "
    "writes SEO blogs, LinkedIn posts and images. Be concise."
)


def query_handler(state: ContentBlitzState) -> ContentBlitzState:
    last = state["messages"][-1].content
    if not get_settings().openai_api_key:
        reply = f"(No model API key configured, echoing) {last}"
    else:
        reply = llm.complete(
            "query_handler",
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": last}],
        )
    return {"messages": [AIMessage(content=reply)], "route": "respond"}
