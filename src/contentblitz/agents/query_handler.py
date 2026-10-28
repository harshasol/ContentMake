"""Query Handler: makes a routing decision for each message and acts on it.

Handles chat and unclear messages. Content and refine requests get a placeholder until plans land.
"""

from langchain_core.messages import AIMessage, AnyMessage

from contentblitz.core.config import get_settings
from contentblitz.integrations import llm
from contentblitz.workflow.routing import RoutingDecision
from contentblitz.workflow.state import ContentBlitzState

HISTORY_WINDOW = 10

SYSTEM_PROMPT = """\
You are the Query Handler of ContentBlitz, a content marketing assistant that researches topics
and writes SEO blogs, LinkedIn posts and images. Read the latest message in the context of the
conversation and return a routing decision:
- new_content: the user wants research, a blog, a LinkedIn post or an image. Set subject and
  formats in the order asked (["research"] for research only).
- refine: the user wants an existing piece changed. Set instructions.
- chat: a question or small talk. Answer it concisely in reply.
- unclear: too vague to act on. Ask one short clarifying_question that suggests two or three
  concrete requests the user could make.
Resolve references like "that" or "it" from the conversation."""

GENERIC_CLARIFYING_QUESTION = (
    "Sorry, I didn't quite get that. What would you like me to do? For example: "
    '"research AI in healthcare", "write a blog about it" or "turn that into a LinkedIn post".'
)
APOLOGY = "Sorry, I couldn't reach the language model just now. Please try again in a moment."
NOT_SUPPORTED_YET = (
    "Creating and refining content is not supported yet. For now I can answer questions about "
    "what I do."
)

_ROLES = {"human": "user", "ai": "assistant"}


def _conversation(messages: list[AnyMessage]) -> list[dict]:
    recent = [m for m in messages if m.type in _ROLES][-HISTORY_WINDOW:]
    return [{"role": _ROLES[m.type], "content": m.content} for m in recent]


def _respond(decision: RoutingDecision) -> str:
    if decision.intent == "chat" and decision.reply:
        return decision.reply
    if decision.intent == "unclear":
        return decision.clarifying_question or GENERIC_CLARIFYING_QUESTION
    if decision.intent in ("new_content", "refine"):
        return NOT_SUPPORTED_YET
    return GENERIC_CLARIFYING_QUESTION


def query_handler(state: ContentBlitzState) -> ContentBlitzState:
    if not get_settings().openai_api_key:
        last = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"(No model API key configured, echoing) {last}")]}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_conversation(state["messages"])]
    try:
        decision = llm.complete_structured("query_handler", messages, RoutingDecision)
    except llm.StructuredOutputError:
        return {"messages": [AIMessage(content=GENERIC_CLARIFYING_QUESTION)], "intent": "unclear"}
    except Exception:
        return {"messages": [AIMessage(content=APOLOGY)]}
    return {"messages": [AIMessage(content=_respond(decision))], "intent": decision.intent}
