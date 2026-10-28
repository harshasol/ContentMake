"""Query Handler: makes a routing decision for each message and acts on it.

Chat and unclear messages get a reply. New content becomes a plan the dispatcher runs.
Refine requests become a one-step plan to the writer of the target Content Piece.
"""

from langchain_core.messages import AIMessage, AnyMessage

from contentblitz.core.config import get_settings
from contentblitz.integrations import llm
from contentblitz.workflow.plan import build_plan, describe
from contentblitz.workflow.refine import (
    describe_refine,
    pieces_context,
    refine_step,
    resolve_target,
    which_piece,
)
from contentblitz.workflow.routing import RoutingDecision
from contentblitz.workflow.state import ContentBlitzState

HISTORY_WINDOW = 10

SYSTEM_PROMPT = """\
You are the Query Handler of ContentBlitz, a content marketing assistant that researches topics
and writes SEO blogs, LinkedIn posts and images. Read the latest message in the context of the
conversation and return a routing decision:
- new_content: the user wants research, a blog, a LinkedIn post or an image. Set subject and
  formats in the order asked (["research"] for research only).
- refine: the user wants an existing piece changed. Set instructions, set refine_target to the
  id of the Content Piece below that they mean (null if unsure), and set formats to its kind
  when they name one ("the LinkedIn post" → ["linkedin_post"]).
- chat: a question or small talk. Answer it concisely in reply.
- unclear: too vague to act on. Ask one short clarifying_question that suggests two or three
  concrete requests the user could make.
Resolve references like "that" or "it" from the conversation.
Answer questions about the research or its sources only from the active Topic's Research Brief
below, never from your own knowledge. If there is no Brief, say the Topic hasn't been researched."""


def _topic_context(state: ContentBlitzState) -> str:
    topic = (state.get("topics") or {}).get(state.get("active_topic_id") or "")
    if not topic:
        return "Active Topic: none yet."
    brief = topic.get("brief")
    if not brief:
        return f"Active Topic: {topic['name']} (no Research Brief yet)."
    findings = "\n".join(f"- {f}" for f in brief["key_findings"]) or "- none"
    sources = "\n".join(f"- {s['title']} ({s['url']})" for s in brief["sources"]) or "- none"
    return (
        f"Active Topic: {topic['name']}\nResearch Brief summary: {brief['summary']}\n"
        f"Key findings:\n{findings}\nSources:\n{sources}"
    )


GENERIC_CLARIFYING_QUESTION = (
    "Sorry, I didn't quite get that. What would you like me to do? For example: "
    '"research AI in healthcare", "write a blog about it" or "turn that into a LinkedIn post".'
)
APOLOGY = "Sorry, I couldn't reach the language model just now. Please try again in a moment."
NOTHING_TO_REFINE = (
    "There's nothing to refine yet. Ask me for a blog, a LinkedIn post or an image first."
)

_ROLES = {"human": "user", "ai": "assistant"}


def _conversation(messages: list[AnyMessage]) -> list[dict]:
    recent = [m for m in messages if m.type in _ROLES][-HISTORY_WINDOW:]
    return [{"role": _ROLES[m.type], "content": m.content} for m in recent]


def _reply(text: str, **update) -> ContentBlitzState:
    """A reply that runs no steps. Every run starts with a fresh plan."""
    return {"messages": [AIMessage(content=text)], "plan": [], "completed_steps": [], **update}


def _respond(decision: RoutingDecision) -> str:
    if decision.intent == "chat" and decision.reply:
        return decision.reply
    if decision.intent == "unclear":
        return decision.clarifying_question or GENERIC_CLARIFYING_QUESTION
    return GENERIC_CLARIFYING_QUESTION


def _new_content(state: ContentBlitzState, decision: RoutingDecision) -> ContentBlitzState:
    topics = dict(state.get("topics") or {})
    topic_id = state.get("active_topic_id")
    if topic_id not in topics:
        topic_id = f"topic-{len(topics) + 1}"
        topics[topic_id] = {"name": decision.subject or "Untitled Topic", "brief": None}
    plan = build_plan(decision, topics[topic_id])
    if not plan:
        return _reply(GENERIC_CLARIFYING_QUESTION, intent="unclear")
    return {
        **_reply(describe(plan, topics[topic_id]["name"]), intent="new_content", plan=plan),
        "topics": topics,
        "active_topic_id": topic_id,
    }


def _refine(state: ContentBlitzState, decision: RoutingDecision) -> ContentBlitzState:
    candidates = resolve_target(state, decision)
    if not candidates:
        return _reply(NOTHING_TO_REFINE, intent="refine")
    if len(candidates) > 1:
        return _reply(which_piece(state, candidates), intent="refine")
    [piece_id] = candidates
    plan = [refine_step(state, piece_id, decision.instructions)]
    return _reply(describe_refine(state, piece_id), intent="refine", plan=plan)


def query_handler(state: ContentBlitzState) -> ContentBlitzState:
    if not get_settings().openai_api_key:
        return _reply(f"(No model API key configured, echoing) {state['messages'][-1].content}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _topic_context(state)},
        {"role": "system", "content": pieces_context(state)},
        *_conversation(state["messages"]),
    ]
    try:
        decision = llm.complete_structured("query_handler", messages, RoutingDecision)
    except llm.StructuredOutputError:
        return _reply(GENERIC_CLARIFYING_QUESTION, intent="unclear")
    except Exception:
        return _reply(APOLOGY)
    if decision.intent == "new_content":
        return _new_content(state, decision)
    if decision.intent == "refine":
        return _refine(state, decision)
    return _reply(_respond(decision), intent=decision.intent)
