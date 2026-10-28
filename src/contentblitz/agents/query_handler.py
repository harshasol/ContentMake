"""Query Handler: makes a routing decision for each message and acts on it.

Chat and unclear messages get a reply. New content becomes a plan the dispatcher runs.
Refine requests become a one-step plan to the writer of the target Content Piece.
"""

from langchain_core.messages import AIMessage, AnyMessage

from contentblitz.core.config import get_settings
from contentblitz.integrations import llm
from contentblitz.workflow.plan import WRITERS, build_plan, describe
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


SAME_SUBJECT_GUIDANCE = (
    "For new_content, set same_subject to true only if the request is about the active Topic "
    "itself; a different or narrower subject is false, and the user will be asked."
)


def _topic_question_context(state: ContentBlitzState) -> list[dict]:
    """Tells the router how to judge the subject, and about any pending Topic question."""
    if (state.get("active_topic_id") or "") not in (state.get("topics") or {}):
        return []
    texts = [SAME_SUBJECT_GUIDANCE]
    pending = state.get("pending_topic_question")
    if pending:
        texts.append(
            f"Pending Topic question: you asked whether to start a new Topic *{pending['name']}* "
            "or continue with the active Topic. Set topic_answer to 'new' or 'continue' if the "
            "latest message answers it; otherwise leave topic_answer empty and route it normally."
        )
    return [{"role": "system", "content": t} for t in texts]


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
    """A reply that runs no steps. Every run starts with a fresh plan and no pending question."""
    return {
        "messages": [AIMessage(content=text)],
        "plan": [],
        "completed_steps": [],
        "pending_topic_question": None,
        **update,
    }


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


def _ask_topic_question(
    state: ContentBlitzState, decision: RoutingDecision
) -> ContentBlitzState | None:
    """Hold the plan and ask, when new content seems to be about a different subject."""
    active = (state.get("topics") or {}).get(state.get("active_topic_id") or "")
    if not active or decision.same_subject:
        return None
    plan = build_plan(decision, active)
    if not plan:
        return None
    name = decision.subject or "Untitled Topic"
    question = (
        f"This sounds like a different subject: *{name}*. "
        f"Start a new Topic, or continue with *{active['name']}*?"
    )
    held = {"name": name, "plan": plan}
    return _reply(question, intent="new_content", pending_topic_question=held)


def _answer_topic_question(
    state: ContentBlitzState, decision: RoutingDecision
) -> ContentBlitzState | None:
    """Run the held plan on a new or the current Topic; None means handle the message normally."""
    pending = state.get("pending_topic_question")
    if not pending or not decision.topic_answer:
        return None
    topics = dict(state.get("topics") or {})
    topic_id, plan = state.get("active_topic_id"), pending["plan"]
    if decision.topic_answer == "new" or topic_id not in topics:
        topic_id = f"topic-{len(topics) + 1}"
        topics[topic_id] = {"name": pending["name"], "brief": None}
        if any(s["agent"] in WRITERS for s in plan) and plan[0]["agent"] != "research":
            plan = [{"agent": "research"}, {"agent": "content_strategist"}, *plan]
    return {
        **_reply(describe(plan, topics[topic_id]["name"]), intent="new_content", plan=plan),
        "topics": topics,
        "active_topic_id": topic_id,
    }


def query_handler(state: ContentBlitzState) -> ContentBlitzState:
    if not get_settings().openai_api_key:
        return _reply(f"(No model API key configured, echoing) {state['messages'][-1].content}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _topic_context(state)},
        {"role": "system", "content": pieces_context(state)},
        *_topic_question_context(state),
        *_conversation(state["messages"]),
    ]
    try:
        decision = llm.complete_structured("query_handler", messages, RoutingDecision)
    except llm.StructuredOutputError:
        return _reply(GENERIC_CLARIFYING_QUESTION, intent="unclear")
    except Exception:
        return _reply(APOLOGY)
    answered = _answer_topic_question(state, decision)
    if answered:
        return answered
    if decision.intent == "new_content":
        return _ask_topic_question(state, decision) or _new_content(state, decision)
    if decision.intent == "refine":
        return _refine(state, decision)
    return _reply(_respond(decision), intent=decision.intent)
