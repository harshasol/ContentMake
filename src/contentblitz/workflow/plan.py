"""Turns a routing decision into an ordered plan of agent steps. The model only names formats."""

from contentblitz.workflow.routing import RoutingDecision
from contentblitz.workflow.state import AgentName, ContentBlitzState, PlanStep, Topic

MAX_STEPS = 6

FORMAT_AGENTS: dict[str, AgentName] = {
    "blog": "blog_writer",
    "linkedin_post": "linkedin_writer",
    "image": "image_generator",
}
WRITERS = {"blog_writer", "linkedin_writer"}
STEP_LABELS: dict[AgentName, str] = {
    "research": "research",
    "content_strategist": "brief",
    "blog_writer": "blog",
    "linkedin_writer": "LinkedIn post",
    "image_generator": "image",
}


def build_plan(decision: RoutingDecision, topic: Topic | None) -> list[PlanStep]:
    """Research first (ADR 0001) when a writer needs a Brief or research was asked for."""
    steps: list[PlanStep] = [
        {"agent": FORMAT_AGENTS[f], "instructions": decision.instructions}
        for f in decision.formats
        if f in FORMAT_AGENTS
    ]
    has_brief = bool(topic and topic.get("brief"))
    needs_research = (
        "research" in decision.formats
        or decision.fresh_research
        or (any(s["agent"] in WRITERS for s in steps) and not has_brief)
    )
    if needs_research:
        steps = [{"agent": "research"}, {"agent": "content_strategist"}, *steps]
    unique: list[PlanStep] = []
    for step in steps:
        if step not in unique:
            unique.append(step)
    return unique[:MAX_STEPS]


def describe(plan: list[PlanStep], topic_name: str) -> str:
    return f"Plan for *{topic_name}*: " + " → ".join(STEP_LABELS[s["agent"]] for s in plan)


def current_step(state: ContentBlitzState) -> PlanStep | None:
    """The next plan step that hasn't run yet, or None when the plan is complete."""
    plan, done = state.get("plan", []), state.get("completed_steps", [])
    return plan[len(done)] if len(done) < len(plan) else None
