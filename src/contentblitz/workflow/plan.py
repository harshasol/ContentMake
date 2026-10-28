"""Turns a routing decision into an ordered plan of agent steps. The model only names formats."""

from contentblitz.workflow.formats import PIECE_FORMATS, format_of, step_label
from contentblitz.workflow.routing import RoutingDecision
from contentblitz.workflow.state import ContentBlitzState, PlanStep, Topic

MAX_STEPS = 6
WRITERS = {"blog_writer", "linkedin_writer"}


def build_plan(decision: RoutingDecision, topic: Topic | None) -> list[PlanStep]:
    """Research first (ADR 0001) when a writer needs a Brief or research was asked for."""
    steps: list[PlanStep] = [
        {"agent": PIECE_FORMATS[f].agent, "instructions": decision.instructions}
        for f in decision.formats
        if f in PIECE_FORMATS
    ]
    has_brief = bool(topic and topic.get("brief"))
    needs_research = (
        "research" in decision.formats
        or decision.fresh_research
        or (any(s["agent"] in WRITERS for s in steps) and not has_brief)
    )
    if needs_research:
        research: PlanStep = {"agent": "research", "instructions": decision.instructions}
        steps = [research, {"agent": "content_strategist"}, *steps]
    unique: list[PlanStep] = []
    for step in steps:
        if step not in unique:
            unique.append(step)
    return unique[:MAX_STEPS]


def replan(plan: list[PlanStep], topic: Topic | None) -> list[PlanStep]:
    """Rebuild a held plan for another Topic, applying build_plan's rules afresh."""
    formats = [f for s in plan if (f := format_of(s["agent"]))]
    instructions = next((s.get("instructions") for s in plan if s.get("instructions")), None)
    decision = RoutingDecision(intent="new_content", formats=formats, instructions=instructions)
    return build_plan(decision, topic)


def describe_plan(plan: list[PlanStep], topic_name: str) -> str:
    return f"Plan for *{topic_name}*: " + " → ".join(step_label(s["agent"]) for s in plan)


def current_step(state: ContentBlitzState) -> PlanStep | None:
    """The next plan step that hasn't run yet, or None when the plan is complete."""
    plan, done = state.get("plan", []), state.get("completed_steps", [])
    return plan[len(done)] if len(done) < len(plan) else None
