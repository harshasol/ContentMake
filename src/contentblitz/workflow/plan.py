"""Turns a routing decision into an ordered plan of agent steps. The model only names formats."""

from typing import get_args

from contentblitz.workflow.formats import PIECE_FORMATS, format_of, step_label
from contentblitz.workflow.routing import BlogTemplate, LinkedInTemplate, RoutingDecision
from contentblitz.workflow.state import ContentBlitzState, PlanStep, Topic

MAX_STEPS = 6
WRITERS = {"blog_writer", "linkedin_writer"}
# What the request may set for a writer: they go on its step, and win over the defaults.
STEP_OPTIONS = ("target_keyword", "template", "length")
TEMPLATES = {"blog": get_args(BlogTemplate), "linkedin_post": get_args(LinkedInTemplate)}


def has_writer(steps: list[PlanStep]) -> bool:
    """Whether any of these steps is a writer's, which needs a Research Brief."""
    return any(step["agent"] in WRITERS for step in steps)


def writer_options(decision: RoutingDecision, fmt: str) -> dict:
    """The options the request sets for the writer of `fmt`, to copy onto its step.

    A blog takes the keyword and a length in words, a LinkedIn post a length in characters, and
    a template goes to the format it is a template of.
    """
    in_characters = decision.length_unit == "characters"
    options = {}
    if fmt == "blog":
        length = None if in_characters else decision.length
        options = {"target_keyword": decision.target_keyword, "length": length}
    if fmt == "linkedin_post" and in_characters:
        options = {"length": decision.length}
    if decision.template in TEMPLATES.get(fmt, ()):
        options["template"] = decision.template
    return {o: v for o, v in options.items() if v is not None}


def _writer_step(decision: RoutingDecision, fmt: str) -> PlanStep:
    step: PlanStep = {"agent": PIECE_FORMATS[fmt].agent, "instructions": decision.instructions}
    return step | writer_options(decision, fmt)


def build_plan(decision: RoutingDecision, topic: Topic | None) -> list[PlanStep]:
    """Research first (ADR 0001) when a writer needs a Brief or research was asked for.

    A writer on a Topic with research but no Brief gets only the Content Strategist first, so a
    failed Brief is rebuilt without searching the web again.
    """
    steps = [_writer_step(decision, f) for f in decision.formats if f in PIECE_FORMATS]
    has_brief = bool(topic and topic.get("brief"))
    has_research = bool(topic and topic.get("research"))
    needs_brief = has_writer(steps) and not has_brief
    needs_research = (
        "research" in decision.formats
        or decision.fresh_research
        or (needs_brief and not has_research)
    )
    brief: PlanStep = {"agent": "content_strategist", "instructions": decision.instructions}
    if needs_research:
        research: PlanStep = {"agent": "research", "instructions": decision.instructions}
        if decision.fresh_research:
            research["fresh"] = True
        steps = [research, brief, *steps]
    elif needs_brief:
        steps = [brief, *steps]
    unique: list[PlanStep] = []
    for step in steps:
        if step not in unique:
            unique.append(step)
    return unique[:MAX_STEPS]


def replan(plan: list[PlanStep], topic: Topic | None) -> list[PlanStep]:
    """Rebuild a held plan for another Topic, applying build_plan's rules afresh.

    Instructions, the fresh research flag and each writer's options carry over to the new plan.
    """
    formats = [f for s in plan if (f := format_of(s["agent"]))]
    instructions = next((s.get("instructions") for s in plan if s.get("instructions")), None)
    fresh = any(s.get("fresh") for s in plan)
    decision = RoutingDecision(
        intent="new_content", formats=formats, instructions=instructions, fresh_research=fresh
    )
    options = {s["agent"]: {o: s[o] for o in STEP_OPTIONS if o in s} for s in plan}
    return [step | options.get(step["agent"], {}) for step in build_plan(decision, topic)]


def describe_plan(plan: list[PlanStep], topic_name: str) -> str:
    return f"Plan for *{topic_name}*: " + " → ".join(step_label(s["agent"]) for s in plan)


def current_step(state: ContentBlitzState) -> PlanStep | None:
    """The next plan step that hasn't run yet, or None when the plan is complete."""
    plan, done = state.get("plan", []), state.get("completed_steps", [])
    return plan[len(done)] if len(done) < len(plan) else None


def writers_follow(state: ContentBlitzState) -> bool:
    """Whether a writer step comes after the current one in the plan."""
    plan, done = state.get("plan", []), state.get("completed_steps", [])
    return has_writer(plan[len(done) + 1 :])
