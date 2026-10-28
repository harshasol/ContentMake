"""The research report: how a Research Brief is shown to the marketer. See spec #6.

Rendering is deterministic Markdown with no model call, so showing the report is instant
and free. The chat, "show me the research" and the Research Panel all use it.
"""

from datetime import datetime

from contentblitz.workflow.state import ResearchBrief, Source


def _date(iso: str) -> str:
    when = datetime.fromisoformat(iso)
    return f"{when.day} {when:%B %Y}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _citations(numbers: list[int], sources: dict[int, Source]) -> str:
    """`[n]` links to each cited Source's page."""
    return " ".join(f"[[{n}]]({sources[n]['url']})" for n in numbers)


def _statistics(brief: ResearchBrief, sources: dict[int, Source]) -> str:
    if not brief["statistics"]:
        return "No statistics were found in the Sources."
    rows = [
        f"| {_cell(s['figure'])} | {_cell(s['description'])} | "
        f"{_citations(s['source_numbers'], sources)} |"
        for s in brief["statistics"]
    ]
    return "\n".join(["| Figure | What it measures | Sources |", "| --- | --- | --- |", *rows])


def _plural(count: int, word: str) -> str:
    return f"{count} {word if count == 1 else word + 's'}"


def summary_line(brief: ResearchBrief) -> str:
    """The one line posted instead of the report when writers follow in the same plan."""
    counts = ", ".join(
        [
            _plural(len(brief["key_findings"]), "finding"),
            _plural(len(brief["statistics"]), "statistic"),
            _plural(len(brief["sources"]), "Source"),
        ]
    )
    return f"Research Brief ready: {counts}. Ask 'show me the research' to read it."


def render_report(topic_name: str, brief: ResearchBrief) -> str:
    """The full research report for a Topic's Research Brief, in Markdown."""
    sources = {s["number"]: s for s in brief["sources"]}
    findings = "\n".join(
        f"- {f['statement']} {_citations(f['source_numbers'], sources)}"
        for f in brief["key_findings"]
    )
    angles = "\n".join(f"- **{a['title']}**: {a['hook']}" for a in brief["angles"])
    source_list = "\n".join(
        f"{s['number']}. [{s['title']}]({s['url']}), {s['site']}, "
        f"retrieved {_date(s['retrieved_at'])}"
        for s in brief["sources"]
    )
    sections = [
        f"# {topic_name}",
        f"*Research gathered on {_date(brief['research_gathered_at'])}.*",
        f"## Summary\n\n{brief['summary']}",
        f"## Key findings\n\n{findings}",
        f"## Statistics\n\n{_statistics(brief, sources)}",
        f"## Content angles\n\n{angles}",
        f"## Keywords\n\n**Target keyword:** {brief['target_keyword']}\n\n"
        f"**Secondary keywords:** {', '.join(brief['secondary_keywords'])}",
        f"## Sources\n\n{source_list}",
    ]
    return "\n\n".join(sections)
