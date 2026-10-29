"""The Usage tab (#73): cost, tokens, cache hits, fallbacks and searches from the usage ledger."""

from datetime import datetime

import pandas as pd
import streamlit as st

from contentblitz.core.usage import Breakdown, Totals
from contentblitz.workflow.workspace import UsageView, Workspace

NO_USAGE = "No usage yet. Ask for something in the Chat tab and its cost will show up here."
ESTIMATE_NOTE = (
    "* estimated: the model's price was unknown, so the cost uses a configured estimate."
)


def cost_text(cost_usd: float, estimated: bool) -> str:
    return f"${cost_usd:.3f}{'*' if estimated else ''}"


def totals_text(totals: Totals) -> str:
    return cost_text(totals.cost_usd, totals.estimated)


def when(ts: str) -> str:
    return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")


def breakdown_table(rows: list[Breakdown], label: str, latency: bool = False) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            label: r.name,
            "Calls": r.calls,
            "Tokens": r.tokens,
            "Cost": cost_text(r.cost_usd, r.estimated),
            "Avg latency (s)": round(r.avg_latency_s, 2),
            "Cache hits": r.cached,
        }
        for r in rows
    )
    return table if latency else table.drop(columns=["Avg latency (s)", "Cache hits"])


def runs_table(view: UsageView) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": when(r.ts),
            "Request": r.request or "",
            "Cost": cost_text(r.cost_usd, r.estimated),
            "Tokens": r.tokens,
            "Duration (s)": round(r.duration_s, 1),
            "Fallback": "yes" if r.fallback else "",
        }
        for r in view.recent_runs
    )


def fallbacks_table(view: UsageView) -> pd.DataFrame:
    return pd.DataFrame(
        {"Time": when(f.ts), "Agent": f.agent, "Failed": f.failed, "Used": f.used, "Error": f.error}
        for f in view.fallback_events
    )


def usage_tab(workspace: Workspace) -> None:
    view = workspace.usage()
    if not view.all_time.calls:
        st.info(NO_USAGE)
        return
    columns = st.columns(5)
    columns[0].metric("This conversation", totals_text(view.conversation))
    columns[1].metric("Today", totals_text(view.today))
    columns[2].metric("All time", totals_text(view.all_time))
    rate = view.cache_hit_rate
    columns[3].metric("Cache hit rate", "n/a" if rate is None else f"{rate:.0%}")
    columns[4].metric("Fallbacks", str(view.fallbacks))
    columns = st.columns(3)
    for column, label, totals in zip(
        columns,
        ("This conversation", "Today", "All time"),
        (view.conversation, view.today, view.all_time),
        strict=True,
    ):
        column.caption(f"{label}: {totals.tokens:,} tokens in {totals.calls} calls")
    if any(r.estimated for r in view.by_agent):
        st.caption(ESTIMATE_NOTE)

    st.subheader("By agent")
    st.dataframe(breakdown_table(view.by_agent, "Agent", latency=True), hide_index=True)
    st.subheader("By model")
    st.dataframe(breakdown_table(view.by_model, "Model"), hide_index=True)
    st.subheader("Recent runs")
    st.dataframe(runs_table(view), hide_index=True)
    st.subheader("Fallback events")
    if view.fallback_events:
        st.dataframe(fallbacks_table(view), hide_index=True)
    else:
        st.caption("No fallbacks.")
    st.subheader("Searches per provider")
    if view.searches:
        searches = breakdown_table(view.searches, "Provider")
        st.dataframe(searches[["Provider", "Calls", "Cost"]], hide_index=True)
    else:
        st.caption("No searches yet.")
