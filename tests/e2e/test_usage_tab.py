from datetime import UTC, datetime, timedelta
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from contentblitz.core.usage import UsageEvent, get_ledger

APP = str(Path(__file__).parents[2] / "src/contentblitz/web_app/streamlit_app.py")


def event(**fields) -> UsageEvent:
    base = dict(
        ts="2026-11-04T10:00:00+00:00",
        agent="blog_writer",
        kind="completion",
        model="openai/gpt-4o",
        primary="openai/gpt-4o",
        tokens_in=1000,
        tokens_out=500,
        cost_usd=0.25,
        latency_s=1.5,
    )
    return UsageEvent(**{**base, **fields})


def seed_usage(conversation_id: str) -> None:
    now = datetime.now(UTC)
    ledger = get_ledger()
    ledger.record_request("old", "An older request")
    ledger.record(
        event(
            run_id="old",
            conversation_id="other",
            ts=(now - timedelta(days=3)).isoformat(),
            cost_usd=1.0,
            tokens_in=2000,
            tokens_out=0,
        )
    )
    ledger.record_request("mine", "Write about caching\nplease")
    ledger.record(
        event(run_id="mine", conversation_id=conversation_id, ts=now.isoformat(), cost_usd=0.25)
    )
    ledger.record(
        event(
            run_id="mine",
            conversation_id=conversation_id,
            ts=now.isoformat(),
            agent="researcher",
            kind="search",
            model="tavily",
            primary="brave",
            cost_usd=0.01,
            estimated=True,
            tokens_in=0,
            tokens_out=0,
            cached=True,
            failed=[{"provider": "brave", "error": "429 rate limit"}],
        )
    )


def test_the_usage_tab_shows_the_metrics_and_tables_from_the_ledger():
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()
    seed_usage(at.query_params["thread"])

    at.run()

    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["This conversation"] == "$0.260*"
    assert metrics["Today"] == "$0.260*"
    assert metrics["All time"] == "$1.260*"
    assert metrics["Cache hit rate"] == "33%"
    assert metrics["Fallbacks"] == "1"
    tables = [df.value for df in at.dataframe]
    [recent] = [t for t in tables if "Request" in t.columns]
    assert list(recent["Request"]) == ["Write about caching", "An older request"]
    assert recent["Fallback"][0] == "yes"
    [fallbacks] = [t for t in tables if "Failed" in t.columns]
    assert list(fallbacks.loc[0, ["Agent", "Failed", "Used", "Error"]]) == [
        "researcher",
        "brave",
        "tavily",
        "429 rate limit",
    ]
    [agents] = [t for t in tables if "Avg latency (s)" in t.columns and "Agent" in t.columns]
    assert set(agents["Agent"]) == {"blog_writer", "researcher"}
    [models] = [t for t in tables if "Model" in t.columns]
    assert set(models["Model"]) == {"openai/gpt-4o", "tavily"}
    [searches] = [t for t in tables if "Provider" in t.columns]
    assert list(searches["Provider"]) == ["tavily"]
    assert any("estimated" in c.value for c in at.caption)


def test_the_usage_tab_says_so_when_nothing_has_been_recorded():
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert any("No usage yet" in i.value for i in at.info)
