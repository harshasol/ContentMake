"""The Research tab (#65), driven through Streamlit's AppTest."""

from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from contentblitz.workflow import graph

APP = str(Path(__file__).parents[2] / "src/contentblitz/web_app/streamlit_app.py")

EMPTY = "No research yet for this Topic. Ask for some in the Chat tab"

BRIEF = {
    "summary": "AI helps radiologists.",
    "key_findings": [{"statement": "Reads scans faster.", "source_numbers": [1]}],
    "statistics": [],
    "angles": [{"title": "Speed", "hook": "Faster reads"}],
    "target_keyword": "ai radiology",
    "secondary_keywords": ["scans"],
    "sources": [
        {
            "number": 1,
            "title": "AI in radiology",
            "url": "https://example.com/ai",
            "site": "example.com",
            "retrieved_at": "2026-10-28T09:00:00+00:00",
        }
    ],
    "created_at": "2026-10-28T09:05:00+00:00",
    "model": "gpt-4o",
    "research_gathered_at": "2026-10-28T09:00:00+00:00",
}


def _app(monkeypatch, brief):
    def handler(state):
        topic = {"name": "AI in radiology", "research": None, "brief": brief, "piece_ids": []}
        return {
            "topics": {"topic-1": topic},
            "active_topic_id": "topic-1",
            "messages": [AIMessage(content="done")],
        }

    st.cache_resource.clear()
    monkeypatch.setattr(graph, "query_handler", handler)
    return AppTest.from_file(APP, default_timeout=30).run()


def _research_tab(at):
    return next(t for t in at.tabs if t.label == "Research")


def test_without_research_the_tab_suggests_asking_for_some():
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    [info] = _research_tab(at).info
    assert EMPTY in info.value


def test_a_topic_without_a_brief_says_there_is_no_research(monkeypatch):
    at = _app(monkeypatch, None)
    at.chat_input[0].set_value("hi").run()

    assert EMPTY in _research_tab(at).info[0].value


def test_after_a_research_run_the_tab_shows_the_report_sources_and_date(monkeypatch):
    at = _app(monkeypatch, BRIEF)
    at.chat_input[0].set_value("research AI in radiology").run()

    assert not at.exception
    tab = _research_tab(at)
    text = "\n".join(m.value for m in tab.markdown)
    assert "# AI in radiology" in text
    assert "AI helps radiologists." in text
    assert "[[1]](https://example.com/ai)" in text
    assert "Research gathered on 28 October 2026" in text
    [table] = tab.dataframe
    row = table.value.iloc[0]
    assert (row["#"], row["Title"], row["Site"]) == (1, "AI in radiology", "example.com")
    assert row["Retrieved at"] == "28 October 2026"
    assert row["Link"] == "https://example.com/ai"
