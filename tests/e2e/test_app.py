from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from contentblitz.agents import query_handler as qh
from contentblitz.core.config import Settings
from contentblitz.workflow import graph

APP = str(Path(__file__).parents[2] / "src/contentblitz/web_app/streamlit_app.py")


def chat_messages(at):
    return [(m.name, m.markdown[0].value) for m in at.chat_message]


def test_chat_round_trip(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key=None))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Chat",
        "Content Dashboard",
        "Research",
        "Brand Profiles",
        "Usage",
    ]

    at.chat_input[0].set_value("hello").run()
    assert at.chat_message[-1].markdown[0].value.endswith("hello")


def test_chat_shows_every_assistant_message_from_a_run_in_order(monkeypatch):
    def two_step_handler(state):
        n = len(state["messages"])
        return {"messages": [AIMessage(content=f"plan {n}"), AIMessage(content=f"step {n}")]}

    st.cache_resource.clear()
    monkeypatch.setattr(graph, "query_handler", two_step_handler)
    at = AppTest.from_file(APP, default_timeout=30).run()

    at.chat_input[0].set_value("first").run()
    at.chat_input[0].set_value("second").run()

    assert chat_messages(at) == [
        ("user", "first"),
        ("assistant", "plan 1"),
        ("assistant", "step 1"),
        ("user", "second"),
        ("assistant", "plan 4"),
        ("assistant", "step 4"),
    ]
