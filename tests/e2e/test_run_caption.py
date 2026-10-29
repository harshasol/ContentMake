from pathlib import Path
from types import SimpleNamespace

import streamlit as st
from streamlit.testing.v1 import AppTest

from contentblitz.agents import query_handler as qh
from contentblitz.core.config import Settings
from contentblitz.core.usage import get_ledger
from contentblitz.integrations import llm

APP = str(Path(__file__).parents[2] / "src/contentblitz/web_app/streamlit_app.py")


def fake_router(monkeypatch):
    """The Query Handler's model call, faked in LiteLLM itself: a chat reply, with usage."""
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key="x"))

    def completion(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"intent": "chat", "reply": "Hello there"}')
                )
            ],
            model=kwargs["model"],
            usage=SimpleNamespace(prompt_tokens=8000, completion_tokens=4400),
            _hidden_params={"response_cost": 0.043},
        )

    monkeypatch.setattr(llm.litellm, "completion", completion)


def test_after_a_run_the_chat_tab_shows_its_cost_and_tokens_as_a_caption(monkeypatch):
    fake_router(monkeypatch)
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()

    at.chat_input[0].set_value("hi").run()

    assert not at.exception
    assert [c.value for c in at.tabs[0].caption] == ["This run: $0.043 · 12.4k tokens"]
    # The caption is not a chat message, so it is never sent to the models.
    assert [(m.name, m.markdown[0].value) for m in at.chat_message] == [
        ("user", "hi"),
        ("assistant", "Hello there"),
    ]
    [event] = get_ledger().events
    assert event.run_id and event.conversation_id == at.query_params["thread"]


def test_a_second_run_gets_its_own_caption(monkeypatch):
    fake_router(monkeypatch)
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()

    at.chat_input[0].set_value("one").run()
    at.chat_input[0].set_value("two").run()

    assert [c.value for c in at.tabs[0].caption] == ["This run: $0.043 · 12.4k tokens"]
    assert len({e.run_id for e in get_ledger().events}) == 2
