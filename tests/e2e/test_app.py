from pathlib import Path

from streamlit.testing.v1 import AppTest

from contentblitz.agents import query_handler as qh
from contentblitz.core.config import Settings

APP = str(Path(__file__).parents[2] / "src/contentblitz/web_app/streamlit_app.py")


def test_chat_round_trip(monkeypatch):
    monkeypatch.setattr(qh, "get_settings", lambda: Settings(_env_file=None, openai_api_key=None))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Chat", "Content Dashboard", "Research", "Brand Profiles", "Usage"
    ]

    at.chat_input[0].set_value("hello").run()
    assert at.chat_message[-1].markdown[0].value.endswith("hello")
