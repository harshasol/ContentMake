from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from contentblitz.core.usage import UsageEvent, get_ledger
from contentblitz.integrations import search_cache
from contentblitz.workflow import brief_cache

APP = str(Path(__file__).parents[2] / "src/contentblitz/web_app/streamlit_app.py")


def test_the_settings_tab_clear_cache_button_empties_the_caches_and_keeps_the_ledger():
    st.cache_resource.clear()
    search_cache.put("search", "k", ["result"])
    brief_cache._cache().set("k", "brief")
    get_ledger().record(
        UsageEvent(ts="2026-11-04T10:00:00+00:00", agent="a", kind="completion", model="m")
    )
    at = AppTest.from_file(APP, default_timeout=30).run()

    at.button(key="clear_cache").click().run()

    assert not at.exception
    assert search_cache.get("search", "k") is None
    assert brief_cache._cache().get("k") is None
    assert len(get_ledger().events) == 1
    assert any("cache cleared" in s.value.lower() for s in at.success)
