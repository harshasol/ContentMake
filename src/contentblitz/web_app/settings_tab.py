"""The Settings tab (#70): read-only configuration and the New conversation button.

Changing models stays a config edit (ADR 0002). Key values are never shown, only set or not.
"""

import streamlit as st

from contentblitz.core.config import get_settings
from contentblitz.workflow.brand_profiles import DEFAULT_PROFILE_ID
from contentblitz.workflow.workspace import Workspace, new_thread_id


def _start_new_conversation() -> None:
    st.query_params["thread"] = new_thread_id()
    st.session_state.brand_profile_id = DEFAULT_PROFILE_ID


def settings_tab(workspace: Workspace) -> None:
    st.button("New conversation", key="new_conversation", on_click=_start_new_conversation)
    view = workspace.configuration(get_settings())

    st.subheader("Providers")
    for name, is_set in view.provider_keys.items():
        st.markdown(f"{'✓' if is_set else '✗'} {name}")

    st.subheader("Models by agent")
    st.table(
        {
            "Agent": list(view.agents),
            "Model": [model for model, _ in view.agents.values()],
            "Fallbacks": [", ".join(f) or "none" for _, f in view.agents.values()],
        }
    )

    st.subheader("Search provider order")
    st.markdown(" → ".join(view.search_providers))

    st.subheader("Image models")
    st.markdown(" → ".join(view.image_models))

    st.subheader("Quality thresholds")
    st.table({"Rule": list(view.thresholds), "Value": list(view.thresholds.values())})
