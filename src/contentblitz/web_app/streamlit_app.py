"""ContentBlitz web UI. Run with: streamlit run src/contentblitz/web_app/streamlit_app.py"""

from pathlib import Path

import streamlit as st

from contentblitz.web_app.brand_profiles_tab import brand_profiles_tab
from contentblitz.web_app.research_tab import research_tab as show_research_tab
from contentblitz.web_app.settings_tab import settings_tab
from contentblitz.workflow.brand_profiles import DEFAULT_PROFILE_ID, BrandProfileStore
from contentblitz.workflow.graph import build_graph
from contentblitz.workflow.image_files import image_caption
from contentblitz.workflow.workspace import ChatMessage, Workspace, new_thread_id

st.set_page_config(page_title="ContentBlitz", page_icon="⚡", layout="wide")

MISSING_IMAGE = "The image file for this message is missing, so it can't be shown."


@st.cache_resource
def get_graph():
    # One graph (and one in-memory checkpointer) per process. See docs/adr/0003.
    return build_graph()


def brand_profile_picker() -> None:
    """The sidebar's Brand Profile picker: the store's profiles, Default first (#22).

    `st.session_state.brand_profile_id` is the conversation's active profile, passed into each
    run. It changes only when the marketer picks another, so a deleted profile's id still
    reaches the workflow, which falls back to Default and says so.
    """
    profiles = {profile.id: profile for profile in BrandProfileStore().list()}
    active = st.session_state.brand_profile_id
    ids = list(profiles)
    st.selectbox(
        "Brand Profile",
        ids,
        index=ids.index(active) if active in profiles else 0,
        format_func=lambda profile_id: profiles[profile_id].name,
        key="brand_profile_picker",
        on_change=lambda: st.session_state.update(
            brand_profile_id=st.session_state.brand_profile_picker
        ),
    )
    shown = profiles.get(active, profiles[DEFAULT_PROFILE_ID])
    st.markdown(f"Writing as **{shown.name}**")


def show_message(message: ChatMessage) -> None:
    """A chat message; an image message also shows the image from its file (provider URLs
    expire), a caption, and the optimised prompt."""
    role, text, image = message.role, message.text, message.image
    with st.chat_message(role):
        st.markdown(text)
        if image is None:
            return
        path = Path(image["file_path"])
        if path.is_file():
            st.image(str(path), caption=image_caption(image))
        else:
            st.warning(MISSING_IMAGE)
        with st.expander("Optimised prompt"):
            st.code(image["prompt"], language=None, wrap_lines=True)


def connect_workspace() -> Workspace:
    """The conversation named by the URL's `thread` parameter; a URL without one starts a new
    conversation and writes its id to the URL, so a reload reconnects to it (ADR 0003)."""
    thread_id = st.query_params.get("thread")
    if not thread_id:
        thread_id = new_thread_id()
        st.query_params["thread"] = thread_id
        st.session_state.brand_profile_id = DEFAULT_PROFILE_ID  # a new conversation uses Default
    st.session_state.setdefault("brand_profile_id", DEFAULT_PROFILE_ID)
    return Workspace(get_graph(), thread_id)


workspace = connect_workspace()

with st.sidebar:
    st.title("⚡ ContentBlitz")
    brand_profile_picker()
    st.selectbox("Topic", ["New Topic"], disabled=True)
    st.button("Refresh research", disabled=True)

chat_tab, dashboard_tab, research_tab, brands_tab, usage_tab, settings_pane = st.tabs(
    ["Chat", "Content Dashboard", "Research", "Brand Profiles", "Usage", "Settings"]
)

with chat_tab:
    for message in workspace.messages():
        show_message(message)

    if prompt := st.chat_input("What would you like to create?"):
        shown = len(workspace.messages())
        show_message(ChatMessage("user", prompt))
        with st.spinner("Thinking..."):
            workspace.send(prompt, st.session_state.brand_profile_id)
        for message in workspace.messages()[shown + 1 :]:
            show_message(message)

with dashboard_tab:
    st.info("Content Pieces will appear here, grouped by Topic.")

with research_tab:
    show_research_tab(workspace)

with brands_tab:
    brand_profiles_tab()

with usage_tab:
    st.info("Token usage, cost and fallback events per agent will appear here.")

with settings_pane:
    settings_tab(workspace)
