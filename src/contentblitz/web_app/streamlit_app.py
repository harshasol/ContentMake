"""ContentBlitz web UI. Run with: streamlit run src/contentblitz/web_app/streamlit_app.py"""

import streamlit as st

from contentblitz.web_app.brand_profiles_tab import brand_profiles_tab
from contentblitz.web_app.content_dashboard_tab import content_dashboard_tab
from contentblitz.web_app.image_view import show_image
from contentblitz.web_app.research_tab import research_tab
from contentblitz.web_app.settings_tab import settings_tab
from contentblitz.web_app.usage_tab import usage_tab
from contentblitz.workflow.brand_profiles import DEFAULT_PROFILE_ID, BrandProfileStore
from contentblitz.workflow.graph import build_graph
from contentblitz.workflow.workspace import ChatMessage, Workspace, new_thread_id

st.set_page_config(page_title="ContentBlitz", page_icon="⚡", layout="wide")


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


NEW_TOPIC = "New Topic"


def topic_selector() -> None:
    """The sidebar's Topic selector: the conversation's Topics plus "New Topic" (#66).

    Choosing one sets the active Topic through the workspace in an `on_change` callback, which
    runs before the rest of the script, so the Refresh button and the tabs see it on the same
    rerun. The widget is reset to the active Topic on every run, because a chat request can
    change it.
    """
    names = dict(workspace.topics())
    active = workspace.active_topic_id()
    st.session_state.topic_selector = active if active in names else NEW_TOPIC
    st.selectbox(
        "Topic",
        [NEW_TOPIC, *names],
        format_func=lambda topic_id: names.get(topic_id, NEW_TOPIC),
        key="topic_selector",
        on_change=lambda: workspace.select_topic(
            None
            if st.session_state.topic_selector == NEW_TOPIC
            else st.session_state.topic_selector
        ),
    )


def show_message(message: ChatMessage) -> None:
    """A chat message; an image message also shows the image from its file (provider URLs
    expire), a caption, and the optimised prompt."""
    role, text, image = message.role, message.text, message.image
    with st.chat_message(role):
        st.markdown(text)
        if image is None:
            return
        show_image(image)
        with st.expander("Optimised prompt"):
            st.code(image["prompt"], language=None, wrap_lines=True)


def show_run_caption(thread_id: str) -> None:
    """The last run's cost and tokens, under the chat. It is not a chat message."""
    caption = st.session_state.get("run_caption")
    if caption and caption[0] == thread_id:
        st.caption(caption[1])


def remember_run_caption(workspace: Workspace, run_id: str) -> None:
    st.session_state.run_caption = (workspace.thread_id, workspace.run_caption(run_id))


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
    topic_selector()
    refresh_slot = st.container()  # filled after the chat runs, so it sees a new Topic

chat_tab, dashboard_tab, research_pane, brands_tab, usage_pane, settings_pane = st.tabs(
    ["Chat", "Content Dashboard", "Research", "Brand Profiles", "Usage", "Settings"]
)

with chat_tab:
    for message in workspace.messages():
        show_message(message)

    if prompt := st.chat_input("What would you like to create?"):
        shown, topics_before = len(workspace.messages()), workspace.topics()
        show_message(ChatMessage("user", prompt))
        with st.spinner("Thinking..."):
            run_id = workspace.send(prompt, st.session_state.brand_profile_id)
        remember_run_caption(workspace, run_id)
        for message in workspace.messages()[shown + 1 :]:
            show_message(message)
        if workspace.topics() != topics_before:
            st.rerun()  # the sidebar selector was drawn before this run made a Topic
    show_run_caption(workspace.thread_id)

with dashboard_tab:
    content_dashboard_tab(workspace)

with research_pane:
    research_tab(workspace)

with brands_tab:
    brand_profiles_tab()

with usage_pane:
    usage_tab(workspace)

with settings_pane:
    settings_tab(workspace)

with refresh_slot:
    if st.button("Refresh research", disabled=not workspace.has_active_topic()):
        with st.spinner("Refreshing research..."):
            run_id = workspace.refresh_research(st.session_state.brand_profile_id)
        remember_run_caption(workspace, run_id)
        st.rerun()  # the Chat tab shows the run's messages like any other run
