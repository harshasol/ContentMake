"""ContentBlitz web UI. Run with: streamlit run src/contentblitz/web_app/streamlit_app.py"""

import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AnyMessage, HumanMessage

from contentblitz.workflow.brand_profiles import DEFAULT_PROFILE_ID, BrandProfileStore
from contentblitz.workflow.graph import build_graph
from contentblitz.workflow.image_files import image_caption
from contentblitz.workflow.state import ContentBlitzState, ImageMetadata
from contentblitz.workflow.topics import find_revision

st.set_page_config(page_title="ContentBlitz", page_icon="⚡", layout="wide")

MISSING_IMAGE = "The image file for this message is missing, so it can't be shown."


@st.cache_resource
def get_graph():
    # One graph (and one in-memory checkpointer) per process. See docs/adr/0003.
    return build_graph()


def new_replies(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The assistant messages a run added after the user's latest message, in order."""
    last_user = max(i for i, m in enumerate(messages) if m.type == "human")
    return [m for m in messages[last_user + 1 :] if m.type == "ai"]


def referenced_image(message: AnyMessage, state: ContentBlitzState) -> ImageMetadata | None:
    """The image metadata of the Content Piece Revision a message references, if any (#21)."""
    reference = message.additional_kwargs.get("content_piece")
    if not reference:
        return None
    revision = find_revision(state, reference["piece_id"], reference["revision"])
    return revision.get("image") if revision else None


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


def show_message(role: str, text: str, image: ImageMetadata | None = None) -> None:
    """A chat message; an image message also shows the image from its file (provider URLs
    expire), a caption, and the optimised prompt."""
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


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []
    st.session_state.brand_profile_id = DEFAULT_PROFILE_ID  # a new conversation uses Default

with st.sidebar:
    st.title("⚡ ContentBlitz")
    brand_profile_picker()
    st.selectbox("Topic", ["New Topic"], disabled=True)
    st.button("Refresh research", disabled=True)

chat_tab, dashboard_tab, research_tab, brands_tab, usage_tab = st.tabs(
    ["Chat", "Content Dashboard", "Research", "Brand Profiles", "Usage"]
)

with chat_tab:
    for entry in st.session_state.chat:
        show_message(*entry)

    if prompt := st.chat_input("What would you like to create?"):
        st.session_state.chat.append(("user", prompt, None))
        show_message("user", prompt)
        with st.spinner("Thinking..."):
            result = get_graph().invoke(
                {
                    "messages": [HumanMessage(content=prompt)],
                    "brand_profile_id": st.session_state.brand_profile_id,
                },
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
        for reply in new_replies(result["messages"]):
            entry = ("assistant", reply.content, referenced_image(reply, result))
            st.session_state.chat.append(entry)
            show_message(*entry)

with dashboard_tab:
    st.info("Content Pieces will appear here, grouped by Topic.")

with research_tab:
    st.info("The active Topic's Research Brief and Sources will appear here.")

with brands_tab:
    st.info("Create and edit Brand Profiles here.")

with usage_tab:
    st.info("Token usage, cost and fallback events per agent will appear here.")
