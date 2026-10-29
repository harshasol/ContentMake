"""ContentBlitz web UI. Run with: streamlit run src/contentblitz/web_app/streamlit_app.py"""

import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AnyMessage, HumanMessage

from contentblitz.workflow.graph import build_graph
from contentblitz.workflow.state import ContentPiece, ImageMetadata

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


def referenced_image(message: AnyMessage, pieces: dict[str, ContentPiece]) -> ImageMetadata | None:
    """The image metadata of the Content Piece Revision a message references, if any (#21)."""
    reference = message.additional_kwargs.get("content_piece")
    if not reference:
        return None
    piece = pieces.get(reference["piece_id"])
    revisions = piece["revisions"] if piece else []
    if not 1 <= reference["revision"] <= len(revisions):
        return None
    return revisions[reference["revision"] - 1].get("image")


def image_caption(image: ImageMetadata) -> str:
    size = image["size"].replace("x", "×")
    return f"Placeholder image · {size}" if image["placeholder"] else f"{image['model']} · {size}"


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

with st.sidebar:
    st.title("⚡ ContentBlitz")
    st.selectbox("Brand Profile", ["(none yet)"], disabled=True)
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
                {"messages": [HumanMessage(content=prompt)]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
        pieces = result.get("content_pieces") or {}
        for reply in new_replies(result["messages"]):
            entry = ("assistant", reply.content, referenced_image(reply, pieces))
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
