"""ContentBlitz web UI. Run with: streamlit run src/contentblitz/web_app/streamlit_app.py"""

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from contentblitz.workflow.graph import build_graph

st.set_page_config(page_title="ContentBlitz", page_icon="⚡", layout="wide")


@st.cache_resource
def get_graph():
    # One graph (and one in-memory checkpointer) per process. See docs/adr/0003.
    return build_graph()


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
    for role, text in st.session_state.chat:
        st.chat_message(role).markdown(text)

    if prompt := st.chat_input("What would you like to create?"):
        st.session_state.chat.append(("user", prompt))
        st.chat_message("user").markdown(prompt)
        with st.spinner("Thinking..."):
            result = get_graph().invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
        reply = result["messages"][-1].content
        st.session_state.chat.append(("assistant", reply))
        st.chat_message("assistant").markdown(reply)

with dashboard_tab:
    st.info("Content Pieces will appear here, grouped by Topic.")

with research_tab:
    st.info("The active Topic's Research Brief and Sources will appear here.")

with brands_tab:
    st.info("Create and edit Brand Profiles here.")

with usage_tab:
    st.info("Token usage, cost and fallback events per agent will appear here.")
