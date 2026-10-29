"""The Research tab (#65): the active Topic's research report and its Sources table."""

import pandas as pd
import streamlit as st

from contentblitz.web_app.export_buttons import export_buttons
from contentblitz.workflow.report import display_date
from contentblitz.workflow.state import Source
from contentblitz.workflow.workspace import Workspace

NO_RESEARCH = (
    "No research yet for this Topic. Ask for some in the Chat tab, "
    "for example *research this Topic*."
)


def sources_table(sources: list[Source]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "#": s["number"],
            "Title": s["title"],
            "Site": s["site"],
            "Retrieved at": display_date(s["retrieved_at"]),
            "Link": s["url"],
        }
        for s in sources
    )


def research_tab(workspace: Workspace) -> None:
    research = workspace.research()
    if research is None:
        st.info(NO_RESEARCH)
        return
    st.markdown(research.report)
    export_buttons(workspace.research_exports(), key="research_export")
    st.subheader("Sources")
    st.dataframe(
        sources_table(research.brief["sources"]),
        hide_index=True,
        column_config={"Link": st.column_config.LinkColumn("Link")},
    )
