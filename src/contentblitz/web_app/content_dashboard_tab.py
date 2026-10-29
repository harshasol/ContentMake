"""The Content Dashboard tab: the Content Pieces of the active Topic or of all Topics (#60)."""

import pandas as pd
import streamlit as st

from contentblitz.web_app.export_buttons import export_buttons
from contentblitz.web_app.piece_detail import piece_detail, revision_history
from contentblitz.workflow.formats import PIECE_FORMATS
from contentblitz.workflow.workspace import PieceRow, TopicPieces, Workspace

ACTIVE_TOPIC = "Active Topic"
ALL_TOPICS = "All Topics"
EMPTY = "No Content Pieces yet. Ask for a blog in the Chat tab"
KIND_ICONS = {"blog": "📝", "linkedin_post": "💼", "image": "🖼️"}
NO_SCORE = "—"


def _table(rows: list[PieceRow]) -> pd.DataFrame:
    """A few columns only, so it stays readable on a phone."""
    return pd.DataFrame(
        {
            "Kind": f"{KIND_ICONS.get(row.kind, '📄')} {PIECE_FORMATS[row.kind].label}",
            "Title": row.title,
            "Revision": row.revision,
            "Quality Score": NO_SCORE if row.score is None else str(row.score),
        }
        for row in rows
    )


def content_dashboard_tab(workspace: Workspace) -> None:
    shown = st.radio("Show", [ACTIVE_TOPIC, ALL_TOPICS], horizontal=True, key="dashboard_filter")
    groups = workspace.dashboard(all_topics=shown == ALL_TOPICS)
    if not groups:
        st.info(EMPTY)
    for group in groups:
        st.subheader(group.topic)
        st.dataframe(_table(group.pieces), hide_index=True, use_container_width=True)
    _open_piece(workspace, groups)


def _open_piece(workspace: Workspace, groups: list[TopicPieces]) -> None:
    """Pick a listed piece (key `open_piece`, value its piece id) to show its detail view below."""
    rows = {row.piece_id: row for group in groups for row in group.pieces}
    if not rows:
        return
    piece_id = st.selectbox(
        "Open a piece",
        list(rows),
        index=None,
        placeholder="Choose a Content Piece",
        format_func=lambda i: f"{PIECE_FORMATS[rows[i].kind].label.capitalize()}: {rows[i].title}",
        key="open_piece",
    )
    if piece_id and (piece := workspace.piece(piece_id)):
        viewed = revision_history(piece, workspace)
        piece_detail(piece, viewed)
        export_buttons(workspace.piece_exports(piece_id, viewed), key=f"export_{piece_id}")
