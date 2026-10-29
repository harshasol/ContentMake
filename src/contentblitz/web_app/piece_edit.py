"""Hand-editing a text Content Piece from the dashboard (#63).

`piece_edit` draws an Edit button under the piece's detail view; it opens a form with the latest
Revision's text (and a blog's meta description). Save goes through `Workspace.edit` and adds a
*hand edit* Revision; Cancel leaves without one. Images offer no Edit.
"""

import streamlit as st

from contentblitz.workflow.workspace import PieceDetail, Workspace

EDITING = "editing_piece"  # session state: the id of the piece being edited, if any


def piece_edit(workspace: Workspace, piece: PieceDetail) -> None:
    if piece.kind == "image":
        return
    if st.session_state.get(EDITING) != piece.piece_id:
        if st.button("Edit", key="edit_piece"):
            st.session_state[EDITING] = piece.piece_id
            st.rerun()
        return
    latest = piece.latest
    with st.form(f"edit_form_{piece.piece_id}"):
        text = st.text_area("Text", value=latest["text"], height=400, key="edit_text")
        meta = None
        if piece.kind == "blog":
            meta = st.text_input(
                "Meta description", value=latest.get("meta_description", ""), key="edit_meta"
            )
        save, cancel = st.columns(2)
        saved = save.form_submit_button("Save", type="primary")
        cancelled = cancel.form_submit_button("Cancel")
    if saved:
        workspace.edit(piece.piece_id, text, meta)
    if saved or cancelled:
        del st.session_state[EDITING]
        st.rerun()
