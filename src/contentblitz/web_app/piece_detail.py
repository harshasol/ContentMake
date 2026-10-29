"""The detail view of one Content Piece: one Revision of it, in a single column (#61).

`piece_detail` renders the Revision it is given, the latest by default, so viewing an older
Revision (#62), hand edit (#63) and restore (#64) add their controls around `show_revision`.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from contentblitz.workflow.image_files import image_caption
from contentblitz.workflow.quality_score import QualityScore
from contentblitz.workflow.report import display_date
from contentblitz.workflow.state import Revision
from contentblitz.workflow.workspace import PieceDetail, Workspace

MISSING_IMAGE = "The image file for this Revision is missing, so it can't be shown."
ORIGIN_LABELS = {
    "draft": "Draft",
    "refinement": "Refinement",
    "hand_edit": "Hand edit",
    "restore": "Restore",
}
NONE = "—"
NO_REVIEW = "The AI reviewer was unavailable, so this draft could not pass."


def piece_detail(piece: PieceDetail, revision_number: int | None = None) -> None:
    """The piece's heading, then Revision `revision_number` (from 1; the latest by default)."""
    number = revision_number or len(piece.revisions)
    st.subheader(piece.title)
    st.caption(f"{piece.topic} · Revision {number} of {len(piece.revisions)}")
    if number != len(piece.revisions):
        latest = len(piece.revisions)
        st.info(f"You are viewing an older Revision {number}. The latest is {latest}.")
    show_revision(piece.kind, piece.revision(number))


def viewed_key(piece_id: str) -> str:
    """Session-state key holding the Revision number chosen with View for `piece_id`."""
    return f"viewed_revision_{piece_id}"


def revision_history(piece: PieceDetail, workspace: Workspace) -> int | None:
    """The Revision history (#62): a table of every Revision, then a View button for each and,
    for the older ones, a Restore button (#64) that calls `workspace.restore`.

    Returns the number to show in `piece_detail`: the one chosen with View (kept in session
    state, so it survives reruns), or None for the latest. Viewing never touches the Workspace.
    """
    st.markdown("**Revision history**")
    st.dataframe(_history_table(piece), hide_index=True, use_container_width=True)
    _view_buttons(piece, workspace)
    chosen = st.session_state.get(viewed_key(piece.piece_id))
    return chosen if chosen and chosen <= len(piece.revisions) else None


def _history_table(piece: PieceDetail) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Revision": number,
            "Origin": ORIGIN_LABELS.get(revision["origin"], revision["origin"]),
            "Time": _time(revision["created_at"]),
            "Brand Profile": (revision.get("brand_profile") or {}).get("name", NONE),
            "Quality Score": _score(piece.kind, revision),
        }
        for number, revision in enumerate(piece.revisions, start=1)
    )


def _time(iso: str) -> str:
    return f"{display_date(iso)} {datetime.fromisoformat(iso):%H:%M} UTC"


def _score(kind: str, revision: Revision) -> str:
    score = revision.get("quality_score")
    return NONE if kind == "image" or not score else str(score["score"])


def _view_buttons(piece: PieceDetail, workspace: Workspace) -> None:
    latest = len(piece.revisions)
    for number, column in enumerate(st.columns(latest), start=1):
        column.button(
            f"View {number}",
            key=f"view_revision_{number}",
            on_click=st.session_state.__setitem__,
            args=(viewed_key(piece.piece_id), number),
        )
        if number != latest:
            column.button(
                f"Restore {number}",
                key=f"restore_revision_{number}",
                on_click=_restore,
                args=(workspace, piece.piece_id, number),
            )


def _restore(workspace: Workspace, piece_id: str, number: int) -> None:
    """Restore Revision `number` (#64) and go back to viewing the latest, the restored copy."""
    workspace.restore(piece_id, number)
    st.session_state[viewed_key(piece_id)] = None


def show_revision(kind: str, revision: Revision) -> None:
    """The content of one Revision of a piece of `kind`, then its Quality Score for text."""
    if kind == "image":
        _image(revision)
        return
    if kind == "blog":
        st.markdown(revision["text"])
        _blog_fields(revision)
    else:
        st.text(revision["text"])
        st.caption(f"{len(revision['text'])} characters")
    if score := revision.get("quality_score"):
        _quality_score(score)


def _blog_fields(revision: Revision) -> None:
    for label, key in (
        ("Meta description", "meta_description"),
        ("Target keyword", "target_keyword"),
        ("Template", "template"),
    ):
        if revision.get(key):
            st.caption(f"{label}: {revision[key]}")


def _image(revision: Revision) -> None:
    image = revision.get("image")
    if image is not None and Path(image["file_path"]).is_file():
        st.image(image["file_path"], caption=image_caption(image))
    else:
        st.warning(MISSING_IMAGE)
    if image is None:
        return
    st.caption(f"Alt text: {image['alt_text']}")
    st.markdown("**Optimised prompt**")
    st.code(image["prompt"], language=None, wrap_lines=True)


def _quality_score(score: QualityScore) -> None:
    st.markdown(
        f"**Quality Score: {score['score']}** ({'passed' if score['passed'] else 'failed'})"
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Check": c["name"],
                "Result": "Pass" if c["passed"] else "Fail",
                "Measured": c["measured"],
                "Suggestion": c["reason"],
            }
            for c in score["checks"]
        ),
        hide_index=True,
        use_container_width=True,
    )
    review = score["review"]
    if review is None:
        st.markdown(NO_REVIEW)
        return
    scores = [f"Engagement {review['engagement']}/10"]
    if "brand_voice" in review:
        scores.append(f"Brand voice {review['brand_voice']}/10")
    st.markdown("**AI reviewer:** " + " · ".join(scores))
    for suggestion in review["suggestions"]:
        st.markdown(f"- {suggestion}")
