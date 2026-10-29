"""The detail view of one Content Piece: one Revision of it, in a single column (#61).

`piece_detail` renders the Revision it is given, the latest by default, so viewing an older
Revision (#62), hand edit (#63) and restore (#64) add their controls around `show_revision`.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from contentblitz.workflow.image_files import image_caption
from contentblitz.workflow.quality_score import QualityScore
from contentblitz.workflow.state import Revision
from contentblitz.workflow.workspace import PieceDetail

MISSING_IMAGE = "The image file for this Revision is missing, so it can't be shown."
NO_REVIEW = "The AI reviewer was unavailable, so this draft could not pass."


def piece_detail(piece: PieceDetail, revision_number: int | None = None) -> None:
    """The piece's heading, then Revision `revision_number` (from 1; the latest by default)."""
    number = revision_number or len(piece.revisions)
    st.subheader(piece.title)
    st.caption(f"{piece.topic} · Revision {number} of {len(piece.revisions)}")
    show_revision(piece.kind, piece.revision(number))


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
    note = " (not re-reviewed)" if score.get("not_re_reviewed") else ""
    st.markdown(f"**AI reviewer{note}:** " + " · ".join(scores))
    for suggestion in review["suggestions"]:
        st.markdown(f"- {suggestion}")
