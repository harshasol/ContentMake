"""Showing a Content Piece's image from its saved file, in the chat and in the dashboard."""

from pathlib import Path

import streamlit as st

from contentblitz.workflow.image_files import image_caption
from contentblitz.workflow.state import ImageMetadata

MISSING_IMAGE = "The image file is missing, so it can't be shown."


def show_image(image: ImageMetadata | None) -> None:
    """The image from its file (provider URLs expire) with its caption, or a notice when the
    image data or the file is missing."""
    if image is not None and Path(image["file_path"]).is_file():
        st.image(image["file_path"], caption=image_caption(image))
    else:
        st.warning(MISSING_IMAGE)
