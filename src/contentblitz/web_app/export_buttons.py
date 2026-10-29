"""Download buttons for exports the Workspace built in memory (#68)."""

import streamlit as st

from contentblitz.workflow.export import Export

FORMAT_LABELS = {"md": "Markdown", "html": "HTML", "txt": "plain text", "png": "PNG"}


def export_buttons(exports: list[Export], key: str) -> None:
    """One download button per export, side by side; `key` keeps each place's buttons distinct.
    Button keys are `<key>_<extension>`."""
    if not exports:
        return
    for export, column in zip(exports, st.columns(len(exports)), strict=True):
        extension = export.file_name.rsplit(".", 1)[-1]
        column.download_button(
            f"Download {FORMAT_LABELS.get(extension, extension)}",
            export.data,
            file_name=export.file_name,
            mime=export.mime,
            key=f"{key}_{extension}",
        )


def bundle_button(bundle: Export | None, key: str = "topic_bundle") -> None:
    """The Topic bundle download (#69); nothing when the Topic has no bundle."""
    if bundle is None:
        return
    st.download_button(
        "Download Topic bundle",
        bundle.data,
        file_name=bundle.file_name,
        mime=bundle.mime,
        key=key,
    )
