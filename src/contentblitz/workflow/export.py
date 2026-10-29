"""Exports of a Revision and of the research report, built in memory (spec #23, #68).

Pure functions: they return an `Export` (file name, media type, bytes) and write nothing. The
Workspace calls them for the viewed Revision; the Topic bundle (#69) reuses `file_name`, `slug`
and `export_piece`.
"""

import html
import io
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import markdown
import yaml

from contentblitz.workflow.formats import FORMATS, PIECE_FORMATS
from contentblitz.workflow.state import Revision

REPORT_EXPORTS = list(FORMATS["research"].exports)
MIME = {
    "md": "text/markdown",
    "html": "text/html",
    "txt": "text/plain",
    "png": "image/png",
    "zip": "application/zip",
}
PAGE_STYLE = (
    "body{font-family:system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem;"
    "line-height:1.6;color:#222}h1,h2,h3{line-height:1.25}a{color:#0b5fff}"
    "table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:.3rem .6rem}"
)


@dataclass(frozen=True)
class Export:
    file_name: str
    mime: str
    data: bytes


def slug(text: str) -> str:
    """A file-name-safe lowercase ASCII slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-") or "untitled"


def file_name(topic: str, kind: str, title: str, revision: int, extension: str) -> str:
    """`<topic>-<kind>-<title>-r<revision>.<ext>`, each part slugged."""
    return f"{slug(topic)}-{slug(kind)}-{slug(title)}-r{revision}.{extension}"


def formats(kind: str) -> list[str]:
    """The export formats (file extensions) offered for a piece of `kind`."""
    return list(PIECE_FORMATS[kind].exports) if kind in PIECE_FORMATS else []


def _page(title: str, body_markdown: str, description: str | None = None) -> str:
    meta = (
        f'<meta name="description" content="{html.escape(description, quote=True)}">\n'
        if description
        else ""
    )
    body = markdown.markdown(body_markdown, extensions=["tables"])
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n{meta}<style>{PAGE_STYLE}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def _front_matter(topic: str, title: str, revision: Revision, number: int) -> str:
    fields = {
        "title": title,
        "meta_description": revision.get("meta_description", ""),
        "target_keyword": revision.get("target_keyword", ""),
        "topic": topic,
        "revision": number,
        "date": datetime.fromisoformat(revision["created_at"]).date().isoformat(),
    }
    return "---\n" + yaml.safe_dump(fields, sort_keys=False, allow_unicode=True) + "---\n\n"


def export_piece(
    topic: str, kind: str, title: str, revision: Revision, number: int, fmt: str
) -> Export:
    """Revision `number` of a piece as `fmt` (one of `formats(kind)`); ValueError otherwise.
    An image export reads its PNG from the Revision's `file_path`."""
    if fmt not in formats(kind):
        raise ValueError(f"Cannot export a {kind} as {fmt}")
    name = file_name(topic, kind, title, number, fmt)
    text = revision["text"]
    if kind == "image":
        data = Path(revision["image"]["file_path"]).read_bytes()
    elif kind == "blog" and fmt == "html":
        data = _page(title, text, revision.get("meta_description")).encode()
    elif kind == "blog":
        data = (_front_matter(topic, title, revision, number) + text + "\n").encode()
    elif fmt == "md":
        data = (text + "\n").encode()
    else:
        data = text.encode()
    return Export(name, MIME[fmt], data)


def export_report(topic: str, report: str, fmt: str) -> Export:
    """The research report (Markdown, as `render_report` makes it) as `md` or `html`."""
    if fmt not in REPORT_EXPORTS:
        raise ValueError(f"Cannot export the research report as {fmt}")
    name = f"{slug(topic)}-research-report.{fmt}"
    data = report if fmt == "md" else _page(f"Research: {topic}", report)
    return Export(name, MIME[fmt], data.encode())


def export_bundle(topic: str, exports: list[Export]) -> Export:
    """A zip of `exports`, each stored under its own file name, as `<topic>-bundle.zip` (#69)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for export in exports:
            archive.writestr(export.file_name, export.data)
    return Export(f"{slug(topic)}-bundle.zip", MIME["zip"], buffer.getvalue())
