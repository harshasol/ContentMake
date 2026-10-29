"""Exports of a Revision and of the research report, built in memory (#68)."""

import pytest
import yaml

from contentblitz.workflow import export

BLOG = {
    "text": "# Heading\n\nBody **bold**",
    "origin": "draft",
    "created_at": "2026-10-29T10:00:00+00:00",
    "meta_description": "A short: meta",
    "target_keyword": "ai health",
}
POST = {"text": "Hook line\n\nSecond para #ai", "origin": "draft", "created_at": BLOG["created_at"]}


def _blog(fmt, revision=BLOG, number=2):
    return export.export_piece("AI in Health", "blog", "AI: a <cure>?", revision, number, fmt)


def test_blog_markdown_has_front_matter_with_every_field():
    out = _blog("md")

    head, body = out.data.decode().removeprefix("---\n").split("---\n", 1)
    assert yaml.safe_load(head) == {
        "title": "AI: a <cure>?",
        "meta_description": "A short: meta",
        "target_keyword": "ai health",
        "topic": "AI in Health",
        "revision": 2,
        "date": "2026-10-29",
    }
    assert body.strip() == BLOG["text"]
    assert out.mime == "text/markdown"


def test_blog_html_is_a_standalone_page_with_title_and_meta_in_the_head():
    page = _blog("html").data.decode()

    head = page.split("</head>")[0]
    assert "<title>AI: a &lt;cure&gt;?</title>" in head
    assert '<meta name="description" content="A short: meta">' in head
    assert "<style>" in head
    assert "<h1>Heading</h1>" in page and "<strong>bold</strong>" in page


def test_linkedin_text_keeps_line_breaks_and_markdown_is_also_offered():
    txt = export.export_piece("T", "linkedin_post", "Post", POST, 1, "txt")
    md = export.export_piece("T", "linkedin_post", "Post", POST, 1, "md")

    assert txt.data.decode() == POST["text"]
    assert md.data.decode().strip() == POST["text"]
    assert export.formats("linkedin_post") == ["txt", "md"]


def test_image_export_returns_the_png_bytes(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(b"\x89PNGdata")
    revision = {**POST, "image": {"file_path": str(path)}}

    out = export.export_piece("T", "image", "Hero", revision, 3, "png")

    assert out.data == b"\x89PNGdata"
    assert out.mime == "image/png"
    assert out.file_name == "t-image-hero-r3.png"


def test_research_report_exports_as_markdown_and_html():
    md = export.export_report("AI in Health", "# Report\n\n- a", "md")
    html = export.export_report("AI in Health", "# Report\n\n- a", "html")

    assert md.data.decode() == "# Report\n\n- a"
    assert "<title>Research: AI in Health</title>" in html.data.decode()
    assert "<li>a</li>" in html.data.decode()
    assert md.file_name == "ai-in-health-research-report.md"


def test_file_names_are_safe_slugs_of_topic_kind_title_and_revision():
    assert (
        export.file_name("AI in Health!", "linkedin_post", "../Hello, World/ é", 4, "txt")
        == "ai-in-health-linkedin-post-hello-world-r4.txt"
    )
    assert export.slug("///") == "untitled"


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError):
        _blog("pdf")
