"""The page fetcher against httpx's mock transport: article text, or no text and no error."""

import httpx

from contentblitz.integrations import search

URL = "https://www.example.com/ai-in-healthcare"

PARAGRAPHS = [
    "Hospitals using AI triage tools cut emergency waiting times by 23% in 2026, according "
    "to a study of 140 hospitals across Europe and North America published this spring.",
    "The FDA has now authorised more than 1,000 AI-enabled medical devices, most of them in "
    "radiology, where algorithms flag suspected strokes and lung nodules for review.",
    "Clinicians remain cautious. In a survey of 2,400 doctors, 61% said they would only trust "
    "an AI recommendation if they could see the evidence behind it.",
    "Regulators in the UK and the EU are drafting rules that would require hospitals to audit "
    "the accuracy of these tools every year and report errors to a national register.",
]

ARTICLE = f"""<!doctype html>
<html><head><title>AI in healthcare: what changed in 2026</title></head>
<body>
  <nav><a href="/">Home</a> <a href="/news">News</a> <a href="/subscribe">Subscribe</a></nav>
  <article>
    <h1>AI in healthcare: what changed in 2026</h1>
    {"".join(f"<p>{p}</p>" for p in PARAGRAPHS)}
  </article>
  <footer>Copyright 2026 Example Media. All rights reserved. Cookie settings.</footer>
</body></html>"""


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def serving(content, content_type="text/html; charset=utf-8", status=200, requests=None):
    def handler(request):
        if requests is not None:
            requests.append(request)
        return httpx.Response(status, content=content, headers={"Content-Type": content_type})

    return client(handler)


def fetch(http_client, max_chars=2000):
    return search.fetch_page(URL, timeout=8, max_chars=max_chars, client=http_client)


def test_an_html_article_gives_its_readable_text():
    requests = []
    text = fetch(serving(ARTICLE.encode(), requests=requests))

    for paragraph in PARAGRAPHS:
        assert paragraph in text
    assert "Subscribe" not in text
    assert "Cookie settings" not in text
    assert str(requests[0].url) == URL
    assert requests[0].method == "GET"
    assert requests[0].headers["User-Agent"].startswith("Mozilla/5.0")


def test_the_text_is_capped_at_the_given_number_of_characters():
    text = fetch(serving(ARTICLE.encode()), max_chars=100)

    assert len(text) <= 100
    assert text.startswith("AI in healthcare") or text.startswith("Hospitals using AI")


def test_a_non_html_page_gives_no_text():
    assert fetch(serving(b"%PDF-1.7 binary", content_type="application/pdf")) == ""


def test_an_http_error_gives_no_text():
    assert fetch(serving(ARTICLE.encode(), status=403)) == ""


def test_a_timeout_gives_no_text():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    assert fetch(client(handler)) == ""


def test_a_connection_failure_gives_no_text():
    def handler(request):
        raise httpx.ConnectError("name not resolved", request=request)

    assert fetch(client(handler)) == ""


def test_a_page_with_nothing_readable_gives_no_text():
    assert fetch(serving(b"<html><body><div id='app'></div></body></html>")) == ""


def test_a_page_over_the_size_limit_is_read_only_up_to_the_limit(monkeypatch):
    monkeypatch.setattr(search, "MAX_PAGE_BYTES", len(ARTICLE.encode()))
    padding = b"<p>" + b"filler " * 200_000 + b"</p>"
    page = ARTICLE.encode().replace(b"</body>", padding + b"</body>")

    text = fetch(serving(page))

    assert PARAGRAPHS[0] in text
    assert "filler" not in text
