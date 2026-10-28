"""SerpAPI and Tavily against httpx's mock transport, with their real response formats."""

import json

import httpx
import pytest

from contentblitz.core.config import DEFAULT_CONFIG_PATH, Settings, load_model_config
from contentblitz.integrations import search
from contentblitz.integrations.search import (
    SearchProviderError,
    SearchResult,
    SerpApiProvider,
    TavilyProvider,
)

SERPAPI_RESPONSE = {
    "search_metadata": {
        "id": "6720a1b2c3d4e5f6a7b8c9d0",
        "status": "Success",
        "json_endpoint": "https://serpapi.com/searches/6720a1b2c3d4e5f6/6720a1b2c3d4e5f6a7b8c9d0.json",
        "total_time_taken": 1.42,
    },
    "search_parameters": {"engine": "google", "q": "AI in healthcare 2026", "num": "10"},
    "search_information": {"total_results": 1250000000, "time_taken_displayed": 0.38},
    "related_questions": [{"question": "How is AI used in healthcare?"}],
    "organic_results": [
        {
            "position": 1,
            "title": "Artificial intelligence in healthcare - WHO",
            "link": "https://www.who.int/health-topics/artificial-intelligence",
            "redirect_link": "https://www.google.com/url?sa=t&url=https://www.who.int/...",
            "displayed_link": "https://www.who.int › health-topics",
            "favicon": "https://serpapi.com/searches/favicon.png",
            "snippet": "AI holds great promise for improving the delivery of healthcare.",
            "snippet_highlighted_words": ["AI", "healthcare"],
            "source": "World Health Organization (WHO)",
        },
        {
            "position": 2,
            "title": "How AI is changing radiology",
            "link": "https://www.nature.com/articles/s41591-026-01234-5",
            "displayed_link": "https://www.nature.com › articles",
            "date": "Jan 12, 2026",
            "source": "Nature",
        },
    ],
    "pagination": {"current": 1, "next": "https://www.google.com/search?q=AI+in+healthcare"},
}

TAVILY_RESPONSE = {
    "query": "AI in healthcare 2026",
    "follow_up_questions": None,
    "answer": None,
    "images": [],
    "results": [
        {
            "title": "AI in Healthcare: 2026 Market Report",
            "url": "https://www.statista.com/topics/ai-healthcare/",
            "content": "The AI in healthcare market is expected to reach $45 billion in 2026.",
            "score": 0.91,
            "raw_content": None,
        },
        {
            "title": "FDA guidance on AI-enabled medical devices",
            "url": "https://www.fda.gov/medical-devices/ai-ml",
            "content": "The FDA has authorised more than 1,000 AI-enabled devices.",
            "score": 0.87,
            "raw_content": None,
        },
    ],
    "response_time": 1.21,
    "request_id": "123e4567-e89b-12d3-a456-426614174111",
}


def client_returning(response, requests=None):
    def handler(request):
        if requests is not None:
            requests.append(request)
        return response(request) if callable(response) else response

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_serpapi_sends_a_google_search_with_the_api_key():
    requests = []
    provider = SerpApiProvider("serp-key", client_returning(httpx.Response(200, json={}), requests))
    provider.search("AI in healthcare 2026", max_results=5)

    [request] = requests
    assert request.method == "GET"
    assert request.url.host == "serpapi.com"
    assert request.url.path == "/search.json"
    params = dict(request.url.params)
    assert params == {
        "engine": "google",
        "q": "AI in healthcare 2026",
        "api_key": "serp-key",
        "num": "5",
    }


def test_a_fresh_serpapi_search_skips_serpapis_cache():
    requests = []
    provider = SerpApiProvider("serp-key", client_returning(httpx.Response(200, json={}), requests))
    provider.search("AI in healthcare 2026", fresh=True)

    [request] = requests
    assert request.url.params["no_cache"] == "true"


def test_serpapi_organic_results_become_search_results():
    provider = SerpApiProvider("k", client_returning(httpx.Response(200, json=SERPAPI_RESPONSE)))
    assert provider.search("AI in healthcare 2026") == [
        SearchResult(
            title="Artificial intelligence in healthcare - WHO",
            url="https://www.who.int/health-topics/artificial-intelligence",
            snippet="AI holds great promise for improving the delivery of healthcare.",
        ),
        SearchResult(
            title="How AI is changing radiology",
            url="https://www.nature.com/articles/s41591-026-01234-5",
            snippet="",
        ),
    ]


def test_serpapi_with_no_results_returns_an_empty_list():
    body = {
        "search_metadata": {"status": "Success"},
        "error": "Google hasn't returned any results for this query.",
    }
    provider = SerpApiProvider("k", client_returning(httpx.Response(200, json=body)))
    assert provider.search("zzqx nonsense") == []


def test_serpapi_reports_its_error_message():
    body = {
        "error": "Invalid API key. Your API key should be here: https://serpapi.com/manage-api-key"
    }
    provider = SerpApiProvider("bad", client_returning(httpx.Response(401, json=body)))
    with pytest.raises(SearchProviderError, match="Invalid API key"):
        provider.search("AI")


def test_serpapi_quota_error_with_a_200_status_is_a_failure():
    body = {"error": "Your account has run out of searches."}
    provider = SerpApiProvider("k", client_returning(httpx.Response(200, json=body)))
    with pytest.raises(SearchProviderError, match="run out of searches"):
        provider.search("AI")


def test_tavily_posts_a_basic_search_with_the_api_key():
    requests = []
    provider = TavilyProvider("tvly-key", client_returning(httpx.Response(200, json={}), requests))
    provider.search("AI in healthcare 2026", max_results=7)

    [request] = requests
    assert request.method == "POST"
    assert str(request.url) == "https://api.tavily.com/search"
    assert request.headers["Authorization"] == "Bearer tvly-key"
    assert json.loads(request.content) == {
        "query": "AI in healthcare 2026",
        "search_depth": "basic",
        "max_results": 7,
    }


def test_tavily_results_become_search_results():
    provider = TavilyProvider("k", client_returning(httpx.Response(200, json=TAVILY_RESPONSE)))
    assert provider.search("AI in healthcare 2026") == [
        SearchResult(
            title="AI in Healthcare: 2026 Market Report",
            url="https://www.statista.com/topics/ai-healthcare/",
            snippet="The AI in healthcare market is expected to reach $45 billion in 2026.",
        ),
        SearchResult(
            title="FDA guidance on AI-enabled medical devices",
            url="https://www.fda.gov/medical-devices/ai-ml",
            snippet="The FDA has authorised more than 1,000 AI-enabled devices.",
        ),
    ]


def test_tavily_reports_its_error_message():
    body = {"detail": {"error": "Unauthorized: missing or invalid API key."}}
    provider = TavilyProvider("bad", client_returning(httpx.Response(401, json=body)))
    with pytest.raises(SearchProviderError, match="invalid API key"):
        provider.search("AI")


@pytest.mark.parametrize("provider_class", [SerpApiProvider, TavilyProvider])
def test_a_timeout_is_a_provider_failure(provider_class):
    def timeout(request):
        raise httpx.ReadTimeout("timed out", request=request)

    provider = provider_class("k", client_returning(timeout))
    with pytest.raises(SearchProviderError, match="timed out"):
        provider.search("AI")


@pytest.mark.parametrize("provider_class", [SerpApiProvider, TavilyProvider])
def test_a_server_error_without_json_is_a_provider_failure(provider_class):
    provider = provider_class("k", client_returning(httpx.Response(502, text="Bad Gateway")))
    with pytest.raises(SearchProviderError, match="502"):
        provider.search("AI")


def config_with(*providers):
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.search.providers = list(providers)
    return config


def test_providers_follow_the_configured_order():
    settings = Settings(_env_file=None, serpapi_api_key="s", tavily_api_key="t")
    providers = search.configured_providers(settings, config_with("tavily", "serpapi"))
    assert [p.name for p in providers] == ["Tavily", "SerpAPI"]


def test_a_provider_without_an_api_key_is_left_out():
    settings = Settings(_env_file=None, serpapi_api_key=None, tavily_api_key="t")
    providers = search.configured_providers(settings, config_with("serpapi", "tavily"))
    assert [p.name for p in providers] == ["Tavily"]


def test_providers_use_the_api_keys_from_settings():
    requests = []
    settings = Settings(_env_file=None, serpapi_api_key="from-settings")
    client = client_returning(httpx.Response(200, json={}), requests)
    [provider] = search.configured_providers(settings, config_with("serpapi", "tavily"), client)
    provider.search("AI")
    assert requests[0].url.params["api_key"] == "from-settings"


def test_the_shipped_config_puts_serpapi_before_tavily():
    assert load_model_config(DEFAULT_CONFIG_PATH).search.providers == ["serpapi", "tavily"]
