from types import SimpleNamespace

from contentblitz.core.config import DEFAULT_CONFIG_PATH, load_model_config
from contentblitz.integrations import llm


def test_complete_passes_primary_and_fallbacks_to_litellm(monkeypatch):
    calls = {}

    def fake_completion(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.agents["blog_writer"].models = ["openai/gpt-4o", "anthropic/claude-sonnet-4-20250514"]

    assert llm.complete("blog_writer", [{"role": "user", "content": "x"}], config=config) == "hi"
    assert calls["model"] == "openai/gpt-4o"
    assert calls["fallbacks"] == ["anthropic/claude-sonnet-4-20250514"]
