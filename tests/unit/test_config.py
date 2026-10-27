import pytest

from contentblitz.core.config import AGENT_NAMES, DEFAULT_CONFIG_PATH, load_model_config

MODELS = "openai/gpt-4o, gemini/gemini-2.0-flash"


def test_shipped_config_covers_every_agent():
    config = load_model_config(DEFAULT_CONFIG_PATH)
    for agent in AGENT_NAMES:
        assert config.for_agent(agent).primary


def test_primary_and_fallbacks_split_the_model_list(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "agents:\n"
        + "".join(f"  {a}:\n    models: [{MODELS}]\n" for a in AGENT_NAMES)
        + "image:\n  models: [openai/gpt-image-1]\nsearch:\n  providers: [serpapi]\n"
    )
    models = load_model_config(path).for_agent("blog_writer")
    assert models.primary == "openai/gpt-4o"
    assert models.fallbacks == ["gemini/gemini-2.0-flash"]


def test_missing_agent_is_rejected(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "agents:\n  research:\n    models: [openai/gpt-4o]\n"
        "image:\n  models: [openai/gpt-image-1]\nsearch:\n  providers: [serpapi]\n"
    )
    with pytest.raises(ValueError, match="missing agents"):
        load_model_config(path)
