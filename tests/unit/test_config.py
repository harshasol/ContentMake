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


def test_shipped_config_caps_each_research_run():
    search = load_model_config(DEFAULT_CONFIG_PATH).search
    assert search.queries_per_run == 4
    assert search.max_sources == 10
    assert search.pages_to_read == 6
    assert search.page_timeout_seconds == 8
    assert search.text_cap_chars == 2000


def test_research_limits_default_when_not_configured(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "agents:\n"
        + "".join(f"  {a}:\n    models: [{MODELS}]\n" for a in AGENT_NAMES)
        + "image:\n  models: [openai/gpt-image-1]\nsearch:\n  providers: [tavily]\n"
    )
    search = load_model_config(path).search
    limits = (search.queries_per_run, search.max_sources, search.pages_to_read)
    assert limits == (4, 10, 6)
    assert (search.page_timeout_seconds, search.text_cap_chars) == (8, 2000)
    blog = load_model_config(path).quality.blog
    assert blog.word_range == (900, 1800)
    assert blog.density_range == (1.0, 2.0)
    assert blog.meta_length_range == (150, 160)


def test_shipped_config_sets_the_blog_thresholds():
    blog = load_model_config(DEFAULT_CONFIG_PATH).quality.blog
    assert blog.word_range == (900, 1800)
    assert blog.density_range == (1.0, 2.0)
    assert blog.meta_length_range == (150, 160)


def test_blog_thresholds_are_read_from_config(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "agents:\n"
        + "".join(f"  {a}:\n    models: [{MODELS}]\n" for a in AGENT_NAMES)
        + "image:\n  models: [openai/gpt-image-1]\nsearch:\n  providers: [tavily]\n"
        + "quality:\n  blog:\n    density_range: [0.5, 1.5]\n    meta_length_range: [120, 155]\n"
    )
    blog = load_model_config(path).quality.blog
    assert blog.density_range == (0.5, 1.5)
    assert blog.meta_length_range == (120, 155)


def test_an_unknown_search_provider_is_rejected(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "agents:\n"
        + "".join(f"  {a}:\n    models: [{MODELS}]\n" for a in AGENT_NAMES)
        + "image:\n  models: [openai/gpt-image-1]\nsearch:\n  providers: [bing]\n"
    )
    with pytest.raises(ValueError, match="serpapi"):
        load_model_config(path)
