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
    assert blog.max_grade == 10
    assert blog.min_citations == 3


def test_shipped_config_sets_the_blog_thresholds():
    blog = load_model_config(DEFAULT_CONFIG_PATH).quality.blog
    assert blog.word_range == (900, 1800)
    assert blog.density_range == (1.0, 2.0)
    assert blog.meta_length_range == (150, 160)
    assert blog.max_grade == 10
    assert blog.min_citations == 3


def test_blog_thresholds_are_read_from_config(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "agents:\n"
        + "".join(f"  {a}:\n    models: [{MODELS}]\n" for a in AGENT_NAMES)
        + "image:\n  models: [openai/gpt-image-1]\nsearch:\n  providers: [tavily]\n"
        + "quality:\n  blog:\n    density_range: [0.5, 1.5]\n    meta_length_range: [120, 155]\n"
        + "    max_grade: 8\n    word_range: [500, 700]\n    min_citations: 5\n"
    )
    blog = load_model_config(path).quality.blog
    assert blog.density_range == (0.5, 1.5)
    assert blog.meta_length_range == (120, 155)
    assert blog.max_grade == 8
    assert blog.word_range == (500, 700)
    assert blog.min_citations == 5


def test_an_unknown_search_provider_is_rejected(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "agents:\n"
        + "".join(f"  {a}:\n    models: [{MODELS}]\n" for a in AGENT_NAMES)
        + "image:\n  models: [openai/gpt-image-1]\nsearch:\n  providers: [bing]\n"
    )
    with pytest.raises(ValueError, match="serpapi"):
        load_model_config(path)


def test_the_quality_judge_has_its_own_model_entry_and_is_required():
    assert "quality_judge" in AGENT_NAMES
    assert load_model_config(DEFAULT_CONFIG_PATH).for_agent("quality_judge").primary


def test_brand_voice_has_its_own_model_entry_and_is_required(tmp_path):
    assert "brand_voice" in AGENT_NAMES
    assert load_model_config(DEFAULT_CONFIG_PATH).for_agent("brand_voice").primary
    path = tmp_path / "models.yaml"
    path.write_text(
        "agents:\n"
        + "".join(f"  {a}:\n    models: [{MODELS}]\n" for a in AGENT_NAMES if a != "brand_voice")
        + "image:\n  models: [openai/gpt-image-1]\nsearch:\n  providers: [tavily]\n"
    )
    with pytest.raises(ValueError, match="brand_voice"):
        load_model_config(path)


def test_the_pass_mark_defaults_to_7_and_is_read_from_config(tmp_path):
    assert load_model_config(DEFAULT_CONFIG_PATH).quality.pass_mark == 7
    path = tmp_path / "models.yaml"
    agents = "".join(f"  {a}:\n    models: [{MODELS}]\n" for a in AGENT_NAMES)
    image_and_search = "image:\n  models: [openai/gpt-image-1]\nsearch:\n  providers: [tavily]\n"
    path.write_text(f"agents:\n{agents}{image_and_search}")
    assert load_model_config(path).quality.pass_mark == 7
    path.write_text(f"agents:\n{agents}{image_and_search}quality:\n  pass_mark: 8\n")
    assert load_model_config(path).quality.pass_mark == 8


def config_with_quality(tmp_path, quality=""):
    path = tmp_path / "models.yaml"
    agents = "".join(f"  {a}:\n    models: [{MODELS}]\n" for a in AGENT_NAMES)
    image_and_search = "image:\n  models: [openai/gpt-image-1]\nsearch:\n  providers: [tavily]\n"
    path.write_text(f"agents:\n{agents}{image_and_search}{quality}")
    return load_model_config(path)


def test_linkedin_ranges_default_and_are_shipped_in_config(tmp_path):
    for config in [config_with_quality(tmp_path), load_model_config(DEFAULT_CONFIG_PATH)]:
        linkedin = config.quality.linkedin
        assert linkedin.char_range == (1300, 1600)
        assert linkedin.hashtag_range == (8, 12)
        assert linkedin.max_hook_chars == 150
        assert linkedin.min_paragraphs == 4
        assert linkedin.max_paragraph_chars == 300
        assert {"comment", "share", "follow", "thoughts", "let me know", "dm", "link in"} <= set(
            linkedin.cta_phrases
        )


def test_linkedin_ranges_are_read_from_config(tmp_path):
    quality = (
        "quality:\n  linkedin:\n    char_range: [1000, 1400]\n    hashtag_range: [3, 5]\n"
        "    max_hook_chars: 120\n    min_paragraphs: 3\n    max_paragraph_chars: 250\n"
        "    cta_phrases: [register, sign up]\n"
    )
    linkedin = config_with_quality(tmp_path, quality).quality.linkedin
    assert linkedin.char_range == (1000, 1400)
    assert linkedin.hashtag_range == (3, 5)
    assert (linkedin.max_hook_chars, linkedin.min_paragraphs) == (120, 3)
    assert linkedin.max_paragraph_chars == 250
    assert linkedin.cta_phrases == ["register", "sign up"]


def test_shipped_config_sizes_each_image_model_for_each_aspect():
    image = load_model_config(DEFAULT_CONFIG_PATH).image
    assert image.models == ["openai/gpt-image-1", "openai/dall-e-3"]
    sizes = {
        model: [image.size_for(model, a) for a in ("landscape", "square", "portrait")]
        for model in image.models
    }
    assert sizes == {
        "openai/gpt-image-1": ["1536x1024", "1024x1024", "1024x1536"],
        "openai/dall-e-3": ["1792x1024", "1024x1024", "1024x1792"],
    }
    assert [image.quality_for(m) for m in image.models] == ["medium", "standard"]


def test_image_settings_are_read_from_config_and_default_when_missing(tmp_path):
    def image_config(image):
        path = tmp_path / "models.yaml"
        path.write_text(
            "agents:\n"
            + "".join(f"  {a}:\n    models: [{MODELS}]\n" for a in AGENT_NAMES)
            + image
            + "search:\n  providers: [serpapi]\n"
        )
        return load_model_config(path).image

    configured = image_config(
        "image:\n  models: [openai/dall-e-2]\n"
        "  sizes:\n    openai/dall-e-2: {landscape: 512x512}\n"
        "  quality:\n    openai/dall-e-2: hd\n"
    )
    assert configured.size_for("openai/dall-e-2", "landscape") == "512x512"
    assert configured.quality_for("openai/dall-e-2") == "hd"

    defaults = image_config("image:\n  models: [openai/gpt-image-1]\n")
    assert defaults.size_for("openai/gpt-image-1", "landscape") == "1536x1024"
    assert defaults.size_for("openai/gpt-image-1", "portrait") == "1024x1536"
    assert defaults.quality_for("openai/gpt-image-1") == "medium"


def test_the_placeholder_colour_is_read_from_config_and_defaults_when_missing(tmp_path):
    assert load_model_config(DEFAULT_CONFIG_PATH).image.placeholder_colour == "#4A5568"

    def placeholder_colour(image):
        path = tmp_path / "models.yaml"
        path.write_text(
            "agents:\n"
            + "".join(f"  {a}:\n    models: [{MODELS}]\n" for a in AGENT_NAMES)
            + "image:\n  models: [openai/gpt-image-1]\n"
            + image
            + "search:\n  providers: [serpapi]\n"
        )
        return load_model_config(path).image.placeholder_colour

    assert placeholder_colour('  placeholder_colour: "#123ABC"\n') == "#123ABC"
    assert placeholder_colour("") == "#4A5568"
    with pytest.raises(ValueError, match="placeholder_colour"):
        placeholder_colour("  placeholder_colour: teal\n")
