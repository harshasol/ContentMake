import pytest

from contentblitz.core import config
from contentblitz.integrations import llm


@pytest.fixture(autouse=True)
def no_real_image_generation(monkeypatch, tmp_path):
    """No test calls a paid image API or writes images outside a temporary data folder; tests
    that make images fake `llm.generate_image` (or LiteLLM's image call) themselves."""

    def refuse(**kwargs):
        raise RuntimeError("tests must not call a real image model")

    monkeypatch.setattr(llm.litellm, "image_generation", refuse)
    monkeypatch.setenv("CONTENTBLITZ_DATA_DIR", str(tmp_path / "data"))


@pytest.fixture(autouse=True)
def clear_config_caches():
    config.get_settings.cache_clear()
    config.get_model_config.cache_clear()
    yield
    config.get_settings.cache_clear()
    config.get_model_config.cache_clear()


@pytest.fixture(autouse=True)
def isolated_model_cache(monkeypatch):
    """LiteLLM's cache is a process global; every test starts with none and leaves none, so the
    per-test data folder (and so cache folder) is what keeps caches apart."""
    monkeypatch.setattr(llm.litellm, "cache", None)
    monkeypatch.setattr(llm, "_cache_key", None)
