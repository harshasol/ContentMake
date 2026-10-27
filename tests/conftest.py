import pytest

from contentblitz.core import config


@pytest.fixture(autouse=True)
def clear_config_caches():
    config.get_settings.cache_clear()
    config.get_model_config.cache_clear()
    yield
    config.get_settings.cache_clear()
    config.get_model_config.cache_clear()
