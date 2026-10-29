"""Where the disk caches live and how they are set up (spec #24, ticket #74).

One place for the cache names, folders and options, so the model response cache, the search and
page text caches (#75), the Research Brief cache (#76) and Clear cache (#77) agree.
"""

from pathlib import Path
from typing import Literal, get_args

from contentblitz.core.config import CacheOptions, ModelConfig, get_model_config, get_settings

CacheName = Literal["model_responses", "search", "page_text", "research_brief"]
CACHE_NAMES: tuple[CacheName, ...] = get_args(CacheName)


def options(name: CacheName, config: ModelConfig | None = None) -> CacheOptions:
    """The on/off switch and expiry configured for the cache `name`."""
    return getattr((config or get_model_config()).cache, name)


def cache_dir(name: CacheName, config: ModelConfig | None = None) -> Path:
    """The cache's folder: `<data folder>/<cache folder>/<name>` (not created here)."""
    folder = (config or get_model_config()).cache.folder
    return get_settings().contentblitz_data_dir / folder / name


def cache_dirs(config: ModelConfig | None = None) -> list[Path]:
    """Every cache folder, for Clear cache (#77)."""
    return [cache_dir(name, config) for name in CACHE_NAMES]
