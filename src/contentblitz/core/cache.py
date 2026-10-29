"""Where the disk caches live and how they are set up (spec #24, ticket #74).

One place for the cache names, folders and options, so the model response cache, the search and
page text caches (#75), the Research Brief cache (#76) and Clear cache (#77) agree.
"""

import threading
from pathlib import Path
from typing import Literal, get_args

import diskcache

from contentblitz.core.config import CacheOptions, ModelConfig, get_model_config, get_settings

CacheName = Literal["model_responses", "search", "page_text", "research_brief"]
CACHE_NAMES: tuple[CacheName, ...] = get_args(CacheName)

_open: dict[str, diskcache.Cache] = {}
_lock = threading.Lock()


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


def open_cache(name: CacheName, config: ModelConfig | None = None) -> diskcache.Cache | None:
    """The open disk cache `name` (one shared handle per folder), or None when it is switched
    off (nothing is created). Safe to call from worker threads."""
    config = config or get_model_config()
    if not options(name, config).enabled:
        return None
    folder = str(cache_dir(name, config))
    with _lock:
        if folder not in _open:
            _open[folder] = diskcache.Cache(folder)
        return _open[folder]


def close_all() -> None:
    """Close every handle `open_cache` gave out; the next use reopens. Clear cache (#77) calls it
    before deleting the folders, and tests use it to stand in for a new process."""
    with _lock:
        for store in _open.values():
            store.close()
        _open.clear()
