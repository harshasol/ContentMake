"""The search and page text caches (spec #24, ticket #75).

Disk caches in the `search` and `page_text` cache folders (`core/cache.py`), each with its own
switch and expiry from config. An entry stores when it was saved, so expiry follows `clock`
(tests replace it). `close_all()` closes the open handles, like `workflow/brief_cache.py`.
"""

import re
import time
from typing import Any

import diskcache

from contentblitz.core import cache
from contentblitz.core.cache import CacheName
from contentblitz.core.config import get_model_config

clock = time.time  # seconds; tests inject a fake

_open: dict[str, diskcache.Cache] = {}


def _store(name: CacheName) -> diskcache.Cache | None:
    """The open cache `name`, or None when it is switched off (nothing is created)."""
    config = get_model_config()
    if not cache.options(name, config).enabled:
        return None
    folder = str(cache.cache_dir(name, config))
    if folder not in _open:
        _open[folder] = diskcache.Cache(folder)
    return _open[folder]


def get(name: CacheName, key: tuple) -> Any | None:
    """The value saved under `key` that hasn't expired, else None."""
    store = _store(name)
    if store is None or (entry := store.get(key)) is None:
        return None
    saved_at, value = entry
    expiry = cache.options(name).expiry_seconds
    if expiry is not None and clock() - saved_at >= expiry:
        return None
    return value


def put(name: CacheName, key: tuple, value: Any) -> None:
    """Save `value` under `key`, replacing any entry."""
    store = _store(name)
    if store is not None:
        store.set(key, (clock(), value))


def normalise_query(query: str) -> str:
    """Lower case with runs of whitespace collapsed, so trivial variants share an entry."""
    return re.sub(r"\s+", " ", query).strip().lower()


def close_all() -> None:
    """Close every open cache (tests use it to stand in for a new process)."""
    for store in _open.values():
        store.close()
    _open.clear()
