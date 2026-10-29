"""The search and page text caches (spec #24, ticket #75).

Disk caches in the `search` and `page_text` cache folders (`core/cache.py`), each with its own
switch and expiry from config. An entry stores when it was saved, so expiry follows `clock`
(tests replace it). The handles come from `cache.open_cache`, which `cache.close_all()` closes.
"""

import re
import time
from typing import Any

import diskcache

from contentblitz.core import cache
from contentblitz.core.cache import CacheName
from contentblitz.core.config import get_model_config

clock = time.time  # seconds; tests inject a fake


def _store(name: CacheName) -> diskcache.Cache | None:
    """The open cache `name`, or None when it is switched off (nothing is created)."""
    return cache.open_cache(name, get_model_config())


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


def delete(name: CacheName, key: tuple) -> None:
    """Drop the entry under `key`, if any."""
    store = _store(name)
    if store is not None:
        store.delete(key)


def normalise_query(query: str) -> str:
    """Lower case with runs of whitespace collapsed, so trivial variants share an entry."""
    return re.sub(r"\s+", " ", query).strip().lower()
