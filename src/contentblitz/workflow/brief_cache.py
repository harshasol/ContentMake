"""The Research Brief cache. See spec #24, ticket #76.

A Brief built from identical research is reused. The key is a hash of the Research Record's
Sources (URLs and text), the Content Strategist step's instructions and the Strategist model;
it never expires, because fresh research has new Sources and so misses. The cache lives in the
`research_brief` cache folder (`core/cache.py`) and is switched off in config.
"""

import hashlib
import json

import diskcache

from contentblitz.core import cache
from contentblitz.core.config import get_model_config
from contentblitz.workflow.state import ResearchBrief, ResearchRecord

AGENT = "content_strategist"

_open: dict[str, diskcache.Cache] = {}


def brief_key(record: ResearchRecord, instructions: str | None, models: list[str]) -> str:
    """Hash of the Sources' URLs and text, the instructions and the Strategist models."""
    sources = [
        [s["url"], s["title"], s["site"], s["snippet"], s["page_text"]] for s in record["sources"]
    ]
    payload = json.dumps([sources, instructions or "", models], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache() -> diskcache.Cache | None:
    """The open Brief cache, or None when it is switched off."""
    config = get_model_config()
    if not cache.options("research_brief", config).enabled:
        return None
    folder = str(cache.cache_dir("research_brief", config))
    if folder not in _open:
        _open[folder] = diskcache.Cache(folder)
    return _open[folder]


def _models() -> list[str]:
    agent = get_model_config().for_agent(AGENT)
    return [agent.primary, *agent.fallbacks]


def cached_brief(record: ResearchRecord, instructions: str | None) -> ResearchBrief | None:
    """The Brief stored for this research, instructions and model, if any."""
    store = _cache()
    if store is None:
        return None
    return store.get(brief_key(record, instructions, _models()))


def store_brief(record: ResearchRecord, instructions: str | None, brief: ResearchBrief) -> None:
    """Keep `brief` for the same research, instructions and model; no expiry."""
    store = _cache()
    if store is not None:
        store.set(brief_key(record, instructions, _models()), brief)


def close_all() -> None:
    """Close every open cache (tests use it to stand in for a new process)."""
    for store in _open.values():
        store.close()
    _open.clear()
