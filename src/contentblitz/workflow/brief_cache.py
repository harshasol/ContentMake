"""The Research Brief cache. See spec #24, ticket #76.

A Brief built from identical research is reused. The key is a hash of the Research Record's
Sources (URLs and text), the Content Strategist step's instructions and the Strategist's primary
model; it never expires, because fresh research has new Sources and so misses. A hit is recorded
in the usage ledger as a cached completion at no cost. The cache lives in the
`research_brief` cache folder (`core/cache.py`) and is switched off in config.
"""

import hashlib
import json

import diskcache

from contentblitz.core import cache, usage
from contentblitz.core.config import get_model_config
from contentblitz.workflow.state import ResearchBrief, ResearchRecord

AGENT = "content_strategist"


def brief_key(record: ResearchRecord, instructions: str | None, model: str) -> str:
    """Hash of the Sources' URLs and text, the instructions and the Strategist's primary model
    (its fallbacks are left out: they only matter when the primary fails)."""
    sources = [
        [s["url"], s["title"], s["site"], s["snippet"], s["page_text"]] for s in record["sources"]
    ]
    payload = json.dumps([sources, instructions or "", model], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache() -> diskcache.Cache | None:
    """The open Brief cache, or None when it is switched off."""
    return cache.open_cache("research_brief", get_model_config())


def primary_model() -> str:
    return get_model_config().for_agent(AGENT).primary


def cached_brief(record: ResearchRecord, instructions: str | None) -> ResearchBrief | None:
    """The Brief stored for this research, instructions and model, if any."""
    store = _cache()
    if store is None:
        return None
    brief = store.get(brief_key(record, instructions, primary_model()))
    if brief is not None:
        usage.record(
            usage.event(
                "completion",
                agent=AGENT,
                model=primary_model(),
                primary=primary_model(),
                cached=True,
            )
        )
    return brief


def store_brief(record: ResearchRecord, instructions: str | None, brief: ResearchBrief) -> None:
    """Keep `brief` for the same research, instructions and model; no expiry."""
    store = _cache()
    if store is not None:
        store.set(brief_key(record, instructions, primary_model()), brief)
