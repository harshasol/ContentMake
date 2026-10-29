"""The usage ledger: an append-only record of every paid call (spec #24, ADR 0003).

One JSON line per `UsageEvent`, in `usage.jsonl` under the data folder (one replica, one writer),
with an in-memory index. The Usage tab and the per-run caption read the same queries from here.
"""

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from contentblitz.core.config import get_settings

LEDGER_FILE = "usage.jsonl"
logger = logging.getLogger(__name__)

EventKind = Literal["completion", "image", "search"]


class UsageEvent(BaseModel):
    """One paid call. `run_id` is None outside a run; `model` is the model or search provider
    that actually answered and `primary` the one configured, so a fallback is when they differ.
    `failed` lists earlier attempts as `{"provider": ..., "error": ...}`."""

    ts: str  # ISO 8601, with a UTC offset
    run_id: str | None = None
    conversation_id: str | None = None
    agent: str
    kind: EventKind
    model: str
    primary: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    estimated: bool = False  # the price was unknown, so the cost is a configured estimate
    latency_s: float = 0.0
    cached: bool = False
    failed: list[dict[str, str]] = Field(default_factory=list)

    @property
    def fallback(self) -> bool:
        return bool(self.failed) or (self.primary is not None and self.model != self.primary)

    @property
    def day(self) -> date:
        return datetime.fromisoformat(self.ts).date()


@dataclass(frozen=True)
class Totals:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cached: int = 0
    fallbacks: int = 0
    estimated: bool = False

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


def _sum(events: list[UsageEvent]) -> Totals:
    return Totals(
        calls=len(events),
        tokens_in=sum(e.tokens_in for e in events),
        tokens_out=sum(e.tokens_out for e in events),
        cost_usd=round(sum(e.cost_usd for e in events), 10),
        cached=sum(e.cached for e in events),
        fallbacks=sum(e.fallback for e in events),
        estimated=any(e.estimated for e in events),
    )


def _short(tokens: int) -> str:
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


def format_caption(cost_usd: float, tokens: int, cached: int = 0, fallbacks: int = 0) -> str:
    """The per-run caption, e.g. "This run: $0.043 · 12.4k tokens · 3 cached · 1 fallback"."""
    parts = [f"This run: ${cost_usd:.3f}", f"{_short(tokens)} tokens"]
    if cached:
        parts.append(f"{cached} cached")
    if fallbacks:
        parts.append(f"{fallbacks} fallback{'s' if fallbacks > 1 else ''}")
    return " · ".join(parts)


class UsageLedger:
    def __init__(self, path: Path):
        self.path = path
        self.events: list[UsageEvent] = self._load()

    def _load(self) -> list[UsageEvent]:
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(UsageEvent.model_validate_json(line))
            except (ValidationError, ValueError):
                continue  # a torn or foreign line must not hide the rest
        return events

    def record(self, event: UsageEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")
        self.events.append(event)

    def clear(self) -> None:
        self.events = []
        self.path.unlink(missing_ok=True)

    def select(
        self,
        run_id: str | None = None,
        conversation_id: str | None = None,
        day: date | None = None,
    ) -> list[UsageEvent]:
        """The events matching every filter given; no filter means all of them."""
        return [
            e
            for e in self.events
            if (run_id is None or e.run_id == run_id)
            and (conversation_id is None or e.conversation_id == conversation_id)
            and (day is None or e.day == day)
        ]

    def totals(
        self,
        run_id: str | None = None,
        conversation_id: str | None = None,
        day: date | None = None,
    ) -> Totals:
        return _sum(self.select(run_id, conversation_id, day))

    def caption(self, run_id: str) -> str | None:
        """The run's one-line cost summary, or None when the run made no recorded call."""
        totals = self.totals(run_id=run_id)
        if not totals.calls:
            return None
        return format_caption(totals.cost_usd, totals.tokens, totals.cached, totals.fallbacks)


_ledgers: dict[Path, UsageLedger] = {}


def get_ledger() -> UsageLedger:
    """The ledger for the configured data folder (one shared instance per file)."""
    path = get_settings().contentblitz_data_dir / LEDGER_FILE
    if path not in _ledgers:
        _ledgers[path] = UsageLedger(path)
    return _ledgers[path]


def record(event: UsageEvent) -> None:
    """Record `event`; a ledger that can't be written never fails the call being recorded."""
    try:
        get_ledger().record(event)
    except OSError:
        logger.exception("couldn't write the usage ledger")
