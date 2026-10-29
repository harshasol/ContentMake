"""The usage ledger: an append-only record of every paid call (spec #24, ADR 0003).

One JSON line per `UsageEvent`, in `usage.jsonl` under the data folder (one replica, one writer),
with an in-memory index. The Usage tab and the per-run caption read the same queries from here.
"""

import json
import logging
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from contentblitz.core import call_context
from contentblitz.core.config import get_settings

LEDGER_FILE = "usage.jsonl"
RUNS_FILE = "runs.jsonl"  # per-run records, next to the ledger
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


def event(kind: EventKind, **fields) -> UsageEvent:
    """A `UsageEvent` of `kind` stamped now and tagged with the current run and conversation."""
    context = call_context.current()
    return UsageEvent(
        ts=datetime.now(UTC).isoformat(),
        run_id=context.run_id,
        conversation_id=context.conversation_id,
        kind=kind,
        **fields,
    )


@dataclass(frozen=True)
class RunRecord:
    """What the workspace knows about a run that its events don't: the marketer's request, when
    it started (ISO 8601) and how long it took."""

    run_id: str
    request: str
    started: str
    duration_s: float


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


@dataclass(frozen=True)
class Breakdown:
    """The events sharing one `name` (an agent, a model or a search provider), summed."""

    name: str
    calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    avg_latency_s: float
    cached: int
    estimated: bool

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass(frozen=True)
class RunRow:
    """One run: when it started, the first line of the marketer's request (None when the run
    wasn't started by a request or predates the record), and what it cost."""

    run_id: str
    ts: str
    request: str | None
    cost_usd: float
    tokens: int
    duration_s: float
    fallback: bool
    estimated: bool


@dataclass(frozen=True)
class FallbackRow:
    ts: str
    agent: str
    failed: str  # the provider that failed (the configured one when no attempt was recorded)
    used: str
    error: str


def _breakdown(name: str, events: list[UsageEvent]) -> Breakdown:
    totals = _sum(events)
    return Breakdown(
        name=name,
        calls=totals.calls,
        tokens_in=totals.tokens_in,
        tokens_out=totals.tokens_out,
        cost_usd=totals.cost_usd,
        avg_latency_s=round(sum(e.latency_s for e in events) / len(events), 10),
        cached=totals.cached,
        estimated=totals.estimated,
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
        self.runs_path = path.with_name(RUNS_FILE)
        self.events: list[UsageEvent] = []
        self.runs: dict[str, RunRecord] = {}
        self._load()

    @staticmethod
    def _lines(path: Path) -> list[dict]:
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                data = json.loads(line)
            except ValueError:
                continue  # a torn line must not hide the rest
            if isinstance(data, dict):
                records.append(data)
        return records

    def _load(self) -> None:
        for data in self._lines(self.path):
            try:
                self.events.append(UsageEvent.model_validate(data))
            except ValidationError:
                continue
        for data in self._lines(self.runs_path):
            try:
                run = RunRecord(
                    str(data["run_id"]),
                    str(data["request"]),
                    str(data["started"]),
                    float(data["duration_s"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            self.runs[run.run_id] = run

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record(self, event: UsageEvent) -> None:
        self._append(self.path, event.model_dump())
        self.events.append(event)

    def record_run(self, run_id: str, request: str, started: str, duration_s: float) -> None:
        """Remember a run's request, start and measured duration for the recent-runs list. The
        record is a label: a file that can't be written is logged, never raised."""
        run = RunRecord(run_id, request, started, duration_s)
        try:
            self._append(self.runs_path, asdict(run))
        except OSError:
            logger.exception("couldn't write the run records")
        self.runs[run_id] = run

    def clear(self) -> None:
        self.events = []
        self.runs = {}
        self.path.unlink(missing_ok=True)
        self.runs_path.unlink(missing_ok=True)

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

    def _grouped(self, key, events: list[UsageEvent]) -> list[Breakdown]:
        groups: dict[str, list[UsageEvent]] = defaultdict(list)
        for e in events:
            groups[key(e)].append(e)
        rows = [_breakdown(name, group) for name, group in groups.items()]
        return sorted(rows, key=lambda r: (-r.cost_usd, r.name))

    def by_agent(self) -> list[Breakdown]:
        """Calls, tokens, cost, average latency and cache hits per agent, dearest first."""
        return self._grouped(lambda e: e.agent, self.events)

    def by_model(self) -> list[Breakdown]:
        """The same per model (or search provider) that answered."""
        return self._grouped(lambda e: e.model, self.events)

    def searches_per_provider(self) -> list[Breakdown]:
        return self._grouped(lambda e: e.model, [e for e in self.events if e.kind == "search"])

    def cache_hit_rate(self) -> float | None:
        """The share of recorded calls answered from a cache, or None before any call. Images are
        left out: they are never cacheable."""
        events = [e for e in self.events if e.kind != "image"]
        return sum(e.cached for e in events) / len(events) if events else None

    def recent_runs(self, limit: int = 20) -> list[RunRow]:
        """The latest runs, newest first; calls outside a run are not listed."""
        runs: dict[str, list[UsageEvent]] = defaultdict(list)
        for e in self.events:
            if e.run_id is not None:
                runs[e.run_id].append(e)
        rows = []
        for run_id, events in runs.items():
            totals = _sum(events)
            record = self.runs.get(run_id)
            request = (record.request if record else "").strip()
            rows.append(
                RunRow(
                    run_id=run_id,
                    ts=record.started if record else min(e.ts for e in events),
                    request=request.splitlines()[0] if request else None,
                    cost_usd=totals.cost_usd,
                    tokens=totals.tokens,
                    duration_s=(
                        record.duration_s if record else round(sum(e.latency_s for e in events), 10)
                    ),
                    fallback=bool(totals.fallbacks),
                    estimated=totals.estimated,
                )
            )
        return sorted(rows, key=lambda r: r.ts, reverse=True)[:limit]

    def fallback_events(self) -> list[FallbackRow]:
        """Every call a fallback answered, newest first."""
        rows = []
        for e in sorted(self.events, key=lambda e: e.ts, reverse=True):
            if not e.fallback:
                continue
            rows.append(
                FallbackRow(
                    ts=e.ts,
                    agent=e.agent,
                    failed=e.failed[0]["provider"] if e.failed else (e.primary or ""),
                    used=e.model,
                    error="; ".join(f["error"] for f in e.failed if f.get("error")),
                )
            )
        return rows

    def caption(self, run_id: str) -> str | None:
        """The run's one-line cost summary, or None when the run made no recorded call."""
        totals = self.totals(run_id=run_id)
        if not totals.calls:
            return None
        return format_caption(totals.cost_usd, totals.tokens, totals.cached, totals.fallbacks)


_ledgers: dict[Path, UsageLedger] = {}
_ledgers_lock = threading.Lock()


def get_ledger() -> UsageLedger:
    """The ledger for the configured data folder (one shared instance per file)."""
    path = get_settings().contentblitz_data_dir / LEDGER_FILE
    with _ledgers_lock:
        if path not in _ledgers:
            _ledgers[path] = UsageLedger(path)
        return _ledgers[path]


def record(event: UsageEvent) -> None:
    """Record `event`; a ledger that can't be written never fails the call being recorded."""
    try:
        get_ledger().record(event)
    except OSError:
        logger.exception("couldn't write the usage ledger")
