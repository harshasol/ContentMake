from datetime import UTC, datetime

from contentblitz.core.usage import UsageEvent, UsageLedger, format_caption


def event(**fields) -> UsageEvent:
    base = dict(
        ts="2026-11-04T10:00:00+00:00",
        run_id="run-1",
        conversation_id="conv-1",
        agent="blog_writer",
        kind="completion",
        model="openai/gpt-4o",
        primary="openai/gpt-4o",
        tokens_in=1000,
        tokens_out=500,
        cost_usd=0.25,
        latency_s=1.5,
    )
    return UsageEvent(**{**base, **fields})


def test_events_append_to_the_file_and_reload_in_a_new_instance(tmp_path):
    path = tmp_path / "usage.jsonl"
    UsageLedger(path).record(event())
    UsageLedger(path).record(event(agent="query_handler"))

    reloaded = UsageLedger(path)

    assert [e.agent for e in reloaded.events] == ["blog_writer", "query_handler"]
    assert reloaded.events[0].cost_usd == 0.25
    assert len(path.read_text().splitlines()) == 2


def test_unparseable_lines_are_skipped(tmp_path):
    path = tmp_path / "usage.jsonl"
    ledger = UsageLedger(path)
    ledger.record(event())
    with path.open("a") as f:
        f.write('not json\n{"agent": 3}\n\n')
    ledger.record(event(run_id="run-2"))

    assert [e.run_id for e in UsageLedger(path).events] == ["run-1", "run-2"]


def test_clear_empties_the_ledger_and_the_file(tmp_path):
    path = tmp_path / "usage.jsonl"
    ledger = UsageLedger(path)
    ledger.record(event())

    ledger.clear()

    assert ledger.events == []
    assert UsageLedger(path).events == []


def test_totals_by_conversation_by_day_all_time_and_by_run(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    ledger.record(event())
    ledger.record(event(run_id="run-2", conversation_id="conv-2", cost_usd=0.5, tokens_in=10))
    ledger.record(event(ts="2026-11-03T09:00:00+00:00", run_id="run-0", cost_usd=1.0))

    assert ledger.totals().cost_usd == 1.75
    assert ledger.totals(conversation_id="conv-1").cost_usd == 1.25
    today = datetime(2026, 11, 4, 23, 0, tzinfo=UTC).date()
    assert ledger.totals(day=today).cost_usd == 0.75
    run = ledger.totals(run_id="run-2")
    assert (run.calls, run.tokens_in, run.tokens_out, run.tokens) == (1, 10, 500, 510)


def test_the_caption_reads_the_runs_cost_and_tokens_from_the_ledger(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    ledger.record(event(cost_usd=0.03, tokens_in=8000, tokens_out=2000))
    ledger.record(event(cost_usd=0.013, tokens_in=2000, tokens_out=400))

    assert ledger.caption("run-1") == "This run: $0.043 · 12.4k tokens"
    assert ledger.caption("no-such-run") is None


def test_caption_format_for_small_and_cached_counts():
    assert format_caption(0, 950) == "This run: $0.000 · 950 tokens"
    assert (
        format_caption(cost_usd=0.1, tokens=1000, cached=3, fallbacks=1)
        == "This run: $0.100 · 1.0k tokens · 3 cached · 1 fallback"
    )
