"""The ledger's Usage-tab queries (#73), on a seeded ledger."""

from contentblitz.core.usage import UsageEvent, UsageLedger


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


def seed(ledger: UsageLedger) -> UsageLedger:
    ledger.record_run(
        "run-1", "Write a blog post about caching\nwith examples", "2026-11-04T09:59:58+00:00", 7.5
    )
    ledger.record(event())
    ledger.record(
        event(
            agent="query_handler",
            model="openai/gpt-4o-mini",
            primary="openai/gpt-4o-mini",
            cost_usd=0.05,
            tokens_in=100,
            tokens_out=50,
            latency_s=0.5,
            cached=True,
            ts="2026-11-04T10:00:05+00:00",
        )
    )
    ledger.record(
        event(
            run_id="run-2",
            conversation_id="conv-2",
            agent="researcher",
            kind="search",
            model="tavily",
            primary="brave",
            cost_usd=0.01,
            estimated=True,
            tokens_in=0,
            tokens_out=0,
            latency_s=1.0,
            ts="2026-11-05T09:00:00+00:00",
            failed=[{"provider": "brave", "error": "429 rate limit"}],
        )
    )
    return ledger


def seeded(tmp_path) -> UsageLedger:
    return seed(UsageLedger(tmp_path / "usage.jsonl"))


def test_by_agent_sums_calls_tokens_cost_latency_and_cache_hits(tmp_path):
    rows = {r.name: r for r in seeded(tmp_path).by_agent()}

    assert rows["blog_writer"].calls == 1
    assert rows["blog_writer"].tokens == 1500
    assert rows["blog_writer"].cost_usd == 0.25
    assert rows["query_handler"].cached == 1
    assert rows["query_handler"].avg_latency_s == 0.5
    assert rows["researcher"].estimated is True
    assert rows["blog_writer"].estimated is False


def test_by_model_and_searches_per_provider(tmp_path):
    ledger = seeded(tmp_path)

    assert {r.name: r.calls for r in ledger.by_model()} == {
        "openai/gpt-4o": 1,
        "openai/gpt-4o-mini": 1,
        "tavily": 1,
    }
    assert [(r.name, r.calls) for r in ledger.searches_per_provider()] == [("tavily", 1)]


def test_cache_hit_rate_leaves_images_out_because_they_are_never_cacheable(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    ledger.record(event(cached=True))
    ledger.record(event())
    ledger.record(event(kind="image"))
    ledger.record(event(kind="image"))

    assert ledger.cache_hit_rate() == 1 / 2


def test_cache_hit_rate_is_none_without_calls(tmp_path):
    assert seeded(tmp_path).cache_hit_rate() == 1 / 3
    assert UsageLedger(tmp_path / "empty.jsonl").cache_hit_rate() is None


def test_recent_runs_newest_first_with_the_first_line_of_the_request(tmp_path):
    ledger = seeded(tmp_path)

    newest, oldest = ledger.recent_runs()
    newest_latencies = sum(e.latency_s for e in ledger.select(run_id="run-2"))

    assert newest.run_id == "run-2" and newest.request is None
    assert newest.fallback is True and newest.estimated is True
    assert oldest.request == "Write a blog post about caching"
    assert (oldest.cost_usd, oldest.tokens) == (0.30, 1650)
    assert (oldest.ts, oldest.duration_s) == ("2026-11-04T09:59:58+00:00", 7.5)  # the run record
    assert newest.duration_s == newest_latencies  # no record: the sum of call latencies
    assert oldest.fallback is False
    assert len(ledger.recent_runs(limit=1)) == 1


def test_run_records_survive_a_reload_and_do_not_count_as_events(tmp_path):
    reloaded = UsageLedger(seeded(tmp_path).path)

    assert len(reloaded.events) == 3
    assert reloaded.recent_runs()[1].request == "Write a blog post about caching"


def test_fallback_events_name_the_agent_the_failed_provider_the_one_used_and_the_error(tmp_path):
    [row] = seeded(tmp_path).fallback_events()

    assert (row.agent, row.failed, row.used, row.error) == (
        "researcher",
        "brave",
        "tavily",
        "429 rate limit",
    )


def test_run_records_live_in_their_own_file_next_to_the_events(tmp_path):
    ledger = seeded(tmp_path)

    assert all('"agent"' in line for line in ledger.path.read_text().splitlines())
    [record] = (tmp_path / "runs.jsonl").read_text().splitlines()
    assert '"run-1"' in record and '"duration_s": 7.5' in record


def test_clear_empties_the_run_records_too(tmp_path):
    ledger = seeded(tmp_path)

    ledger.clear()

    assert not (tmp_path / "runs.jsonl").exists()
    assert ledger.runs == {} and UsageLedger(ledger.path).runs == {}


def test_a_run_record_that_cannot_be_written_is_swallowed_by_the_ledger(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the folder should be")
    ledger = UsageLedger(blocker / "usage.jsonl")

    ledger.record_run("run-1", "a request", "2026-11-04T10:00:00+00:00", 1.0)  # must not raise
