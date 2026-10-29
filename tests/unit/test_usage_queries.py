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
    ledger.record_request("run-1", "Write a blog post about caching\nwith examples")
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


def test_cache_hit_rate_is_none_without_calls(tmp_path):
    assert seeded(tmp_path).cache_hit_rate() == 1 / 3
    assert UsageLedger(tmp_path / "empty.jsonl").cache_hit_rate() is None


def test_recent_runs_newest_first_with_the_first_line_of_the_request(tmp_path):
    ledger = seeded(tmp_path)

    newest, oldest = ledger.recent_runs()

    assert newest.run_id == "run-2" and newest.request is None
    assert newest.fallback is True and newest.estimated is True
    assert oldest.request == "Write a blog post about caching"
    assert (oldest.cost_usd, oldest.tokens, oldest.duration_s) == (0.30, 1650, 2.0)
    assert oldest.fallback is False
    assert len(ledger.recent_runs(limit=1)) == 1


def test_requests_survive_a_reload_and_do_not_count_as_events(tmp_path):
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
