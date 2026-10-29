from types import SimpleNamespace

from pydantic import BaseModel

from contentblitz.core import call_context
from contentblitz.core.config import DEFAULT_CONFIG_PATH, load_model_config
from contentblitz.core.usage import get_ledger
from contentblitz.integrations import llm

MESSAGES = [{"role": "user", "content": "x"}]


def _response(content, model="openai/gpt-4o", tokens=(100, 50), cost=0.002):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model=model,
        usage=SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1]),
        _hidden_params={"response_cost": cost},
    )


def _fake(monkeypatch, *responses):
    queue = list(responses)
    monkeypatch.setattr(llm.litellm, "completion", lambda **kwargs: queue.pop(0))


def _config():
    config = load_model_config(DEFAULT_CONFIG_PATH)
    config.agents["blog_writer"].models = ["openai/gpt-4o", "anthropic/claude-x"]
    return config


def test_a_completion_records_tokens_cost_latency_and_models(monkeypatch):
    _fake(monkeypatch, _response("hi", model="anthropic/claude-x"))

    llm.complete("blog_writer", MESSAGES, config=_config())

    (event,) = get_ledger().events
    assert (event.agent, event.kind, event.cached) == ("blog_writer", "completion", False)
    assert (event.model, event.primary) == ("anthropic/claude-x", "openai/gpt-4o")
    assert event.fallback
    assert (event.tokens_in, event.tokens_out, event.cost_usd) == (100, 50, 0.002)
    assert event.latency_s >= 0
    assert (event.run_id, event.conversation_id) == (None, None)  # outside a run


class Answer(BaseModel):
    value: int


def test_structured_completions_record_each_attempt_inside_a_run(monkeypatch):
    _fake(monkeypatch, _response("not json"), _response('{"value": 1}', tokens=(7, 3)))

    with call_context.run("conv-1", "run-1"), call_context.agent("blog_writer"):
        llm.complete_structured("blog_writer", MESSAGES, Answer, config=_config())

    events = get_ledger().events
    assert [(e.run_id, e.conversation_id, e.agent) for e in events] == [
        ("run-1", "conv-1", "blog_writer")
    ] * 2
    assert events[1].tokens_in == 7


def test_a_response_without_usage_or_price_records_zeros(monkeypatch):
    bare = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
    _fake(monkeypatch, bare)

    llm.complete("blog_writer", MESSAGES, config=_config())

    (event,) = get_ledger().events
    assert (event.tokens_in, event.tokens_out, event.cost_usd) == (0, 0, 0.0)
    assert event.model == "openai/gpt-4o"
