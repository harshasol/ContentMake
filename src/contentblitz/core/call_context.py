"""Call context: which run, conversation and agent a paid call belongs to (spec #24).

The workspace opens a `run` for each workflow invocation, and the graph opens an `agent` for each
step. The integration modules read `current()` when they record usage, so agents pass nothing.
A call outside a run (e.g. "Learn voice") has no run id.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CallContext:
    run_id: str | None = None
    conversation_id: str | None = None
    agent: str | None = None


_current: ContextVar[CallContext | None] = ContextVar("call_context", default=None)


def current() -> CallContext:
    return _current.get() or CallContext()


def new_run_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def run(conversation_id: str, run_id: str | None = None) -> Iterator[str]:
    """Make everything called inside belong to one run of `conversation_id`; yields the run id."""
    run_id = run_id or new_run_id()
    token = _current.set(CallContext(run_id, conversation_id, None))
    try:
        yield run_id
    finally:
        _current.reset(token)


@contextmanager
def agent(name: str) -> Iterator[None]:
    """Make calls inside belong to the agent `name`, within the current run (if any)."""
    token = _current.set(replace(current(), agent=name))
    try:
        yield
    finally:
        _current.reset(token)
