"""LangSmith tracing: one trace per run, switched on by the settings (spec #24, ticket #78).

Tracing is on only when `LANGSMITH_TRACING` is true and a LangSmith API key is set; otherwise the
workspace runs inside `tracing_context(enabled=False)`, so the spans the integration modules open
send nothing. The `run_trace` run is the root trace (its id is the run id), each plan step is nested
under it (inside LangGraph's own run), and `span` is how the integration modules nest provider
calls under the step.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import langsmith
from langsmith import Client, tracing_context

from contentblitz.core.config import Settings, get_settings

_clients: dict[str, Client] = {}


def is_enabled(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return settings.langsmith_tracing and bool(settings.langsmith_api_key)


def make_client(settings: Settings) -> Client:
    """The LangSmith client for the configured key, built once."""
    key = settings.langsmith_api_key or ""
    if key not in _clients:
        _clients[key] = Client(api_key=key)
    return _clients[key]


@contextmanager
def tracing() -> Iterator[None]:
    """Everything inside is traced to LangSmith when tracing is on, and sends nothing otherwise."""
    settings = get_settings()
    if is_enabled(settings):
        with tracing_context(
            enabled=True, client=make_client(settings), project_name=settings.langsmith_project
        ):
            yield
    else:
        with tracing_context(enabled=False):
            yield


@contextmanager
def run_trace(
    run_id: str, conversation_id: str, brand_profile_id: str | None
) -> Iterator[langsmith.RunTree]:
    """The run's root trace: its id is the ledger run id, tagged for finding it. The graph's
    steps nest under it; add the `topic` metadata before leaving, once the run has settled on
    one."""
    with langsmith.trace(
        "ContentBlitz run",
        "chain",
        run_id=uuid.UUID(run_id),
        tags=[f"conversation:{conversation_id}", f"run:{run_id}"],
        metadata={
            "conversation_id": conversation_id,
            "run_id": run_id,
            "brand_profile_id": brand_profile_id,
        },
    ) as root:
        yield root


@contextmanager
def span(
    name: str, run_type: str = "tool", inputs: dict | None = None, metadata: dict | None = None
) -> Iterator[langsmith.RunTree]:
    """A span nested under the current step. Call `.add_metadata` and `.end(outputs=...)` on it."""
    with langsmith.trace(name, run_type, inputs=inputs or {}, metadata=metadata) as run:
        yield run


def annotate(metadata: dict) -> None:
    """Put `metadata` on the span the caller is running in (a graph node, say); a no-op when
    tracing is off."""
    if run := langsmith.get_current_run_tree():
        run.add_metadata(metadata)
