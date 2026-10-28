"""Optional LLM tracing via Langfuse.

`observe()` opens a Langfuse observation (trace / span / generation) when
LANGFUSE_* keys are configured and the `langfuse` package is installed;
otherwise it is a no-op. Call sites look identical either way:

    with observe("llm_call", as_type="generation", model=...) as obs:
        ...
        obs.update(output=..., usage_details={...})

Nested `observe()` calls become nested spans, so one claim shows up in
Langfuse as a single trace with its LLM calls, tool calls and retrieval.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from app.config import get_settings
from app.core.logging import claim_id_var, get_logger, request_id_var

logger = get_logger(__name__)


class _NoOpObservation:
    def update(self, **_: Any) -> None:
        return None


_client: Any = None
_initialized = False


def _get_client() -> Any:
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("Langfuse keys are set but the 'langfuse' package is not installed; tracing disabled")
        return None
    _client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )
    logger.info("Langfuse tracing enabled")
    return _client


@contextmanager
def observe(name: str, *, as_type: str = "span", **attributes: Any):
    client = _get_client()
    if client is None:
        yield _NoOpObservation()
        return
    from langfuse import propagate_attributes

    # Tag spans with the same ids the JSON logs carry, so a run can be found in
    # Langfuse by session (= claim_id) or by metadata request_id.
    request_id = request_id_var.get()
    with (
        propagate_attributes(
            session_id=claim_id_var.get(),
            metadata={"request_id": request_id} if request_id else None,
        ),
        client.start_as_current_observation(name=name, as_type=as_type, **attributes) as observation,
    ):
        yield observation


def trace_carrier() -> dict[str, str]:
    """The current trace context as W3C headers (`traceparent`), for handing
    to another request -- e.g. an MCP tool call's `_meta` -- so spans created
    there join this trace instead of starting a new one."""
    if _get_client() is None:
        return {}
    from opentelemetry import propagate

    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


@contextmanager
def continue_trace(carrier: dict[str, Any] | None):
    """Make the trace context in `carrier` (from `trace_carrier()`) current,
    so `observe()` calls inside nest under the caller's span."""
    if not carrier or _get_client() is None:
        yield
        return
    from opentelemetry import context, propagate

    token = context.attach(propagate.extract(carrier))
    try:
        yield
    finally:
        context.detach(token)


def flush() -> None:
    if _client is not None:
        _client.flush()
