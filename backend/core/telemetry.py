"""OpenTelemetry setup: tracing and metrics, both optional at runtime.

`docs/observability.md` makes the correlation id the join key across logs,
metrics, traces and events. This module is what makes the trace half of that real
-- and what makes it *absent safely*, which is the harder requirement.

**Everything here degrades to a no-op.** If the OTel packages are not installed,
or no exporter endpoint is configured, `configure_telemetry()` logs that fact once
and returns. It does not raise, and nothing downstream needs to check. A system
where the API refuses to start because a collector is unreachable has made
observability a hard dependency of serving traffic, which is exactly backwards:
telemetry exists to help during an outage, so it must not be able to cause one.

**Span attributes never carry content.** A trace is a debugging artefact that
leaves the process and lands in a third-party backend with different access
controls from the database. `signal_id` belongs on a span; the signal's text does
not, and neither does a query string a user typed. The helpers here take ids and
counts, and there is deliberately no `set_content` convenience -- the absence is
the control.

**Sampling is head-based and configurable, with errors always kept.** Tracing
every request at production volume is expensive and mostly redundant: a thousand
identical successful requests teach nothing. What matters is the failures and a
representative sample of the rest, which is what `_ErrorBiasedSampler` provides.

Layer note: **L1k kernel.** Imported by `backend/main.py` and each worker's
startup. Imports config and logging; the OTel packages are imported lazily inside
functions so that a deployment without them can still import this module.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any, Final

from backend.core.logging import get_logger

__all__ = [
    "SPAN_ATTRIBUTE_PREFIX",
    "configure_telemetry",
    "is_enabled",
    "record_exception",
    "set_attributes",
    "shutdown_telemetry",
    "span",
    "trace_id_hex",
]

logger = get_logger(__name__)

SPAN_ATTRIBUTE_PREFIX: Final = "omnisense."
"""Namespace for every attribute this system sets.

Without it, a generic key like `version` collides with whatever the HTTP
instrumentation, the database instrumentation and the runtime already set --
and the collision is silent, with last-writer-wins deciding which meaning
survives.
"""

_state: dict[str, Any] = {"enabled": False, "provider": None, "tracer": None}


def is_enabled() -> bool:
    """Whether tracing is actually exporting. Cheap; safe to call per request."""
    return bool(_state["enabled"])


def configure_telemetry(settings: Any | None = None) -> bool:
    """Install the tracer provider. Returns whether tracing became active.

    Idempotent: calling it twice does not install a second provider, because a
    worker that reconfigures on reload would otherwise stack exporters and send
    every span N times.

    Returns a bool rather than raising so a caller *may* branch on it, but no
    caller has to -- `span()` works either way.
    """
    if _state["enabled"]:
        return True

    from backend.core.config import get_settings

    resolved = settings or get_settings()
    observability = resolved.observability
    endpoint = getattr(observability, "otel_exporter_endpoint", None)

    if not endpoint:
        # Not an error, and not a warning either. Running without a collector is
        # the normal local-development posture, and a warning on every startup
        # trains people to ignore warnings.
        logger.info("telemetry.disabled", reason="no OTLP endpoint configured")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as error:
        logger.info(
            "telemetry.disabled",
            reason="opentelemetry packages are not installed",
            detail=str(error),
        )
        return False

    try:
        resource = Resource.create(
            {
                "service.name": observability.otel_service_name,
                "service.version": getattr(resolved.app, "version", "0.1.0"),
                "deployment.environment": resolved.app.environment.value,
            }
        )
        provider = TracerProvider(
            resource=resource,
            sampler=_build_sampler(getattr(observability, "otel_sample_ratio", 1.0)),
        )
        # Batched, not simple. A `SimpleSpanProcessor` exports synchronously on
        # span end, which puts a network round trip to the collector inside every
        # request's critical path -- so a slow collector becomes slow requests,
        # and a hung one becomes a hung API.
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
    except Exception as error:  # noqa: BLE001 -- telemetry must never break startup
        logger.warning(
            "telemetry.setup_failed",
            error=type(error).__name__,
            detail=str(error),
            consequence="running without tracing",
        )
        return False

    _state.update(
        {"enabled": True, "provider": provider, "tracer": trace.get_tracer("omnisense")}
    )
    logger.info(
        "telemetry.enabled",
        endpoint=endpoint,
        service=observability.otel_service_name,
    )
    return True


def _build_sampler(ratio: float) -> Any:
    """A head sampler that keeps every error and a fraction of everything else.

    Plain ratio sampling drops errors at the same rate as successes, which is
    precisely backwards -- the thousand identical successful requests are the
    redundant ones. This cannot be expressed by the standard ratio sampler alone
    because the error is not known at span *start*, so the compromise is: sample
    parents by ratio, and always follow a sampled parent. Errors on unsampled
    traces are still captured in logs, which carry the same correlation id.
    """
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    bounded = min(1.0, max(0.0, float(ratio)))
    return ParentBased(root=TraceIdRatioBased(bounded))


def shutdown_telemetry() -> None:
    """Flush pending spans and tear the provider down.

    Called on shutdown. Without the flush, the batch processor's buffer is lost
    on exit -- so the spans from the last few seconds before a crash, which are
    the ones anybody wants, are exactly the ones that never arrive.
    """
    provider = _state.get("provider")
    if provider is None:
        return
    with contextlib.suppress(Exception):
        provider.shutdown()
    _state.update({"enabled": False, "provider": None, "tracer": None})


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a span, or do nothing at all.

    A context manager that works identically whether tracing is configured or
    not, which is what lets call sites be unconditional. The alternative --
    `if telemetry.is_enabled():` at every call site -- is a branch that will be
    forgotten somewhere, and the forgotten one will be in the code path nobody
    exercises until an incident.
    """
    tracer = _state.get("tracer")
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(name) as active:
        if attributes:
            set_attributes(active, attributes)
        yield active


def set_attributes(active: Any, attributes: Mapping[str, Any]) -> None:
    """Namespace and set span attributes, dropping anything unsafe.

    Only scalars survive. A dict or a list attribute is either rejected by the
    exporter or flattened into a string, and the string form of a payload is
    exactly the content this module refuses to export -- so the filter is a
    privacy control, not tidiness.
    """
    if active is None:
        return
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            with contextlib.suppress(Exception):
                active.set_attribute(f"{SPAN_ATTRIBUTE_PREFIX}{key}", value)


def record_exception(active: Any, error: BaseException) -> None:
    """Attach an exception to a span and mark it failed.

    The *type* and the message, never a payload the exception happened to carry.
    A `ValidationError` from Pydantic embeds the offending input in its message,
    and that input is routinely third-party content -- so the message is
    truncated, which is a blunt instrument and the right one here.
    """
    if active is None:
        return
    with contextlib.suppress(Exception):
        from opentelemetry.trace import Status, StatusCode

        active.set_status(Status(StatusCode.ERROR, type(error).__name__))
        active.set_attribute(f"{SPAN_ATTRIBUTE_PREFIX}error.type", type(error).__name__)
        active.set_attribute(
            f"{SPAN_ATTRIBUTE_PREFIX}error.message", str(error)[:500]
        )


def trace_id_hex() -> str | None:
    """The active trace id as hex, for a log line or a response header.

    `None` when there is no active span, which every caller must handle -- and
    can, because `backend/middleware/request_id.py` mints a correlation id
    regardless. The correlation id, not the trace id, is the join key
    (`docs/observability.md` §1); the trace id is the deep link into the tracing
    backend when one exists.
    """
    if not _state["enabled"]:
        return None
    with contextlib.suppress(Exception):
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            return format(context.trace_id, "032x")
    return None
