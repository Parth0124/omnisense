"""Renders finished investigations into stored, downloadable reports.

Separate from the investigation graph on purpose. The Report *agent* produces the
structured document inside the run; this worker turns that structure into the
artefacts a person downloads -- Markdown, HTML, PDF -- and puts them in object
storage.

**Why that is not a step in the graph.** Rendering is slow, format-specific, and
occasionally fails for reasons that have nothing to do with the investigation: a
PDF engine that cannot find a font, an object store that is briefly unwritable.
Making it a graph node would mean a font problem marks a completed investigation
as failed, destroying an analysis that is finished and correct. Here, a render
failure leaves the investigation completed and the artefact absent, which is
recoverable by re-running the render.

**Idempotent by content, not by attempt.** The artefact key is derived from the
investigation id and the report version, so re-rendering overwrites in place. A
timestamped key would accumulate one copy per retry, and the API would have to
guess which was current.

**Formats degrade independently.** Markdown always works -- it is string
concatenation. HTML nearly always works. PDF needs an engine that may not be
installed. Rendering them in that order and recording each success separately
means a missing PDF engine costs the PDF, not the Markdown.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from backend.core.logging import get_logger
from services.events.consumer import ConsumedMessage
from services.events.topics import TopicRole
from workers.runtime.base_worker import ConsumerWorker, run_worker

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from backend.core.config import Settings
    from workers.runtime.health import DependencyProbe

__all__ = ["RenderFormat", "RenderedArtifact", "ReportWorker", "main", "render_markdown"]

logger = get_logger(__name__)

WORKER_NAME: Final = "report"


class RenderFormat(enum.StrEnum):
    """Output formats, in the order they are attempted.

    Ordered by how likely each is to work. Markdown is string concatenation and
    cannot fail; PDF needs an engine that may not be installed in a given image.
    Attempting them in this order means a partial success is always the *most*
    useful subset rather than an arbitrary one.
    """

    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"


@dataclass(frozen=True, slots=True)
class RenderedArtifact:
    format: RenderFormat
    key: str
    content_type: str
    size_bytes: int


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the structured report as Markdown.

    A pure function over the stored document -- no template engine, no I/O, no
    model. That makes it testable against a fixture and means a rendering bug is
    found by a unit test rather than by a reader.

    The gaps section is rendered *last and always*, matching
    `agents/report/schemas.py`: the promise that a smaller answer is honestly
    labelled only holds if the label survives every layer, and a renderer that
    skipped an empty-looking section would break it at the last step.
    """
    lines: list[str] = []
    title = report.get("title") or "Investigation report"
    lines.append(f"# {title}")
    lines.append("")

    band = report.get("confidence_band")
    confidence = report.get("confidence")
    if band or confidence is not None:
        parts = []
        if band:
            parts.append(f"**Confidence: {str(band).upper()}**")
        if isinstance(confidence, (int, float)):
            parts.append(f"({confidence:.2f})")
        lines.append(" ".join(parts))
        lines.append("")

    summary = report.get("executive_summary")
    if summary:
        lines.extend(["## Executive summary", "", str(summary), ""])

    sections = report.get("sections") or []
    for section in sorted(
        (s for s in sections if isinstance(s, Mapping)),
        key=lambda s: s.get("order", 0),
    ):
        if section.get("kind") == "gaps":
            continue  # rendered last, unconditionally
        lines.append(f"## {section.get('title', 'Section')}")
        lines.append("")
        lines.append(str(section.get("body", "")))
        lines.append("")
        for claim in section.get("claims") or []:
            if not isinstance(claim, Mapping):
                continue
            citations = ", ".join(str(c) for c in claim.get("citations") or [])
            hedge = " *(hedged)*" if claim.get("hedged") else ""
            lines.append(f"- {claim.get('text')}{hedge}  \n  _Sources: {citations}_")
        if section.get("claims"):
            lines.append("")

    gaps = report.get("gaps") or []
    if gaps:
        lines.extend(["## What this investigation could not establish", ""])
        lines.extend(f"- {gap}" for gap in gaps)
        lines.append("")

    return "\n".join(lines)


def render_html(report: Mapping[str, Any]) -> str:
    """Wrap the Markdown in a minimal, self-contained HTML document.

    Escaped, and that is the whole reason this is not a one-line template. The
    report contains quoted passages from scraped web pages, and a passage
    containing `<script>` reaching an HTML file unescaped is stored XSS delivered
    by our own artefact -- served from our own origin to whoever opens it.
    """
    from html import escape

    body = escape(render_markdown(report))
    title = escape(str(report.get("title") or "Investigation report"))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:48rem;"
        "margin:2rem auto;padding:0 1rem;line-height:1.6}"
        "pre{white-space:pre-wrap}</style></head>"
        f"<body><pre>{body}</pre></body></html>"
    )


class ReportWorker(ConsumerWorker):
    """Renders and stores report artefacts for completed investigations."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        object_store: Any | None = None,
        formats: Sequence[RenderFormat] = (RenderFormat.MARKDOWN, RenderFormat.HTML),
        settings: Settings | None = None,
        name: str = WORKER_NAME,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, topics=[TopicRole.SIGNALS], settings=settings, **kwargs
        )
        self._session_factory = session_factory
        self._store = object_store
        self._formats = tuple(formats)
        self.rendered = 0
        self.render_failures = 0

    def readiness_probes(self) -> Mapping[str, DependencyProbe]:
        from backend.db.session import check_postgres

        return {"postgres": check_postgres}

    async def handle(self, message: ConsumedMessage) -> None:
        """Render one investigation's report into every configured format."""
        payload = message.envelope.payload
        investigation_id = _investigation_id_of(payload)
        if investigation_id is None:
            logger.debug("report.not_an_investigation_event")
            return

        report = await self._load_report(investigation_id)
        if report is None:
            # Completed with no report is legitimate: an investigation can be
            # cancelled or fail after the graph opens. Nothing to render, and
            # raising would send a perfectly ordinary event to the DLQ.
            logger.info("report.no_report_stored", investigation_id=investigation_id)
            return

        artifacts = await self._render_all(investigation_id, report)
        if artifacts:
            await self._record(investigation_id, artifacts)

    # ------------------------------------------------------------ internals --

    async def _render_all(
        self, investigation_id: str, report: Mapping[str, Any]
    ) -> list[RenderedArtifact]:
        """Render each format, keeping whatever succeeds.

        Each format is wrapped individually. A PDF engine missing from the image
        must cost the PDF and nothing else -- failing the whole handler would
        send the message to the DLQ and lose the Markdown that rendered
        perfectly.
        """
        version = str(report.get("version") or "v1")
        artifacts: list[RenderedArtifact] = []

        for fmt in self._formats:
            try:
                body, content_type = self._render_one(fmt, report)
            except Exception as error:  # noqa: BLE001 -- one format must not lose the rest
                self.render_failures += 1
                logger.warning(
                    "report.render_failed",
                    investigation_id=investigation_id,
                    format=fmt.value,
                    error=type(error).__name__,
                )
                continue

            # Derived from (investigation, version, format), so a re-render
            # overwrites in place. A timestamped key would accumulate a copy per
            # retry and force the API to guess which is current.
            key = f"reports/{investigation_id}/{version}.{fmt.value}"
            encoded = body.encode("utf-8")
            if self._store is not None:
                await self._store.put(key=key, body=encoded, content_type=content_type)
            artifacts.append(
                RenderedArtifact(
                    format=fmt,
                    key=key,
                    content_type=content_type,
                    size_bytes=len(encoded),
                )
            )
            self.rendered += 1

        return artifacts

    def _render_one(
        self, fmt: RenderFormat, report: Mapping[str, Any]
    ) -> tuple[str, str]:
        match fmt:
            case RenderFormat.MARKDOWN:
                return render_markdown(report), "text/markdown; charset=utf-8"
            case RenderFormat.HTML:
                return render_html(report), "text/html; charset=utf-8"
            case RenderFormat.PDF:
                raise NotImplementedError(
                    "PDF rendering needs an engine that is not in the base image. "
                    "Enable it deliberately rather than having it fail per report."
                )
        raise ValueError(f"unhandled format {fmt!r}")

    async def _load_report(self, investigation_id: str) -> Mapping[str, Any] | None:
        from sqlalchemy import select

        from models.orm.signal import InvestigationRow  # type: ignore[attr-defined]

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(InvestigationRow).where(InvestigationRow.id == investigation_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            report = getattr(row, "report", None)
            return report if isinstance(report, Mapping) else None

    async def _record(
        self, investigation_id: str, artifacts: Sequence[RenderedArtifact]
    ) -> None:
        logger.info(
            "report.rendered",
            investigation_id=investigation_id,
            formats=[artifact.format.value for artifact in artifacts],
            total_bytes=sum(artifact.size_bytes for artifact in artifacts),
            rendered_at=datetime.now(UTC).isoformat(),
        )


def _investigation_id_of(payload: Any) -> str | None:
    for attribute in ("investigation_id", "id"):
        value = getattr(payload, attribute, None)
        if value is None and isinstance(payload, Mapping):
            value = payload.get(attribute)
        if isinstance(value, str) and value:
            return value
    return None


def main() -> None:  # pragma: no cover -- process entry point
    from backend.db.session import get_sessionmaker

    run_worker(ReportWorker(session_factory=get_sessionmaker()))


if __name__ == "__main__":  # pragma: no cover
    main()
