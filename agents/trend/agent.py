"""The Trend agent: measures change over time, and refuses to assert it otherwise.

A trend is the most quotable output an investigation produces. "Complaints about
battery life rose 40% this quarter" goes straight into a slide, and once it has
left the system nothing distinguishes a measured 40% from an invented one. So
this node is built around one rule: **every number it reports came back from a
tool.**

The rule is enforced in three places, because one is not enough:

1. `agents/trend/schemas.py` shapes a trend so that an unsupported one is
   visibly unsupported -- no window, no observation count, no signal ids.
2. The schema validator refuses a `rising`/`falling` claim with fewer than three
   observations, because two points are a line, not a trend.
3. `_verify_against_series` here cross-checks the direction the model asserted
   against the series that was actually retrieved, and downgrades a claim the
   data does not support.

The third is the one that catches the interesting failure. A model handed a
series that wobbles will confidently describe it as rising, because "rising" is
the more useful-sounding answer and it has no incentive to say "this is noise".
Checking the assertion against the numbers costs nothing and is the difference
between a trend report and a horoscope.

**Degradation is total-loss-tolerant.** If `timeseries` is unavailable this node
returns no trends and says so. That is correct: a trend section absent from a
report is honest, and a trend section written from the model's prior beliefs
about the industry is not.

`docs/agent-system.md` §5.4.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Final

from agents.base import AgentContext, BaseAgent, StateDelta
from agents.errors import ToolExecutionError
from agents.state import InvestigationState
from agents.trend.schemas import (
    MAX_TRENDS,
    MIN_OBSERVATIONS_FOR_TREND,
    DetectedTrend,
    TrendDirection,
    TrendInput,
    TrendOutput,
)
from backend.core.logging import get_logger
from models.enums import AgentName
from services.llm.router import ModelTier

__all__ = ["STABLE_BAND_PCT", "TrendAgent"]

_log = get_logger(__name__)

STABLE_BAND_PCT: Final = 10.0
"""Below this absolute percentage change, a series is stable rather than moving.

Ten percent is a judgement, and it is written down here rather than left to the
model precisely because the model's judgement varies between calls. A fixed band
means "stable" means the same thing in every report this system produces, which
is what makes two reports comparable.
"""

MAX_SERIES: Final = 6
"""Time series fetched in one pass. Each is a database aggregation over a window."""


class TrendAgent(BaseAgent[TrendInput, TrendOutput]):
    """Fetches series, asks the model to interpret them, then checks the answer."""

    name: ClassVar[AgentName] = AgentName.TREND
    tier: ClassVar[ModelTier] = ModelTier.WORKER
    output_model: ClassVar[type[TrendOutput]] = TrendOutput
    tools: ClassVar[frozenset[str]] = frozenset(
        {"timeseries", "aggregate", "describe", "neighbours"}
    )

    def build_input(self, state: InvestigationState) -> TrendInput:
        graph_context = state.get("graph_context")
        entity_ids = list(getattr(graph_context, "expanded_entity_ids", ()) or ()) or list(
            getattr(graph_context, "seed_entity_ids", ()) or ()
        )
        return TrendInput(
            query=state["query"],
            objective=state.get("objective", ""),
            tenant_id=state["tenant_id"],
            entity_ids=entity_ids[:32],
            evidence_count=len(state.get("evidence") or []),
        )

    async def execute(self, request: TrendInput, ctx: AgentContext) -> TrendOutput:
        """Fetch first, interpret second, verify third.

        The ordering is the control. A model asked to identify trends *and then*
        given the data will anchor on what it said first; given the data first,
        it is describing rather than predicting.
        """
        series = await self._fetch_series(request, ctx)
        if not series:
            _log.info("trend.no_series_available", entities=len(request.entity_ids))
            return TrendOutput(
                series_retrieved=0,
                notes=(
                    "No time series could be retrieved, so no trend is asserted. "
                    "This is an absence of measurement, not a finding of stability."
                ),
            )

        rendered = self.render_prompt(
            ctx, query=request.query, objective=request.objective, window_days=request.window_days
        )
        interpreted = await self.call_model(
            ctx,
            prompt=self._describe_series(request, series),
            schema=TrendOutput,
            system=rendered.text,
        )

        verified = [
            self._verify_against_series(trend, series) for trend in interpreted.trends
        ]
        return TrendOutput(
            trends=[trend for trend in verified if trend is not None][:MAX_TRENDS],
            series_retrieved=len(series),
            notes=interpreted.notes,
        )

    def to_delta(self, output: TrendOutput, state: InvestigationState) -> StateDelta:
        """`trends` is `operator.add`-reduced: return only this pass's findings."""
        return {"trends": [trend.model_dump(mode="json") for trend in output.trends]}

    # ------------------------------------------------------------ internals --

    async def _fetch_series(
        self, request: TrendInput, ctx: AgentContext
    ) -> list[dict[str, Any]]:
        """Retrieve one series per topic or entity, concurrently and bounded."""
        subjects = (request.topics or request.entity_ids)[:MAX_SERIES]
        if not subjects:
            return []

        semaphore = asyncio.Semaphore(3)

        async def fetch_one(subject: str) -> dict[str, Any] | None:
            async with semaphore:
                try:
                    result = await self.use_tool(
                        ctx,
                        "timeseries",
                        {"subject": subject, "window_days": request.window_days},
                    )
                except ToolExecutionError as error:
                    _log.warning("trend.series_failed", subject=subject, error=str(error))
                    return None
                points = _points_from(result)
                if not points:
                    return None
                return {"subject": subject, "points": points}

        fetched = await asyncio.gather(*(fetch_one(subject) for subject in subjects))
        return [item for item in fetched if item is not None]

    def _describe_series(self, request: TrendInput, series: list[dict[str, Any]]) -> str:
        """Render the retrieved series as the model's input.

        The numbers are rendered as text rather than summarised, because a
        summary computed here and handed to the model is a second place where
        the arithmetic could be wrong -- and the model would then be describing
        my summary rather than the data.
        """
        lines = [
            f"Investigation: {request.query}",
            f"Objective: {request.objective}" if request.objective else "",
            "",
            f"Observed series over the last {request.window_days} days:",
        ]
        for entry in series:
            points = entry["points"]
            rendered = ", ".join(f"{value:g}" for _, value in points)
            lines.append(f"- {entry['subject']}: [{rendered}] ({len(points)} observations)")
        lines.extend(
            [
                "",
                "Describe what each series shows. Report only what these numbers "
                "support. Every trend must cite its observation count. If a series "
                "is noisy, say volatile rather than choosing a direction.",
            ]
        )
        return "\n".join(line for line in lines if line != "")

    def _verify_against_series(
        self, trend: DetectedTrend, series: list[dict[str, Any]]
    ) -> DetectedTrend | None:
        """Check the model's claim against the numbers, and correct it if needed.

        The check that matters. A model handed a wobbling series will describe it
        as rising, because "rising" sounds like a finding and it has no incentive
        to report noise. Three outcomes:

        * No matching series -- the trend is about something that was never
          measured. Dropped entirely; a fabricated subject is not recoverable by
          softening the direction.
        * Too few observations -- downgraded to `VOLATILE`, which is the honest
          description of a series too short to call.
        * Direction contradicts the endpoints -- corrected, and the confidence
          halved because the model's reading was demonstrably wrong once.
        """
        matched = next(
            (
                entry
                for entry in series
                if entry["subject"].casefold() in trend.topic.casefold()
                or trend.topic.casefold() in entry["subject"].casefold()
            ),
            None,
        )
        if matched is None:
            _log.warning("trend.unmatched_subject_dropped", topic=trend.topic)
            return None

        points = [value for _, value in matched["points"]]
        observed = len(points)

        if observed < MIN_OBSERVATIONS_FOR_TREND:
            return trend.model_copy(
                update={
                    "direction": TrendDirection.VOLATILE,
                    "observation_count": observed,
                    "change_pct": None,
                    "confidence": min(trend.confidence, 0.3),
                }
            )

        first, last = points[0], points[-1]
        # A zero baseline makes percentage change undefined. Reporting "infinite
        # growth" from one prior mention is the classic way this metric
        # embarrasses a report, so it is left null and the direction carries the
        # claim instead.
        change_pct = None if first == 0 else ((last - first) / abs(first)) * 100.0

        if change_pct is None:
            actual = TrendDirection.RISING if last > first else TrendDirection.STABLE
        elif abs(change_pct) < STABLE_BAND_PCT:
            actual = TrendDirection.STABLE
        elif change_pct > 0:
            actual = TrendDirection.RISING
        else:
            actual = TrendDirection.FALLING

        if actual is trend.direction:
            return trend.model_copy(
                update={"observation_count": observed, "change_pct": change_pct}
            )

        _log.info(
            "trend.direction_corrected",
            topic=trend.topic,
            claimed=trend.direction.value,
            actual=actual.value,
        )
        return trend.model_copy(
            update={
                "direction": actual,
                "observation_count": observed,
                "change_pct": change_pct,
                "confidence": trend.confidence / 2,
            }
        )


def _points_from(result: Any) -> list[tuple[Any, float]]:
    """Extract `(timestamp, value)` pairs from a `timeseries` tool result."""
    data = getattr(result, "data", result)
    raw = getattr(data, "points", None)
    if raw is None and isinstance(data, dict):
        raw = data.get("points") or data.get("series")
    if not isinstance(raw, (list, tuple)):
        return []

    points: list[tuple[Any, float]] = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            bucket, value = entry
        else:
            bucket = getattr(entry, "bucket", None)
            value = getattr(entry, "value", None)
            if isinstance(entry, dict):
                bucket = entry.get("bucket", entry.get("t"))
                value = entry.get("value", entry.get("count"))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            points.append((bucket, float(value)))
    return points
