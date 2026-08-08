"""The Forecast agent: projects series, and never invents the numbers.

`docs/agent-system.md` §5.6 is structural about this node: the model selects a
method and writes the caveats; `fit_forecast` produces the numbers. This module
is where that separation is made real, and it is made real by *construction*
rather than by instruction -- `execute` builds every `ForecastPoint` from the
tool result and the model is never given the opportunity to write one.

That matters more here than anywhere else in the system. A hallucinated trend is
a wrong statement about the past, which someone can check against the corpus. A
hallucinated forecast is a wrong statement about the future, which nobody can
check until it is too late to matter, and which reads as authoritative precisely
because it carries a confidence interval.

**Refusing is a first-class outcome.** Below `MIN_HISTORY_POINTS` observations
this node emits `INSUFFICIENT_DATA` with no points at all. The schema forbids
carrying numbers alongside that label, because a label next to a number is a
label that gets ignored. A report section reading "not enough history to
forecast" is a useful sentence; a projection from five noisy observations is
worse than silence.

`docs/agent-system.md` §5.6.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, ClassVar

from agents.base import AgentContext, BaseAgent, StateDelta
from agents.errors import ToolExecutionError
from agents.forecast.schemas import (
    MAX_FORECASTS,
    MIN_HISTORY_POINTS,
    ForecastInput,
    ForecastMethod,
    ForecastOutput,
    ForecastPoint,
    SeriesForecast,
)
from agents.state import InvestigationState
from backend.core.logging import get_logger
from models.base import StrictModel
from models.enums import AgentName
from services.llm.router import ModelTier

__all__ = ["ForecastAgent", "ForecastNarrative"]

_log = get_logger(__name__)


from pydantic import Field  # noqa: E402 -- kept beside the model it annotates


class ForecastNarrative(StrictModel):
    """What the model is allowed to contribute: judgement, never numbers.

    Deliberately shaped so it *cannot* carry a projection. There is no points
    field, no value field, nothing numeric beyond a confidence. A model that
    wanted to assert a number has nowhere to put it, which is a stronger control
    than a prompt asking it not to.
    """

    subject: str = Field(min_length=1, max_length=200)
    caveats: list[str] = Field(min_length=1, max_length=6)
    interpretation: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ForecastAgent(BaseAgent[ForecastInput, ForecastOutput]):
    """Fits forecasts with a tool, then asks the model only to interpret them."""

    name: ClassVar[AgentName] = AgentName.FORECAST
    tier: ClassVar[ModelTier] = ModelTier.WORKER
    output_model: ClassVar[type[ForecastOutput]] = ForecastOutput
    tools: ClassVar[frozenset[str]] = frozenset({"timeseries", "fit_forecast", "hybrid_search"})

    def build_input(self, state: InvestigationState) -> ForecastInput:
        """Forecast the subjects the Trend agent actually measured.

        Reading subjects off `trends` rather than off the plan is what keeps
        this node honest: a subject the Trend agent could not measure has no
        series, so forecasting it would mean fitting to nothing.
        """
        trends = state.get("trends") or []
        subjects: list[str] = []
        for trend in trends:
            topic = trend.get("topic") if isinstance(trend, dict) else None
            if isinstance(topic, str) and topic and topic not in subjects:
                subjects.append(topic)
        return ForecastInput(
            query=state["query"],
            objective=state.get("objective", ""),
            tenant_id=state["tenant_id"],
            subjects=subjects[:MAX_FORECASTS],
            trend_count=len(trends),
        )

    async def execute(self, request: ForecastInput, ctx: AgentContext) -> ForecastOutput:
        if not request.subjects:
            return ForecastOutput(
                horizon_days=request.horizon_days,
                notes=(
                    "No measured series were available to forecast. Nothing is "
                    "projected; this is an absence of data, not a prediction of "
                    "stability."
                ),
            )

        fitted = await self._fit_all(request, ctx)
        forecasts: list[SeriesForecast] = []

        for subject, result in fitted:
            if result is None:
                forecasts.append(
                    SeriesForecast(
                        subject=subject,
                        method=ForecastMethod.INSUFFICIENT_DATA,
                        interpretation="The forecasting tool could not fit this series.",
                    )
                )
                continue

            points, history, method = _forecast_from(result)
            if history < MIN_HISTORY_POINTS or not points:
                forecasts.append(
                    SeriesForecast(
                        subject=subject,
                        method=ForecastMethod.INSUFFICIENT_DATA,
                        history_points=history,
                        interpretation=(
                            f"{history} observations is below the {MIN_HISTORY_POINTS} "
                            "needed; any fit here would be to noise, and its interval "
                            "would understate the uncertainty."
                        ),
                    )
                )
                continue

            narrative = await self._interpret(subject, points, history, method, request, ctx)
            forecasts.append(
                SeriesForecast(
                    subject=subject,
                    method=method,
                    # Points come from the tool. The model never sees a slot to
                    # write one into -- see `ForecastNarrative`.
                    points=points,
                    history_points=history,
                    confidence=narrative.confidence if narrative else 0.3,
                    caveats=(
                        narrative.caveats
                        if narrative
                        else ["Interpretation unavailable; the projection is unreviewed."]
                    ),
                    interpretation=narrative.interpretation if narrative else None,
                )
            )

        return ForecastOutput(
            forecasts=forecasts[:MAX_FORECASTS], horizon_days=request.horizon_days
        )

    def to_delta(self, output: ForecastOutput, state: InvestigationState) -> StateDelta:
        """`forecasts` is `operator.add`-reduced: return only this pass's output."""
        return {"forecasts": [item.model_dump(mode="json") for item in output.forecasts]}

    # ------------------------------------------------------------ internals --

    async def _fit_all(
        self, request: ForecastInput, ctx: AgentContext
    ) -> list[tuple[str, Any | None]]:
        semaphore = asyncio.Semaphore(3)

        async def fit_one(subject: str) -> tuple[str, Any | None]:
            async with semaphore:
                try:
                    return subject, await self.use_tool(
                        ctx,
                        "fit_forecast",
                        {"subject": subject, "horizon_days": request.horizon_days},
                    )
                except ToolExecutionError as error:
                    _log.warning("forecast.fit_failed", subject=subject, error=str(error))
                    return subject, None

        return list(await asyncio.gather(*(fit_one(s) for s in request.subjects)))

    async def _interpret(
        self,
        subject: str,
        points: list[ForecastPoint],
        history: int,
        method: ForecastMethod,
        request: ForecastInput,
        ctx: AgentContext,
    ) -> ForecastNarrative | None:
        """Ask the model what the projection means and how it could be wrong.

        Returns `None` on failure rather than raising. A projection whose
        interpretation call failed is still a valid projection -- the numbers came
        from the tool -- so it is emitted with a caveat saying it is unreviewed,
        which is more useful than discarding a successful fit.
        """
        rendered = self.render_prompt(
            ctx, query=request.query, objective=request.objective, method=method.value
        )
        first, last = points[0], points[-1]
        prompt = (
            f"Investigation: {request.query}\n\n"
            f"Subject: {subject}\n"
            f"Method: {method.value}\n"
            f"History: {history} observations\n"
            f"Horizon: {request.horizon_days} days\n"
            f"Projection start: {first.value:g} (band {first.lower:g}-{first.upper:g})\n"
            f"Projection end:   {last.value:g} (band {last.lower:g}-{last.upper:g})\n\n"
            "State what this projection means and list what would make it wrong. "
            "Do not restate or adjust the numbers -- they are fixed. If the band "
            "is wide relative to the movement, say the projection is not "
            "actionable."
        )
        try:
            return await self.call_model(
                ctx, prompt=prompt, schema=ForecastNarrative, system=rendered.text
            )
        except Exception as error:  # noqa: BLE001 -- a valid fit survives a failed narration
            _log.warning("forecast.interpretation_failed", subject=subject, error=str(error))
            return None


def _forecast_from(result: Any) -> tuple[list[ForecastPoint], int, ForecastMethod]:
    """Build points from the tool result, discarding anything malformed.

    A point whose interval does not contain its estimate is dropped rather than
    repaired. Repairing it -- widening the band to fit -- would manufacture an
    uncertainty statement the fit never made, which is precisely the fabrication
    this node exists to prevent.
    """
    data = getattr(result, "data", result)
    raw = _attr(data, "points") or _attr(data, "forecast") or []
    history = _attr(data, "history_points")
    if not isinstance(history, int) or isinstance(history, bool):
        history = 0

    raw_method = _attr(data, "method")
    try:
        method = ForecastMethod(raw_method) if isinstance(raw_method, str) else ForecastMethod.NAIVE
    except ValueError:
        method = ForecastMethod.NAIVE

    points: list[ForecastPoint] = []
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            at = _attr(entry, "at") or _attr(entry, "t")
            value = _attr(entry, "value")
            lower = _attr(entry, "lower")
            upper = _attr(entry, "upper")
            if not isinstance(at, datetime):
                continue
            if not all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in (value, lower, upper)
            ):
                continue
            try:
                points.append(
                    ForecastPoint(
                        at=at, lower=float(lower), value=float(value), upper=float(upper)
                    )
                )
            except ValueError:
                _log.warning("forecast.malformed_interval_dropped", at=str(at))
                continue
    return points, history, method


def _attr(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        return obj.get(name)
    return value
