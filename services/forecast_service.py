"""Forecasting: time-series projection with intervals and assumptions.

`docs/glossary.md` draws the line this module lives on: a **Trend** is a
measurement of the past and can only be wrong if the data is wrong; a
**Forecast** projects forward and can be wrong with perfect data. Everything
here exists to make that second kind of wrongness visible rather than hidden.

Three rules, and each of them is a refusal to do the easy thing
---------------------------------------------------------------
**1. A point estimate is never returned on its own.** `Forecast.points` carries
`lower` and `upper` on every point and there is no accessor that yields the mean
alone. A bare number reads as a fact; "42, somewhere between 11 and 96" reads as
what it is. `docs/roadmap.md` §Phase 4 makes this a shipping criterion -- "every
forecast surfaced in the UI shows its interval and its horizon; a point estimate
alone never renders" -- and the only way an interface can guarantee that is to
never produce the naked number in the first place.

The interval is a **prediction** interval, not a confidence interval on the
fitted mean. That distinction is the whole point: the confidence interval on an
OLS mean at 30 observations is roughly a third the width of the prediction
interval, and reporting it would understate the uncertainty by exactly the factor
that makes a forecast look trustworthy when it is not. For the linear model that
means `obs_ci_*` rather than `mean_ci_*`; for the state-space model the forecast
variance already includes the innovation term.

**2. Assumptions ship with the numbers.** `Forecast.assumptions` is never empty.
A projection is a conditional statement, and a reader who cannot see the
conditions cannot tell when it has stopped applying. The assumptions here are the
ones that actually break in this system -- a connector enabled mid-window is a
level shift the model will happily project forward as organic growth, and a
change to dedup behaviour rewrites the meaning of every historical count
(`docs/signal-model.md` §4.3).

**3. Too little history is a refusal, not a wider interval.** Below
`ForecastConfig.min_observations` this module raises `InsufficientHistoryError`.
Four points fit a straight line perfectly and produce an interval whose width is
governed by two residual degrees of freedom -- confident nonsense with a
statistical veneer, which is worse than no answer because it is quotable.
`docs/roadmap.md` names "overfitting to a short history" as a Phase 4 risk and
"minimum-history gate before a forecast is emitted at all" as the mitigation.
That gate is `_require_forecastable`.

Why two methods rather than one
-------------------------------
A short series cannot support a fitted ARIMA -- there is nothing to estimate the
autocorrelation from, and the optimizer will happily return a fit that is noise.
So the selection rule is by history length, stated as a constant rather than
chosen adaptively: below `arima_min_observations` the method is a linear trend
with a prediction interval, above it a small ARIMA, and an ARIMA that fails to
fit falls back to the linear trend with the fallback recorded as a caveat rather
than swallowed. A caller that names a method explicitly gets it or gets a
refusal; it never gets a different method under the same name.

Layer note: `services/` (L2). Pure computation over a series it is handed -- no
database, no session, no network. That is deliberate: the Layer 5 Forecast agent
must be able to re-fit a series it already holds, and `docs/agent-system.md` §5.6
makes the agent's numbers traceable to a recorded tool call, which is only
possible if the fit is a function of its arguments.
"""

from __future__ import annotations

import enum
import itertools
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from backend.core.exceptions import ValidationError
from backend.core.logging import get_logger

__all__ = [
    "Forecast",
    "ForecastConfig",
    "ForecastMethod",
    "ForecastPoint",
    "ForecastService",
    "InsufficientHistoryError",
    "Observation",
    "observations_from_buckets",
]

logger = get_logger(__name__)


DEFAULT_INTERVAL_LEVEL: Final[float] = 0.95
"""Nominal coverage of the reported interval.

95% rather than 80%: this number is read by people deciding whether to fund
something, and an 80% interval is wrong one time in five, which is far more often
than the word "interval" suggests to a non-statistician.
"""

_SCALE_EPSILON: Final[float] = 1e-9


class InsufficientHistoryError(ValidationError):
    """The series is too short, too sparse or too irregular to forecast from.

    A distinct class rather than a bare `ValidationError` because the caller's
    correct response is specific and automatable: widen the window, wait for more
    data, or present the trend without a projection. `details` carries the counts
    so a handler can say which of the three applies without parsing a message
    (`docs/coding-standards.md` §2.7).
    """

    status_code = 422
    code = "insufficient_history"
    default_message = "Not enough history to produce a forecast worth reporting."


class ForecastMethod(enum.StrEnum):
    """The fitted model. Reported on every `Forecast` because it changes the meaning.

    Not in `models/enums.py`: this is a private vocabulary of one module and does
    not cross a process boundary today. It moves there the day `models/forecast.py`
    stops being a stub.
    """

    LINEAR_TREND = "linear_trend"
    """OLS on the bucket index with a prediction interval. Assumes the trend is
    locally linear and the residuals are roughly homoscedastic."""

    ARIMA = "arima"
    """A small ARIMA fitted by state space. Assumes the differenced series is
    stationary; captures autocorrelation the linear fit treats as noise."""


@dataclass(frozen=True, slots=True)
class Observation:
    """One historical point: a bucket start and the value observed in it.

    Deliberately not coupled to `services/trend_service.VolumeSeries`. The Layer 5
    Forecast agent forecasts volume today and will forecast sentiment share and
    engagement tomorrow, and a forecaster that only accepts one producer's type
    would have to be rewritten for the second.
    """

    at: datetime
    value: float


@dataclass(frozen=True, slots=True)
class ForecastPoint:
    """One projected bucket. There is no constructor path that omits the interval."""

    at: datetime
    mean: float
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    """Gates and tuning. Every value here is a documented refusal boundary."""

    min_observations: int = 12
    """The minimum-history gate.

    Twelve is the smallest number that leaves a usable residual degrees-of-freedom
    count after fitting an intercept and a slope, and that spans a weekly cycle
    plus change at daily buckets. It is a judgement, not a theorem -- what is not
    a judgement is that four points must be refused, because four points fit a
    line with two residual degrees of freedom and an interval narrow enough to
    quote."""

    min_nonzero_observations: int = 3
    """A history of one spike and eleven zeros is not a series.

    The linear fit through it is dominated entirely by the single non-zero point,
    and the projection is that point's position on a line nothing else
    constrains. Refused rather than caveated: there is no interval width that
    makes it informative."""

    arima_min_observations: int = 24
    """Below this, ARIMA is not offered. There is not enough data to estimate an
    autocorrelation structure from, and the optimizer will fit one anyway."""

    arima_order: tuple[int, int, int] = (1, 1, 1)
    """A deliberately small order. Anything richer overfits at the sample sizes
    this system has, and `docs/roadmap.md` names overfitting as the Phase 4 risk."""

    max_horizon_ratio: float = 1.0
    """Refuse a horizon longer than `ratio * len(history)`.

    Forecasting 90 days from 30 days of data is not a forecast with a wide
    interval; it is an extrapolation whose interval is itself extrapolated."""

    caveat_horizon_ratio: float = 0.34
    """Above this the horizon is flagged in `caveats` while still being served.
    Roughly the classical "do not project beyond a third of your history"."""

    zero_share_caveat: float = 0.5
    """A series that is mostly zeros gets a caveat: a Gaussian interval around a
    near-zero mean is symmetric where the data cannot be."""

    min_resolvable_width: float = 1.0
    """Below this full band width the model's interval is treated as degenerate.

    Not an epsilon guarding against float noise -- a statement about what the
    forecast quantity can mean. Volume series count distinct
    `(dedup_cluster_id, platform)` pairs (`services/trend_service.py`), so they
    move in whole units; a band spanning less than one of them claims to know
    next bucket's count *exactly*, and the fact that the claim arrived with two
    decimal places rather than zero width does not make it weaker.

    An absolute threshold rather than a fraction of the level, because the
    failure being caught is a fit with no residual variation to speak of, and a
    genuinely well-fitted noisy series clears one whole unit easily at any level.
    A caller forecasting a *continuous* quantity -- a sentiment delta on
    [-1, 1], where a band of 0.4 is both legitimate and narrower than a unit --
    must lower this, which is why it is a knob and not a literal."""

    max_spacing_jitter: timedelta = timedelta(seconds=1)
    """Tolerance on the evenness of the history's spacing.

    Not zero, because a caller may build buckets from wall-clock arithmetic, and
    not generous, because an irregular series silently reinterprets the horizon:
    "12 buckets ahead" means nothing if the buckets are not the same length."""

    def __post_init__(self) -> None:
        if self.min_observations < 3:
            raise ValidationError("min_observations must be at least 3")
        if self.min_nonzero_observations < 1:
            raise ValidationError("min_nonzero_observations must be at least 1")
        if self.arima_min_observations < self.min_observations:
            raise ValidationError("arima_min_observations must be >= min_observations")
        if self.max_horizon_ratio <= 0:
            raise ValidationError("max_horizon_ratio must be positive")
        if self.min_resolvable_width < 0:
            # Zero is permitted -- it disables the floor for a caller who has
            # decided their quantity really is resolvable without limit. Negative
            # is incoherent: no band width could ever fall below it.
            raise ValidationError("min_resolvable_width must not be negative")


@dataclass(frozen=True, slots=True)
class Forecast:
    """A projected trajectory, its interval, and the conditions it depends on.

    Every field below is part of the contract with the Layer 5 Forecast agent,
    whose output schema takes trajectory values only from a recorded tool result
    (`docs/agent-system.md` §5.6). An agent may narrate these numbers; it may
    never invent them, and the Critic rejects any forecast whose numbers do not
    match a recorded call.
    """

    method: ForecastMethod
    points: tuple[ForecastPoint, ...]
    interval_level: float
    bucket: timedelta
    horizon: int
    history_start: datetime
    history_end: datetime
    observations_used: int

    assumptions: tuple[str, ...]
    """Conditions under which the projection holds. Never empty -- a projection
    with no stated conditions reads as a fact."""

    caveats: tuple[str, ...]
    """Things that were true of *this* fit and weaken it. Empty is meaningful."""

    confidence: float
    """A heuristic belief in `[0.05, 0.95]`, never 0 and never 1.

    Explicitly **not calibrated** -- no backtest has scored it yet
    (`docs/roadmap.md` Phase 4 exit criteria list the harness as outstanding), so
    the only property it promises is ordering: more history, a shorter horizon
    and a tighter interval each raise it. `docs/testing-strategy.md` asks for
    exactly that assertion rather than an equality against a literal."""

    diagnostics: dict[str, float] = field(default_factory=dict)
    """Fit statistics -- residual scale, AIC where the method has one. For an
    operator comparing two runs, not for a report."""

    @property
    def horizon_end(self) -> datetime:
        return self.points[-1].at if self.points else self.history_end

    @property
    def mean_interval_width(self) -> float:
        if not self.points:
            return 0.0
        return sum(point.width for point in self.points) / len(self.points)


def observations_from_buckets(
    start: datetime,
    bucket: timedelta,
    values: Sequence[float | int],
) -> list[Observation]:
    """Build a regular history from a bucket start and a contiguous value list.

    The adapter for `services/trend_service.VolumeSeries`, kept as a free function
    so the two modules stay independent: trend detection must not acquire a
    dependency on statsmodels, and a caller with volumes from anywhere else gets
    the same path.
    """
    return [
        Observation(at=start + index * bucket, value=float(value))
        for index, value in enumerate(values)
    ]


class ForecastService:
    """Fits a projection to a history and reports what it assumed to do it.

    Stateless and cheap to construct. Holds configuration only, so one instance
    is shared freely across concurrent requests.
    """

    def __init__(self, *, config: ForecastConfig | None = None) -> None:
        self._config = config or ForecastConfig()

    @property
    def config(self) -> ForecastConfig:
        return self._config

    def forecast(
        self,
        observations: Sequence[Observation],
        *,
        horizon: int,
        interval_level: float = DEFAULT_INTERVAL_LEVEL,
        method: ForecastMethod | None = None,
        non_negative: bool = True,
    ) -> Forecast:
        """Project `horizon` buckets forward, or refuse.

        `non_negative` clips the lower bound at zero and defaults to `True`
        because every quantity this system forecasts today is a count. Clipping
        is recorded as an assumption rather than applied silently: it makes the
        interval asymmetric, and an asymmetric interval that nobody explained
        looks like a bug in the model.

        Raises `InsufficientHistoryError` when the history cannot support a
        forecast, and `ValidationError` when the *request* is malformed (a
        non-positive horizon, an interval level outside `(0, 1)`, a method that
        this history cannot support). The two are separated because only the
        second is the caller's mistake.
        """
        if horizon < 1:
            raise ValidationError("horizon must be at least 1 bucket")
        if not 0.0 < interval_level < 1.0:
            raise ValidationError(
                f"interval_level must be strictly between 0 and 1; got {interval_level}"
            )

        history = self._require_forecastable(observations, horizon=horizon)
        bucket = _infer_bucket(history, jitter=self._config.max_spacing_jitter)
        values = [point.value for point in history]

        chosen = self._choose_method(method, observations=len(values))
        caveats: list[str] = []

        # A fitted model is allowed to complain -- non-convergence and a
        # non-invertible starting polynomial are *results*, not exceptions, and
        # the honest place for them is the caveat list a reader sees. Recording
        # rather than raising also keeps the unit suite's `filterwarnings=error`
        # from turning a legitimately noisy fit into a test failure that says
        # nothing about correctness.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                mean, lower, upper, diagnostics = _fit(
                    chosen, values, horizon=horizon, level=interval_level, config=self._config
                )
            except Exception as exc:  # noqa: BLE001 - the fallback is the point
                if chosen is not ForecastMethod.ARIMA:
                    raise
                logger.warning(
                    "forecast_service.arima_failed",
                    error=type(exc).__name__,
                    observations=len(values),
                )
                caveats.append(
                    f"The {ForecastMethod.ARIMA.value} fit failed "
                    f"({type(exc).__name__}) and the projection fell back to "
                    f"{ForecastMethod.LINEAR_TREND.value}, which cannot represent "
                    "autocorrelation; treat any cyclical structure as unmodelled."
                )
                chosen = ForecastMethod.LINEAR_TREND
                mean, lower, upper, diagnostics = _fit(
                    chosen, values, horizon=horizon, level=interval_level, config=self._config
                )
            caveats.extend(_warning_caveats(caught))

        lower, upper, degenerate = _widen_degenerate_interval(
            mean,
            lower,
            upper,
            values=values,
            level=interval_level,
            min_width=self._config.min_resolvable_width,
        )
        if degenerate:
            caveats.append(
                "The fit left effectively no residual variation, so the model's own "
                "interval was narrower than a single countable unit. It was widened "
                "to the dispersion a Poisson process at this level would show; the "
                "true uncertainty is at least that and is not measurable from this "
                "history."
            )

        if non_negative:
            lower = [max(0.0, value) for value in lower]
            mean = [max(0.0, value) for value in mean]
            upper = [max(0.0, value) for value in upper]

        points = tuple(
            ForecastPoint(
                at=history[-1].at + (step + 1) * bucket,
                mean=mean[step],
                lower=lower[step],
                upper=upper[step],
            )
            for step in range(horizon)
        )

        caveats.extend(self._data_caveats(values, horizon=horizon))
        confidence = _confidence(
            observations=len(values),
            horizon=horizon,
            points=points,
            config=self._config,
        )

        return Forecast(
            method=chosen,
            points=points,
            interval_level=interval_level,
            bucket=bucket,
            horizon=horizon,
            history_start=history[0].at,
            history_end=history[-1].at,
            observations_used=len(values),
            assumptions=self._assumptions(
                chosen, bucket=bucket, level=interval_level, non_negative=non_negative
            ),
            caveats=tuple(caveats),
            confidence=confidence,
            diagnostics=diagnostics,
        )

    # -- gates ------------------------------------------------------------- #

    def _require_forecastable(
        self, observations: Sequence[Observation], *, horizon: int
    ) -> list[Observation]:
        """The minimum-history gate. Refuses rather than widening the interval.

        Sorting is done here rather than demanded of the caller because an
        out-of-order history is a silent catastrophe otherwise: the fit still
        succeeds, the slope is meaningless, and nothing in the output says so.
        Duplicated timestamps are a refusal, because collapsing or averaging them
        would be this module inventing data.
        """
        config = self._config
        if len(observations) < config.min_observations:
            raise InsufficientHistoryError(
                f"{len(observations)} observations is below the minimum of "
                f"{config.min_observations}. A shorter history fits a line "
                "perfectly and produces an interval narrow enough to quote, which "
                "is worse than no forecast. Widen the window or use a coarser "
                "bucket.",
                details={
                    "observations": len(observations),
                    "minimum": config.min_observations,
                    "reason": "too_short",
                },
            )

        history = sorted(observations, key=lambda point: _as_utc(point.at))
        history = [Observation(at=_as_utc(point.at), value=float(point.value)) for point in history]

        stamps = [point.at for point in history]
        if len(set(stamps)) != len(stamps):
            raise ValidationError(
                "the history contains duplicate timestamps; two observations of "
                "the same bucket cannot be reconciled here without inventing a "
                "rule the caller did not state (sum? mean? last?)."
            )

        nonzero = sum(1 for point in history if abs(point.value) > _SCALE_EPSILON)
        if nonzero < config.min_nonzero_observations:
            raise InsufficientHistoryError(
                f"only {nonzero} of {len(history)} observations are non-zero, below "
                f"the minimum of {config.min_nonzero_observations}. A fit through "
                "one or two isolated points is that point's position on a line "
                "nothing else constrains.",
                details={
                    "observations": len(history),
                    "nonzero": nonzero,
                    "minimum_nonzero": config.min_nonzero_observations,
                    "reason": "too_sparse",
                },
            )

        limit = int(len(history) * config.max_horizon_ratio)
        if horizon > limit:
            raise InsufficientHistoryError(
                f"a horizon of {horizon} buckets from {len(history)} observations "
                f"exceeds the limit of {limit}. Beyond it the interval is itself "
                "an extrapolation, so it stops being a statement about "
                "uncertainty.",
                details={
                    "observations": len(history),
                    "horizon": horizon,
                    "max_horizon": limit,
                    "reason": "horizon_too_long",
                },
            )
        return history

    def _choose_method(
        self, requested: ForecastMethod | None, *, observations: int
    ) -> ForecastMethod:
        """Pick a model by history length, or honour an explicit choice or refuse it.

        An explicitly requested method that the history cannot support is a
        refusal rather than a silent downgrade: the caller recorded which model it
        asked for, and returning a different one under that name is how a
        reproducibility claim quietly stops being true.
        """
        if requested is ForecastMethod.ARIMA and observations < self._config.arima_min_observations:
            raise InsufficientHistoryError(
                f"{ForecastMethod.ARIMA.value} needs at least "
                f"{self._config.arima_min_observations} observations to estimate an "
                f"autocorrelation structure; {observations} were given.",
                details={
                    "observations": observations,
                    "minimum": self._config.arima_min_observations,
                    "reason": "method_unsupported",
                },
            )
        if requested is not None:
            return requested
        if observations >= self._config.arima_min_observations:
            return ForecastMethod.ARIMA
        return ForecastMethod.LINEAR_TREND

    # -- narration --------------------------------------------------------- #

    def _assumptions(
        self,
        method: ForecastMethod,
        *,
        bucket: timedelta,
        level: float,
        non_negative: bool,
    ) -> tuple[str, ...]:
        """The conditions the projection is contingent on. Never empty.

        These are the ones that actually break in *this* system, not a textbook
        list. The connector one in particular: enabling a source mid-window is a
        level shift, and every model here will read it as growth and project it
        forward.
        """
        stated = [
            f"The history is evenly spaced at {bucket}, and an empty bucket means "
            "genuine silence rather than missing data. A gap that was filled with "
            "zeros because a connector was down is read as a real decline.",
            "Collection coverage is unchanged across the history and the horizon: "
            "the same connectors enabled, the same scope, the same rate limits. "
            "Enabling a source mid-window is a level shift this model will project "
            "forward as organic growth.",
            "The de-duplication rule behind the counts is unchanged "
            "(docs/signal-model.md §4.3). Changing what counts as one mention "
            "rewrites the meaning of every historical value, so a forecast fitted "
            "before the change does not describe the series after it.",
            "No structural break -- launch, outage, acquisition, policy change -- "
            "occurs within the horizon. The model has no way to anticipate one and "
            "will not widen its interval for it.",
            f"The reported band is a {level:.0%} *prediction* interval under the "
            "fitted model. It covers sampling and innovation variance only; it "
            "does not cover the model being the wrong model, and a wrong model is "
            "usually wrong outside its own interval.",
        ]
        if method is ForecastMethod.LINEAR_TREND:
            stated.append(
                "The trend is locally linear over the horizon and the residuals "
                "are roughly constant in spread. Any cycle in the history is "
                "being treated as noise."
            )
        else:
            stated.append(
                "The once-differenced series is stationary, so the level may "
                "wander but its rate of change does not systematically."
            )
        if non_negative:
            stated.append(
                "The quantity cannot be negative, so the lower bound is clipped at "
                "zero. Near zero the interval is therefore asymmetric by "
                "construction, which is deliberate: a negative lower bound is not "
                "a wider interval, it is a meaningless one."
            )
        return tuple(stated)

    def _data_caveats(self, values: Sequence[float], *, horizon: int) -> list[str]:
        """Weaknesses of this particular fit, read off the data."""
        config = self._config
        found: list[str] = []

        if len(values) < config.min_observations * 2:
            found.append(
                f"The history is {len(values)} buckets, close to the minimum of "
                f"{config.min_observations}. The interval is honest about sampling "
                "error but there is not enough data to detect a wrong model shape."
            )
        if horizon > len(values) * config.caveat_horizon_ratio:
            found.append(
                f"The horizon ({horizon} buckets) exceeds "
                f"{config.caveat_horizon_ratio:.0%} of the history "
                f"({len(values)} buckets). Accuracy is reported per horizon for a "
                "reason: a model that is good at 7 buckets can be useless at 90."
            )
        zeros = sum(1 for value in values if abs(value) <= _SCALE_EPSILON)
        if zeros > len(values) * config.zero_share_caveat:
            found.append(
                f"{zeros} of {len(values)} buckets are empty. A Gaussian interval "
                "around a near-zero level is symmetric where the data cannot be, "
                "so the upper half of the band is the informative one."
            )
        return found


# --------------------------------------------------------------------------- #
# Fitting -- statsmodels lives behind these two functions and nowhere else
# --------------------------------------------------------------------------- #


def _fit(
    method: ForecastMethod,
    values: Sequence[float],
    *,
    horizon: int,
    level: float,
    config: ForecastConfig,
) -> tuple[list[float], list[float], list[float], dict[str, float]]:
    """Dispatch to a fitter. Returns `(mean, lower, upper, diagnostics)`."""
    if method is ForecastMethod.LINEAR_TREND:
        return _fit_linear(values, horizon=horizon, level=level)
    return _fit_arima(values, horizon=horizon, level=level, order=config.arima_order)


def _fit_linear(
    values: Sequence[float], *, horizon: int, level: float
) -> tuple[list[float], list[float], list[float], dict[str, float]]:
    """OLS on the bucket index, reported with an *observation* interval.

    `summary_frame` offers two bands and picking the wrong one is the classic way
    to publish a misleading forecast. `mean_ci_*` is the interval on the fitted
    regression line -- where the average sits. `obs_ci_*` is the interval a *new
    observation* is expected to fall in, which is the question a forecast asks,
    and it is materially wider because it adds the residual variance to the
    parameter uncertainty. This function returns `obs_ci_*`.
    """
    numpy, statsmodels_api, _ = _statsmodels()

    endog = numpy.asarray(values, dtype=float)
    index = numpy.arange(len(endog), dtype=float)
    design = statsmodels_api.add_constant(index, has_constant="add")

    fitted = statsmodels_api.OLS(endog, design).fit()
    future_index = numpy.arange(len(endog), len(endog) + horizon, dtype=float)
    future = statsmodels_api.add_constant(future_index, has_constant="add")

    frame = fitted.get_prediction(future).summary_frame(alpha=1.0 - level)
    diagnostics = {
        "residual_scale": float(numpy.sqrt(fitted.mse_resid))
        if numpy.isfinite(fitted.mse_resid)
        else 0.0,
        "r_squared": float(fitted.rsquared) if numpy.isfinite(fitted.rsquared) else 0.0,
        "slope_per_bucket": float(fitted.params[1]),
    }
    return (
        [float(value) for value in frame["mean"]],
        [float(value) for value in frame["obs_ci_lower"]],
        [float(value) for value in frame["obs_ci_upper"]],
        diagnostics,
    )


def _fit_arima(
    values: Sequence[float], *, horizon: int, level: float, order: tuple[int, int, int]
) -> tuple[list[float], list[float], list[float], dict[str, float]]:
    """A small ARIMA by state space.

    `get_forecast(...).summary_frame()` bands are already prediction intervals
    here -- the state-space forecast variance includes the innovation term -- so
    unlike the OLS path there is no second, narrower band to pick by mistake.
    """
    numpy, _, arima = _statsmodels()

    endog = numpy.asarray(values, dtype=float)
    fitted = arima(endog, order=order).fit()
    frame = fitted.get_forecast(steps=horizon).summary_frame(alpha=1.0 - level)

    diagnostics = {
        "aic": float(fitted.aic) if numpy.isfinite(fitted.aic) else 0.0,
        "residual_scale": float(numpy.std(fitted.resid, ddof=1)) if len(endog) > 1 else 0.0,
    }
    return (
        [float(value) for value in frame["mean"]],
        [float(value) for value in frame["mean_ci_lower"]],
        [float(value) for value in frame["mean_ci_upper"]],
        diagnostics,
    )


def _statsmodels() -> tuple[Any, Any, Any]:
    """Import numpy and statsmodels lazily.

    statsmodels drags scipy and pandas in behind it, which is roughly a second of
    import time and a large resident set. Every process that imports `services/`
    would pay it -- including the API, which forecasts on a small fraction of
    requests, and the enrichment worker, which never does. The import is cached by
    `sys.modules` after the first call, so the cost is paid once and only by a
    process that actually fits something.
    """
    import numpy
    import statsmodels.api as statsmodels_api
    from statsmodels.tsa.arima.model import ARIMA

    return numpy, statsmodels_api, ARIMA


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _infer_bucket(history: Sequence[Observation], *, jitter: timedelta) -> timedelta:
    """Derive the bucket width and prove the history is evenly spaced.

    An irregular history silently reinterprets the horizon: "12 buckets ahead"
    is not a duration if the buckets are not the same length, and every method
    here indexes by position rather than by time. Caught at the input boundary,
    where the caller still knows what it passed.
    """
    bucket = history[1].at - history[0].at
    if bucket <= timedelta(0):
        raise ValidationError("history timestamps must strictly increase")

    for previous, current in itertools.pairwise(history):
        delta = current.at - previous.at
        if abs(delta - bucket) > jitter:
            raise ValidationError(
                "the history is not evenly spaced: expected a bucket of "
                f"{bucket} but found {delta} at {current.at.isoformat()}. Fill the "
                "gaps explicitly -- a forecaster cannot tell a missing bucket from "
                "an empty one, and guessing is how a collection outage becomes a "
                "predicted decline.",
                details={"expected_bucket_seconds": bucket.total_seconds()},
            )
    return bucket


def _widen_degenerate_interval(
    mean: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    values: Sequence[float],
    level: float,
    min_width: float,
) -> tuple[list[float], list[float], bool]:
    """Replace an unresolvably narrow band with a Poisson floor.

    A perfectly linear (or perfectly constant) history leaves no residual
    variance to speak of, and the model then reports an interval narrow enough to
    be a claim that the future is known. That is arithmetically defensible and
    editorially indefensible. The floor is the dispersion a Poisson process at the
    projected level would show, which is the least uncertainty a count series can
    honestly have.

    **The trigger is `min_width`, not an exact zero, and the difference is the
    whole point.** Only OLS on an exactly-collinear design produces a literal
    zero; the state-space ARIMA path does not. Fitted against the same perfectly
    linear history it returns a band of width ~0.009 around a level of 70 -- the
    optimizer driving the innovation variance toward zero and stopping at machine
    noise instead of on it. An exact-zero test passes that straight through to a
    reader as an interval of one part in seven thousand on next week's count,
    which is the most confident and least justified number this service could
    emit. Anything under one countable unit is the same claim.
    """
    # Keyed on the *widest* step, because degeneracy is a property of the fit
    # rather than of a horizon position. A prediction band is non-decreasing in
    # the horizon, so if the furthest step still cannot resolve one unit then no
    # step had residual variation behind it. Testing every step instead would
    # floor an honest fit -- widths 0.5, 1.2, 2.0 -- on the strength of its
    # nearest, most certain point, replacing real estimates with a Poisson band
    # an order of magnitude wider and non-monotone against the steps beyond it.
    if any(high - low >= min_width for low, high in zip(lower, upper, strict=True)):
        return list(lower), list(upper), False

    z_score = _normal_quantile(0.5 + level / 2.0)
    level_estimate = max(abs(sum(values) / len(values)), 1.0)
    half_width = z_score * math.sqrt(level_estimate)
    return (
        [value - half_width for value in mean],
        [value + half_width for value in mean],
        True,
    )


def _normal_quantile(probability: float) -> float:
    """Inverse standard normal CDF, via the error function.

    Stdlib rather than scipy: this is called once per forecast and pulling in a
    second numerical stack for one quantile is not a trade worth making. Bisection
    on `erf` converges to machine precision in well under a hundred iterations
    over this bracket.
    """
    low, high = -10.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0))) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _warning_caveats(caught: Sequence[warnings.WarningMessage]) -> list[str]:
    """Turn fit-time warnings into reader-facing caveats, de-duplicated."""
    seen: dict[str, None] = {}
    for record in caught:
        name = record.category.__name__
        if "Convergence" in name:
            seen.setdefault(
                "The optimizer did not converge, so the fitted parameters are "
                "wherever it stopped. Treat the interval as a lower bound on the "
                "uncertainty.",
                None,
            )
        elif "Value" in name or name == "Warning":
            seen.setdefault(
                f"The fit reported {name}: {str(record.message)[:200]}",
                None,
            )
    return list(seen)


def _confidence(
    *,
    observations: int,
    horizon: int,
    points: Sequence[ForecastPoint],
    config: ForecastConfig,
) -> float:
    """A heuristic belief, clamped away from both certainties.

    Three terms, each in `[0, 1]`: how much history there was relative to a
    comfortable amount, how far the projection reaches relative to that history,
    and how tight the resulting band is relative to its own level. Weighted rather
    than multiplied so that one weak term degrades the score instead of
    annihilating it.

    Never 0 and never 1. A forecast that claims certainty is the failure this
    whole module is built against, and one that claims none should not have been
    returned at all -- the gates above would have refused it.
    """
    history_term = min(1.0, observations / (4.0 * config.min_observations))
    horizon_term = 1.0 / (1.0 + horizon / max(observations, 1))

    widths = []
    for point in points:
        level_estimate = max(abs(point.mean), 1.0)
        widths.append(point.width / (2.0 * level_estimate))
    relative_width = sum(widths) / len(widths) if widths else 0.0
    precision_term = 1.0 / (1.0 + relative_width)

    score = 0.4 * history_term + 0.3 * horizon_term + 0.3 * precision_term
    return min(0.95, max(0.05, score))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValidationError(
            "observation timestamps must be timezone-aware; a naive datetime is "
            "ambiguous the moment it crosses a process boundary (models/base.py)."
        )
    return value.astimezone(UTC)
