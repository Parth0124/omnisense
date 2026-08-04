"""Unit tests for `services/forecast_service.py`.

The three properties asserted hardest are the three the module was written to
guarantee, because each of them is what makes a forecast honest rather than
merely available:

1. **There is no path to a bare point estimate.** Every returned point carries a
   `lower` and an `upper`, the band is a *prediction* interval rather than the
   much narrower confidence interval on the fitted mean, and the interval widens
   with the horizon. A forecast whose interval does not grow with distance is not
   quantifying uncertainty, it is decorating a number.
2. **Assumptions are never empty**, and they name the conditions that actually
   break here -- coverage changes, dedup-rule changes, structural breaks.
3. **Too little history is refused.** Four points are refused, a history that is
   almost all zeros is refused, an unevenly spaced history is refused, and a
   horizon longer than the history is refused. Each refusal carries structured
   `details` so a handler can branch without parsing English.

Following `docs/testing-strategy.md`, numeric assertions are about *ordering and
bounds* rather than equality against a literal -- "0.5 < confidence <= 1.0, and
that the ordering between two scenarios holds" -- because a fitted value pinned
to four decimal places is a test of the statsmodels release, not of this module.

statsmodels, numpy and scipy are local, pinned dependencies. Nothing here opens a
socket and nothing needs a service running.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from backend.core.exceptions import ValidationError
from services.forecast_service import (
    Forecast,
    ForecastConfig,
    ForecastMethod,
    ForecastService,
    InsufficientHistoryError,
    Observation,
    observations_from_buckets,
)

pytestmark = pytest.mark.unit


START = datetime(2026, 5, 1, tzinfo=UTC)
DAY = timedelta(days=1)


def history(values: list[float], *, bucket: timedelta = DAY) -> list[Observation]:
    return observations_from_buckets(START, bucket, values)


def rising(n: int = 30, *, slope: float = 2.0, base: float = 10.0) -> list[Observation]:
    """A clean upward trend with a little structure, so a fit has residuals."""
    wobble = [0.0, 1.0, -1.0, 0.5, -0.5]
    return history([base + slope * i + wobble[i % len(wobble)] for i in range(n)])


@pytest.fixture
def service() -> ForecastService:
    return ForecastService()


# --------------------------------------------------------------------------- #
# 1. The interval is never optional
# --------------------------------------------------------------------------- #


def test_every_point_carries_an_interval(service: ForecastService) -> None:
    result = service.forecast(rising(), horizon=7)
    assert len(result.points) == 7
    for point in result.points:
        assert point.lower <= point.mean <= point.upper
        assert point.width > 0.0


def test_forecast_exposes_no_bare_point_estimate() -> None:
    """Structural, not stylistic.

    `docs/roadmap.md` makes "a point estimate alone never renders" a Phase 4
    shipping criterion, and an interface that offers a means-only accessor makes
    violating it the path of least resistance for every caller downstream.
    """
    accessors = {name for name in dir(Forecast) if not name.startswith("_")}
    assert "means" not in accessors
    assert "trajectory" not in accessors
    assert "points" in accessors


def test_interval_widens_with_the_horizon(service: ForecastService) -> None:
    """Uncertainty compounds. A flat band would be the tell that it is decoration."""
    result = service.forecast(rising(40), horizon=12)
    assert result.points[-1].width > result.points[0].width


def test_band_is_a_prediction_interval_not_a_mean_interval(service: ForecastService) -> None:
    """The single most consequential line in the module.

    The confidence interval on an OLS *mean* is far narrower than the interval a
    new observation falls in, and publishing the first as a forecast understates
    the uncertainty by exactly the factor that makes it look trustworthy. The
    prediction interval must be at least as wide as the residual spread; the mean
    interval at n=40 is a fraction of it.
    """
    observations = rising(40)
    result = service.forecast(observations, horizon=1, method=ForecastMethod.LINEAR_TREND)
    residual_scale = result.diagnostics["residual_scale"]
    assert residual_scale > 0.0
    # ~1.96 sigma either side at 95%, plus parameter uncertainty on top.
    assert result.points[0].width > 3.5 * residual_scale


def test_a_wider_level_gives_a_wider_band(service: ForecastService) -> None:
    narrow = service.forecast(rising(40), horizon=5, interval_level=0.80)
    wide = service.forecast(rising(40), horizon=5, interval_level=0.99)
    assert wide.points[0].width > narrow.points[0].width
    assert wide.interval_level == 0.99


def test_a_perfectly_linear_history_still_gets_a_band(service: ForecastService) -> None:
    """Zero residual variance is arithmetically a zero-width interval and
    editorially a claim that the future is known. The Poisson floor replaces it."""
    perfect = history([float(10 + 2 * i) for i in range(30)])
    result = service.forecast(perfect, horizon=5)
    assert all(point.width > 0.0 for point in result.points)
    assert any("no residual variation" in caveat for caveat in result.caveats)


def test_a_near_zero_band_is_floored_even_though_it_is_not_exactly_zero(
    service: ForecastService,
) -> None:
    """The floor cannot key on an exact zero, because the ARIMA path never emits one.

    Fitted against a perfectly linear history the state-space optimizer drives the
    innovation variance toward zero and halts at machine noise rather than on it,
    yielding a band of ~0.01 around a level near 70. An exact-zero guard passes
    that through as a claim to know next bucket's count to one part in thousands,
    which is a more confident statement than the zero-width band it was written to
    catch, arrived at by a rounding accident.
    """
    perfect = history([float(10 + 2 * i) for i in range(30)])
    result = service.forecast(perfect, horizon=5, method=ForecastMethod.ARIMA)
    assert result.method is ForecastMethod.ARIMA
    # The unfloored ARIMA band here is ~0.01 wide; the Poisson floor at this
    # level is several units. Asserting a bound, not a fitted literal.
    assert all(point.width > 1.0 for point in result.points)
    assert any("no residual variation" in caveat for caveat in result.caveats)


def test_a_genuinely_noisy_fit_is_not_floored(service: ForecastService) -> None:
    """The counterpart property, and the reason the floor is not simply a `max`.

    A Poisson floor applied unconditionally would widen every honest fit to
    `z * sqrt(level)` -- at a level near 50 that is a band of roughly 28, an order
    of magnitude past what the residuals support -- and overstating uncertainty
    misleads exactly as effectively as understating it. The floor is a repair for
    a degenerate fit, so a fit with real residual variation must come back
    untouched.
    """
    result = service.forecast(rising(40), horizon=5)
    assert not any("no residual variation" in caveat for caveat in result.caveats)
    poisson_floor_width = 2 * 1.96 * (result.points[0].mean ** 0.5)
    assert result.points[0].width < poisson_floor_width


def test_lower_bound_is_clipped_at_zero_for_counts(service: ForecastService) -> None:
    falling = history([float(max(0, 40 - 2 * i)) for i in range(30)])
    result = service.forecast(falling, horizon=10)
    assert all(point.lower >= 0.0 for point in result.points)
    assert any("cannot be negative" in assumption for assumption in result.assumptions)


def test_clipping_can_be_switched_off(service: ForecastService) -> None:
    """A quantity that genuinely can go negative -- a sentiment delta -- must not
    be silently floored."""
    falling = history([float(40 - 2 * i) for i in range(30)])
    result = service.forecast(falling, horizon=10, non_negative=False)
    assert min(point.lower for point in result.points) < 0.0


# --------------------------------------------------------------------------- #
# 2. Assumptions travel with the numbers
# --------------------------------------------------------------------------- #


def test_assumptions_are_never_empty(service: ForecastService) -> None:
    result = service.forecast(rising(), horizon=5)
    assert result.assumptions
    assert all(assumption.strip() for assumption in result.assumptions)


def test_assumptions_name_the_failure_modes_of_this_system(service: ForecastService) -> None:
    """Generic textbook assumptions would be true and useless. These are the ones
    that actually break: coverage changes, dedup-rule changes, structural breaks."""
    joined = " ".join(service.forecast(rising(), horizon=5).assumptions).lower()
    assert "connector" in joined
    assert "de-duplication" in joined or "dedup" in joined
    assert "structural break" in joined
    assert "prediction* interval" in joined or "prediction" in joined


def test_assumptions_describe_the_method_actually_used(service: ForecastService) -> None:
    linear = service.forecast(rising(20), horizon=5)
    assert linear.method is ForecastMethod.LINEAR_TREND
    assert any("locally linear" in text for text in linear.assumptions)

    arima = service.forecast(rising(40), horizon=5, method=ForecastMethod.ARIMA)
    assert arima.method is ForecastMethod.ARIMA
    assert any("stationary" in text for text in arima.assumptions)


def test_a_long_horizon_is_caveated(service: ForecastService) -> None:
    short = service.forecast(rising(40), horizon=2)
    stretched = service.forecast(rising(40), horizon=20)
    assert not any("exceeds" in caveat for caveat in short.caveats)
    assert any("exceeds" in caveat for caveat in stretched.caveats)


def test_a_mostly_empty_history_is_caveated(service: ForecastService) -> None:
    sparse = history([0, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0])
    result = service.forecast(sparse, horizon=4)
    assert any("empty" in caveat for caveat in result.caveats)


def test_a_history_near_the_minimum_is_caveated(service: ForecastService) -> None:
    result = service.forecast(rising(13), horizon=3)
    assert any("close to the minimum" in caveat for caveat in result.caveats)


# --------------------------------------------------------------------------- #
# 3. Refusals
# --------------------------------------------------------------------------- #


def test_four_observations_are_refused(service: ForecastService) -> None:
    """The named case. Four points fit a line perfectly and produce a quotable
    interval from two residual degrees of freedom."""
    with pytest.raises(InsufficientHistoryError) as raised:
        service.forecast(history([1, 2, 3, 4]), horizon=3)
    assert raised.value.details["observations"] == 4
    assert raised.value.details["minimum"] == 12
    assert raised.value.details["reason"] == "too_short"
    assert raised.value.code == "insufficient_history"
    assert raised.value.status_code == 422


def test_the_minimum_is_configurable_but_still_enforced() -> None:
    lenient = ForecastService(config=ForecastConfig(min_observations=6))
    result = lenient.forecast(history([1, 2, 3, 4, 5, 6, 7, 8]), horizon=2)
    assert result.observations_used == 8
    with pytest.raises(InsufficientHistoryError):
        lenient.forecast(history([1, 2, 3, 4, 5]), horizon=2)


def test_an_almost_entirely_empty_history_is_refused(service: ForecastService) -> None:
    """One spike and fifteen zeros is not a series: the fit is that one point's
    position on a line nothing else constrains."""
    with pytest.raises(InsufficientHistoryError) as raised:
        service.forecast(history([0] * 15 + [90]), horizon=3)
    assert raised.value.details["reason"] == "too_sparse"
    assert raised.value.details["nonzero"] == 1


def test_a_horizon_longer_than_the_history_is_refused(service: ForecastService) -> None:
    with pytest.raises(InsufficientHistoryError) as raised:
        service.forecast(rising(20), horizon=40)
    assert raised.value.details["reason"] == "horizon_too_long"
    assert raised.value.details["max_horizon"] == 20


def test_an_unevenly_spaced_history_is_refused(service: ForecastService) -> None:
    """A forecaster cannot tell a missing bucket from an empty one, and guessing
    is how a collection outage becomes a predicted decline."""
    points = rising(20)
    gapped = points[:10] + [
        Observation(at=point.at + timedelta(days=3), value=point.value) for point in points[10:]
    ]
    with pytest.raises(ValidationError, match="evenly spaced"):
        service.forecast(gapped, horizon=3)


def test_duplicate_timestamps_are_refused(service: ForecastService) -> None:
    points = rising(20)
    with pytest.raises(ValidationError, match="duplicate timestamps"):
        service.forecast([*points, Observation(at=points[0].at, value=99.0)], horizon=3)


def test_a_naive_timestamp_is_refused(service: ForecastService) -> None:
    points = rising(20)
    naive = [Observation(at=point.at.replace(tzinfo=None), value=point.value) for point in points]
    with pytest.raises(ValidationError, match="timezone-aware"):
        service.forecast(naive, horizon=3)


@pytest.mark.parametrize("horizon", [0, -1])
def test_a_non_positive_horizon_is_a_request_error(
    service: ForecastService, horizon: int
) -> None:
    with pytest.raises(ValidationError):
        service.forecast(rising(), horizon=horizon)


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5, -0.2])
def test_an_impossible_interval_level_is_rejected(
    service: ForecastService, level: float
) -> None:
    with pytest.raises(ValidationError):
        service.forecast(rising(), horizon=3, interval_level=level)


def test_requesting_arima_without_enough_history_refuses_rather_than_downgrades(
    service: ForecastService,
) -> None:
    """A silent downgrade would return a different model under the requested
    name, which is how a reproducibility claim quietly stops being true."""
    with pytest.raises(InsufficientHistoryError) as raised:
        service.forecast(rising(14), horizon=3, method=ForecastMethod.ARIMA)
    assert raised.value.details["reason"] == "method_unsupported"


def test_forecast_config_rejects_incoherent_gates() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(min_observations=2)
    with pytest.raises(ValidationError):
        ForecastConfig(min_observations=30, arima_min_observations=10)
    with pytest.raises(ValidationError):
        ForecastConfig(max_horizon_ratio=0.0)


# --------------------------------------------------------------------------- #
# Method selection and mechanics
# --------------------------------------------------------------------------- #


def test_method_is_selected_by_history_length(service: ForecastService) -> None:
    assert service.forecast(rising(20), horizon=3).method is ForecastMethod.LINEAR_TREND
    assert service.forecast(rising(40), horizon=3).method is ForecastMethod.ARIMA


def test_history_is_sorted_before_fitting(service: ForecastService) -> None:
    """An out-of-order history still fits, still produces a slope, and says
    nothing about being wrong -- so it is normalized rather than trusted."""
    ordered = rising(20)
    shuffled = [ordered[i] for i in (5, 0, 12, 3, *range(20)) if i < 20]
    unique = {point.at: point for point in shuffled}
    result = service.forecast(list(unique.values()), horizon=3)
    assert result.history_start == ordered[0].at
    assert result.history_end == ordered[-1].at


def test_projected_timestamps_continue_the_bucket_grid(service: ForecastService) -> None:
    result = service.forecast(rising(20), horizon=3)
    assert result.bucket == DAY
    assert result.points[0].at == result.history_end + DAY
    assert result.points[-1].at == result.history_end + 3 * DAY
    assert result.horizon_end == result.points[-1].at


def test_an_upward_trend_projects_upward(service: ForecastService) -> None:
    result = service.forecast(rising(30, slope=3.0), horizon=5)
    assert result.points[-1].mean > result.points[0].mean


def test_hourly_buckets_are_supported(service: ForecastService) -> None:
    hourly = history([float(5 + i) for i in range(24)], bucket=timedelta(hours=1))
    result = service.forecast(hourly, horizon=4)
    assert result.bucket == timedelta(hours=1)
    assert result.points[0].at == result.history_end + timedelta(hours=1)


# --------------------------------------------------------------------------- #
# Confidence: ordering, not equality
# --------------------------------------------------------------------------- #


def test_confidence_is_bounded_away_from_both_certainties(service: ForecastService) -> None:
    result = service.forecast(rising(40), horizon=5)
    assert 0.05 <= result.confidence <= 0.95


def test_more_history_raises_confidence(service: ForecastService) -> None:
    short = service.forecast(rising(14), horizon=3)
    long = service.forecast(rising(60), horizon=3)
    assert long.confidence > short.confidence


def test_a_longer_horizon_lowers_confidence(service: ForecastService) -> None:
    near = service.forecast(rising(60), horizon=2)
    far = service.forecast(rising(60), horizon=30)
    assert far.confidence < near.confidence


def test_a_noisier_history_lowers_confidence() -> None:
    """Same length, same horizon; only the spread differs."""
    service = ForecastService()
    calm = history([10.0 + 0.1 * (i % 3) for i in range(30)])
    noisy = history([10.0 + 25.0 * ((i * 7) % 5) for i in range(30)])
    assert service.forecast(noisy, horizon=5).confidence < (
        service.forecast(calm, horizon=5).confidence
    )
