"""Runs golden cases against agents and scores them. The measurement loop.

Ties together the two halves: `golden_sets.py` says what to run and what must be
true; `rubrics.py` says how to weight what is found. This module executes cases,
checks the mechanical properties itself, optionally asks a judge about the rest,
and produces a result somebody can act on.

**Mechanical checks run here, in Python, with no model involved.** "Do these
citations resolve" is arithmetic. Handing it to an LLM judge would be slower,
more expensive, and *wrong more often* -- a judge reading a plausible-looking
signal id has no way to know it is fabricated and will say it looks fine. Every
property in `ExpectedProperties` is checked by code.

**Judged criteria are optional and clearly labelled.** With no judge configured,
the harness reports mechanical scores only and says so. That is a genuinely
useful mode -- it runs in CI, it is deterministic, and it catches the failures
that matter most. What it must not do is report a partial score as if it were
complete, which is why `EvaluationResult` carries `judged` explicitly.

**A failing case is never a raised exception.** A run over sixteen cases where
the third throws tells you nothing about the other thirteen. Every case is
isolated; a crash is recorded as a failure with its exception type and the run
continues.

**Nothing here writes to the golden set.** A harness that could update expected
values would, the first time a case failed inconveniently -- and the suite would
become a transcript of current behaviour rather than a check on it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from backend.core.logging import get_logger
from models.enums import AgentName

from agents.evaluation.golden_sets import (
    GOLDEN_CASES,
    CaseKind,
    ExpectedProperties,
    GoldenCase,
)
from agents.evaluation.rubrics import PASS_THRESHOLD, RubricScore, rubric_for

__all__ = [
    "CaseResult",
    "EvaluationHarness",
    "EvaluationResult",
    "Judge",
    "PropertyCheck",
    "check_properties",
]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PropertyCheck:
    """One mechanical assertion and whether it held."""

    name: str
    passed: bool
    detail: str = ""

    @property
    def as_score(self) -> float:
        return 1.0 if self.passed else 0.0


class Judge(Protocol):
    """Scores a judged criterion. An LLM or a human, from this module's view.

    Returns 0-1 for one criterion against one output. Narrow deliberately: a
    judge that received the rubric and returned a total would be doing the
    weighting too, and the weighting is `rubrics.py`'s job -- kept there so a
    weight change does not require re-running the judge.
    """

    async def score(
        self, *, criterion_key: str, guidance: str, case: GoldenCase, output: Any
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One case, run and scored."""

    case_id: str
    agent: AgentName
    kind: CaseKind
    checks: tuple[PropertyCheck, ...]
    score: RubricScore | None
    duration_ms: float
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Mechanical checks all held, and the rubric score cleared the bar.

        Both, not either. A case whose properties held but which scored poorly is
        still a problem -- the properties are a floor, not the definition of good.
        """
        if self.error is not None:
            return False
        if not all(check.passed for check in self.checks):
            return False
        return self.score is None or self.score.passed

    @property
    def failed_checks(self) -> tuple[PropertyCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """A whole run. Every number here is meant to be acted on."""

    results: tuple[CaseResult, ...]
    judged: bool
    """Whether judged criteria were scored.

    Carried explicitly so a mechanical-only run cannot be mistaken for a complete
    one. A 0.8 covering 70% of the weight is not the same claim as a 0.8 covering
    all of it.
    """

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def regressions(self) -> tuple[CaseResult, ...]:
        """Failing cases that pin a bug which has already happened once.

        Separated because they are the ones that must block. A capability case
        failing may mean the feature changed; a regression case failing means a
        fixed bug came back.
        """
        return tuple(
            result
            for result in self.results
            if result.kind is CaseKind.REGRESSION and not result.passed
        )

    @property
    def adversarial_failures(self) -> tuple[CaseResult, ...]:
        """Failing cases designed to induce a specific failure.

        Reported separately because these are the ones most likely to be quietly
        dropped from a suite -- they are the ones that fail.
        """
        return tuple(
            result
            for result in self.results
            if result.kind is CaseKind.ADVERSARIAL and not result.passed
        )

    def format_report(self) -> str:
        """A fixed-width summary for a CI log.

        Present because a result nobody reads changes nothing, and a nested dict
        of dataclasses is not read by anyone.
        """
        lines = [
            f"cases={self.total}  passed={self.passed}  "
            f"rate={self.pass_rate:.0%}  judged={'yes' if self.judged else 'MECHANICAL ONLY'}",
        ]
        if not self.judged:
            lines.append(
                "  (no judge configured -- judged criteria scored 0; the totals "
                "below are lower bounds, not measurements)"
            )
        for result in self.results:
            if result.passed:
                continue
            mark = "REGRESSION" if result.kind is CaseKind.REGRESSION else result.kind.value
            score = f"{result.score.total:.2f}" if result.score else "n/a"
            lines.append(f"  FAIL [{mark}] {result.case_id} (score {score})")
            if result.error:
                lines.append(f"        error: {result.error}")
            for check in result.failed_checks:
                lines.append(f"        {check.name}: {check.detail}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Mechanical property checking
# --------------------------------------------------------------------------- #


def check_properties(
    expected: ExpectedProperties, output: Any, *, resolvable_signals: Sequence[str] = ()
) -> tuple[PropertyCheck, ...]:
    """Check every asserted property against the output. No model involved.

    A property left `None` is not asserted and produces no check -- as opposed to
    producing a passing one, which would inflate the count and make a case with
    two assertions look as thorough as one with eight.
    """
    checks: list[PropertyCheck] = []
    items = _items_of(output)
    text = _text_of(output)

    if expected.min_items is not None:
        checks.append(
            PropertyCheck(
                "min_items",
                len(items) >= expected.min_items,
                f"got {len(items)}, expected at least {expected.min_items}",
            )
        )
    if expected.max_items is not None:
        checks.append(
            PropertyCheck(
                "max_items",
                len(items) <= expected.max_items,
                f"got {len(items)}, expected at most {expected.max_items}",
            )
        )

    if expected.must_cite_signals:
        uncited = [
            index for index, item in enumerate(items) if not _signal_ids_of(item)
        ]
        checks.append(
            PropertyCheck(
                "must_cite_signals",
                not uncited,
                f"{len(uncited)} item(s) carry no citation",
            )
        )

    if expected.all_citations_resolve:
        known = set(resolvable_signals)
        dangling = [
            signal_id
            for item in items
            for signal_id in _signal_ids_of(item)
            if signal_id not in known
        ]
        checks.append(
            PropertyCheck(
                "all_citations_resolve",
                not dangling,
                f"unresolvable: {sorted(set(dangling))[:5]}",
            )
        )

    for name in expected.must_mention_entities:
        checks.append(
            PropertyCheck(
                f"mentions:{name}",
                name.casefold() in text.casefold(),
                f"{name!r} not found in output",
            )
        )

    for forbidden in expected.must_not_mention:
        # Case-insensitive, because an injection that survives as
        # "Ignore Previous Instructions" is exactly as successful as the
        # uppercase form.
        checks.append(
            PropertyCheck(
                f"absent:{forbidden}",
                forbidden.casefold() not in text.casefold(),
                f"{forbidden!r} leaked into the output",
            )
        )

    if expected.must_flag_degraded is not None:
        flagged = bool(_attr(output, "degraded_backends") or _attr(output, "is_degraded"))
        checks.append(
            PropertyCheck(
                "must_flag_degraded",
                flagged == expected.must_flag_degraded,
                f"degraded flag was {flagged}",
            )
        )

    if expected.must_report_gaps is not None:
        gaps = _attr(output, "gaps") or _attr(output, "unanswered_sub_questions") or []
        checks.append(
            PropertyCheck(
                "must_report_gaps",
                bool(gaps) == expected.must_report_gaps,
                f"{len(gaps)} gap(s) reported",
            )
        )

    if expected.must_refuse is not None:
        refused = bool(
            _attr(output, "withheld_reason")
            or _attr(output, "skipped_reason")
            or (not items and _attr(output, "notes"))
        )
        checks.append(
            PropertyCheck(
                "must_refuse",
                refused == expected.must_refuse,
                "expected an explicit refusal with a stated reason"
                if expected.must_refuse
                else "unexpected refusal",
            )
        )

    for bound, name, comparator in (
        (expected.max_confidence, "max_confidence", lambda v, b: v <= b),
        (expected.min_confidence, "min_confidence", lambda v, b: v >= b),
    ):
        if bound is None:
            continue
        confidence = _attr(output, "confidence")
        value = float(confidence) if isinstance(confidence, (int, float)) else 0.0
        checks.append(
            PropertyCheck(name, comparator(value, bound), f"confidence was {value:.2f}")
        )

    return tuple(checks)


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class EvaluationHarness:
    """Runs cases against a callable that produces an agent's output."""

    runner: Callable[[GoldenCase], Any]
    """Produces output for one case. Async or sync.

    A callable rather than an agent instance so the harness can be driven against
    a real agent, a recorded fixture, or a deliberately broken stub -- which is
    how the harness itself gets tested.
    """

    judge: Judge | None = None
    resolvable_signals: Sequence[str] = ()
    concurrency: int = 4

    async def run(
        self, cases: Sequence[GoldenCase] = GOLDEN_CASES
    ) -> EvaluationResult:
        """Run every case, bounded, isolating failures.

        Bounded because each case may make model calls, and sixteen at once
        against a rate-limited provider produces sixteen 429s rather than
        sixteen results.
        """
        semaphore = asyncio.Semaphore(max(1, self.concurrency))

        async def run_one(case: GoldenCase) -> CaseResult:
            async with semaphore:
                return await self._run_case(case)

        results = await asyncio.gather(*(run_one(case) for case in cases))
        outcome = EvaluationResult(results=tuple(results), judged=self.judge is not None)
        logger.info(
            "evaluation.complete",
            cases=outcome.total,
            passed=outcome.passed,
            regressions=len(outcome.regressions),
            judged=outcome.judged,
        )
        return outcome

    async def _run_case(self, case: GoldenCase) -> CaseResult:
        """Run and score one case. Never raises.

        A crash is recorded as a failure carrying its exception type. Letting it
        propagate would abandon every other case in the run, so one broken agent
        would tell you nothing about the other nine.
        """
        started = time.perf_counter()
        try:
            output = self.runner(case)
            if asyncio.iscoroutine(output):
                output = await output
        except Exception as error:  # noqa: BLE001 -- see the docstring
            logger.warning(
                "evaluation.case_crashed", case=case.id, error=type(error).__name__
            )
            return CaseResult(
                case_id=case.id,
                agent=case.agent,
                kind=case.kind,
                checks=(),
                score=None,
                duration_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(error).__name__}: {error}",
            )

        checks = check_properties(
            case.expected,
            output,
            resolvable_signals=case.fixture_signals or self.resolvable_signals,
        )
        score = await self._score(case, output, checks)
        return CaseResult(
            case_id=case.id,
            agent=case.agent,
            kind=case.kind,
            checks=checks,
            score=score,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def _score(
        self, case: GoldenCase, output: Any, checks: Sequence[PropertyCheck]
    ) -> RubricScore | None:
        """Score against the agent's rubric.

        Mechanical criteria take the aggregate of the property checks; judged ones
        are asked of the judge, or scored zero when there is none. Zero rather
        than omitted, because omitting would renormalise the weights and make an
        unjudged run score *higher* than a judged one -- which is exactly the
        wrong incentive.
        """
        try:
            rubric = rubric_for(case.agent)
        except KeyError:
            return None

        mechanical = (
            sum(check.as_score for check in checks) / len(checks) if checks else 1.0
        )
        values: dict[str, float] = {}
        for criterion in rubric.criteria:
            if criterion.kind.value == "mechanical":
                values[criterion.key] = mechanical
                continue
            if self.judge is None:
                values[criterion.key] = 0.0
                continue
            try:
                values[criterion.key] = await self.judge.score(
                    criterion_key=criterion.key,
                    guidance=criterion.guidance,
                    case=case,
                    output=output,
                )
            except Exception as error:  # noqa: BLE001 -- one criterion, not the run
                logger.warning(
                    "evaluation.judge_failed",
                    case=case.id,
                    criterion=criterion.key,
                    error=type(error).__name__,
                )
                values[criterion.key] = 0.0
        return rubric.score(values)


# --------------------------------------------------------------------------- #
# Output introspection
# --------------------------------------------------------------------------- #
#
# Duck-typed because the harness runs against ten different output models plus
# whatever a fixture returns. An isinstance ladder over every agent's schema
# would need editing each time an agent is added, and the edit would be forgotten.


def _attr(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, Mapping):
        return obj.get(name)
    return value


def _items_of(output: Any) -> list[Any]:
    """The output's produced items, whatever that agent calls them."""
    for name in (
        "insights",
        "recommendations",
        "findings",
        "trends",
        "items",
        "competitors",
        "forecasts",
        "steps",
    ):
        value = _attr(output, name)
        if isinstance(value, (list, tuple)):
            return list(value)
    return []


def _signal_ids_of(item: Any) -> list[str]:
    for name in ("signal_ids", "citations", "supporting_signal_ids"):
        value = _attr(item, name)
        if isinstance(value, (list, tuple)):
            return [entry for entry in value if isinstance(entry, str)]
    return []


def _text_of(output: Any) -> str:
    """Everything the output says, flattened, for absence assertions.

    `repr` rather than a field walk. An injection that leaked into an unexpected
    field is exactly as successful as one in the expected place, and a walk over
    known fields would miss it -- which is the failure mode of a check that only
    looks where it expects trouble.
    """
    try:
        return repr(output)
    except Exception:  # noqa: BLE001 -- an unreprable output asserts nothing
        return ""
