"""The enrichment pipeline: raw record in, enriched Signal out.

Design Doc §6 fixes the stage order:

    Clean -> Normalize -> Language -> Entities -> Sentiment -> Embedding -> Store

with Scoring folded in as stage 6b. This module owns the *orchestration* of those
stages; each stage's logic lives in its own sibling module and is swapped freely
because every one of them satisfies the same `Stage` protocol.

The single most important behaviour here is **partial failure**
(`docs/signal-model.md` §5.2). The pipeline never discards a Signal because an
optional enrichment failed -- a news article with no sentiment is still
retrievable, citable and countable. Three stages are fatal (Clean, Normalize,
Store) because without them no usable Signal exists at all. Every other stage
degrades to a documented empty value, records the failure in
`lineage.stages[]`, and the Signal is stored as `partial` with a reduced
confidence.

That asymmetry is the whole design. A pipeline that treated every stage as fatal
would drop a week of ingestion the first time an embedding provider rate-limited
us; one that treated every stage as optional would silently store Signals with no
text.

Why a protocol rather than a base class: stages differ in what they need
(`Language` needs nothing, `Embedding` needs a provider and a chunker, `Store`
needs five clients). A shared base class would either take every dependency in
one constructor or push them into globals. A `Protocol` lets each stage be
constructed with exactly its own dependencies and lets a test substitute a
two-line fake.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from models.base import utcnow
from models.enums import FATAL_STAGES, SignalStatus, StageName, StageStatus
from models.lineage import StageRecord
from models.signal import Signal
from services.events.schemas import RawRecordEvent

__all__ = [
    "EnrichmentContext",
    "PipelineResult",
    "SignalPipeline",
    "Stage",
    "StageOutcome",
]


@dataclass(slots=True)
class EnrichmentContext:
    """Mutable state threaded through one pass of the pipeline.

    Carries the Signal being built plus the bookkeeping stages need to cooperate
    without knowing about each other. `signal` is `None` until stage 2
    (Normalize) constructs it -- stage 1 operates on raw bytes, which is exactly
    why Clean and Normalize are separate stages rather than one.
    """

    raw_bytes: bytes | None = None
    content_type: str = "application/json"
    payload: dict[str, object] = field(default_factory=dict)
    cleaned_text: str | None = None
    signal: Signal | None = None
    pipeline_version: str = "1.0.0"
    skip: frozenset[StageName] = frozenset()

    record: RawRecordEvent | None = None
    """The event that sent this record here, carried verbatim for stage 2.

    `docs/data-stores.md` §5.1 publishes the raw payload's *address* and its
    provenance, never the connector's mapped output, so the connector slug, the
    sync run, the fetch time and the R2 key exist nowhere else on this context --
    and `services/signal_engine/normalize.py` needs all four to rebuild `Lineage`.

    Optional because a pipeline can legitimately be driven over a Signal that
    already exists: `workers/embedding_worker.py` re-runs stage 6 alone against a
    row read back from PostgreSQL, and there is no raw record event in sight. It
    is `None` for exactly those runs, and stage 2 says so plainly when it is
    absent rather than failing several frames deeper on a missing slug.
    """

    def require_signal(self) -> Signal:
        """Fetch the Signal, or fail loudly if a stage ran out of order.

        A stage that reads `ctx.signal` before Normalize has run is a wiring bug,
        and an `AttributeError` on `None` several frames deeper is a poor way to
        discover it.
        """
        if self.signal is None:
            raise RuntimeError(
                "no Signal on the context yet; a stage after Normalize ran before it. "
                "Check the stage order passed to SignalPipeline."
            )
        return self.signal


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """What one stage did, for `lineage.stages[]` and for the scoring component."""

    name: StageName
    status: StageStatus
    duration_ms: int
    model: str | None = None
    error: str | None = None


@runtime_checkable
class Stage(Protocol):
    """One enrichment step.

    Implementations mutate `ctx` in place and return nothing. Raising is how a
    stage reports failure; the pipeline decides whether that is fatal based on
    `FATAL_STAGES`, not on anything the stage says. Keeping that decision out of
    the stages means a stage cannot accidentally promote itself to fatal and take
    down ingestion.
    """

    name: StageName
    version: str

    @property
    def model_id(self) -> str | None:
        """Model this stage used, when it used one.

        Recorded per stage because stages 4-6 are non-deterministic: reproducing
        a result later requires knowing which model produced it
        (`docs/signal-model.md` §5.1).
        """
        ...

    async def apply(self, ctx: EnrichmentContext) -> None:
        """Run this stage against the context, mutating it in place."""
        ...


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Outcome of one full pass."""

    signal: Signal | None
    status: SignalStatus
    outcomes: Sequence[StageOutcome]
    fatal_stage: StageName | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether a usable Signal exists. `partial` counts -- it is still citable."""
        return self.signal is not None and self.fatal_stage is None

    @property
    def failed_stages(self) -> list[StageName]:
        return [o.name for o in self.outcomes if o.status is StageStatus.FAILED]


class SignalPipeline:
    """Runs an ordered list of stages over one raw record.

    Stateless and reusable: one instance is constructed per worker and driven
    concurrently, so nothing per-record may be stored on `self`.
    """

    def __init__(self, stages: Sequence[Stage], *, pipeline_version: str = "1.0.0") -> None:
        self._stages = tuple(stages)
        self.pipeline_version = pipeline_version
        self._validate_order()

    def _validate_order(self) -> None:
        """Reject a mis-ordered stage list at construction, not mid-record.

        The order is fixed by Design Doc §6 and every stage after Normalize
        assumes a Signal exists. Catching an inversion here costs one assertion
        at startup; catching it in production costs a DLQ full of records that
        failed for a reason nobody can read off the error.
        """
        canonical = [
            StageName.CLEAN,
            StageName.NORMALIZE,
            StageName.LANGUAGE,
            StageName.ENTITIES,
            StageName.SENTIMENT,
            StageName.EMBEDDING,
            StageName.SCORING,
            StageName.STORE,
        ]
        rank = {name: i for i, name in enumerate(canonical)}
        seen = [s.name for s in self._stages]

        unknown = [n for n in seen if n not in rank]
        if unknown:
            raise ValueError(f"unknown stages in pipeline: {unknown}")
        if len(set(seen)) != len(seen):
            raise ValueError(f"duplicate stages in pipeline: {seen}")

        ranks = [rank[n] for n in seen]
        if ranks != sorted(ranks):
            raise ValueError(
                f"stages are out of order: {[n.value for n in seen]}. "
                f"Design Doc §6 fixes the order as {[n.value for n in canonical]}."
            )

        self._require_normalize(seen, canonical, rank)

    @staticmethod
    def _require_normalize(
        seen: Sequence[StageName],
        canonical: Sequence[StageName],
        rank: dict[StageName, int],
    ) -> None:
        """Reject an ingest pipeline that never builds a Signal.

        Stage 2 is the only stage that assigns `ctx.signal`, so a list that
        starts at Clean, skips Normalize and then runs Language, Entities,
        Sentiment, Embedding, Scoring or Store produces *nothing at all*. Without
        this check the symptom is a run of `RuntimeError`s from
        `require_signal()`: five degradable stages fail one after another, each
        recorded as a `partial`-making failure that cannot be attached to any
        lineage because there is no Signal to attach it to, and the pipeline only
        turns fatal on reaching Store. Every one of those errors describes a
        consequence; none names the cause. Catching it at construction costs one
        comparison at startup.

        **The check is anchored on Clean, not applied unconditionally**, and the
        exception is load-bearing rather than a convenience. A list that starts
        *after* Normalize is a re-drive, not an ingest: `workers/embedding_worker.py`
        re-runs stage 6 over a Signal read back from PostgreSQL, the enrichment
        sweeper re-runs the degradable stages over a `partial` row, and both are
        handed a context whose `signal` is already populated. Rejecting those at
        construction would forbid the retry path that
        `docs/signal-model.md` §5.2 requires. A fragment that genuinely is
        mis-wired -- no Clean, no Normalize, no Signal on the context -- is caught
        on the first record by `require_signal()`, which is the right place for it
        because only the caller knows whether it meant to supply one.
        """
        if StageName.NORMALIZE in seen or StageName.CLEAN not in seen:
            return
        dependent = [
            name.value for name in seen if rank[name] > rank[StageName.NORMALIZE]
        ]
        if not dependent:
            return
        raise ValueError(
            f"pipeline starts at {StageName.CLEAN.value!r} and runs {dependent} but has "
            f"no {StageName.NORMALIZE.value!r} stage. Stage 2 is the only stage that "
            "builds the Signal, so every one of those stages would fail on "
            "require_signal() for every record until Store made it fatal. "
            f"Design Doc §6 fixes the order as {[n.value for n in canonical]}."
        )

    async def run(self, ctx: EnrichmentContext) -> PipelineResult:
        """Execute every stage, degrading where the contract allows.

        Returns rather than raises even on a fatal stage: the caller
        (`workers/enrichment_worker.py`) needs the partial outcome list to build
        a useful DLQ record, and an exception would discard exactly the
        diagnostic information that makes a replay possible.
        """
        ctx.pipeline_version = self.pipeline_version
        outcomes: list[StageOutcome] = []

        for stage in self._stages:
            if stage.name in ctx.skip:
                outcomes.append(
                    StageOutcome(
                        name=stage.name,
                        status=StageStatus.SKIPPED,
                        duration_ms=0,
                        model=stage.model_id,
                    )
                )
                self._record(ctx, outcomes[-1], stage)
                continue

            started = time.perf_counter()
            try:
                await stage.apply(ctx)
            except Exception as exc:  # noqa: BLE001 -- classification is the point
                duration = _elapsed_ms(started)
                # Only the exception *class* is recorded. A provider message can
                # echo the request that caused it, and requests carry fetched
                # content (`docs/security-and-privacy.md`).
                outcome = StageOutcome(
                    name=stage.name,
                    status=StageStatus.FAILED,
                    duration_ms=duration,
                    model=stage.model_id,
                    error=type(exc).__name__,
                )
                outcomes.append(outcome)
                self._record(ctx, outcome, stage)

                if stage.name in FATAL_STAGES:
                    return PipelineResult(
                        signal=ctx.signal,
                        status=SignalStatus.QUARANTINED,
                        outcomes=outcomes,
                        fatal_stage=stage.name,
                        error=type(exc).__name__,
                    )
                # Degradable: the field keeps its documented empty value, which
                # the model already defaults to, so there is nothing to reset.
                continue

            outcome = StageOutcome(
                name=stage.name,
                status=StageStatus.OK,
                duration_ms=_elapsed_ms(started),
                model=stage.model_id,
            )
            outcomes.append(outcome)
            self._record(ctx, outcome, stage)

        return PipelineResult(
            signal=ctx.signal,
            status=self._final_status(outcomes),
            outcomes=outcomes,
        )

    def _record(self, ctx: EnrichmentContext, outcome: StageOutcome, stage: Stage) -> None:
        """Append the stage record to lineage, once a Signal exists to hold it.

        Stage 1 runs before there is a Signal, so its record cannot be attached.
        That is not a loss: a Clean failure is fatal and the raw record goes to
        the DLQ carrying the outcome list directly.
        """
        if ctx.signal is None:
            return
        ctx.signal.lineage.append_stage(
            StageRecord(
                name=outcome.name,
                version=stage.version,
                model=outcome.model,
                started_at=utcnow(),
                duration_ms=outcome.duration_ms,
                status=outcome.status,
                error=outcome.error,
            )
        )

    @staticmethod
    def _final_status(outcomes: Sequence[StageOutcome]) -> SignalStatus:
        """`enriched` only if nothing failed; otherwise `partial`.

        `SKIPPED` does not demote to partial -- a stage disabled by configuration
        is not a degraded result, and treating it as one would make a
        deliberately cheap pipeline permanently look untrustworthy.
        """
        if any(o.status is StageStatus.FAILED for o in outcomes):
            return SignalStatus.PARTIAL
        return SignalStatus.ENRICHED


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
