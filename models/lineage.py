"""Provenance: how a Signal came to exist, and what was done to it since.

Lineage exists so that any sentence in a generated report can be walked back to
bytes fetched from the internet at a known time by a known code version
(`docs/signal-model.md` §3.6). Without it, "our competitor's NPS is falling" is an
assertion; with it, it is a claim with a receipt.

Two envelope concerns live here rather than on `Signal` itself -- `schema_version`
and `status` -- so that the Design Doc §6 field list stays exactly as specified
(`docs/signal-model.md` §2).

`stages` is **append-only**. Reprocessing a Signal appends a new run's records
rather than replacing the old ones, which is what makes "this claim was made when
the sentiment model was v3" answerable after a model upgrade.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from models.base import Score, Sha256Hex, StrictModel, UtcDatetime
from models.enums import (
    STAGE_QUALITY_WEIGHTS,
    SignalStatus,
    StageName,
    StageStatus,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ConfidenceComponents",
    "Lineage",
    "StageRecord",
]


CURRENT_SCHEMA_VERSION = 1
"""Monotonic integer stamped on every Signal at normalization time.

The only thing that lets a consumer decide whether it understands a message
(`docs/signal-model.md` §7). Bumping it requires a dual-read window, because six
stores cannot be migrated atomically.
"""


class StageRecord(StrictModel):
    """One execution of one enrichment stage.

    `model` is populated only for stages that call a model (entities, sentiment,
    embedding). It is what makes a result reproducible-in-principle after a model
    upgrade: stages 1-3 and 6b are deterministic and replay identically, stages
    4-6 do not, so the model id is recorded rather than assumed
    (`docs/signal-model.md` §5.1).
    """

    name: StageName
    version: str = Field(description="Semantic version of the stage implementation.")
    model: str | None = Field(
        default=None,
        description="Model identifier, for stages that call a model. None otherwise.",
    )
    started_at: UtcDatetime
    duration_ms: int = Field(ge=0)
    status: StageStatus
    error: str | None = Field(
        default=None,
        description="Exception class name when status is FAILED. Never the full message: "
        "provider errors can echo request bodies, which may contain fetched content.",
    )

    @model_validator(mode="after")
    def _error_only_when_failed(self) -> Self:
        """A stage that succeeded must not carry an error, and vice versa.

        Caught here rather than in review because a `failed` record with no error
        makes the DLQ un-triageable, and an `ok` record carrying an error string
        silently misleads whoever reads the trace six months later.
        """
        if self.status is StageStatus.FAILED and not self.error:
            raise ValueError(f"stage {self.name!r} failed but recorded no error class")
        if self.status is not StageStatus.FAILED and self.error:
            raise ValueError(
                f"stage {self.name!r} has status {self.status!r} but recorded an error"
            )
        return self


class ConfidenceComponents(StrictModel):
    """The four inputs to `Signal.confidence` (`docs/signal-model.md` §3.5).

    Stored alongside the scalar so that a low score is explainable in the UI and
    in `agents/critic/` reasoning. A bare float would leave "why is this 0.31?"
    unanswerable.
    """

    source_credibility: Score = Field(
        description="Per-platform prior combined with author signals "
        "(follower count, account age, verification)."
    )
    extraction_quality: Score = Field(
        description="Stage-weighted fraction of enrichment stages that succeeded."
    )
    content_integrity: Score = Field(
        description="1.0 full body; 0.5 truncated; 0.2 title-only."
    )
    corroboration: Score = Field(
        description="Log-scaled count of independent near-duplicates. The only "
        "component that can rise after enrichment, as cluster members arrive."
    )


class Lineage(StrictModel):
    """Full provenance chain from raw payload to the current Signal."""

    # -- Processing ---------------------------------------------------------
    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, ge=1)
    pipeline_version: str = Field(
        description="Semantic version of the enrichment pipeline as a whole."
    )
    stages: list[StageRecord] = Field(
        default_factory=list,
        description="Append-only. Reprocessing appends; it never replaces.",
    )

    # -- Acquisition --------------------------------------------------------
    connector_slug: str
    connector_version: str
    sync_run_id: str = Field(
        description="Groups every Signal produced by one connector run, so a bad "
        "run can be identified and reverted wholesale."
    )
    fetched_at: UtcDatetime = Field(
        description="Ingestion time. Distinct from Signal.timestamp, which is "
        "event time at the source."
    )
    request_fingerprint: str | None = Field(
        default=None,
        description="Hash of the request that produced this record (endpoint + "
        "normalized params, never credentials). Makes a fetch reproducible.",
    )

    # -- Raw payload --------------------------------------------------------
    raw_object_key: str | None = Field(
        default=None,
        description="R2 key of the immutable original, so a cleaning bug is "
        "repairable by reprocessing rather than re-fetching.",
    )
    raw_sha256: Sha256Hex | None = None
    raw_bytes: int | None = Field(default=None, ge=0)
    raw_content_type: str | None = None

    # -- Identity -----------------------------------------------------------
    native_id: str = Field(
        description="The platform's own identifier, or the derived substitute. "
        "Input to Signal.id (`docs/signal-model.md` §4.1)."
    )
    status: SignalStatus = SignalStatus.RAW
    dedup_cluster_id: str | None = None
    duplicate_of: str | None = Field(
        default=None,
        description="Signal.id of the canonical cluster member, when this Signal "
        "is not itself canonical.",
    )

    # -- Scoring ------------------------------------------------------------
    confidence_components: ConfidenceComponents | None = None
    engagement_baseline: str | None = Field(
        default=None,
        description="Cohort and window used to normalize engagement, e.g. "
        "'reddit:text_post:30d'. A score computed against a cold-start cohort is "
        "not comparable to one computed against a mature cohort.",
    )

    # ------------------------------------------------------------------ API --

    @model_validator(mode="after")
    def _duplicate_consistency(self) -> Self:
        """`duplicate_of` and `status == DUPLICATE` must agree, and a Signal
        cannot be its own duplicate.

        Enforced because `docs/signal-model.md` §4.3 makes only the canonical
        member retrievable. A Signal marked `DUPLICATE` with no pointer would be
        unreachable through either path -- invisible rather than deduplicated.
        """
        if self.status is SignalStatus.DUPLICATE and self.duplicate_of is None:
            raise ValueError(
                "status is 'duplicate' but duplicate_of is unset; the canonical "
                "Signal would be unreachable"
            )
        if self.duplicate_of is not None and self.dedup_cluster_id is None:
            raise ValueError("duplicate_of is set but dedup_cluster_id is not")
        return self

    def latest_stages(self) -> dict[StageName, StageRecord]:
        """Most recent record per stage.

        `stages` is append-only, so a reprocessed Signal holds several records
        for the same stage. Scoring and status decisions must read the newest.
        """
        latest: dict[StageName, StageRecord] = {}
        for record in self.stages:
            latest[record.name] = record
        return latest

    def append_stage(self, record: StageRecord) -> None:
        """Append a stage record.

        Uses an explicit reassignment rather than `list.append` because
        `StrictModel` enables `validate_assignment`: mutating the list in place
        would bypass validation entirely.
        """
        self.stages = [*self.stages, record]

    def compute_extraction_quality(self) -> float:
        """Stage-weighted fraction of degradable enrichment stages that succeeded.

        Feeds `ConfidenceComponents.extraction_quality`. Only the four degradable
        stages carry weight (`STAGE_QUALITY_WEIGHTS`); the fatal stages -- clean,
        normalize, store -- are excluded because a Signal cannot exist without
        them, so crediting them would compress the useful range of the score.

        `SKIPPED` earns full credit. A stage disabled by configuration is not a
        quality failure, and penalizing it would make a deliberately cheap
        pipeline look untrustworthy. A stage never attempted earns nothing, which
        is why a mid-flight Signal scores low until the pipeline completes.
        """
        latest = self.latest_stages()
        earned = 0.0
        for stage, weight in STAGE_QUALITY_WEIGHTS.items():
            record = latest.get(stage)
            if record is not None and record.status in (StageStatus.OK, StageStatus.SKIPPED):
                earned += weight
        return round(earned, 6)

    def failed_stages(self) -> list[StageName]:
        """Stages whose most recent attempt failed. Drives the `partial` status."""
        return [
            name
            for name, record in self.latest_stages().items()
            if record.status is StageStatus.FAILED
        ]
