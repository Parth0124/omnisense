"""Entities: surface mentions in text, and the canonical things they refer to.

Two distinct models, and conflating them is the most common modelling mistake in
this area:

`EntityMention`
    A *span of text* in one Signal. "Datadog" at characters 4-11. Produced by
    enrichment stage 4 (`services/signal_engine/entities.py`). Cheap, local, and
    possibly wrong -- it carries candidates and a link score, not a verdict.

`Entity`
    A *thing in the world* that many mentions across many Signals refer to.
    Produced by `graph/resolution/` and persisted as a Neo4j node
    (`docs/knowledge-graph.md`). This is what `COMPETES_WITH` connects.

The split is what allows resolution to be revisited: re-running entity resolution
changes which `Entity` a mention points at without re-running extraction, and
without touching the Signal's text.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from models.base import LenientModel, NonEmptyStr, Score, StrictModel, UtcDatetime
from models.enums import EntityType

__all__ = ["Entity", "EntityMention"]


class EntityMention(StrictModel):
    """One surface mention of an entity within a single Signal's text.

    Character offsets are into `Signal.content.text` -- the *cleaned* body, not
    the raw payload. This matters: offsets taken against raw HTML would drift the
    moment the cleaner changed, silently mis-highlighting every citation in the
    UI. Because `content.text` is what gets embedded and chunked, offsets stay
    valid for exactly as long as the text does.
    """

    surface: NonEmptyStr = Field(description="The literal text as it appeared.")
    type: EntityType
    start: int = Field(ge=0, description="Inclusive start offset into content.text.")
    end: int = Field(gt=0, description="Exclusive end offset into content.text.")
    candidate_ids: list[str] = Field(
        default_factory=list,
        description="Knowledge-graph entity ids this mention might refer to, best "
        "first. Retained even after resolution so a wrong link is diagnosable.",
    )
    resolved_id: str | None = Field(
        default=None,
        description="The chosen Entity.id, or None when resolution was ambiguous "
        "or has not run. None is a legitimate outcome, not a failure.",
    )
    link_score: Score | None = Field(
        default=None,
        description="Confidence in `resolved_id`. None when unresolved.",
    )

    @model_validator(mode="after")
    def _check_span_and_resolution(self) -> Self:
        """Offsets must describe a real span, and resolution must be coherent."""
        if self.end <= self.start:
            raise ValueError(
                f"empty or inverted span: start={self.start}, end={self.end}"
            )
        if self.resolved_id is not None and self.resolved_id not in self.candidate_ids:
            # Not fatal -- a resolver may legitimately introduce an id that
            # extraction never proposed (e.g. via an alias table). But the
            # candidate list is the audit trail, so keep it complete.
            self.candidate_ids = [self.resolved_id, *self.candidate_ids]
        if self.resolved_id is None and self.link_score is not None:
            raise ValueError("link_score is set but the mention is unresolved")
        return self

    @property
    def is_resolved(self) -> bool:
        """Whether this mention points at a canonical `Entity`."""
        return self.resolved_id is not None


class Entity(LenientModel):
    """A canonical thing in the knowledge graph -- one node in Neo4j.

    `LenientModel` rather than `StrictModel`: entities are *read* far more than
    they are written, and graph nodes accumulate properties from several writers
    (resolution, analytics, enrichment). A reader must not break because a
    centrality score it does not know about was added last week.
    """

    id: NonEmptyStr = Field(
        description="Stable canonical id, e.g. 'ent_datadog'. Survives merges: "
        "when two entities are merged the loser gains a SAME_AS edge rather than "
        "being deleted, so historical references stay resolvable."
    )
    type: EntityType
    canonical_name: NonEmptyStr = Field(
        description="The preferred display name. Indexed for lookup in "
        "graph/schema/constraints.py."
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Known alternative surfaces, used for blocking during "
        "resolution and for query expansion in retrieval/graph_retrieval/.",
    )
    description: str | None = None

    # -- Resolution provenance ---------------------------------------------
    merged_from: list[str] = Field(
        default_factory=list,
        description="Entity ids absorbed into this one. Makes a merge reversible, "
        "which `docs/knowledge-graph.md` requires: resolution errors are corrected "
        "by un-merging, not by re-ingesting.",
    )
    resolution_confidence: Score | None = None

    # -- Temporal ----------------------------------------------------------
    first_seen: UtcDatetime | None = None
    last_seen: UtcDatetime | None = None

    # -- Open extension ----------------------------------------------------
    properties: dict[str, object] = Field(
        default_factory=dict,
        description="Type-specific properties: ticker for a Company, version for "
        "a Product, ISO code for a Region. Kept open because the seven node "
        "labels have little in common beyond identity.",
    )

    @model_validator(mode="after")
    def _check_temporal_order(self) -> Self:
        if (
            self.first_seen is not None
            and self.last_seen is not None
            and self.last_seen < self.first_seen
        ):
            raise ValueError("last_seen precedes first_seen")
        return self

    def all_surfaces(self) -> set[str]:
        """Every string that should match this entity, for blocking and expansion."""
        return {self.canonical_name, *self.aliases}
