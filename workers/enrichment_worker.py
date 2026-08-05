"""Runs the Signal Engine enrichment pipeline.

This module owns the **canonical eight-stage assembly**. That matters more than
it sounds: before it existed, the only place the stage order was written down was
a helper inside `tests/unit/services/test_pipeline_end_to_end.py`. The pipeline
provably worked in-process and nothing shipped it, and the ordering assertions
guarded a list that no production code path used.

So `build_pipeline()` is the single definition, and the end-to-end test imports
it rather than rebuilding the list. A stage added, removed or reordered is now
caught by that test because the test and the worker are looking at the same
object -- which is the whole point of having the assertion.

Design Doc §6 fixes the order:

    Clean -> Normalize -> Language -> Entities -> Sentiment -> Embedding -> Store

with Scoring folded in as stage 6b, immediately before Store so that
`extraction_quality` sees every enrichment outcome. `SignalPipeline` validates
the order at construction, so a mis-assembly here fails at worker startup rather
than as a run of confusing per-record errors.

Every dependency is injected. The worker resolves real providers from settings at
its composition root (`build_default_pipeline()`); `build_pipeline()` itself
takes ports, so the test suite assembles the identical stage list with fakes and
no network.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from services.signal_engine.cleaning import CleaningStage
from services.signal_engine.embeddings import EmbeddingStage
from services.signal_engine.enrichment import ScoringStage
from services.signal_engine.entities import EntityExtractionStage
from services.signal_engine.language import LanguageStage
from services.signal_engine.normalize import NormalizeStage
from services.signal_engine.pipeline import SignalPipeline, Stage
from services.signal_engine.sentiment import SentimentStage
from services.signal_engine.store import StoreStage

if TYPE_CHECKING:  # pragma: no cover -- import-cycle avoidance only
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from services.llm.embeddings import EmbeddingProvider
    from services.llm.provider import LLMProvider
    from services.signal_engine.enrichment import CohortBaseline
    from services.signal_engine.store import SignalPublisher

__all__ = ["CANONICAL_STAGE_ORDER", "PIPELINE_VERSION", "build_pipeline"]

PIPELINE_VERSION = "1.0.0"
"""Stamped onto every Signal this assembly produces.

Bump the minor component when a stage's behaviour changes in a way that would
produce a different result for the same input -- new weights, a new prompt
version, a different chunker. `models.lineage.pipeline_version_ordinal` turns
this into the integer the upsert guard in `services/signal_engine/store.py`
compares, so a slow reprocess running old code cannot overwrite newer output.
"""

CANONICAL_STAGE_ORDER: Sequence[str] = (
    "clean",
    "normalize",
    "language",
    "entities",
    "sentiment",
    "embedding",
    "scoring",
    "store",
)
"""The order, as names, for assertions that should not import eight classes."""


def build_pipeline(
    *,
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    baseline: CohortBaseline,
    session_factory: async_sessionmaker[AsyncSession],
    publisher: SignalPublisher,
    url_resolver: Callable[[str], str | None] | None = None,
    language_detector: object | None = None,
    collection: str | None = None,
    vector_sink: object | None = None,
    baseline_weights: dict[str, float] | None = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> SignalPipeline:
    """Assemble the eight stages in the order Design Doc §6 fixes.

    Ports in, pipeline out. Nothing here reads settings or constructs a client,
    so the unit suite builds the *same* stage list with fakes -- if this function
    and the tested one ever diverge, the divergence is the bug, and having one
    function is how that is prevented.

    `keywords.py` is deliberately absent from the list. Topic and keyword
    extraction folds into `EntityExtractionStage` rather than running as a ninth
    stage, because a separate per-Signal LLM call for keywords would roughly
    double ingestion cost for a result the entities call already has the text in
    context to produce.
    """
    stages: list[Stage] = [
        CleaningStage(),
        NormalizeStage(url_resolver=url_resolver),
        # Every optional port below is threaded through rather than defaulted
        # inside the stage, because the test suite needs to substitute each one
        # and a port it cannot reach is a port that forces a second, drifting
        # copy of this list.
        LanguageStage(language_detector) if language_detector else LanguageStage(),
        EntityExtractionStage(llm),
        SentimentStage(llm),
        EmbeddingStage(embeddings, collection=collection, sink=vector_sink),
        # Scoring sits immediately before Store so `extraction_quality` reflects
        # every enrichment outcome. Run any earlier and a stage that failed after
        # it would not be counted, inflating the confidence of exactly the
        # Signals whose enrichment degraded.
        ScoringStage(baseline=baseline),
        StoreStage(session_factory, publisher),
    ]
    # `SignalPipeline.__init__` re-validates the order and rejects duplicates or
    # a post-Normalize stage with no Normalize. Constructing it here means a
    # mis-assembly fails at worker startup, not per record.
    return SignalPipeline(stages, pipeline_version=pipeline_version)
