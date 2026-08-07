"""The ingestion process: fetch side and enrich side of one pipeline.

Two halves live here because they are two ends of one flow and neither is a whole
process on its own.

**The fetch side** (`run_account_sync`) drives `services/connector_service.py`:
it turns a `connector_accounts` row into a running connector -- credentials
decrypted, rate-limit buckets bound, cursor loaded -- and lets `ConnectorRuntime`
archive each payload to R2, publish a `RawRecordEvent`, and commit the cursor
*after* the broker acknowledges. `workers/scheduler.py` decides *when* to call
this; nothing here has an opinion about time.

**The enrich side** (`IngestionWorker`) consumes `omnisense.records.raw` and
drives the enrichment pipeline that `workers/enrichment_worker.py` assembles. The
architecture diagram (`docs/architecture.md` §4) shows exactly that: the raw
topic feeds the ingestion worker, which feeds the enrichment pipeline, with no
topic between them. A second hop would buy nothing -- enrichment is CPU and LLM
bound, not IO bound on a broker -- and would cost another at-least-once boundary
to make idempotent.

Why the payload is re-read from R2 rather than carried on the message
---------------------------------------------------------------------
`docs/data-stores.md` §5.1 publishes the *address* of the raw payload, never the
bytes. So this worker's first act is a `GET` against R2 for the exact bytes the
connector fetched, verified against the digest embedded in the key. That is the
difference between reprocessing what was fetched and re-fetching what the
provider serves today -- posts get deleted, edited and rate-limited, so a
re-fetch is lossy in a way that only shows up as Signals that quietly disagree
with their own citations.

A record whose `raw_object_key` is `None` is the documented degraded case:
`docs/architecture.md` §7.3 lets ingestion continue when R2 is unavailable, so
the event carries provenance without a payload. There is nothing to enrich, and
inventing an empty body would write a Signal that claims to be an article with no
text. It goes to the DLQ, where a replay after the R2 outage can produce it.

Idempotency
-----------
Redelivery is guaranteed, not hypothetical. The whole path converges because
`Signal.id` is derived from `(platform, native_id)` (`docs/signal-model.md` §4.1)
and stage 7 is an `ON CONFLICT (id) DO UPDATE` guarded on the pipeline-version
ordinal. Reprocessing the same raw record therefore rewrites one row rather than
inserting a second, and re-publishes one `SignalEnrichedEvent` whose consumers
are themselves keyed by derived ids. Nothing in this module needs a dedup table.

Layer note: `workers/` (L4).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Protocol

from backend.core.config import Settings, get_settings
from backend.core.exceptions import ConfigurationError
from backend.core.logging import get_logger
from connectors.protocol import SyncMode, SyncResult
from models.orm.mixins import DEFAULT_TENANT
from services.events.consumer import ConsumedMessage
from services.events.schemas import RawRecordEvent
from services.events.topics import TopicRole
from services.signal_engine.pipeline import EnrichmentContext, PipelineResult, SignalPipeline
from workers.runtime.base_worker import ConsumerWorker, run_worker
from workers.runtime.health import DependencyProbe

if TYPE_CHECKING:  # pragma: no cover -- typing only, keeps import cost off the hot path
    from models.orm.connector_account import ConnectorAccountRow
    from services.connector_service import CursorStore

__all__ = [
    "IngestionFailedError",
    "IngestionWorker",
    "RawPayloadLoader",
    "build_default_pipeline",
    "build_worker",
    "run_account_sync",
]

logger = get_logger(__name__)

WORKER_NAME: Final = "ingestion"
"""Consumer-group suffix and metric label. Stable: changing it re-joins the
group under a new name, which resets every committed offset to
`auto_offset_reset` and replays or skips the whole retention window."""


class RawPayloadLoader(Protocol):
    """How the archived bytes are fetched back. Injectable so tests need no R2."""

    async def __call__(self, key: str) -> bytes: ...


async def _load_from_r2(key: str) -> bytes:
    """Production `RawPayloadLoader`: read and verify against the key's digest.

    Function-local import: `services/storage/object_store.py` reaches a boto3
    client, and importing it at module scope would mean that merely importing
    this worker -- which the test suite does at collection time -- constructs
    one.
    """
    from services.storage.object_store import get_raw_payload

    return await get_raw_payload(key)


class IngestionWorker(ConsumerWorker):
    """Consumes `omnisense.records.raw` and runs the enrichment pipeline.

    The pipeline is injected rather than built here, because
    `workers/enrichment_worker.py` owns the canonical stage order and a second
    assembly is a second thing to keep in step. `build_worker()` below is the
    composition root that resolves real providers from settings.
    """

    def __init__(
        self,
        pipeline: SignalPipeline,
        *,
        payload_loader: RawPayloadLoader | None = None,
        settings: Settings | None = None,
        name: str = WORKER_NAME,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            topics=[TopicRole.RAW_RECORDS],
            settings=settings,
            **kwargs,
        )
        self._pipeline = pipeline
        self._load_payload = payload_loader or _load_from_r2

    def readiness_probes(self) -> Mapping[str, DependencyProbe]:
        """PostgreSQL only.

        `docs/architecture.md` §7.3 makes PostgreSQL a hard failure on the
        ingestion path -- stage 7 is the commit point and there is no degraded
        mode for it. R2 is deliberately absent: an R2 read failure sends one
        record to the DLQ, which is a per-message outcome rather than a reason to
        take the whole replica out of rotation.
        """
        from backend.db.session import check_postgres

        return {"postgres": check_postgres}

    async def handle(self, message: ConsumedMessage) -> None:
        """Enrich one raw record. Raises to send the record to the DLQ.

        Raising is the whole error contract. `services/events/consumer.py`
        retries a bounded number of times and then parks the message with its
        original bytes intact, which is what makes a replay after a fix possible
        -- and `docs/signal-model.md` §5.2 requires exactly that for a fatal
        stage, "with the exception and the R2 key".
        """
        event = message.envelope.payload_as(RawRecordEvent)

        if not event.raw_object_key:
            # See the module docstring. Provenance without a payload: there is
            # nothing to enrich and no honest way to invent one.
            raise ValueError(
                f"raw record {event.platform.value}:{event.native_id} carries no "
                "raw_object_key, so the archived payload cannot be read. The R2 "
                "PUT was deferred (docs/architecture.md §7.3); replay this record "
                "once the object exists."
            )

        raw_bytes = await self._load_payload(event.raw_object_key)
        result = await self._pipeline.run(self._context_for(event, raw_bytes))
        self._report(event, result)

    def _context_for(self, event: RawRecordEvent, raw_bytes: bytes) -> EnrichmentContext:
        """Build the pipeline's context from the event and the archived bytes.

        `payload` is decoded here rather than inside a stage because the decode
        can fail on bytes that are perfectly valid as an *archive* -- a provider
        that served HTML where JSON was declared, say. A failure here is a DLQ
        record naming the record; a failure three stages deeper is a
        `JSONDecodeError` with no provenance attached.

        A non-JSON content type is not an error: `CleaningStage` handles HTML and
        text bodies, and the payload map stays empty for them because there is no
        structured provider object to map from.
        """
        payload: dict[str, object] = {}
        if event.raw_content_type.startswith("application/json"):
            try:
                decoded = json.loads(raw_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as err:
                raise ValueError(
                    f"raw object {event.raw_object_key!r} is declared "
                    f"{event.raw_content_type!r} but does not parse as JSON; the "
                    "archived bytes are intact, so this is a mapping problem "
                    "rather than a corruption one"
                ) from err
            # A provider that returns a bare list or scalar at the top level gets
            # wrapped: `EnrichmentContext.payload` is a mapping, and that is what
            # every field spec in `connectors/normalize/mapper.py` indexes into.
            payload = decoded if isinstance(decoded, dict) else {"items": decoded}

        return EnrichmentContext(
            raw_bytes=raw_bytes,
            content_type=event.raw_content_type,
            payload=payload,
            record=event,
        )

    def _report(self, event: RawRecordEvent, result: PipelineResult) -> None:
        """Turn the pipeline's outcome into a log line, or into a raise.

        `PipelineResult` reports rather than raises so the caller can build a
        useful DLQ record (`services/signal_engine/pipeline.py`). This is that
        caller: a fatal stage becomes an exception carrying the stage name, and
        the runtime parks the original bytes.
        """
        if not result.succeeded:
            raise IngestionFailedError(
                f"enrichment failed at stage {result.fatal_stage.value if result.fatal_stage else 'unknown'} "
                f"for {event.platform.value}:{event.native_id} "
                f"(raw_object_key={event.raw_object_key})",
                stage=result.fatal_stage.value if result.fatal_stage else "unknown",
                error_class=result.error or "unknown",
            )

        signal = result.signal
        if signal is None:  # pragma: no cover -- `succeeded` already excluded this
            # Not an `assert`: assertions vanish under `python -O`, and this is
            # the one place that turns "the pipeline reported success" into a
            # committed Signal. A wrong answer here is a Signal nobody wrote.
            raise IngestionFailedError(
                f"pipeline reported success with no Signal for "
                f"{event.platform.value}:{event.native_id}",
                stage="unknown",
                error_class="InvariantViolation",
            )
        logger.info(
            "ingestion.enriched",
            signal_id=signal.id,
            platform=event.platform.value,
            connector=event.connector_slug,
            status=signal.lineage.status.value,
            failed_stages=[stage.value for stage in result.failed_stages],
            confidence=round(signal.confidence, 4),
        )


class IngestionFailedError(RuntimeError):
    """A raw record could not be turned into a Signal.

    Carries the stage and the failing exception's *class name* rather than a
    message, for the same reason `DlqEvent` does: a provider or driver message
    can echo the request that produced it, and requests carry fetched content.
    """

    def __init__(self, message: str, *, stage: str, error_class: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.error_class = error_class


# --------------------------------------------------------------------------- #
# The fetch side: driving connector_service for one account
# --------------------------------------------------------------------------- #


async def run_account_sync(
    account: ConnectorAccountRow,
    *,
    mode: SyncMode = SyncMode.INCREMENTAL,
    cursors: CursorStore | None = None,
    runtime: Any | None = None,
    settings: Settings | None = None,
    redis: Any | None = None,
    max_pages: int | None = None,
    max_records: int | None = None,
) -> SyncResult:
    """Fetch one connector account to completion and report what became durable.

    This is the composition root for the fetch side: registry lookup, credential
    decryption, shared rate-limit buckets, shared dedup store, cursor store, and
    the runtime that performs the R2-then-Kafka-then-cursor ordering. Every piece
    already exists in `services/connector_service.py`; assembling them is a
    process-level concern and therefore lives in `workers/`.

    **`ConnectorRuntime.run` does not raise for anticipated failures.** The four
    families in `connectors/exceptions.py` come back inside `SyncResult` with an
    `error_class`, because the scheduler's next decision -- retry sooner, back
    off, stop scheduling, prompt a human -- is a function of that class and
    nothing else. Only genuine defects propagate.

    Args:
        account: The row to sync. Must carry usable credentials for its auth
            type; a `needs_reauth` account is refused rather than attempted,
            because a failed auth attempt against most providers counts toward a
            lockout the operator then has to wait out.
        mode: `INCREMENTAL` resumes from the stored cursor; `FULL` ignores it.
        cursors: Cursor store. Defaults to the SQL-backed one, which is the only
            implementation whose commits survive a restart.
        runtime: A pre-built `ConnectorRuntime`, for tests and for a caller that
            wants to share one circuit breaker across many accounts.

    Raises:
        ConfigurationError: The account names a connector this build does not
            have, or its credentials cannot be decrypted.
    """
    # Imported inside the function: `services/connector_service.py` pulls in the
    # Redis client factory and the R2 archiver at module scope, and the scheduler
    # imports this module merely to reach `run_account_sync`.
    from connectors import registry
    from services.connector_service import (
        ConnectorRuntime,
        SqlAccountWriter,
        SqlCursorStore,
        build_sync_context,
        decrypt_credentials,
    )

    resolved = settings or get_settings()

    try:
        connector_cls = registry.get(account.connector_slug)
    except KeyError as err:
        raise ConfigurationError(
            f"connector account {account.id!r} names slug "
            f"{account.connector_slug!r}, which is not registered in this build. "
            "The account outlived the connector, or the connector module is not "
            "imported by connectors/__init__.py.",
            details={"account_id": account.id, "slug": account.connector_slug},
        ) from err

    credentials = decrypt_credentials(account, settings=resolved)
    context = build_sync_context(
        connector_cls,
        account_id=account.id,
        params=account.params,
        mode=mode,
        redis=redis,
        settings=resolved,
        max_pages=max_pages,
        max_records=max_records,
    )
    connector = registry.create(account.connector_slug, context, credentials)

    driver = runtime or ConnectorRuntime(
        cursors or SqlCursorStore(),
        accounts=SqlAccountWriter(),
        tenant_id=account.tenant_id or DEFAULT_TENANT,
    )
    result = await driver.run(connector)

    logger.info(
        "ingestion.sync.finished",
        account_id=account.id,
        connector=account.connector_slug,
        run_id=context.run_id,
        mode=mode.value,
        emitted=result.emitted,
        error_class=result.error_class.value if result.error_class else None,
    )
    return result


# --------------------------------------------------------------------------- #
# Composition root
# --------------------------------------------------------------------------- #


def build_default_pipeline(
    *,
    settings: Settings | None = None,
    llm: Any | None = None,
    embeddings: Any | None = None,
    baseline: Any | None = None,
    vector_sink: Any | None = None,
) -> SignalPipeline:
    """Resolve real providers from settings and hand them to `build_pipeline()`.

    `workers/enrichment_worker.py` deliberately takes ports and constructs
    nothing, which is what lets the unit suite assemble the identical stage list
    with fakes. Resolution has to happen *somewhere*, and it belongs in the
    process that runs the pipeline rather than in the module that defines it --
    otherwise importing the stage order drags an Anthropic client and a database
    pool into every test that wants to assert on stage names.

    One dependency is knowingly degraded here. `CohortBaseline` has no
    PostgreSQL-backed implementation yet (`services/signal_engine/enrichment.py`
    says so plainly), so the in-memory one is used and every cohort answers
    "empty". Under the default `ColdStartPolicy.OMIT` that means the engagement
    axis is *omitted* from `confidence` rather than fabricated, which is the
    documented cold-start behaviour and the reason the warning below names the
    missing store instead of this being silent.
    """
    from backend.db.session import get_sessionmaker
    from services.llm.embeddings import OpenAICompatibleEmbeddingProvider
    from services.signal_engine.enrichment import InMemoryCohortBaseline
    from services.signal_engine.store import publish_enriched
    from workers.enrichment_worker import build_pipeline

    resolved = settings or get_settings()

    if baseline is None:
        logger.warning(
            "ingestion.cohort_baseline_missing",
            reason="no PostgreSQL trailing-window CohortBaseline implementation exists",
            consequence="engagement percentiles are omitted from confidence (ColdStartPolicy.OMIT)",
        )
        baseline = InMemoryCohortBaseline()

    return build_pipeline(
        llm=llm or _resolve_llm_provider(resolved),
        embeddings=embeddings or OpenAICompatibleEmbeddingProvider(
            settings=resolved.embedding
        ),
        baseline=baseline,
        session_factory=get_sessionmaker(),
        publisher=publish_enriched,
        collection=resolved.qdrant.collection,
        vector_sink=vector_sink,
    )


def _resolve_llm_provider(settings: Settings) -> Any:
    """Build the configured chat provider.

    Only Anthropic is implemented. The other seven members of `LLMProvider` are
    declared in `backend/core/config.py` because the AI layer is model-agnostic
    by design (Design Doc §15), but declaring an enum member is not implementing
    a client -- and a deployment that set `LLM_PROVIDER=openai` and silently got
    Anthropic would be billed against the wrong account and produce output
    attributed to the wrong model in `lineage.stages[]`.
    """
    from backend.core.config import LLMProvider as LLMProviderChoice
    from services.llm.anthropic_provider import AnthropicProvider

    choice = settings.llm.provider
    if choice is LLMProviderChoice.ANTHROPIC:
        return AnthropicProvider(settings=settings.llm)
    raise ConfigurationError(
        f"LLM_PROVIDER={choice.value!r} has no client in this build; only "
        "'anthropic' is implemented (services/llm/anthropic_provider.py). Add a "
        "provider module implementing services.llm.provider.LLMProvider, or set "
        "LLM_PROVIDER=anthropic.",
        details={"provider": choice.value},
    )


def build_worker(
    *,
    settings: Settings | None = None,
    pipeline: SignalPipeline | None = None,
    payload_loader: RawPayloadLoader | None = None,
    **kwargs: Any,
) -> IngestionWorker:
    """Resolve production dependencies and assemble the worker.

    Separate from `IngestionWorker.__init__` because construction and *resolution*
    are different concerns: the constructor takes ports and is therefore testable
    with fakes, while this function reads settings and opens clients and is
    therefore only runnable in a real deployment.
    """
    resolved = settings or get_settings()
    return IngestionWorker(
        pipeline or build_default_pipeline(settings=resolved),
        payload_loader=payload_loader,
        settings=resolved,
        **kwargs,
    )


def main() -> None:  # pragma: no cover -- process entrypoint
    """`python -m workers.ingestion_worker`, per `docs/deployment.md` §3."""
    run_worker(build_worker())


if __name__ == "__main__":  # pragma: no cover
    main()
