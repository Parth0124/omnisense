"""Connector tools: what can be collected, start a collection, check on it.

The Collector is the only agent that causes the system to reach the outside
world, so this is the narrowest tool surface in the layer and the one with the
most rules attached.

**Never `connectors/` directly.** These tools call `services/connector_service.py`
and nothing else. Credentials, rate limits, cursors, retries and the DLQ are the
service layer's job (`docs/architecture.md` §6.2, `docs/connector-spec.md` §2.6);
an agent that constructed a connector would hold decrypted credentials in a
context window that also contains attacker-authored text. `agents/` importing
`connectors/` is the single import that would make that possible, so it does not
happen -- the port below is what the service satisfies.

**A platform, never a URL.** `docs/security-and-privacy.md` §8.2: connector tools
accept a slug from a fixed enum, never a URL and never a credential, and there is
no generic `http_get` tool anywhere in this package. If an agent could name a
host, an injected passage could name an attacker's host and exfiltrate the
context in a query string. `Platform` is a closed enum, and the platform still
has to resolve to a connector this tenant has enabled -- so the reachable set is
the intersection of two things the model does not control.

**Collection is asynchronous.** `fetch` enqueues a sync and returns a run id;
`sync_status` polls it. `docs/agent-system.md` §5.2 requires this: a full ingest
is minutes of connector fetch plus enrichment, and a node that blocked on it
would hold a checkpoint boundary open for the whole time and lose all of it to
one restart.

`services/connector_service.py` today exposes `ConnectorRuntime`, which drives a
*constructed* connector -- useful to a worker, unusable from here, because
constructing one means importing `connectors/`. The slug-level facade this module
needs does not exist yet, so `load_connector_gateway()` raises
`NotImplementedError` naming the three methods it must grow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Protocol, runtime_checkable

from pydantic import Field, field_validator

from agents.tools.registry import BoundedResult, ProvenanceStr, ToolSpec
from backend.core.exceptions import ConfigurationError, ValidationError
from backend.core.logging import get_logger
from models.base import StrictModel
from models.enums import ConnectorErrorClass, Platform, SourceCategory

__all__ = [
    "MAX_CONNECTORS",
    "MAX_QUERY_TERMS",
    "AvailableConnectors",
    "ConnectorDescriptor",
    "ConnectorGateway",
    "ConnectorInfo",
    "FetchInput",
    "FetchResult",
    "ListAvailableInput",
    "SyncHandle",
    "SyncStatusInput",
    "SyncStatusRecord",
    "SyncStatusResult",
    "ConnectorToolset",
    "load_connector_gateway",
]

logger = get_logger(__name__)

MAX_CONNECTORS: Final = 50
MAX_QUERY_TERMS: Final = 10
MAX_TERM_CHARS: Final = 100
MAX_ITEMS_PER_FETCH: Final = 1_000
"""Ceiling on one collection request.

Not a performance number. `docs/agent-system.md` §5.2 names unbounded fan-out
against a rate-limited API as the Collector's characteristic failure, and the
per-connector concurrency caps bound *parallelism* while this bounds *depth*. A
plan step that asks for 100,000 items is a plan defect, and discovering it after
the quota is spent is too late.
"""


# --------------------------------------------------------------------------- #
# Shapes the service layer will return
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ConnectorDescriptor:
    """One connector as the service layer describes it to an agent.

    Deliberately carries no credential, no endpoint and no configuration --
    everything here is safe in a context window that also holds hostile text.
    """

    slug: str
    platform: Platform
    category: SourceCategory
    enabled: bool = False
    requires_tos_review: bool = False
    """Mirrors `connectors/base.py::BaseConnector.requires_tos_review`.

    Carried all the way to the agent so a refusal to collect from a source with
    no viable official API is visible in the run rather than looking like an
    outage. `docs/security-and-privacy.md` §7 makes that a hard gate.
    """

    needs_reauth: bool = False
    supports_incremental: bool = True
    supports_backfill: bool = False
    last_synced_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SyncHandle:
    """The receipt for an enqueued collection."""

    run_id: str
    slug: str
    accepted: bool = True
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SyncStatusRecord:
    """Where one enqueued collection has got to."""

    run_id: str
    slug: str
    state: str
    emitted: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_class: ConnectorErrorClass | None = None
    error_message: str = ""
    is_partial: bool = False
    cursor_watermark: datetime | None = None
    stats: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Port
# --------------------------------------------------------------------------- #


@runtime_checkable
class ConnectorGateway(Protocol):
    """The slug-level facade `services/connector_service.py` must expose.

    Three methods, all tenant-scoped by parameter rather than by construction:
    one gateway instance serves every concurrent investigation in a worker, and a
    tenant remembered on `self` is a cross-tenant fetch waiting for two runs to
    overlap.
    """

    async def list_connectors(self, *, tenant_id: str) -> Sequence[ConnectorDescriptor]: ...

    async def start_sync(
        self,
        *,
        tenant_id: str,
        slug: str,
        query_terms: Sequence[str] = (),
        since: datetime | None = None,
        until: datetime | None = None,
        max_items: int = 200,
        idempotency_key: str | None = None,
    ) -> SyncHandle: ...

    async def sync_status(self, *, tenant_id: str, run_id: str) -> SyncStatusRecord | None: ...


def load_connector_gateway(**kwargs: object) -> ConnectorGateway:
    """Construct the real gateway, or say exactly what is missing.

    `NotImplementedError` rather than a stub that reports zero connectors: an
    empty connector list is a *meaningful* answer -- this tenant has configured
    none -- so a stub would make "collection is not wired yet" indistinguishable
    from "there is nothing to collect from", and the Planner would then decide
    the corpus is sufficient and skip the Collector entirely. The run would look
    successful and would be answering from stale data.
    """
    import services.connector_service as connector_service

    gateway_cls = getattr(connector_service, "ConnectorGatewayService", None)
    if gateway_cls is None:
        raise NotImplementedError(
            "services/connector_service.py exposes ConnectorRuntime, which drives an "
            "already-constructed connector and so cannot be called from agents/ "
            "(constructing one means importing connectors/, which agents must never "
            "do). It needs a slug-level facade -- ConnectorGatewayService with "
            "list_connectors / start_sync / sync_status per "
            "agents.tools.connector_tools.ConnectorGateway -- before the connector "
            "tools can be bound."
        )
    gateway = gateway_cls(**kwargs)
    if not isinstance(gateway, ConnectorGateway):
        raise NotImplementedError(
            "services.connector_service.ConnectorGatewayService does not satisfy "
            "agents.tools.connector_tools.ConnectorGateway; all three methods are "
            "required before the connector tools can be registered."
        )
    return gateway


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


def _reject_urls(terms: list[str]) -> list[str]:
    """Refuse a query term that is a URL or carries control characters.

    The connector layer builds its own requests, so a URL here would not be
    fetched -- but a search term that *is* a URL is either a confused plan or an
    injected instruction testing whether this tool is a disguised `http_get`, and
    both deserve a loud refusal rather than a query for a nonsense string.
    """
    for term in terms:
        if "://" in term or any(character in term for character in "\r\n\x00"):
            raise ValueError(
                f"query term {term[:60]!r} looks like a URL or contains control "
                "characters; connector tools take search terms, never addresses."
            )
    return terms


class ListAvailableInput(StrictModel):
    include_disabled: bool = Field(
        default=False,
        description="Include connectors this tenant has not enabled, so a plan can "
        "say what it could not reach rather than silently omitting it.",
    )


class ConnectorInfo(StrictModel):
    slug: ProvenanceStr
    platform: Platform
    category: SourceCategory
    enabled: bool = False
    collectable: bool = False
    """Whether `fetch` will accept this connector right now.

    A single field rather than leaving the agent to combine `enabled`,
    `needs_reauth` and `requires_tos_review` correctly. It will get that
    conjunction wrong eventually, and the failure -- planning a collection that
    is then refused -- costs a whole plan step to discover.
    """

    unavailable_reason: str = ""
    supports_incremental: bool = True
    supports_backfill: bool = False
    last_synced_at: datetime | None = None


class AvailableConnectors(BoundedResult):
    ITEMS_FIELD = "connectors"

    connectors: list[ConnectorInfo] = Field(default_factory=list)


class FetchInput(StrictModel):
    """Arguments for `fetch`. A platform, terms and a window -- nothing else.

    No URL, no host, no credential, no header, no connector configuration. The
    schema is the enforcement: `additionalProperties: false` plus a closed enum
    means there is no field an injected instruction could use to redirect where
    this call goes.
    """

    platform: Platform
    query_terms: list[str] = Field(default_factory=list, max_length=MAX_QUERY_TERMS)
    since: datetime | None = None
    until: datetime | None = None
    max_items: int = Field(default=200, ge=1, le=MAX_ITEMS_PER_FETCH)

    @field_validator("query_terms")
    @classmethod
    def _validate_terms(cls, value: list[str]) -> list[str]:
        trimmed = [term.strip() for term in value if term.strip()]
        for term in trimmed:
            if len(term) > MAX_TERM_CHARS:
                raise ValueError(f"query term exceeds {MAX_TERM_CHARS} characters")
        return _reject_urls(trimmed)


class FetchResult(BoundedResult):
    """The receipt, not the data.

    Collection is asynchronous (§5.2): records land in Kafka and are enriched by
    the pipeline. Returning documents here would both block the node and bypass
    enrichment, dedup and indexing -- the agent would be reading raw text that
    nothing had cleaned, scored or de-duplicated.
    """

    run_id: str
    slug: ProvenanceStr
    accepted: bool
    platform: Platform
    detail: str = ""


class SyncStatusInput(StrictModel):
    run_id: str = Field(min_length=1, max_length=128)


class SyncStatusResult(BoundedResult):
    run_id: str
    slug: ProvenanceStr = ""
    state: str = "unknown"
    found: bool = True
    emitted: int = 0
    is_partial: bool = False
    error_class: ConnectorErrorClass | None = None
    error_message: ProvenanceStr = ""
    """A provider's error string is third-party text.

    Scrubbed and length-capped by its type: an API that echoes a query back
    inside its error message is echoing text the plan may itself have taken from
    a hostile passage, and that round trip is a real path into the context.
    """

    started_at: datetime | None = None
    finished_at: datetime | None = None


# --------------------------------------------------------------------------- #
# The toolset
# --------------------------------------------------------------------------- #


class ConnectorToolset:
    """Binds a `ConnectorGateway` to the three connector tools.

    `allowed_platforms` is an optional second gate on top of what the tenant has
    enabled. It exists for the case a deployment wants a run confined to a subset
    -- an investigation scoped to public sources should not be able to reach a
    customer's Slack even if the connector is configured -- and it is a
    constructor argument rather than a tool argument for the usual reason: an
    argument is something an injected instruction can set.
    """

    def __init__(
        self,
        *,
        gateway: ConnectorGateway,
        tenant_id: str,
        investigation_id: str | None = None,
        allowed_platforms: frozenset[Platform] | None = None,
    ) -> None:
        if not tenant_id:
            raise ConfigurationError("ConnectorToolset requires an explicit tenant_id")
        self._gateway = gateway
        self._tenant_id = tenant_id
        self._investigation_id = investigation_id
        self._allowed_platforms = allowed_platforms

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="list_available",
                description=(
                    "List the connectors this workspace can collect from, and why any "
                    "of them cannot be used right now."
                ),
                input_model=ListAvailableInput,
                output_model=AvailableConnectors,
                handler=self._list_available,
            ),
            ToolSpec(
                name="fetch",
                description=(
                    "Enqueue a collection from one platform for the given search terms "
                    "and time window. Returns a run id immediately; poll sync_status. "
                    "Accepts a platform, never a URL."
                ),
                input_model=FetchInput,
                output_model=FetchResult,
                handler=self._fetch,
            ),
            ToolSpec(
                name="sync_status",
                description="Report progress of a collection previously enqueued by fetch.",
                input_model=SyncStatusInput,
                output_model=SyncStatusResult,
                handler=self._sync_status,
            ),
        ]

    # ------------------------------------------------------------ handlers --

    async def _list_available(self, args: ListAvailableInput) -> AvailableConnectors:
        descriptors = await self._gateway.list_connectors(tenant_id=self._tenant_id)
        infos: list[ConnectorInfo] = []
        for descriptor in descriptors:
            collectable, reason = self._collectable(descriptor)
            if not descriptor.enabled and not args.include_disabled:
                continue
            infos.append(
                ConnectorInfo(
                    slug=descriptor.slug,
                    platform=descriptor.platform,
                    category=descriptor.category,
                    enabled=descriptor.enabled,
                    collectable=collectable,
                    unavailable_reason=reason,
                    supports_incremental=descriptor.supports_incremental,
                    supports_backfill=descriptor.supports_backfill,
                    last_synced_at=descriptor.last_synced_at,
                )
            )
        infos.sort(key=lambda info: info.slug)
        kept = infos[:MAX_CONNECTORS]
        return AvailableConnectors(
            connectors=kept,
            truncated=len(infos) > len(kept),
            dropped=max(0, len(infos) - len(kept)),
        )

    async def _fetch(self, args: FetchInput) -> FetchResult:
        descriptor = await self._resolve_platform(args.platform)
        collectable, reason = self._collectable(descriptor)
        if not collectable:
            # A refusal, not an empty result. A Collector that reports zero
            # emitted records for a connector that was never going to run makes
            # "the source said nothing" and "we never asked" identical in the
            # report, and only one of those is evidence of anything.
            raise ValidationError(
                f"connector {descriptor.slug!r} cannot be collected from: {reason}"
            )

        handle = await self._gateway.start_sync(
            tenant_id=self._tenant_id,
            slug=descriptor.slug,
            query_terms=tuple(args.query_terms),
            since=args.since,
            until=args.until,
            max_items=min(args.max_items, MAX_ITEMS_PER_FETCH),
            # Nodes must be idempotent across a resume (`docs/agent-system.md`
            # §7): a replayed Collector step whose sync already landed must not
            # start a second one. The key is derived from the run, the connector
            # and the request rather than generated, so the replay produces the
            # same key and the service can recognise it.
            idempotency_key=self._idempotency_key(descriptor.slug, args),
        )
        logger.info(
            "agent.tool.connector.sync_enqueued",
            tenant_id=self._tenant_id,
            slug=descriptor.slug,
            run_id=handle.run_id,
            accepted=handle.accepted,
        )
        return FetchResult(
            run_id=handle.run_id,
            slug=handle.slug,
            accepted=handle.accepted,
            platform=descriptor.platform,
            detail=handle.detail[:200],
        )

    async def _sync_status(self, args: SyncStatusInput) -> SyncStatusResult:
        record = await self._gateway.sync_status(
            tenant_id=self._tenant_id, run_id=args.run_id
        )
        if record is None:
            # Not an error: a run id from a previous investigation, or one whose
            # record has aged out, is a legitimate miss. `found=False` lets the
            # agent stop polling instead of retrying a lookup that will never
            # succeed.
            return SyncStatusResult(run_id=args.run_id, found=False, state="unknown")
        return SyncStatusResult(
            run_id=record.run_id,
            slug=record.slug,
            state=record.state,
            found=True,
            emitted=record.emitted,
            is_partial=record.is_partial,
            error_class=record.error_class,
            error_message=record.error_message,
            started_at=record.started_at,
            finished_at=record.finished_at,
        )

    # ----------------------------------------------------------- internals --

    async def _resolve_platform(self, platform: Platform) -> ConnectorDescriptor:
        """Map a platform onto a connector this tenant may actually use.

        The lookup is against the gateway's live listing rather than a static
        table, so a connector disabled or de-authorised since the plan was made
        is refused here rather than attempted.
        """
        if platform is Platform.UNKNOWN:
            # `Platform` is a `TolerantStrEnum`, so an unrecognised string
            # validates as UNKNOWN instead of failing. Left unhandled, that turns
            # "fetch from evil.example" into a silently accepted argument.
            raise ValidationError("unknown platform; fetch takes a known platform slug")
        if self._allowed_platforms is not None and platform not in self._allowed_platforms:
            raise ValidationError(
                f"platform {platform} is outside this investigation's collection scope"
            )
        descriptors = await self._gateway.list_connectors(tenant_id=self._tenant_id)
        for descriptor in descriptors:
            if descriptor.platform is platform:
                return descriptor
        raise ValidationError(f"no connector configured for platform {platform}")

    def _collectable(self, descriptor: ConnectorDescriptor) -> tuple[bool, str]:
        """Whether a fetch would be accepted, and the reason when it would not."""
        if descriptor.requires_tos_review:
            return False, "source requires a documented legal review before use"
        if not descriptor.enabled:
            return False, "connector is not enabled for this workspace"
        if descriptor.needs_reauth:
            return False, "credentials need re-authorisation"
        if (
            self._allowed_platforms is not None
            and descriptor.platform not in self._allowed_platforms
        ):
            return False, "platform is outside this investigation's collection scope"
        return True, ""

    def _idempotency_key(self, slug: str, args: FetchInput) -> str:
        """Stable across a replay of the same node with the same request."""
        parts = [
            self._investigation_id or "no-investigation",
            slug,
            "|".join(sorted(args.query_terms)),
            args.since.isoformat() if args.since else "-",
            args.until.isoformat() if args.until else "-",
            str(args.max_items),
        ]
        return ":".join(parts)
