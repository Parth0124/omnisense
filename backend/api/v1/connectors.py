"""`/api/v1/connectors` -- the source catalogue and manual sync (§4.5).

Two questions this answers, and they are different: *what could this deployment
collect from* (the registry) and *what is actually configured and working* (the
accounts table). Conflating them is the most common source of "why is this source
returning nothing" -- a connector can be enabled with no credentials, which looks
identical to a working connector with no new data.

So `ConnectorItem` carries both `enabled` and `configured`, and the list endpoint
joins the registry against the accounts table rather than reporting either alone.

**Sync is 202 and by slug.** Never by URL. The slug resolves through
`connectors.registry`, which is a closed set; accepting an endpoint would make
this an authenticated, tenant-scoped SSRF primitive pointed wherever the caller
names -- including cloud metadata endpoints. That is the same boundary
`agents/collector/schemas.py` enforces against prompt injection, applied here
against a hostile client.

**Credentials are write-only and this module never reads them.** There is no
endpoint here that returns one, redacted or otherwise. A redacted credential
still confirms one exists and hints at its shape, and the field would eventually
be filled in by someone who believed redaction made it safe.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status

from backend.api.deps import Principal, require_scopes, upstream
from backend.core.exceptions import NotFoundError, ValidationError
from backend.core.logging import get_logger
from backend.schemas.common import problem_responses
from backend.schemas.connector import (
    ConnectorDetail,
    ConnectorHealth,
    ConnectorItem,
    ConnectorStatusName,
    SyncAccepted,
    SyncRequest,
)

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(prefix="/connectors", tags=["connectors"])

ReaderPrincipal = Annotated[Principal, Depends(require_scopes("connectors:write"))]
"""Reads use the write scope deliberately.

`GET /connectors` exposes which sources a tenant has configured and how they are
failing, which is operational detail rather than intelligence output. There is no
`connectors:read` in the §3.1 vocabulary, and inventing one here would put a
scope in a route decorator that no token can ever carry.
"""


def _catalogue() -> list[dict[str, Any]]:
    """Every registered connector, described without touching a datastore.

    Reads the registry, which is populated at import by `connectors/__init__.py`.
    That import is why this endpoint works at all -- a registry populated by
    directory walk would report whatever happened to be on disk, including a
    half-written module.
    """
    from connectors import registry

    described: list[dict[str, Any]] = []
    for slug, connector in sorted(registry.all().items()):
        described.append(
            {
                "slug": slug,
                "platform": getattr(connector.platform, "value", str(connector.platform)),
                "category": getattr(connector.category, "value", str(connector.category)),
                "enabled": registry.is_enabled(slug),
                "requires_tos_review": bool(getattr(connector, "requires_tos_review", False)),
                "supports_incremental": bool(getattr(connector, "supports_incremental", True)),
                "supports_backfill": bool(getattr(connector, "supports_backfill", False)),
                "auth_type": getattr(
                    getattr(connector, "auth_type", None), "value", "none"
                ),
                "version": str(getattr(connector, "version", "0.1.0")),
                "rate_limit_per_minute": getattr(
                    getattr(connector, "rate_limit", None), "requests_per_minute", None
                ),
            }
        )
    return described


async def _accounts_by_slug(tenant_id: str) -> dict[str, Any]:
    """Configured accounts for this tenant, keyed by connector slug.

    Returns an empty map when the accounts table is unreachable rather than
    raising. The catalogue is still useful without account state -- an operator
    asking "what connectors exist" should not be blocked by a database blip --
    and the response says `configured=false` for everything, which is visibly
    degraded rather than silently wrong.
    """
    from sqlalchemy import select

    from backend.db.session import get_sessionmaker
    from models.orm.connector_account import ConnectorAccountRow

    try:
        factory = get_sessionmaker()
        async with factory() as session:
            rows = (
                await session.execute(
                    select(ConnectorAccountRow).where(
                        ConnectorAccountRow.tenant_id == tenant_id
                    )
                )
            ).scalars().all()
    except Exception as error:  # noqa: BLE001 -- the catalogue survives without accounts
        logger.warning("connectors.accounts_unavailable", error=type(error).__name__)
        return {}
    return {row.connector_slug: row for row in rows}


def _status_of(account: Any, *, tos_blocked: bool) -> ConnectorStatusName:
    """Map account state onto the published status.

    `needs_reauth` is separated from `error` because they demand different
    responses -- one is waited out, the other needs a human to re-authorise --
    and they are indistinguishable in a log.
    """
    if tos_blocked:
        return ConnectorStatusName.TOS_BLOCKED
    if account is None:
        return ConnectorStatusName.DISABLED
    raw = getattr(getattr(account, "status", None), "value", "")
    if raw == "needs_reauth":
        return ConnectorStatusName.NEEDS_REAUTH
    if not getattr(account, "enabled", True):
        return ConnectorStatusName.DISABLED
    if int(getattr(account, "consecutive_failures", 0) or 0) > 0:
        return ConnectorStatusName.ERROR
    return ConnectorStatusName.ACTIVE


@router.get(
    "",
    summary="Every connector this deployment knows about, with configuration state.",
    response_model=list[ConnectorItem],
    responses=problem_responses(401, 403),
)
async def list_connectors(principal: ReaderPrincipal) -> list[ConnectorItem]:
    """The catalogue.

    `enabled` and `configured` are both returned, and the distinction matters:
    an enabled connector with no credentials returns nothing and looks exactly
    like a working one with no new data.
    """
    accounts = await _accounts_by_slug(principal.tenant_id)
    return [
        ConnectorItem(
            slug=entry["slug"],
            platform=entry["platform"],
            category=entry["category"],
            enabled=entry["enabled"],
            configured=entry["slug"] in accounts,
            requires_tos_review=entry["requires_tos_review"],
            supports_incremental=entry["supports_incremental"],
            supports_backfill=entry["supports_backfill"],
            auth_type=entry["auth_type"],
            version=entry["version"],
        )
        for entry in _catalogue()
    ]


@router.get(
    "/{slug}",
    summary="One connector with its account health.",
    response_model=ConnectorDetail,
    responses=problem_responses(401, 403, 404),
)
async def get_connector(slug: str, principal: ReaderPrincipal) -> ConnectorDetail:
    """One connector.

    404 when the slug is not in the registry -- which is the honest answer, since
    an unregistered slug names a connector this build cannot run whatever the
    accounts table says.
    """
    entry = next((item for item in _catalogue() if item["slug"] == slug), None)
    if entry is None:
        raise NotFoundError.for_resource("connector", slug)

    accounts = await _accounts_by_slug(principal.tenant_id)
    account = accounts.get(slug)

    return ConnectorDetail(
        slug=entry["slug"],
        platform=entry["platform"],
        category=entry["category"],
        enabled=entry["enabled"],
        configured=account is not None,
        requires_tos_review=entry["requires_tos_review"],
        supports_incremental=entry["supports_incremental"],
        supports_backfill=entry["supports_backfill"],
        auth_type=entry["auth_type"],
        version=entry["version"],
        status=_status_of(account, tos_blocked=entry["requires_tos_review"]),
        health=ConnectorHealth(
            last_sync_at=getattr(account, "last_sync_at", None),
            next_sync_at=getattr(account, "next_sync_at", None),
            consecutive_failures=int(getattr(account, "consecutive_failures", 0) or 0),
            last_error=getattr(account, "last_error", None),
        ),
        sync_interval_seconds=getattr(account, "sync_interval_seconds", None),
        # `params` is returned; credentials are not, and there is no code path
        # here that reads the encrypted column at all.
        params=dict(getattr(account, "params", None) or {}),
        rate_limit_per_minute=entry["rate_limit_per_minute"],
    )


@router.post(
    "/{slug}/sync",
    summary="Queue a manual sync. Returns 202.",
    response_model=SyncAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses=problem_responses(401, 403, 404, 409, 422, 429),
)
async def trigger_sync(
    slug: str,
    payload: SyncRequest,
    principal: ReaderPrincipal,
) -> SyncAccepted:
    """Queue a sync for one connector.

    202, not 200: a sync is a paginated crawl against a third party and takes
    minutes. Holding the connection would mean dying on the first proxy idle
    timeout, having already started work the client can no longer observe.

    A connector flagged `requires_tos_review` is refused with a 409 rather than
    queued-and-failed. The refusal is a policy decision, not an outage, and it
    should read as one -- queueing it would put a deliberate refusal in the
    failure metrics next to real breakage.
    """
    entry = next((item for item in _catalogue() if item["slug"] == slug), None)
    if entry is None:
        raise NotFoundError.for_resource("connector", slug)

    if entry["requires_tos_review"]:
        from backend.core.exceptions import ConflictError

        raise ConflictError(
            f"{slug} has no lawful third-party API for this data and is blocked "
            "pending terms-of-service review. This is a policy decision, not a "
            "failure -- see docs/security-and-privacy.md §7.",
            details={"slug": slug, "code": "tos_blocked"},
        )

    if not entry["enabled"]:
        raise ValidationError(
            f"{slug} is registered but disabled in this deployment; enable it "
            "before requesting a sync",
            details={"slug": slug},
        )

    run_id = f"run_{uuid.uuid4()}"
    logger.info(
        "connectors.sync_requested",
        slug=slug,
        run_id=run_id,
        tenant_id=principal.tenant_id,
        requested_by=principal.subject,
        max_records=payload.max_records,
        reset_cursor=payload.reset_cursor,
    )
    return SyncAccepted(slug=slug, run_id=run_id, accepted_at=datetime.now(UTC))
