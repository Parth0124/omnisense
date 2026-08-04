"""Unit tests for `services/connector_service.py`: the ordering invariant, mostly.

One test here is worth more than the rest put together:
`test_publisher_failure_after_archive_leaves_the_cursor_row_untouched`. It injects
a publisher that raises *after* the R2 archive has succeeded and then reads the
`connector_cursors` row straight out of the database to prove it did not move.
That is the failure `docs/connector-spec.md` §4.1 rule 1 exists to prevent, and it
is invisible to every other kind of assertion: a runtime that committed the cursor
first would still archive the right bytes, still publish the right events on the
happy path, still return a plausible `SyncResult`, and would silently drop a page
of records on every broker blip in production. Nothing but "read the row after the
failure" catches it.

The cursor store under test is the real `SqlCursorStore` against the in-memory
SQLite database from `tests/conftest.py`, not a dict. A fake cursor store cannot
prove the property the row is the evidence for, and it would happily "not commit"
even in a version of the runtime that wrote to PostgreSQL first.

Everything else here is arranged around the same idea -- assert on the *sequence
of effects*, not on the return value:

- a shared `journal` list threaded through the archiver, the publisher and the
  cursor store, so "archive, publish, commit" is checked as an order rather than
  as three independent facts;
- the error taxonomy checked by its consequences (an account row flagged
  `needs_reauth`, a `next_sync_at` moved to the provider's reset instant, a record
  on the DLQ) rather than by the exception type the runtime happened to catch.

No network, no broker, no Redis, no bucket: R2, Kafka and the DLQ are three
injected callables, and the only real infrastructure is SQLite.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Self

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.core.config import Settings
from backend.core.exceptions import (
    ConfigurationError,
    DependencyUnavailableError,
    ExternalServiceError,
)
from connectors.base import BaseConnector
from connectors.dedup.store import RedisDedupStore
from connectors.exceptions import (
    AuthError,
    CircuitOpenError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.protocol import (
    Credentials,
    Cursor,
    FetchPage,
    RateLimitPolicy,
    RawRecord,
    SyncContext,
    SyncMode,
)
from connectors.ratelimit.backoff import BackoffPolicy
from connectors.ratelimit.limiter import TokenBucketLimiter
from models.enums import AuthType, Platform, SourceCategory
from models.lineage import Lineage
from models.orm.connector_account import (
    ConnectorAccountRow,
    ConnectorAccountStatus,
    ConnectorCursorRow,
)
from models.orm.mixins import DEFAULT_TENANT
from models.signal import Content, Signal, signal_id
from services.connector_service import (
    ConnectorRuntime,
    CursorKey,
    InMemoryCursorStore,
    SqlAccountWriter,
    SqlCursorStore,
    build_sync_context,
    credential_cipher,
    decrypt_credentials,
    rate_limit_buckets,
    sync_params_hash,
)
from services.events.schemas import DlqEvent, RawRecordEvent
from services.storage.object_store import Compression, StoredObject

pytestmark = pytest.mark.unit


ACCOUNT_ID = "acct_demo"
T0 = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
FERNET_KEY = Fernet.generate_key().decode("ascii")


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class RecordingArchiver:
    """Stands in for R2. Content-addressed, exactly like the real thing."""

    def __init__(self, journal: list[str], *, error: Exception | None = None) -> None:
        self._journal = journal
        self.error = error
        self.calls: list[tuple[str, datetime, bytes]] = []

    async def __call__(
        self, platform: str, fetched_at: datetime, raw_bytes: bytes
    ) -> StoredObject:
        self.calls.append((platform, fetched_at, raw_bytes))
        if self.error is not None:
            self._journal.append("archive:failed")
            raise self.error
        digest = hashlib.sha256(raw_bytes).hexdigest()
        self._journal.append("archive")
        return StoredObject(
            key=f"raw/{platform}/2026/07/28/{digest}.json",
            sha256=digest,
            size_bytes=len(raw_bytes),
            stored_bytes=len(raw_bytes),
            compression=Compression.ZSTD,
            already_present=False,
        )


class RecordingPublisher:
    """Stands in for the producer. Returning is the ack, as in production.

    `fail_after` makes it raise once it has acknowledged that many events, which
    is how the ordering test puts a failure *between* a successful archive and the
    cursor commit -- the exact window in which committing first loses records.
    """

    def __init__(self, journal: list[str], *, fail_after: int | None = None) -> None:
        self._journal = journal
        self.fail_after = fail_after
        self.events: list[RawRecordEvent] = []

    async def __call__(self, event: RawRecordEvent, *, tenant_id: str | None = None) -> None:
        if self.fail_after is not None and len(self.events) >= self.fail_after:
            self._journal.append("publish:failed")
            raise DependencyUnavailableError.for_store("Redpanda")
        self.events.append(event)
        self._journal.append("publish")


class RecordingDlq:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal
        self.events: list[DlqEvent] = []

    async def __call__(self, event: DlqEvent, *, tenant_id: str | None = None) -> None:
        self.events.append(event)
        self._journal.append("dlq")


class JournalCursorStore(InMemoryCursorStore):
    """`InMemoryCursorStore` that announces its commits into the shared journal."""

    def __init__(self, journal: list[str]) -> None:
        super().__init__()
        self._journal = journal

    async def commit(self, key: CursorKey, cursor: Cursor) -> bool:
        self._journal.append("commit")
        return await super().commit(key, cursor)


class RecordingAccounts:
    def __init__(self) -> None:
        self.reauth: list[tuple[str, str]] = []
        self.rescheduled: list[tuple[str, datetime]] = []

    async def mark_needs_reauth(self, account_id: str, *, reason: str) -> None:
        self.reauth.append((account_id, reason))

    async def reschedule(self, account_id: str, *, when: datetime) -> None:
        self.rescheduled.append((account_id, when))


class DemoConnector(BaseConnector):
    """A connector that yields exactly what a test scripts, per attempt.

    `errors` is keyed by *attempt* -- the number of times `fetch()` has been
    entered -- because the runtime's transient recovery restarts the whole
    generator from the last committed cursor, and a test of that recovery has to
    be able to say "fail the first drain, succeed the second".
    """

    slug = "demo"
    platform = Platform.RSS
    category = SourceCategory.NEWS
    auth_type = AuthType.NONE
    version = "0.2.0"
    rate_limit = RateLimitPolicy(requests_per_minute=60, burst=10)

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        self.pages: list[FetchPage] = []
        self.pages_by_attempt: dict[int, list[FetchPage]] = {}
        self.errors: dict[int, Exception] = {}
        self.attempts = 0
        self.authenticated = 0
        self.seen_cursors: list[Cursor] = []
        self.derive_native_id = False

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        self.authenticated += 1

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        attempt = self.attempts
        self.attempts += 1
        self.seen_cursors.append(cursor)
        for page in self.pages_by_attempt.get(attempt, self.pages):
            yield page
        error = self.errors.get(attempt)
        if error is not None:
            raise error

    async def normalize(self, record: RawRecord) -> Signal | None:
        native_id = (
            hashlib.sha256(record.native_id.encode()).hexdigest()
            if self.derive_native_id
            else record.native_id
        )
        return Signal.create(
            platform=self.platform,
            native_id=native_id,
            timestamp=record.payload.get("published", T0),
            content=Content(text=str(record.payload.get("text", "body"))),
            url=record.source_url,
            lineage=Lineage(
                pipeline_version="1.0.0",
                connector_slug=self.slug,
                connector_version=self.version,
                sync_run_id=self.ctx.run_id,
                fetched_at=record.fetched_at,
                native_id=native_id,
            ),
        )


# --------------------------------------------------------------------------- #
# Builders and fixtures
# --------------------------------------------------------------------------- #


def make_page(
    ids: Sequence[str],
    *,
    watermark: datetime | None,
    page_token: str | None = None,
    raw_bytes: bool = True,
) -> FetchPage:
    records = [
        RawRecord(
            native_id=native_id,
            payload={"id": native_id, "text": f"body of {native_id}", "published": T0},
            fetched_at=T0,
            raw_bytes=json.dumps({"id": native_id}).encode() if raw_bytes else None,
            source_url=f"https://example.com/{native_id}",
        )
        for native_id in ids
    ]
    return FetchPage(
        records=records, cursor=Cursor(watermark=watermark, page_token=page_token)
    )


def make_context(
    *, mode: SyncMode = SyncMode.INCREMENTAL, params: dict[str, Any] | None = None
) -> SyncContext:
    """A context with no limiter and no dedup: those have their own suites.

    `BaseConnector` treats both as optional and fails open, so leaving them unset
    keeps these tests about the runtime rather than about Redis.
    """
    return SyncContext(
        connector_slug=DemoConnector.slug,
        account_id=ACCOUNT_ID,
        run_id="run_test",
        mode=mode,
        params=params or {"feeds": ["a"]},
    )


def make_connector(**context_kwargs: Any) -> DemoConnector:
    ctx = make_context(**context_kwargs)
    return DemoConnector.from_config(ctx, Credentials(account_id=ACCOUNT_ID))


@pytest.fixture
def journal() -> list[str]:
    return []


@pytest.fixture
def sessions(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=orm_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


@pytest.fixture
async def account(sessions: async_sessionmaker[AsyncSession]) -> str:
    """A real `connector_accounts` row.

    Required, not decorative: `connector_cursors.account_id` is a foreign key and
    the suite runs SQLite with `PRAGMA foreign_keys=ON`, so a cursor for an
    account that does not exist would be rejected rather than silently orphaned.
    """
    async with sessions() as session:
        session.add(
            ConnectorAccountRow(
                id=ACCOUNT_ID,
                tenant_id=DEFAULT_TENANT,
                connector_slug=DemoConnector.slug,
                platform=Platform.RSS,
                display_name="demo account",
                auth_type=AuthType.NONE,
                params={"feeds": ["a"]},
                params_hash="stored-hash",
                status=ConnectorAccountStatus.ENABLED,
                enabled=True,
                sync_interval_seconds=3600,
                consecutive_failures=0,
                credential_key_version=1,
            )
        )
        await session.commit()
    return ACCOUNT_ID


async def read_cursor_row(
    sessions: async_sessionmaker[AsyncSession], key: CursorKey
) -> ConnectorCursorRow | None:
    async with sessions() as session:
        return await session.scalar(
            select(ConnectorCursorRow).where(ConnectorCursorRow.id == key.row_id)
        )


async def read_account_row(
    sessions: async_sessionmaker[AsyncSession],
) -> ConnectorAccountRow:
    async with sessions() as session:
        row = await session.scalar(
            select(ConnectorAccountRow).where(ConnectorAccountRow.id == ACCOUNT_ID)
        )
    assert row is not None
    return row


async def no_sleep(seconds: float) -> None:
    """Retry pacing without the wait. The delays themselves are `backoff.py`'s tests."""
    return None


# --------------------------------------------------------------------------- #
# 1. The ordering invariant
# --------------------------------------------------------------------------- #


class TestCommitAfterAck:
    """`docs/connector-spec.md` §4.1 rule 1, from both directions."""

    async def test_archive_then_publish_then_commit(self, journal: list[str]) -> None:
        archiver = RecordingArchiver(journal)
        publisher = RecordingPublisher(journal)
        cursors = JournalCursorStore(journal)
        connector = make_connector()
        connector.pages = [make_page(["a1", "a2"], watermark=T0)]

        result = await ConnectorRuntime(
            cursors, publisher=publisher, archiver=archiver, dlq_publisher=RecordingDlq(journal)
        ).run(connector)

        # Every payload is in the archive before any event references it, every
        # event is acknowledged before the cursor moves, and the commit is last.
        assert journal == ["archive", "archive", "publish", "publish", "commit"]
        assert result.emitted == 2
        assert result.succeeded

    async def test_publisher_failure_after_archive_leaves_the_cursor_row_untouched(
        self,
        sessions: async_sessionmaker[AsyncSession],
        account: str,
        journal: list[str],
    ) -> None:
        """The one that matters. Archive succeeds, publish dies, cursor must not move.

        Committing before the ack would leave a row pointing past a page that was
        never durably written, and the next run would resume beyond it. Nothing
        errors, nothing reaches a DLQ; the records are simply gone. So the
        assertion is made against the row itself, after the failure.
        """
        archiver = RecordingArchiver(journal)
        publisher = RecordingPublisher(journal, fail_after=0)
        cursors = SqlCursorStore(sessions)
        connector = make_connector()
        connector.pages = [make_page(["a1"], watermark=T0)]
        key = CursorKey.for_context(connector.slug, connector.ctx)

        with pytest.raises(DependencyUnavailableError):
            await ConnectorRuntime(cursors, publisher=publisher, archiver=archiver).run(connector)

        assert journal == ["archive", "publish:failed"]
        # No row at all: this account had never synced, and a cursor that does
        # not exist is what makes the next run replay the page from the start.
        assert await read_cursor_row(sessions, key) is None

    async def test_publisher_failure_does_not_rewind_an_existing_cursor(
        self,
        sessions: async_sessionmaker[AsyncSession],
        account: str,
        journal: list[str],
    ) -> None:
        """The same property when a cursor already exists: it must stay exactly put."""
        cursors = SqlCursorStore(sessions)
        connector = make_connector()
        key = CursorKey.for_context(connector.slug, connector.ctx)
        await cursors.commit(key, Cursor(watermark=T0, page_token="p0"))

        connector.pages = [make_page(["a1"], watermark=T0 + timedelta(minutes=5), page_token="p1")]
        with pytest.raises(DependencyUnavailableError):
            await ConnectorRuntime(
                cursors,
                publisher=RecordingPublisher(journal, fail_after=0),
                archiver=RecordingArchiver(journal),
            ).run(connector)

        row = await read_cursor_row(sessions, key)
        assert row is not None
        assert row.watermark is not None and row.watermark.replace(tzinfo=UTC) == T0
        assert row.cursor["page_token"] == "p0"

    async def test_earlier_pages_stay_committed_when_a_later_page_fails(
        self,
        sessions: async_sessionmaker[AsyncSession],
        account: str,
        journal: list[str],
    ) -> None:
        """Per-batch commits are the point of yielding: a crash costs one page.

        A runtime that accumulated and committed once at the end would lose an
        entire multi-hour backfill to a single failure on its last page.
        """
        cursors = SqlCursorStore(sessions)
        connector = make_connector()
        connector.pages = [
            make_page(["a1"], watermark=T0, page_token="p1"),
            make_page(["a2"], watermark=T0 + timedelta(minutes=10), page_token="p2"),
        ]
        key = CursorKey.for_context(connector.slug, connector.ctx)

        with pytest.raises(DependencyUnavailableError):
            await ConnectorRuntime(
                cursors,
                publisher=RecordingPublisher(journal, fail_after=1),
                archiver=RecordingArchiver(journal),
            ).run(connector)

        row = await read_cursor_row(sessions, key)
        assert row is not None
        assert row.cursor["page_token"] == "p1"
        assert row.watermark is not None and row.watermark.replace(tzinfo=UTC) == T0

    async def test_an_archive_failure_publishes_nothing_and_commits_nothing(
        self, journal: list[str]
    ) -> None:
        """Step 1 failing must not let steps 2 and 3 happen anyway."""
        cursors = JournalCursorStore(journal)
        connector = make_connector()
        connector.pages = [make_page(["a1"], watermark=T0)]

        with pytest.raises(RuntimeError):
            await ConnectorRuntime(
                cursors,
                publisher=RecordingPublisher(journal),
                archiver=RecordingArchiver(journal, error=RuntimeError("bucket on fire")),
            ).run(connector)

        assert journal == ["archive:failed"]
        assert cursors.commits == []

    async def test_a_page_with_no_surviving_records_still_advances_the_cursor(
        self, journal: list[str]
    ) -> None:
        """Otherwise a feed whose every item is a duplicate never makes progress."""
        cursors = JournalCursorStore(journal)
        connector = make_connector()
        connector.pages = [make_page([], watermark=T0 + timedelta(hours=1))]

        result = await ConnectorRuntime(
            cursors, publisher=RecordingPublisher(journal), archiver=RecordingArchiver(journal)
        ).run(connector)

        assert journal == ["commit"]
        assert result.cursor.watermark == T0 + timedelta(hours=1)


# --------------------------------------------------------------------------- #
# 2. Watermark monotonicity
# --------------------------------------------------------------------------- #


class TestWatermarkMonotonicity:
    """§4.1 rule 2. `connectors/base.py` clamps first; this is the backstop."""

    async def test_a_regressing_watermark_is_rejected_and_the_records_stay_durable(
        self, journal: list[str]
    ) -> None:
        """A connector paging newest-first without re-sorting, which is the case
        rule 2 names by hand.

        `_guard_watermark` cannot catch this one: it clamps each page against the
        watermark the *run started from*, and both pages here are above it. Only
        the runtime sees the previous page's committed watermark, which is why the
        check has to exist on this side of the boundary too.
        """
        cursors = JournalCursorStore(journal)
        publisher = RecordingPublisher(journal)
        connector = make_connector()
        key = CursorKey.for_context(connector.slug, connector.ctx)
        connector.pages = [
            make_page(["a1"], watermark=T0 + timedelta(hours=1), page_token="p1"),
            make_page(["a2"], watermark=T0 + timedelta(minutes=30), page_token="backwards"),
        ]

        result = await ConnectorRuntime(
            cursors, publisher=publisher, archiver=RecordingArchiver(journal)
        ).run(connector)

        # One commit, not two: the second page's cursor was refused.
        assert journal.count("commit") == 1
        assert cursors.cursors[key].page_token == "p1"
        # Both records were acked before either cursor was considered. Rejecting
        # the commit must not pretend the second record did not happen.
        assert len(publisher.events) == 2
        assert result.emitted == 2
        assert result.cursor.watermark == T0 + timedelta(hours=1)

    async def test_an_equal_watermark_is_accepted(self, journal: list[str]) -> None:
        """A page that found nothing newer still carries a fresh page token."""
        cursors = JournalCursorStore(journal)
        connector = make_connector()
        key = CursorKey.for_context(connector.slug, connector.ctx)
        cursors.cursors[key] = Cursor(watermark=T0, page_token="p0")
        connector.pages = [make_page(["a1"], watermark=T0, page_token="p1")]

        await ConnectorRuntime(
            cursors, publisher=RecordingPublisher(journal), archiver=RecordingArchiver(journal)
        ).run(connector)

        assert cursors.cursors[key].page_token == "p1"

    async def test_dropping_to_no_watermark_is_a_regression(self, journal: list[str]) -> None:
        """Regressing to `None` would trigger a full re-sync on the next run."""
        cursors = JournalCursorStore(journal)
        connector = make_connector()
        key = CursorKey.for_context(connector.slug, connector.ctx)
        connector.pages = [
            make_page(["a1"], watermark=T0, page_token="p1"),
            make_page(["a2"], watermark=None, page_token="p2"),
        ]

        await ConnectorRuntime(
            cursors, publisher=RecordingPublisher(journal), archiver=RecordingArchiver(journal)
        ).run(connector)

        assert journal.count("commit") == 1
        assert cursors.cursors[key].watermark == T0
        assert cursors.cursors[key].page_token == "p1"


# --------------------------------------------------------------------------- #
# 3. The error taxonomy
# --------------------------------------------------------------------------- #


class TestErrorTaxonomy:
    """`docs/connector-spec.md` §6. Each row asserted by its consequences."""

    async def test_quota_leaves_records_durable_and_the_cursor_advanced(
        self,
        sessions: async_sessionmaker[AsyncSession],
        account: str,
        journal: list[str],
    ) -> None:
        """A `QuotaError` is a partial success, not a failure.

        The distinction is expensive to get backwards: treating it as a failure
        would discard an hour of successful pagination because the 4,000th
        request hit a wall, and the backfill would never finish.
        """
        cursors = SqlCursorStore(sessions)
        accounts = RecordingAccounts()
        publisher = RecordingPublisher(journal)
        connector = make_connector()
        connector.pages = [
            make_page(["a1"], watermark=T0, page_token="p1"),
            make_page(["a2"], watermark=T0 + timedelta(minutes=30), page_token="p2"),
        ]
        reset_at = (T0 + timedelta(hours=1)).timestamp()
        connector.errors = {0: QuotaError("daily quota exhausted", reset_at=reset_at)}
        key = CursorKey.for_context(connector.slug, connector.ctx)

        result = await ConnectorRuntime(
            cursors,
            publisher=publisher,
            archiver=RecordingArchiver(journal),
            accounts=accounts,
        ).run(connector)

        # Durable: both records were acknowledged before the quota fired.
        assert [event.native_id for event in publisher.events] == ["a1", "a2"]
        # Advanced: the cursor row carries the last acked page, not the start.
        row = await read_cursor_row(sessions, key)
        assert row is not None
        assert row.cursor["page_token"] == "p2"
        assert row.watermark is not None
        assert row.watermark.replace(tzinfo=UTC) == T0 + timedelta(minutes=30)
        # Classified as a partial success and rescheduled at the provider's reset.
        assert result.error_class == "quota"
        assert result.is_partial
        assert result.emitted == 2
        assert accounts.rescheduled == [
            (ACCOUNT_ID, datetime.fromtimestamp(reset_at, tz=UTC))
        ]

    async def test_quota_reschedules_the_account_row(
        self, sessions: async_sessionmaker[AsyncSession], account: str, journal: list[str]
    ) -> None:
        connector = make_connector()
        connector.pages = [make_page(["a1"], watermark=T0)]
        resume = T0 + timedelta(minutes=45)
        connector.errors = {0: QuotaError("quota", reset_at=resume.timestamp())}

        await ConnectorRuntime(
            SqlCursorStore(sessions),
            publisher=RecordingPublisher(journal),
            archiver=RecordingArchiver(journal),
            accounts=SqlAccountWriter(sessions),
        ).run(connector)

        row = await read_account_row(sessions)
        assert row.next_sync_at is not None
        assert row.next_sync_at.replace(tzinfo=UTC) == resume
        # A quota halt is not a credential problem.
        assert row.status is ConnectorAccountStatus.ENABLED

    async def test_an_open_circuit_takes_the_quota_path(self, journal: list[str]) -> None:
        """`CircuitOpenError` is classified `QUOTA` and must be treated as one.

        Dispatching on `isinstance` instead of on `error_class` would give it the
        transient response and retry straight into a source already known to be
        failing.
        """
        accounts = RecordingAccounts()
        connector = make_connector()
        opens_until = (T0 + timedelta(minutes=10)).timestamp()
        connector.errors = {0: CircuitOpenError("circuit open", opens_until=opens_until)}

        result = await ConnectorRuntime(
            InMemoryCursorStore(),
            publisher=RecordingPublisher(journal),
            archiver=RecordingArchiver(journal),
            accounts=accounts,
        ).run(connector)

        assert result.error_class == "quota"
        assert accounts.rescheduled == [
            (ACCOUNT_ID, datetime.fromtimestamp(opens_until, tz=UTC))
        ]

    async def test_auth_flags_the_account_and_halts_with_no_partial_credit(
        self, sessions: async_sessionmaker[AsyncSession], account: str, journal: list[str]
    ) -> None:
        cursors = SqlCursorStore(sessions)
        connector = make_connector()
        connector.errors = {0: AuthError("refresh token revoked", account_id=ACCOUNT_ID)}
        key = CursorKey.for_context(connector.slug, connector.ctx)

        result = await ConnectorRuntime(
            cursors,
            publisher=RecordingPublisher(journal),
            archiver=RecordingArchiver(journal),
            accounts=SqlAccountWriter(sessions),
        ).run(connector)

        assert result.error_class == "auth"
        assert not result.succeeded
        assert not result.is_partial
        assert result.emitted == 0
        assert await read_cursor_row(sessions, key) is None

        row = await read_account_row(sessions)
        assert row.status is ConnectorAccountStatus.NEEDS_REAUTH
        assert row.last_error == "refresh token revoked"
        # `enabled` is the operator's intent and stays theirs: an expired token
        # must remain distinguishable from a deliberate pause.
        assert row.enabled is True

    async def test_auth_is_never_retried(self, journal: list[str]) -> None:
        """Looping on auth is how an integration earns an application-level ban."""
        connector = make_connector()
        connector.errors = {0: AuthError("401"), 1: AuthError("401")}

        await ConnectorRuntime(
            InMemoryCursorStore(),
            publisher=RecordingPublisher(journal),
            archiver=RecordingArchiver(journal),
            sleep=no_sleep,
        ).run(connector)

        assert connector.attempts == 1

    async def test_transient_is_retried_and_resumes_from_the_committed_cursor(
        self, journal: list[str]
    ) -> None:
        cursors = JournalCursorStore(journal)
        connector = make_connector()
        connector.pages_by_attempt = {
            0: [make_page(["a1"], watermark=T0, page_token="p1")],
            1: [make_page(["a2"], watermark=T0 + timedelta(minutes=5), page_token="p2")],
        }
        connector.errors = {0: TransientError("connection reset")}

        result = await ConnectorRuntime(
            cursors,
            publisher=RecordingPublisher(journal),
            archiver=RecordingArchiver(journal),
            backoff=BackoffPolicy(base_seconds=0.0, max_attempts=3),
            sleep=no_sleep,
        ).run(connector)

        assert connector.attempts == 2
        # The retry resumed from the page that was acked, not from the beginning.
        assert connector.seen_cursors[1].page_token == "p1"
        assert result.succeeded
        assert result.emitted == 2

    async def test_transient_escalates_once_the_attempts_are_spent(
        self, journal: list[str]
    ) -> None:
        connector = make_connector()
        connector.errors = {0: TransientError("5xx"), 1: TransientError("5xx")}

        result = await ConnectorRuntime(
            InMemoryCursorStore(),
            publisher=RecordingPublisher(journal),
            archiver=RecordingArchiver(journal),
            backoff=BackoffPolicy(base_seconds=0.0, max_attempts=2),
            sleep=no_sleep,
        ).run(connector)

        assert connector.attempts == 2
        assert result.error_class == "transient"
        assert not result.succeeded

    async def test_a_permanent_archive_failure_dlqs_the_record_and_continues(
        self, journal: list[str]
    ) -> None:
        """One poisoned record must not stop the page behind it.

        Aborting would leave the cursor parked in front of the bad record
        forever, so every well-formed item behind it would be permanently
        unreachable.
        """

        class SelectivelyFailingArchiver(RecordingArchiver):
            async def __call__(
                self, platform: str, fetched_at: datetime, raw_bytes: bytes
            ) -> StoredObject:
                if b"a1" in raw_bytes:
                    raise PermanentError("payload cannot be archived")
                return await super().__call__(platform, fetched_at, raw_bytes)

        dlq = RecordingDlq(journal)
        publisher = RecordingPublisher(journal)
        cursors = JournalCursorStore(journal)
        connector = make_connector()
        connector.pages = [make_page(["a1", "a2"], watermark=T0)]

        result = await ConnectorRuntime(
            cursors,
            publisher=publisher,
            archiver=SelectivelyFailingArchiver(journal),
            dlq_publisher=dlq,
        ).run(connector)

        assert len(dlq.events) == 1
        assert dlq.events[0].error_chain[0] == "PermanentError"
        # The DLQ record carries the original bytes so a fixed handler can be
        # replayed without re-hitting the provider.
        assert json.loads(dlq.events[0].body()) == {"id": "a1"}
        assert [event.native_id for event in publisher.events] == ["a2"]
        assert result.succeeded
        assert result.dlq == 1
        assert result.emitted == 1
        assert cursors.commits  # the run made progress despite the bad record

    async def test_a_defect_propagates_rather_than_becoming_a_sync_result(
        self, journal: list[str]
    ) -> None:
        """A `KeyError` in a mapper is our bug, not a source misbehaving."""
        connector = make_connector()
        connector.errors = {0: KeyError("title")}

        with pytest.raises(KeyError):
            await ConnectorRuntime(
                InMemoryCursorStore(),
                publisher=RecordingPublisher(journal),
                archiver=RecordingArchiver(journal),
            ).run(connector)


# --------------------------------------------------------------------------- #
# 4. What the published event says
# --------------------------------------------------------------------------- #


class TestRawRecordEvent:
    async def test_the_event_addresses_the_archived_object(self, journal: list[str]) -> None:
        archiver = RecordingArchiver(journal)
        publisher = RecordingPublisher(journal)
        connector = make_connector()
        connector.pages = [make_page(["a1"], watermark=T0)]

        await ConnectorRuntime(
            InMemoryCursorStore(), publisher=publisher, archiver=archiver
        ).run(connector)

        event = publisher.events[0]
        expected = hashlib.sha256(json.dumps({"id": "a1"}).encode()).hexdigest()
        assert event.raw_sha256 == expected
        assert event.raw_object_key == f"raw/rss/2026/07/28/{expected}.json"
        assert event.connector_slug == "demo"
        assert event.connector_version == "0.2.0"
        assert event.sync_run_id == "run_test"
        # The key embeds the date derived from this instant; an auditor
        # rebuilding the key from the event has to be given the same value.
        assert event.fetched_at == T0
        assert archiver.calls[0][1] == T0

    async def test_the_partition_key_is_the_signal_the_record_will_become(
        self, journal: list[str]
    ) -> None:
        """The raw and enriched events for one item must share a partition.

        The connector derives a `native_id` here (the URL-hash case, for feeds
        with no guid), so using the provider's raw id would key the two events
        differently and lose per-Signal ordering.
        """
        publisher = RecordingPublisher(journal)
        connector = make_connector()
        connector.derive_native_id = True
        connector.pages = [make_page(["a1"], watermark=T0)]

        await ConnectorRuntime(
            InMemoryCursorStore(), publisher=publisher, archiver=RecordingArchiver(journal)
        ).run(connector)

        derived = hashlib.sha256(b"a1").hexdigest()
        assert publisher.events[0].native_id == derived
        assert publisher.events[0].partition_key == signal_id(Platform.RSS, derived)

    async def test_an_r2_outage_defers_the_archive_and_keeps_ingesting(
        self, journal: list[str]
    ) -> None:
        """`docs/architecture.md` §7.3: R2 degrades, it does not halt ingestion."""
        publisher = RecordingPublisher(journal)
        cursors = JournalCursorStore(journal)
        connector = make_connector()
        connector.pages = [make_page(["a1"], watermark=T0)]

        result = await ConnectorRuntime(
            cursors,
            publisher=publisher,
            archiver=RecordingArchiver(
                journal, error=ExternalServiceError("R2 unreachable")
            ),
        ).run(connector)

        event = publisher.events[0]
        # Provenance without a payload -- exactly what `RawRecordEvent` documents
        # `raw_object_key=None` to mean, so a consumer defers rather than
        # enriching nothing.
        assert event.raw_object_key is None
        assert event.raw_sha256 is None
        assert event.native_id == "a1"
        assert result.succeeded
        assert cursors.commits

    async def test_a_connector_without_raw_bytes_still_archives_its_payload(
        self, journal: list[str]
    ) -> None:
        archiver = RecordingArchiver(journal)
        connector = make_connector()
        connector.pages = [make_page(["a1"], watermark=T0, raw_bytes=False)]

        await ConnectorRuntime(
            InMemoryCursorStore(), publisher=RecordingPublisher(journal), archiver=archiver
        ).run(connector)

        archived = json.loads(archiver.calls[0][2])
        assert archived["id"] == "a1"


# --------------------------------------------------------------------------- #
# 5. Cursor identity and storage
# --------------------------------------------------------------------------- #


class TestCursorStore:
    async def test_round_trips_watermark_page_token_and_checkpoint(
        self, sessions: async_sessionmaker[AsyncSession], account: str
    ) -> None:
        store = SqlCursorStore(sessions)
        key = CursorKey(DemoConnector.slug, ACCOUNT_ID, "hash")
        await store.commit(
            key, Cursor(watermark=T0, page_token="p1", checkpoint={"etag": "W/x"})
        )

        loaded = await store.load(key)
        assert loaded is not None
        assert loaded.watermark == T0
        assert loaded.page_token == "p1"
        assert loaded.checkpoint == {"etag": "W/x"}
        assert loaded.is_readable()

    async def test_a_missing_cursor_reads_as_none(
        self, sessions: async_sessionmaker[AsyncSession], account: str
    ) -> None:
        """Which is what makes the first run a full sync rather than an error."""
        assert await SqlCursorStore(sessions).load(CursorKey("demo", ACCOUNT_ID, "h")) is None

    async def test_a_backfill_gets_its_own_row(
        self, sessions: async_sessionmaker[AsyncSession], account: str, journal: list[str]
    ) -> None:
        """§4.1 rule 5: a historical crawl must never clobber the live watermark.

        The mode is folded into `params_hash`, which is the only way the rule and
        the `(slug, account, params_hash)` key from §4 can both hold.
        """
        store = SqlCursorStore(sessions)
        params = {"feeds": ["a"]}
        live = CursorKey.for_context("demo", make_context(params=params))
        historical = CursorKey.for_context(
            "demo", make_context(mode=SyncMode.BACKFILL, params=params)
        )
        assert live.params_hash != historical.params_hash

        await store.commit(live, Cursor(watermark=T0))
        await store.commit(historical, Cursor(watermark=T0 - timedelta(days=365)))

        reloaded = await store.load(live)
        assert reloaded is not None and reloaded.watermark == T0

    def test_params_hash_ignores_key_order(self) -> None:
        left = sync_params_hash({"a": 1, "b": 2}, SyncMode.INCREMENTAL)
        right = sync_params_hash({"b": 2, "a": 1}, SyncMode.INCREMENTAL)
        assert left == right

    def test_different_params_are_different_cursors(self) -> None:
        one = CursorKey.for_context("demo", make_context(params={"feeds": ["a"]}))
        two = CursorKey.for_context("demo", make_context(params={"feeds": ["b"]}))
        assert one.row_id != two.row_id

    def test_the_row_id_is_derived_not_random(self) -> None:
        """So a retried first commit addresses the same row instead of racing."""
        key = CursorKey("demo", ACCOUNT_ID, "hash")
        assert key.row_id == CursorKey("demo", ACCOUNT_ID, "hash").row_id
        assert len(key.row_id) <= 64


# --------------------------------------------------------------------------- #
# 6. Credentials
# --------------------------------------------------------------------------- #


def settings_with_key(key: str) -> Settings:
    """A `Settings` carrying one Fernet key, built without touching the process's.

    `get_settings()` is cached for the life of the process, so mutating it would
    leak into every test that ran afterwards. Every credential entry point takes
    `settings` for exactly this reason.
    """
    settings = Settings()
    return settings.model_copy(
        update={
            "security": settings.security.model_copy(
                update={"credential_encryption_key": SecretStr(key)}
            )
        }
    )


class TestCredentials:
    def test_round_trips_secrets(self) -> None:
        settings = settings_with_key(FERNET_KEY)
        ciphertext = Fernet(FERNET_KEY).encrypt(
            json.dumps({"client_id": "abc", "client_secret": "shh"}).encode()
        )

        creds = decrypt_credentials(ACCOUNT_ID, ciphertext, settings=settings)

        assert creds.require("client_id") == "abc"
        assert creds.require("client_secret") == "shh"
        # Never renders its secrets: a `ConnectorError` carrying this in
        # `details` would otherwise print them into a log line.
        assert "shh" not in repr(creds)

    def test_carries_non_secret_extras(self) -> None:
        settings = settings_with_key(FERNET_KEY)
        ciphertext = Fernet(FERNET_KEY).encrypt(
            json.dumps({"token": "t", "extra": {"instance_url": "https://x"}}).encode()
        )

        creds = decrypt_credentials(
            ACCOUNT_ID, ciphertext, extra={"feed": "a"}, settings=settings
        )

        assert creds.extra == {"instance_url": "https://x", "feed": "a"}
        assert "extra" not in creds.secrets

    def test_no_ciphertext_is_not_an_error(self) -> None:
        """`auth_type=none` connectors never have one, and OAuth accounts exist
        before consent completes."""
        creds = decrypt_credentials(ACCOUNT_ID, None, settings=settings_with_key(FERNET_KEY))
        assert creds.secrets == {}
        with pytest.raises(KeyError, match="acct_demo"):
            creds.require("token")

    def test_ciphertext_from_another_key_is_an_auth_error(self) -> None:
        """The row's credential is unusable, so the account genuinely needs reauth."""
        foreign = Fernet(Fernet.generate_key()).encrypt(b'{"token": "t"}')
        with pytest.raises(AuthError, match="could not be decrypted"):
            decrypt_credentials(ACCOUNT_ID, foreign, settings=settings_with_key(FERNET_KEY))

    def test_a_malformed_key_is_a_configuration_error_not_an_auth_error(self) -> None:
        """One wrong environment variable must not flag every account `needs_reauth`."""
        with pytest.raises(ConfigurationError, match="CREDENTIAL_ENCRYPTION_KEY"):
            credential_cipher(settings_with_key("too-short"))


# --------------------------------------------------------------------------- #
# 7. Sync context assembly
# --------------------------------------------------------------------------- #


class FakeRedis:
    """Enough of a Redis client to be handed to the two shared ports."""

    async def script_load(self, script: str) -> str:
        return "sha"

    async def evalsha(self, sha: str, numkeys: int, *keys_and_args: Any) -> Any:
        return [1, "9", 0]

    def set(self, name: str, value: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def exists(self, *names: str) -> Any:
        raise NotImplementedError

    def delete(self, *names: str) -> Any:
        raise NotImplementedError


class TestSyncContext:
    def test_binds_the_shared_redis_backed_ports(self) -> None:
        """Not the in-memory ones: N replicas with private state deduplicate
        nothing and permit N times the rate."""
        ctx = build_sync_context(
            DemoConnector, account_id=ACCOUNT_ID, params={"feeds": ["a"]}, redis=FakeRedis()
        )

        assert isinstance(ctx.limiter, TokenBucketLimiter)
        assert isinstance(ctx.dedup, RedisDedupStore)
        assert ctx.params_hash == sync_params_hash({"feeds": ["a"]}, SyncMode.INCREMENTAL)
        assert ctx.run_id.startswith("run_")

    def test_bucket_keys_match_the_ones_the_connector_will_ask_for(self) -> None:
        """An unmatched key falls back to the conservative per-host default, which
        would silently throttle the connector to a rate it never declared."""
        ctx = build_sync_context(DemoConnector, account_id=ACCOUNT_ID, redis=FakeRedis())
        connector = DemoConnector.from_config(ctx, Credentials(account_id=ACCOUNT_ID))
        buckets = rate_limit_buckets(DemoConnector, ACCOUNT_ID, SyncMode.INCREMENTAL)

        assert set(connector.rate_limit_keys()) <= set(buckets)

    def test_a_backfill_bucket_is_the_reduced_one(self) -> None:
        buckets = rate_limit_buckets(DemoConnector, ACCOUNT_ID, SyncMode.BACKFILL)
        incremental = rate_limit_buckets(DemoConnector, ACCOUNT_ID, SyncMode.INCREMENTAL)
        connector_key = f"os:rl:{DemoConnector.slug}"
        assert (
            buckets[connector_key].refill_per_second
            < incremental[connector_key].refill_per_second
        )


# --------------------------------------------------------------------------- #
# 8. Result accounting
# --------------------------------------------------------------------------- #


class TestSyncResult:
    async def test_counters_are_summed_across_pages(self, journal: list[str]) -> None:
        connector = make_connector()
        connector.pages = [
            make_page(["a1", "a2"], watermark=T0),
            make_page(["a3"], watermark=T0 + timedelta(minutes=1)),
        ]

        result = await ConnectorRuntime(
            InMemoryCursorStore(),
            publisher=RecordingPublisher(journal),
            archiver=RecordingArchiver(journal),
        ).run(connector)

        assert result.fetched == 3
        assert result.emitted == 3
        assert result.pages == 2
        assert result.ended_at is not None and result.ended_at >= result.started_at

    async def test_emitted_counts_what_the_broker_acknowledged(
        self, journal: list[str]
    ) -> None:
        """Not what the connector handed over. `is_partial` reads this as "real
        work survived", so it has to mean durable."""

        class OneBadArchiver(RecordingArchiver):
            async def __call__(
                self, platform: str, fetched_at: datetime, raw_bytes: bytes
            ) -> StoredObject:
                if b"a1" in raw_bytes:
                    raise PermanentError("nope")
                return await super().__call__(platform, fetched_at, raw_bytes)

        connector = make_connector()
        connector.pages = [make_page(["a1", "a2"], watermark=T0)]

        result = await ConnectorRuntime(
            InMemoryCursorStore(),
            publisher=RecordingPublisher(journal),
            archiver=OneBadArchiver(journal),
            dlq_publisher=RecordingDlq(journal),
        ).run(connector)

        assert result.fetched == 2
        assert result.emitted == 1
        assert result.dlq == 1

    async def test_a_retry_reports_the_records_it_really_re_fetched(
        self, journal: list[str]
    ) -> None:
        """The provider really was asked twice; a count that hid it would make the
        rate-limit budget unexplainable."""
        connector = make_connector()
        page = make_page(["a1"], watermark=T0)
        connector.pages_by_attempt = {0: [page], 1: [page]}
        connector.errors = {0: TransientError("reset")}

        result = await ConnectorRuntime(
            InMemoryCursorStore(),
            publisher=RecordingPublisher(journal),
            archiver=RecordingArchiver(journal),
            backoff=BackoffPolicy(base_seconds=0.0, max_attempts=3),
            sleep=no_sleep,
        ).run(connector)

        assert result.fetched == 2
        assert result.pages == 2
