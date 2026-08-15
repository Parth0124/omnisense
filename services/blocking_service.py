"""What is standing in the way, answered in SQL.

This module exists to settle an architecture question as much as to ship a
feature. `docs/architecture.md` provisioned Neo4j on the reasoning that
cross-source links are *inferred* rather than foreign keys, and that traversing
them is a graph problem. That is a plausible argument and it deserved a
measurement rather than a vote, so `docs/tasks.md` step 7 says: write the
"what's blocking X" traversal in PostgreSQL first, and keep Neo4j only if
PostgreSQL genuinely cannot do it.

What the traversal actually is
------------------------------
    version -> features -> feature_links -> artifacts -> state/outcome

Four joins, all on indexed columns, all one-to-many in a fixed direction. There
is no recursion, no variable-length path, and no "find any route between these
two nodes" -- the shape is known at compile time because the schema fixes it.
That is a relational query wearing a graph's vocabulary.

Neo4j earns its place when the *depth is unknown*: "what eventually depends on
this decision" or "shortest path from this incident to a root cause". Nothing in
steps 8 or 9 asks for that yet. This module is the evidence for that claim, and
if a later question does need unbounded traversal, it will be equally obvious.

The four kinds of blocker
-------------------------
Chosen by the person who has to read them (12 Aug 2026), and ordered here by how
reliably they can be detected rather than by how bad they are:

| Blocker             | Detection                                   | Ships |
| ------------------- | ------------------------------------------- | ----- |
| CI failing          | latest run for the feature failed           | now   |
| Work stopped        | nothing touched it in `STALE_AFTER`         | now   |
| PR unreviewed       | open pull request with no review artifact   | now*  |
| Unanswered question | needs Slack                                 | 10    |

*The pull-request rule is live but its input is not: the `reviews` stream has
never completed a pass against real GitHub, so "no review" currently cannot be
distinguished from "reviews were never fetched". `Blocker.confidence` says so
rather than the reader having to know.

Layer note: **L2 service.**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.exceptions import NotFoundError
from backend.core.logging import get_logger
from models.artifact import ArtifactKind, ArtifactOutcome, ArtifactState
from models.feature import FeatureState, MembershipMethod
from models.orm.artifact import ArtifactRow
from models.orm.feature import FeatureLinkRow, FeatureRow, VersionRow
from models.orm.project import ProjectRow

__all__ = ["Blocker", "BlockingReport", "BlockingService", "build_blocking_service"]

_log = get_logger(__name__)

DEFAULT_TENANT = "local"

STALE_AFTER = timedelta(days=21)
"""How long without activity before unfinished work counts as stalled.

Three weeks rather than one, because "stalled" fires on *deliberate* pauses too
and there is no way to tell them apart from the data. Set it short and the report
fills with things somebody parked on purpose, which is how a blocker list becomes
something people scroll past. Muting is `feature state --dropped`.
"""

REVIEW_OVERDUE_AFTER = timedelta(days=3)
"""How long an open pull request may sit unreviewed before it is a blocker.

A working day plus the weekend. Shorter and every pull request opened on a Friday
is a blocker by Monday.
"""


@dataclass(frozen=True, slots=True)
class Blocker:
    """One thing in the way, and how sure we are it is real."""

    kind: str
    feature: str
    summary: str
    since: datetime | None
    artifact_id: str | None = None

    severity: int = 0
    """Higher sorts first. Ordering is the point of the list -- four blockers with
    no ranking is a list somebody reads top to bottom and acts on at random."""

    confidence: float = 1.0
    """Below 1.0 when the *absence* of data is doing the work.

    "No review exists" means one thing when reviews are being fetched and nothing
    at all when they are not, and a reader cannot tell which from the sentence.
    """

    caveat: str | None = None


@dataclass(frozen=True, slots=True)
class BlockingReport:
    target: str
    blockers: tuple[Blocker, ...] = ()
    features_checked: int = 0
    query_ms: float = 0.0

    @property
    def is_clear(self) -> bool:
        return not self.blockers


class BlockingService:
    """Answers "what is blocking X" for a version or a feature."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: str = DEFAULT_TENANT,
        now: datetime | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id
        self._now = now

    def _clock(self) -> datetime:
        return self._now or datetime.now(UTC)

    async def blocking(self, project: str, target: str) -> BlockingReport:
        """Everything standing in the way of one version or one feature.

        `target` is matched against version names first and feature names second,
        which is the order somebody would mean them: `blocking v1.1` is the
        common question and `blocking "image upload"` the zoomed-in one.
        """
        started = datetime.now(UTC)

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(ProjectRow).where(
                        ProjectRow.slug == project, ProjectRow.tenant_id == self._tenant_id
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                raise NotFoundError.for_resource("project", project)

            # `exists` is separate from the list, deliberately. A version that has
            # been declared but holds no features yet is a perfectly ordinary
            # state, and collapsing it into "not found" tells somebody their
            # version is missing when they are looking straight at it in
            # `feature list`.
            exists, features = await self._features_for(session, row.id, target)
            if not exists:
                raise NotFoundError.for_resource("version or feature", target)

            blockers: list[Blocker] = []
            for feature in features:
                blockers.extend(await self._ci_failures(session, feature))
                blockers.extend(await self._unreviewed(session, feature))
                blockers.extend(await self._stalled(session, feature))

        elapsed = (datetime.now(UTC) - started).total_seconds() * 1000
        _log.info(
            "blocking.queried", project=project, target=target, found=len(blockers), ms=elapsed
        )
        return BlockingReport(
            target=target,
            blockers=tuple(sorted(blockers, key=lambda b: (-b.severity, b.feature))),
            features_checked=len(features),
            query_ms=elapsed,
        )

    async def _features_for(
        self, session: AsyncSession, project_id: str, target: str
    ) -> tuple[bool, list[FeatureRow]]:
        """Whether the target exists, and which features it covers.

        Two return values because "no such version" and "a version holding
        nothing yet" need opposite answers -- one is an error and the other is
        "nothing is blocking it".
        """
        version = (
            await session.execute(
                select(VersionRow).where(
                    VersionRow.project_id == project_id,
                    func.lower(VersionRow.name) == target.casefold(),
                )
            )
        ).scalar_one_or_none()
        if version is not None:
            return True, list(
                (
                    await session.execute(
                        select(FeatureRow).where(FeatureRow.version_id == version.id)
                    )
                )
                .scalars()
                .all()
            )

        features = list(
            (
                await session.execute(
                    select(FeatureRow).where(
                        FeatureRow.project_id == project_id,
                        func.lower(FeatureRow.name) == target.casefold(),
                    )
                )
            )
            .scalars()
            .all()
        )
        return bool(features), features

    def _members(self, feature_id: str) -> Select[tuple[ArtifactRow]]:
        """Artifacts belonging to one feature, excluding what a person ruled out."""
        return (
            select(ArtifactRow)
            .join(FeatureLinkRow, FeatureLinkRow.artifact_id == ArtifactRow.id)
            .where(
                FeatureLinkRow.feature_id == feature_id,
                FeatureLinkRow.method != MembershipMethod.EXCLUDED,
            )
        )

    async def _ci_failures(self, session: AsyncSession, feature: FeatureRow) -> list[Blocker]:
        """A failed run with nothing green after it.

        The "after it" is the whole rule. Reporting every failure that ever
        happened would mean a feature fixed on Tuesday is still blocked on Friday,
        and a blocker list that never clears is one nobody trusts.
        """
        runs = list(
            (
                await session.execute(
                    self._members(feature.id)
                    .where(ArtifactRow.kind == ArtifactKind.CI_RUN)
                    .order_by(ArtifactRow.occurred_at.desc())
                    .limit(20)
                )
            )
            .scalars()
            .all()
        )
        for run in runs:
            if run.outcome is ArtifactOutcome.SUCCESS:
                return []
            if run.outcome is ArtifactOutcome.FAILURE:
                return [
                    Blocker(
                        kind="ci_failing",
                        feature=feature.name,
                        summary=f"CI failing since {run.title or 'a run'}",
                        since=run.occurred_at,
                        artifact_id=run.id,
                        severity=30,
                    )
                ]
        return []

    async def _unreviewed(self, session: AsyncSession, feature: FeatureRow) -> list[Blocker]:
        """Open pull requests nobody has reviewed."""
        cutoff = self._clock() - REVIEW_OVERDUE_AFTER
        pulls = list(
            (
                await session.execute(
                    self._members(feature.id).where(
                        ArtifactRow.kind == ArtifactKind.PULL_REQUEST,
                        ArtifactRow.state == ArtifactState.OPEN,
                        ArtifactRow.occurred_at <= cutoff,
                    )
                )
            )
            .scalars()
            .all()
        )
        if not pulls:
            return []

        reviewed = set(
            (
                await session.execute(
                    select(ArtifactRow.native_id).where(ArtifactRow.kind == ArtifactKind.REVIEW)
                )
            )
            .scalars()
            .all()
        )

        return [
            Blocker(
                kind="unreviewed",
                feature=feature.name,
                summary=f"{pull.title or 'a pull request'} has had no review",
                since=pull.occurred_at,
                artifact_id=pull.id,
                severity=20,
                # Lowered because the *absence* of a review is doing the work
                # here, and absence is indistinguishable from "never fetched"
                # while the reviews stream is unverified.
                confidence=0.6,
                caveat="reviews have never completed a live sync — absence may mean unfetched",
            )
            for pull in pulls
            if pull.native_id not in reviewed
        ]

    async def _stalled(self, session: AsyncSession, feature: FeatureRow) -> list[Blocker]:
        """Unfinished work nobody has touched."""
        if feature.state in (FeatureState.DONE, FeatureState.DROPPED):
            return []

        latest = (
            await session.execute(
                self._members(feature.id).order_by(ArtifactRow.occurred_at.desc()).limit(1)
            )
        ).scalar_one_or_none()
        if latest is None:
            return []

        # SQLite hands back naive datetimes for a `timezone=True` column, so a
        # comparison against an aware `now` raises rather than answering. The unit
        # suite runs on SQLite by design, and a service that only works on
        # PostgreSQL would be untested where it is cheapest to test.
        occurred = latest.occurred_at
        if occurred.tzinfo is None:
            occurred = occurred.replace(tzinfo=UTC)

        idle = self._clock() - occurred
        if idle < STALE_AFTER:
            return []

        return [
            Blocker(
                kind="stalled",
                feature=feature.name,
                summary=f"nothing has touched this in {idle.days} days",
                since=occurred,
                severity=10,
                # A deliberate pause looks exactly like an abandoned one, and no
                # column distinguishes them.
                confidence=0.5,
                caveat="a deliberate pause looks the same; mark it dropped to silence this",
            )
        ]


def build_blocking_service(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    *,
    tenant_id: str = DEFAULT_TENANT,
) -> BlockingService:
    if session_factory is None:
        from backend.db.session import get_sessionmaker

        session_factory = get_sessionmaker()
    return BlockingService(session_factory, tenant_id=tenant_id)
