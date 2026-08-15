"""Group artifacts into features, by guessing and then being corrected.

`omnisense feature add v1 "image upload" --keyword cloudinary` declares the thing;
`sort()` walks the project's artifacts and attaches whichever ones mention it.
Nothing in a commit says which feature it belongs to, so every attachment here is
an inference carrying its evidence -- and a person overrules it whenever it is
wrong.

Why this writes its guesses and `identity_service` does not
-----------------------------------------------------------
The two look like the same problem and are not, because their wrong answers cost
different amounts.

A wrong *identity* merge puts somebody else's commits under your name. The totals
still add up, nothing looks broken, and you would have to already suspect it to
find it. So that one never happens without a person saying so.

A wrong *feature* tag puts a commit in the wrong bucket. You see it the moment you
look at the feature, and fixing it is one command. Meanwhile there are hundreds of
them -- one per artifact per feature -- and a system that demanded a person
confirm each would simply never be used.

So features are sorted automatically and corrected afterwards; identities are
proposed and confirmed. The asymmetry is the point, not an inconsistency.

Corrections are permanent
-------------------------
Rejecting a guess writes an `EXCLUDED` link rather than deleting the row. A
deleted rejection is proposed again on the next sync, and rejected again, and
after the third time nobody corrects anything -- at which point the guesses are
all that is left and the feature is quietly wrong. `sort()` never touches a link a
person has decided, in either direction.

Layer note: **L2 service.**
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.exceptions import ConflictError, NotFoundError
from backend.core.logging import get_logger
from models.feature import (
    DEFAULT_MEMBERSHIP_CONFIDENCE,
    FeatureState,
    MembershipMethod,
    VersionState,
    feature_id,
    version_id,
)
from models.orm.artifact import ArtifactRow, SourceRow
from models.orm.feature import FeatureLinkRow, FeatureRow, VersionRow
from models.orm.project import ProjectRow

__all__ = [
    "FeatureService",
    "FeatureSummary",
    "SortReport",
    "VersionSummary",
    "build_feature_service",
]

_log = get_logger(__name__)

DEFAULT_TENANT = "local"

MIN_TERM_LENGTH = 3
"""Shorter terms match everything.

A feature keyworded `ui` would claim every commit containing "build", "guide" or
"requirements". Three characters is the shortest that is usually a word rather
than a fragment, and anything below it is dropped with a warning rather than
silently ignored.
"""


@dataclass(frozen=True, slots=True)
class VersionSummary:
    id: str
    name: str
    state: VersionState
    feature_count: int = 0
    artifact_count: int = 0


@dataclass(frozen=True, slots=True)
class FeatureSummary:
    id: str
    name: str
    state: FeatureState
    version_name: str | None
    keywords: tuple[str, ...] = ()
    artifact_count: int = 0
    guessed_count: int = 0
    """How many of those were inferred rather than decided by a person.

    Reported so a feature that is entirely guesswork is visibly different from one
    somebody has been through -- the same count with two very different meanings.
    """


@dataclass(frozen=True, slots=True)
class SortReport:
    """What one sorting pass did."""

    linked: int = 0
    already_linked: int = 0
    protected: int = 0
    """Artifacts left alone because a person had already decided them."""
    scanned: int = 0

    @property
    def is_empty(self) -> bool:
        return self.linked == 0


class FeatureService:
    """Declares versions and features, and works out what belongs to them."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id

    # ------------------------------------------------------------ declaring --

    async def _project(self, session: AsyncSession, slug: str) -> ProjectRow:
        project = (
            await session.execute(
                select(ProjectRow).where(
                    ProjectRow.slug == slug, ProjectRow.tenant_id == self._tenant_id
                )
            )
        ).scalar_one_or_none()
        if project is None:
            raise NotFoundError.for_resource("project", slug)
        return project

    async def add_version(self, *, project: str, name: str, description: str | None = None) -> str:
        async with self._session_factory() as session, session.begin():
            row = await self._project(session, project)
            new_id = version_id(self._tenant_id, row.id, name)
            if await session.get(VersionRow, new_id) is not None:
                raise ConflictError(
                    f"{project} already has a version called {name!r}.",
                    details={"project": project, "name": name},
                )
            session.add(
                VersionRow(
                    id=new_id,
                    tenant_id=self._tenant_id,
                    project_id=row.id,
                    name=name,
                    description=description,
                )
            )
        _log.info("version.created", version_id=new_id, name=name)
        return new_id

    async def add_feature(
        self,
        *,
        project: str,
        name: str,
        version: str | None = None,
        keywords: Sequence[str] = (),
        description: str | None = None,
    ) -> str:
        async with self._session_factory() as session, session.begin():
            row = await self._project(session, project)

            resolved_version: str | None = None
            if version:
                found = await session.get(VersionRow, version_id(self._tenant_id, row.id, version))
                if found is None:
                    raise NotFoundError.for_resource("version", version)
                resolved_version = found.id

            new_id = feature_id(self._tenant_id, row.id, name)
            if await session.get(FeatureRow, new_id) is not None:
                raise ConflictError(
                    f"{project} already has a feature called {name!r}.",
                    details={"project": project, "name": name},
                )
            session.add(
                FeatureRow(
                    id=new_id,
                    tenant_id=self._tenant_id,
                    project_id=row.id,
                    version_id=resolved_version,
                    name=name,
                    description=description,
                    keywords=_clean_terms(keywords),
                )
            )
        _log.info("feature.created", feature_id=new_id, name=name)
        return new_id

    # --------------------------------------------------------------- sorting --

    async def sort(self, project: str) -> SortReport:
        """Attach the project's artifacts to whichever features mention them.

        Safe to run after every sync. Existing links are left alone, and a link a
        person has confirmed or excluded is never revisited -- which is what makes
        correcting the system worth doing more than once.
        """
        async with self._session_factory() as session:
            row = await self._project(session, project)

            features = list(
                (await session.execute(select(FeatureRow).where(FeatureRow.project_id == row.id)))
                .scalars()
                .all()
            )
            if not features:
                return SortReport()

            artifacts = list(
                (
                    await session.execute(
                        select(ArtifactRow)
                        .join(SourceRow, SourceRow.id == ArtifactRow.source_id)
                        .where(SourceRow.project_id == row.id)
                    )
                )
                .scalars()
                .all()
            )

            existing = {
                (link.feature_id, link.artifact_id): link
                for link in (
                    await session.execute(
                        select(FeatureLinkRow).where(
                            FeatureLinkRow.feature_id.in_([f.id for f in features])
                        )
                    )
                )
                .scalars()
                .all()
            }

        matchers = [(feature, _terms_for(feature)) for feature in features]
        report = SortReport(scanned=len(artifacts))
        fresh: list[FeatureLinkRow] = []

        for artifact in artifacts:
            haystacks = _haystacks(artifact)
            for feature, terms in matchers:
                if not terms:
                    continue
                key = (feature.id, artifact.id)
                if key in existing:
                    link = existing[key]
                    if link.method.is_decided:
                        report = _bump(report, protected=1)
                    else:
                        report = _bump(report, already_linked=1)
                    continue

                match = _match(terms, haystacks)
                if match is None:
                    continue
                method, evidence = match
                fresh.append(
                    FeatureLinkRow(
                        feature_id=feature.id,
                        artifact_id=artifact.id,
                        tenant_id=self._tenant_id,
                        method=method,
                        confidence=DEFAULT_MEMBERSHIP_CONFIDENCE[method],
                        evidence=evidence[:256],
                    )
                )
                report = _bump(report, linked=1)

        if fresh:
            async with self._session_factory() as session, session.begin():
                session.add_all(fresh)

        _log.info(
            "feature.sorted",
            project=project,
            linked=report.linked,
            protected=report.protected,
        )
        return report

    # ----------------------------------------------------------- correcting --

    async def resolve_artifact(self, prefix: str) -> str:
        """A full artifact id from a unique prefix, git-style.

        Ids are `art_` plus 32 hex characters, which does not fit on a listing
        line beside a title -- so the listing shows a prefix, and this turns it
        back into the real thing. Without it the interface tells somebody to run
        `feature reject ... <artifact-id>` while never showing them one, which is
        an instruction with no way to follow it.

        An ambiguous prefix raises rather than picking. Two artifacts is exactly
        when the wrong choice is silent.
        """
        cleaned = prefix.strip()
        if not cleaned:
            raise NotFoundError.for_resource("artifact", prefix)

        async with self._session_factory() as session:
            matches = list(
                (
                    await session.execute(
                        select(ArtifactRow.id)
                        .where(
                            ArtifactRow.id.startswith(cleaned),
                            ArtifactRow.tenant_id == self._tenant_id,
                        )
                        .limit(2)
                    )
                )
                .scalars()
                .all()
            )

        if not matches:
            raise NotFoundError.for_resource("artifact", prefix)
        if len(matches) > 1:
            raise ConflictError(
                f"{prefix!r} matches more than one artifact. Use a longer prefix.",
                details={"prefix": prefix},
            )
        return matches[0]

    async def decide(
        self, *, feature: str, artifact: str, belongs: bool, decided_by: str | None = None
    ) -> None:
        """Settle one membership, either way.

        `belongs=False` writes an `EXCLUDED` row rather than deleting. A deletion
        is undone by the next `sort()`, so the correction would have to be made
        again every time -- and a correction that does not stick is one nobody
        makes twice.
        """
        method = MembershipMethod.CONFIRMED if belongs else MembershipMethod.EXCLUDED
        async with self._session_factory() as session, session.begin():
            if await session.get(FeatureRow, feature) is None:
                raise NotFoundError.for_resource("feature", feature)
            if await session.get(ArtifactRow, artifact) is None:
                raise NotFoundError.for_resource("artifact", artifact)

            link = await session.get(FeatureLinkRow, (feature, artifact))
            if link is None:
                session.add(
                    FeatureLinkRow(
                        feature_id=feature,
                        artifact_id=artifact,
                        tenant_id=self._tenant_id,
                        method=method,
                        confidence=1.0,
                        decided_at=datetime.now(UTC),
                        decided_by=decided_by,
                    )
                )
            else:
                link.method = method
                link.confidence = 1.0
                link.decided_at = datetime.now(UTC)
                link.decided_by = decided_by

        _log.info("feature.decided", feature_id=feature, artifact_id=artifact, belongs=belongs)

    # ----------------------------------------------------------------- reads --

    async def versions(self, project: str) -> list[VersionSummary]:
        async with self._session_factory() as session:
            row = await self._project(session, project)
            rows = list(
                (
                    await session.execute(
                        select(VersionRow)
                        .where(VersionRow.project_id == row.id)
                        .order_by(VersionRow.name)
                    )
                )
                .scalars()
                .all()
            )
            summaries = []
            for version in rows:
                features = list(
                    (
                        await session.execute(
                            select(FeatureRow.id).where(FeatureRow.version_id == version.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                artifacts = 0
                if features:
                    artifacts = len(
                        (
                            await session.execute(
                                select(FeatureLinkRow.artifact_id)
                                .where(
                                    FeatureLinkRow.feature_id.in_(features),
                                    FeatureLinkRow.method != MembershipMethod.EXCLUDED,
                                )
                                .distinct()
                            )
                        )
                        .scalars()
                        .all()
                    )
                summaries.append(
                    VersionSummary(
                        id=version.id,
                        name=version.name,
                        state=version.state,
                        feature_count=len(features),
                        artifact_count=artifacts,
                    )
                )
            return summaries

    async def features(self, project: str) -> list[FeatureSummary]:
        async with self._session_factory() as session:
            row = await self._project(session, project)
            rows = list(
                (
                    await session.execute(
                        select(FeatureRow, VersionRow)
                        .outerjoin(VersionRow, VersionRow.id == FeatureRow.version_id)
                        .where(FeatureRow.project_id == row.id)
                        .order_by(FeatureRow.name)
                    )
                ).all()
            )
            if not rows:
                return []

            links = list(
                (
                    await session.execute(
                        select(FeatureLinkRow).where(
                            FeatureLinkRow.feature_id.in_([f.id for f, _ in rows])
                        )
                    )
                )
                .scalars()
                .all()
            )

        counts: dict[str, list[int]] = {}
        for link in links:
            if link.method is MembershipMethod.EXCLUDED:
                continue
            tally = counts.setdefault(link.feature_id, [0, 0])
            tally[0] += 1
            if not link.method.is_decided:
                tally[1] += 1

        return [
            FeatureSummary(
                id=feature.id,
                name=feature.name,
                state=feature.state,
                version_name=version.name if version else None,
                keywords=tuple(feature.keywords or ()),
                artifact_count=counts.get(feature.id, [0, 0])[0],
                guessed_count=counts.get(feature.id, [0, 0])[1],
            )
            for feature, version in rows
        ]

    async def members(self, feature: str) -> list[tuple[ArtifactRow, FeatureLinkRow]]:
        """Everything in one feature, newest first. Excludes what was ruled out."""
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(ArtifactRow, FeatureLinkRow)
                    .join(FeatureLinkRow, FeatureLinkRow.artifact_id == ArtifactRow.id)
                    .where(
                        FeatureLinkRow.feature_id == feature,
                        FeatureLinkRow.method != MembershipMethod.EXCLUDED,
                    )
                    .order_by(ArtifactRow.occurred_at.desc())
                )
            ).all()
        # Unpacked into plain tuples: `Row` is a named-tuple-alike, and returning
        # it leaks a SQLAlchemy type into the CLI's signature for no benefit.
        return [(artifact, link) for artifact, link in rows]


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #


def _clean_terms(terms: Iterable[str]) -> list[str]:
    cleaned = []
    for term in terms:
        value = term.strip().casefold()
        if len(value) >= MIN_TERM_LENGTH and value not in cleaned:
            cleaned.append(value)
    return cleaned


def _terms_for(feature: FeatureRow) -> list[str]:
    """A feature's own words, plus its name split into them.

    "image upload" matches a commit saying "upload the image" only if the phrase
    is broken up, and most commit messages do not quote a feature name verbatim.
    Words shorter than `MIN_TERM_LENGTH` are dropped -- "to" and "of" would match
    everything.
    """
    words = re.split(r"[^a-z0-9]+", feature.name.casefold())
    return _clean_terms([feature.name, *words, *(feature.keywords or ())])


def _haystacks(artifact: ArtifactRow) -> dict[MembershipMethod, list[str]]:
    """Where to look, grouped by how much a hit there is worth."""
    details: dict[str, Any] = artifact.details or {}

    branches = [str(value) for key in ("head_ref", "head_branch") if (value := details.get(key))]
    titles = [text for text in (artifact.title, artifact.body) if text]
    paths = [
        str(entry.get("path", ""))
        for entry in (details.get("files") or [])
        if isinstance(entry, dict)
    ]

    return {
        MembershipMethod.BRANCH: [value.casefold() for value in branches],
        MembershipMethod.TITLE: [value.casefold() for value in titles],
        MembershipMethod.PATH: [value.casefold() for value in paths],
    }


def _match(
    terms: Sequence[str], haystacks: dict[MembershipMethod, list[str]]
) -> tuple[MembershipMethod, str] | None:
    """The strongest evidence, or nothing.

    Checked branch-first because a branch name is *chosen*: `feature/image-upload`
    is somebody declaring intent, while a title mention may be incidental.
    Returning on the first hit keeps one link per pair, carrying the best reason
    rather than the last one found.
    """
    for method in (MembershipMethod.BRANCH, MembershipMethod.TITLE, MembershipMethod.PATH):
        for text in haystacks[method]:
            for term in terms:
                if term in text:
                    return method, f"{method.value}: {term}"
    return None


def _bump(report: SortReport, **deltas: int) -> SortReport:
    return SortReport(
        linked=report.linked + deltas.get("linked", 0),
        already_linked=report.already_linked + deltas.get("already_linked", 0),
        protected=report.protected + deltas.get("protected", 0),
        scanned=report.scanned,
    )


def build_feature_service(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    *,
    tenant_id: str = DEFAULT_TENANT,
) -> FeatureService:
    if session_factory is None:
        from backend.db.session import get_sessionmaker

        session_factory = get_sessionmaker()
    return FeatureService(session_factory, tenant_id=tenant_id)
