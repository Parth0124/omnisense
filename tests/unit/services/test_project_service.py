"""Projects: creating them, and deciding which one a source belongs to.

The service is small, and almost all of its value is in what it *refuses* and
what it declines to destroy. So the tests here are weighted that way: a duplicate
slug, a source that moves between projects, a project that is paused rather than
deleted. The happy path is one test; the rest are the edges somebody hits during
`omnisense init` at 11pm.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from models.artifact import ArtifactKind, artifact_id, source_id
from models.enums import Platform
from models.orm.artifact import ArtifactRow, SourceRow
from models.project import normalize_slug, project_id
from services.project_service import ProjectService

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 11, 9, 0, tzinfo=UTC)


@pytest.fixture
def factory(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=orm_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


@pytest.fixture
def service(factory: async_sessionmaker[AsyncSession]) -> ProjectService:
    return ProjectService(factory, tenant_id="local")


async def add_source(
    factory: async_sessionmaker[AsyncSession],
    external_id: str,
    name: str,
    *,
    tenant_id: str = "local",
    artifacts: int = 0,
) -> str:
    """A source, optionally with artifacts, so counts have something to count."""
    identifier = source_id(Platform.GITHUB, external_id)
    async with factory() as session:
        session.add(
            SourceRow(
                id=identifier,
                tenant_id=tenant_id,
                platform=Platform.GITHUB,
                external_id=external_id,
                name=name,
            )
        )
        # Flushed before the artifacts so the foreign key has something to point
        # at. Within one flush SQLAlchemy orders inserts by table dependency, but
        # it batches the artifact rows into an executemany that SQLite evaluates
        # against the connection state at statement time -- so the source has to
        # already be there.
        await session.flush()
        for index in range(artifacts):
            session.add(
                ArtifactRow(
                    id=artifact_id(Platform.GITHUB, f"{external_id}-{index}"),
                    tenant_id=tenant_id,
                    kind=ArtifactKind.COMMIT,
                    source_id=identifier,
                    platform=Platform.GITHUB,
                    native_id=f"{external_id}-{index}",
                    occurred_at=NOW,
                )
            )
        await session.commit()
    return identifier


class TestCreating:
    async def test_a_project_is_created_with_a_derived_slug(self, service: ProjectService) -> None:
        project = await service.create(name="OmniSense API")

        assert project.slug == "omnisense-api"
        assert project.name == "OmniSense API"
        assert project.is_active
        assert project.id == project_id("local", "omnisense-api")

    async def test_an_explicit_slug_wins_over_the_derived_one(
        self, service: ProjectService
    ) -> None:
        project = await service.create(name="OmniSense API", slug="api")
        assert project.slug == "api"

    async def test_the_id_is_derived_so_the_same_slug_addresses_the_same_project(
        self, service: ProjectService
    ) -> None:
        """What lets the CLI name a project before creating it, and makes a second
        `init` with the same slug a conflict rather than a duplicate."""
        first = await service.create(name="One", slug="shared")
        with pytest.raises(ConflictError):
            await service.create(name="Two", slug="shared")

        assert first.id == project_id("local", "shared")

    async def test_a_duplicate_slug_says_what_to_do_about_it(self, service: ProjectService) -> None:
        """The message is the point. This is read in a terminal by somebody who
        has just been stopped mid-setup."""
        await service.create(name="OmniSense", slug="omnisense")
        with pytest.raises(ConflictError, match="already exists"):
            await service.create(name="OmniSense Again", slug="omnisense")

    async def test_a_name_with_no_usable_characters_is_refused_clearly(
        self, service: ProjectService
    ) -> None:
        """`normalize_slug("!!!")` is empty, and an empty slug would derive an id
        that every unnameable project collides on."""
        with pytest.raises(ValidationError, match="cannot derive"):
            await service.create(name="!!!")

    async def test_an_invalid_explicit_slug_is_refused_by_the_model(
        self, service: ProjectService
    ) -> None:
        """Capitals and spaces make a handle that has to be quoted and whose
        casing has to be remembered -- both are ways to be told 'no such project'
        for a project that exists."""
        for bad in ("Omni Sense", "OMNISENSE", "-leading", "trailing-"):
            with pytest.raises(ValueError):
                await service.create(name="x", slug=bad)

    async def test_two_tenants_may_use_the_same_slug(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """Uniqueness is per tenant. Two customers both having a project called
        `api` is expected, and they are different projects."""
        await ProjectService(factory, tenant_id="a").create(name="API", slug="api")
        await ProjectService(factory, tenant_id="b").create(name="API", slug="api")

        assert project_id("a", "api") != project_id("b", "api")


class TestAttachingSources:
    async def test_a_source_joins_a_project(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        await service.create(name="OmniSense", slug="omnisense")
        source = await add_source(factory, "R_1", "omnisense/api")

        attached = await service.attach_source(slug="omnisense", source_id=source)

        assert attached.project_id == project_id("local", "omnisense")
        assert [s.source_id for s in await service.sources("omnisense")] == [source]

    async def test_attaching_twice_is_a_no_op(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """Which is what makes `PUT` the right verb: re-running `omnisense init`
        must not produce a second membership or an error."""
        await service.create(name="OmniSense", slug="omnisense")
        source = await add_source(factory, "R_1", "omnisense/api")

        await service.attach_source(slug="omnisense", source_id=source)
        await service.attach_source(slug="omnisense", source_id=source)

        assert len(await service.sources("omnisense")) == 1

    async def test_a_source_moves_rather_than_being_refused(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """A repository reassigned to a different product is ordinary. Refusing it
        would mean deleting and re-adding the source, which would take its
        artifacts with it."""
        await service.create(name="Old", slug="old")
        await service.create(name="New", slug="new")
        source = await add_source(factory, "R_1", "omnisense/api")

        await service.attach_source(slug="old", source_id=source)
        await service.attach_source(slug="new", source_id=source)

        assert await service.sources("old") == []
        assert [s.source_id for s in await service.sources("new")] == [source]

    async def test_detaching_keeps_the_source_and_its_artifacts(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """ "This repository is not part of that product any more" does not mean
        the work never happened."""
        await service.create(name="OmniSense", slug="omnisense")
        source = await add_source(factory, "R_1", "omnisense/api", artifacts=3)
        await service.attach_source(slug="omnisense", source_id=source)

        await service.detach_source(source_id=source)

        assert await service.sources("omnisense") == []
        async with factory() as session:
            assert await session.get(SourceRow, source) is not None

    async def test_a_source_from_another_tenant_is_reported_as_absent(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """Not "forbidden": confirming a source exists in another tenant is itself
        a disclosure."""
        await service.create(name="OmniSense", slug="omnisense")
        foreign = await add_source(factory, "R_9", "other/repo", tenant_id="somebody-else")

        with pytest.raises(NotFoundError):
            await service.attach_source(slug="omnisense", source_id=foreign)

    async def test_attaching_to_a_project_that_does_not_exist_is_a_404(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        source = await add_source(factory, "R_1", "omnisense/api")
        with pytest.raises(NotFoundError):
            await service.attach_source(slug="nope", source_id=source)


class TestReading:
    async def test_a_project_spans_every_source_it_owns(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """The reason projects exist: one question, three repositories."""
        await service.create(name="OmniSense", slug="omnisense")
        for external, name, count in (
            ("R_1", "omnisense/api", 5),
            ("R_2", "omnisense/web", 3),
            ("R_3", "omnisense/infra", 1),
        ):
            source = await add_source(factory, external, name, artifacts=count)
            await service.attach_source(slug="omnisense", source_id=source)

        sources = await service.sources("omnisense")

        assert [s.name for s in sources] == [
            "omnisense/api",
            "omnisense/infra",
            "omnisense/web",
        ]
        assert sum(s.artifact_count for s in sources) == 9

    async def test_artifact_counts_come_from_one_query_not_one_per_source(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """A project with a dozen repositories would otherwise issue thirteen
        round trips to render one screen -- the shape of loop that only shows up
        as slowness once somebody has real data."""
        await service.create(name="OmniSense", slug="omnisense")
        for index in range(4):
            source = await add_source(factory, f"R_{index}", f"repo/{index}", artifacts=index)
            await service.attach_source(slug="omnisense", source_id=source)

        sources = await service.sources("omnisense")
        assert [s.artifact_count for s in sources] == [0, 1, 2, 3]

    async def test_a_source_with_no_artifacts_counts_zero_rather_than_vanishing(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """An outer join, not an inner one. A repository that is attached but has
        never synced is exactly what somebody debugging setup needs to see, and an
        inner join would hide it."""
        await service.create(name="OmniSense", slug="omnisense")
        source = await add_source(factory, "R_1", "omnisense/api", artifacts=0)
        await service.attach_source(slug="omnisense", source_id=source)

        sources = await service.sources("omnisense")
        assert len(sources) == 1
        assert sources[0].artifact_count == 0

    async def test_unassigned_sources_are_findable(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """Otherwise they are invisible: present, collecting artifacts, and absent
        from every project-scoped answer."""
        await service.create(name="OmniSense", slug="omnisense")
        attached = await add_source(factory, "R_1", "omnisense/api")
        await add_source(factory, "R_2", "omnisense/orphan")
        await service.attach_source(slug="omnisense", source_id=attached)

        loose = await service.unassigned_sources()
        assert [s.name for s in loose] == ["omnisense/orphan"]

    async def test_resolve_source_ids_is_what_makes_a_project_query_flat(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """Callers filter artifacts by these ids rather than joining through
        `sources` on what will be the largest table in the system."""
        await service.create(name="OmniSense", slug="omnisense")
        first = await add_source(factory, "R_1", "omnisense/api")
        second = await add_source(factory, "R_2", "omnisense/web")
        for source in (first, second):
            await service.attach_source(slug="omnisense", source_id=source)

        assert set(await service.resolve_source_ids("omnisense")) == {first, second}

    async def test_listing_hides_paused_projects_unless_asked(
        self, service: ProjectService
    ) -> None:
        await service.create(name="Live", slug="live")
        await service.create(name="Paused", slug="paused")
        await service.set_active(slug="paused", is_active=False)

        assert [p.slug for p in await service.list()] == ["live"]
        assert [p.slug for p in await service.list(include_inactive=True)] == [
            "live",
            "paused",
        ]

    async def test_projects_are_scoped_to_their_tenant(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """The check that matters most in this file. A service constructed with
        the wrong tenant returns another customer's projects and looks correct
        doing it."""
        await ProjectService(factory, tenant_id="a").create(name="Mine", slug="mine")

        assert await ProjectService(factory, tenant_id="b").list() == []
        with pytest.raises(NotFoundError):
            await ProjectService(factory, tenant_id="b").get("mine")


class TestPausingRatherThanDeleting:
    async def test_a_paused_project_keeps_its_sources(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        await service.create(name="OmniSense", slug="omnisense")
        source = await add_source(factory, "R_1", "omnisense/api", artifacts=2)
        await service.attach_source(slug="omnisense", source_id=source)

        await service.set_active(slug="omnisense", is_active=False)

        assert [s.source_id for s in await service.sources("omnisense")] == [source]

    async def test_a_paused_project_can_be_resumed(self, service: ProjectService) -> None:
        await service.create(name="OmniSense", slug="omnisense")
        await service.set_active(slug="omnisense", is_active=False)
        resumed = await service.set_active(slug="omnisense", is_active=True)
        assert resumed.is_active

    async def test_a_project_holding_artifacts_refuses_to_be_deleted(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """And names what to do instead. Deleting would either orphan the sources
        or cascade into their history, and every citation the system produces
        resolves against those rows."""
        await service.create(name="OmniSense", slug="omnisense")
        source = await add_source(factory, "R_1", "omnisense/api", artifacts=3)
        await service.attach_source(slug="omnisense", source_id=source)

        with pytest.raises(ConflictError, match="pause"):
            await service.delete(slug="omnisense")

        assert await service.get("omnisense")

    async def test_an_empty_project_can_be_deleted(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """Because there is nothing to protect. Refusing would leave somebody
        editing SQL to undo a typo made thirty seconds ago."""
        await service.create(name="Typo", slug="typoo")
        await service.delete(slug="typoo")

        with pytest.raises(NotFoundError):
            await service.get("typoo")

    async def test_deleting_detaches_its_sources_rather_than_destroying_them(
        self, service: ProjectService, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """A repository is not owned by the project that grouped it. With no
        artifacts yet, the source itself is still worth keeping -- it may be
        re-attached elsewhere in a moment."""
        await service.create(name="Wrong", slug="wrong")
        source = await add_source(factory, "R_1", "omnisense/api")
        await service.attach_source(slug="wrong", source_id=source)

        await service.delete(slug="wrong")

        async with factory() as session:
            stored = await session.get(SourceRow, source)
        assert stored is not None
        assert stored.project_id is None


class TestSlugNormalisation:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("OmniSense API", "omnisense-api"),
            ("  spaced  out  ", "spaced-out"),
            ("Already-Fine", "already-fine"),
            ("weird!!!chars", "weird-chars"),
            ("trailing---", "trailing"),
            ("emoji 🚀 name", "emoji-name"),
        ],
    )
    def test_a_display_name_becomes_a_usable_handle(self, given: str, expected: str) -> None:
        """`omnisense init` accepts what a person types; it should not refuse and
        ask again over a space."""
        assert normalize_slug(given) == expected

    def test_an_unusable_name_normalises_to_empty_rather_than_to_junk(self) -> None:
        """Empty is caught by the caller with a message. A junk slug would be
        accepted and then be impossible to type."""
        assert normalize_slug("!!!") == ""
