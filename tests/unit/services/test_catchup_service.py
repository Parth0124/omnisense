"""Briefing on a window, and refusing to make things up in the process.

Most of this file is about the citation check, because that is the only defence
against the failure this feature invites. A summariser that invents an outcome
does not crash, does not log, and does not look wrong -- it reads *better* than
the truth, which is precisely why it gets repeated.

The concrete case this was written against: four commits reading `establishing
ci/cd`, `testing ci/cd`, `testing ci/cd`, `Delete .github/workflows/deploy.yml`.
The obvious narrative is "CI/CD attempted and abandoned". Every workflow run had
in fact succeeded and the pipeline was removed on purpose. Nothing in the
artifacts supported the story, and nothing but a citation check would catch it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.core.exceptions import NotFoundError
from models.artifact import (
    Artifact,
    ArtifactKind,
    ArtifactOutcome,
    ArtifactProvenance,
    ArtifactState,
    artifact_id,
    source_id,
)
from models.enums import Platform
from models.orm.artifact import SourceRow
from models.orm.project import ProjectRow
from services.artifact_store import ArtifactStore
from services.catchup_service import (
    CatchupService,
    _Draft,
    _DraftPhase,
    parse_since,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
SRC = source_id(Platform.GITHUB, "R_1")
PROJECT_ID = "prj_test"


class ScriptedProvider:
    """Returns a fixed draft. The model's judgement is not what is under test."""

    def __init__(self, draft: _Draft | None = None) -> None:
        self.draft = draft or _Draft(
            headline="Work happened.",
            phases=[_DraftPhase(label="Building", period="Aug", narrative="Things.", refs=[1])],
        )
        self.prompts: list[str] = []

    async def structured(self, *, prompt, schema, system=None, model=None, max_tokens=None):
        self.prompts.append(prompt)
        return self.draft

    async def complete(self, **kwargs):  # pragma: no cover -- unused
        raise NotImplementedError

    async def aclose(self) -> None:  # pragma: no cover -- unused
        return None


@pytest.fixture
def factory(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=orm_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


@pytest.fixture
async def project(factory) -> str:
    async with factory() as session:
        session.add(ProjectRow(id=PROJECT_ID, tenant_id="local", slug="proj", name="Proj"))
        # Flushed before the source rather than added alongside it. `bind_for_testing()`
        # strips the `omnisense` schema qualifier so SQLite can read the DDL, and that
        # also erases the mapper-level dependency SQLAlchemy would have used to order
        # these two inserts -- leaving the foreign key to fail on a project that does
        # exist, one statement later.
        await session.flush()
        session.add(
            SourceRow(
                id=SRC,
                tenant_id="local",
                project_id=PROJECT_ID,
                platform=Platform.GITHUB,
                external_id="R_1",
                name="me/repo",
            )
        )
        await session.commit()
    return "proj"


async def store(factory, *artifacts: Artifact) -> None:
    await ArtifactStore(factory).write(list(artifacts))


def commit(native: str, *, days_ago: int = 0, title: str = "work") -> Artifact:
    return Artifact(
        id=artifact_id(Platform.GITHUB, native),
        tenant_id="local",
        kind=ArtifactKind.COMMIT,
        source_id=SRC,
        platform=Platform.GITHUB,
        native_id=native,
        title=title,
        occurred_at=NOW - timedelta(days=days_ago),
        provenance=ArtifactProvenance(connector_slug="github", fetched_at=NOW),
    )


def service(factory, provider: ScriptedProvider) -> CatchupService:
    return CatchupService(factory, provider=provider, model="test-model")


class TestCitationChecking:
    """The part that makes the rest of the output trustworthy."""

    async def test_a_phase_citing_nothing_real_is_dropped(self, factory, project) -> None:
        """The whole point. A model that cites `[9]` when eight artifacts exist has
        invented the paragraph, and printing it unattributed is how a fabrication
        reaches somebody who repeats it in a standup."""
        await store(factory, commit("c1", days_ago=1))
        provider = ScriptedProvider(
            _Draft(
                headline="h",
                phases=[
                    _DraftPhase(label="Real", period="Aug", narrative="n", refs=[1]),
                    _DraftPhase(label="Invented", period="Aug", narrative="n", refs=[9]),
                ],
            )
        )

        brief = await service(factory, provider).brief("proj", since=NOW - timedelta(days=7))

        assert [p.label for p in brief.phases] == ["Real"]
        assert brief.dropped_phases == ("Invented",)

    async def test_an_out_of_range_reference_does_not_take_the_phase_with_it(
        self, factory, project
    ) -> None:
        """A paragraph resting on three real artifacts and one imagined one is
        still worth keeping -- with the imagined one removed."""
        await store(factory, commit("c1", days_ago=3), commit("c2", days_ago=2))
        provider = ScriptedProvider(
            _Draft(
                headline="h",
                phases=[_DraftPhase(label="P", period="Aug", narrative="n", refs=[1, 2, 47])],
            )
        )

        brief = await service(factory, provider).brief("proj", since=NOW - timedelta(days=7))

        assert len(brief.phases[0].citations) == 2
        assert brief.dropped_phases == ()

    async def test_citations_resolve_to_the_artifact_the_number_pointed_at(
        self, factory, project
    ) -> None:
        """Off-by-one here is the worst possible bug: every claim would be attached
        to the wrong artifact and still look perfectly cited."""
        await store(
            factory,
            commit("c1", days_ago=3, title="oldest"),
            commit("c2", days_ago=2, title="middle"),
            commit("c3", days_ago=1, title="newest"),
        )
        provider = ScriptedProvider(
            _Draft(
                headline="h",
                phases=[_DraftPhase(label="P", period="Aug", narrative="n", refs=[1, 3])],
            )
        )

        brief = await service(factory, provider).brief("proj", since=NOW - timedelta(days=7))

        # The digest is oldest-first, so [1] is the oldest and [3] the newest.
        assert [c.title for c in brief.phases[0].citations] == ["oldest", "newest"]

    async def test_a_repeated_reference_is_cited_once(self, factory, project) -> None:
        await store(factory, commit("c1", days_ago=1))
        provider = ScriptedProvider(
            _Draft(
                headline="h",
                phases=[_DraftPhase(label="P", period="Aug", narrative="n", refs=[1, 1, 1])],
            )
        )

        brief = await service(factory, provider).brief("proj", since=NOW - timedelta(days=7))
        assert len(brief.phases[0].citations) == 1

    async def test_zero_is_not_a_valid_reference(self, factory, project) -> None:
        """Numbering starts at one. A zero would index the *last* artifact in
        Python and cite something plausible but unrelated."""
        await store(factory, commit("c1", days_ago=1), commit("c2", days_ago=2))
        provider = ScriptedProvider(
            _Draft(
                headline="h",
                phases=[_DraftPhase(label="P", period="Aug", narrative="n", refs=[0])],
            )
        )

        brief = await service(factory, provider).brief("proj", since=NOW - timedelta(days=7))
        assert brief.phases == ()
        assert brief.dropped_phases == ("P",)


class TestWindow:
    async def test_only_artifacts_inside_the_window_are_considered(self, factory, project) -> None:
        await store(factory, commit("recent", days_ago=2), commit("ancient", days_ago=400))

        provider = ScriptedProvider()
        await service(factory, provider).brief("proj", since=NOW - timedelta(days=7))

        assert "recent" not in provider.prompts[0] or "ancient" not in provider.prompts[0]
        assert "1 artifacts" in provider.prompts[0]

    async def test_the_digest_runs_oldest_first(self, factory, project) -> None:
        """A narrative is read forwards. Handing the model a reverse-chronological
        list asks it to invert time in its head while also not inventing
        anything, and it will do one of those two things."""
        await store(
            factory,
            commit("c1", days_ago=3, title="first thing"),
            commit("c2", days_ago=1, title="last thing"),
        )

        provider = ScriptedProvider()
        await service(factory, provider).brief("proj", since=NOW - timedelta(days=7))

        digest = provider.prompts[0]
        assert digest.index("first thing") < digest.index("last thing")

    async def test_an_empty_window_says_so_without_calling_the_model(
        self, factory, project
    ) -> None:
        """Nothing happened is a real answer, and paying a model to phrase it is
        both wasteful and an invitation to invent something."""
        provider = ScriptedProvider()

        brief = await service(factory, provider).brief("proj", since=NOW - timedelta(days=7))

        assert brief.is_empty
        assert provider.prompts == []
        assert "Nothing happened" in brief.headline

    async def test_the_bound_keeps_the_recent_end_and_reports_the_rest(
        self, factory, project
    ) -> None:
        """ "What happened while I was away" is a question about the near past, so
        truncating from the front would answer about the far one. And the count
        is printed, because a briefing that silently dropped three weeks reads
        exactly like a briefing of a quiet three weeks."""
        await store(factory, *[commit(f"c{i}", days_ago=i + 1) for i in range(10)])

        provider = ScriptedProvider()
        brief = await service(factory, provider).brief(
            "proj", since=NOW - timedelta(days=30), limit=4
        )

        assert brief.considered == 4
        assert brief.omitted == 6
        # `c1` is the most recent (1 day ago); `c10` the oldest.
        assert "c9" not in provider.prompts[0]

    async def test_artifacts_from_another_project_are_not_included(self, factory, project) -> None:
        """The reason projects exist. A briefing that leaked another product's
        commits would be worse than no briefing."""
        other = source_id(Platform.GITHUB, "R_2")
        async with factory() as session:
            session.add(ProjectRow(id="prj_other", tenant_id="local", slug="other", name="Other"))
            await session.flush()  # see the `project` fixture for why
            session.add(
                SourceRow(
                    id=other,
                    tenant_id="local",
                    project_id="prj_other",
                    platform=Platform.GITHUB,
                    external_id="R_2",
                    name="them/repo",
                )
            )
            await session.commit()

        await store(factory, commit("mine", days_ago=1, title="mine"))
        await ArtifactStore(factory).write(
            [
                Artifact(
                    id=artifact_id(Platform.GITHUB, "theirs"),
                    tenant_id="local",
                    kind=ArtifactKind.COMMIT,
                    source_id=other,
                    platform=Platform.GITHUB,
                    native_id="theirs",
                    title="theirs",
                    occurred_at=NOW - timedelta(days=1),
                    provenance=ArtifactProvenance(connector_slug="github", fetched_at=NOW),
                )
            ]
        )

        provider = ScriptedProvider()
        await service(factory, provider).brief("proj", since=NOW - timedelta(days=7))

        assert "mine" in provider.prompts[0]
        assert "theirs" not in provider.prompts[0]

    async def test_an_unknown_project_is_reported_clearly(self, factory) -> None:
        with pytest.raises(NotFoundError):
            await service(factory, ScriptedProvider()).brief("ghost", since=NOW)


class TestDigest:
    async def test_state_and_outcome_reach_the_model(self, factory, project) -> None:
        """The pair that stops the CI story from being invented: a run that says
        `success` cannot be narrated as a failure by anything reading carefully."""
        await ArtifactStore(factory).write(
            [
                Artifact(
                    id=artifact_id(Platform.GITHUB, "run1"),
                    tenant_id="local",
                    kind=ArtifactKind.CI_RUN,
                    source_id=SRC,
                    platform=Platform.GITHUB,
                    native_id="run1",
                    title="Delete deploy.yml",
                    state=ArtifactState.COMPLETED,
                    outcome=ArtifactOutcome.SUCCESS,
                    occurred_at=NOW - timedelta(days=1),
                    provenance=ArtifactProvenance(connector_slug="github", fetched_at=NOW),
                )
            ]
        )

        provider = ScriptedProvider()
        await service(factory, provider).brief("proj", since=NOW - timedelta(days=7))

        digest = provider.prompts[0]
        assert "ci_run" in digest
        assert "success" in digest

    async def test_a_multiline_title_is_flattened(self, factory, project) -> None:
        """One artifact must be one line, or the numbering the citations depend on
        stops lining up with what the model sees."""
        await store(factory, commit("c1", days_ago=1, title="subject\n\nbody paragraph"))

        provider = ScriptedProvider()
        await service(factory, provider).brief("proj", since=NOW - timedelta(days=7))

        assert "body paragraph" not in provider.prompts[0]
        assert len([ln for ln in provider.prompts[0].splitlines() if ln.startswith("[")]) == 1


class TestParseSince:
    @pytest.mark.parametrize(
        ("text", "days"),
        [("2w", 14), ("10d", 10), ("1w", 7)],
    )
    def test_durations(self, text: str, days: int) -> None:
        assert parse_since(text, now=NOW) == NOW - timedelta(days=days)

    def test_hours(self) -> None:
        assert parse_since("36h", now=NOW) == NOW - timedelta(hours=36)

    def test_an_iso_date_is_read_as_utc(self) -> None:
        """A naive date must not be compared against timezone-aware `occurred_at`
        -- that raises, and it would raise only for the person who typed a date
        instead of a duration."""
        parsed = parse_since("2026-08-01", now=NOW)
        assert parsed == datetime(2026, 8, 1, tzinfo=UTC)
        assert parsed.tzinfo is not None

    def test_nonsense_says_what_to_type_instead(self) -> None:
        with pytest.raises(ValueError, match="2w, 10d, 36h"):
            parse_since("last tuesday", now=NOW)
