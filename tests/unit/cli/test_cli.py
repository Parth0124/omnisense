"""The `omnisense` commands, driven the way a person drives them.

Run through Typer's `CliRunner` against a real (SQLite) database and a faked
GitHub, so what is under test is the whole path: parse the argument, validate
against GitHub, write the source, attach it, print something a human can act on.

**The output is asserted, not just the exit code.** This is the only part of the
system somebody reads while they are still deciding whether it works, and a
command that succeeds silently or fails with a traceback fails at the job it
exists for.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from typer.testing import CliRunner

from cli import main as cli_main
from cli.github_probe import RepoProbe, RepoStatus
from models.orm.artifact import SourceRow
from services.project_service import ProjectService

pytestmark = pytest.mark.unit

runner = CliRunner()

REAL_GITHUB_TOKEN = cli_main.github_token
"""Captured before the autouse fixture below replaces it.

`TestTokenResolution` tests this exact function, and the fixture stubs it for
every other test in the file -- so without holding a reference here those tests
would assert against the stub and pass no matter what the real one did."""


@pytest.fixture
def factory(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=orm_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


@pytest.fixture(autouse=True)
def wired(factory, monkeypatch: pytest.MonkeyPatch):
    """Point the CLI at the test database, and stop it dialling GitHub.

    `_check_database` is stubbed out because it deliberately opens the *process*
    engine -- the one pointed at PostgreSQL -- which is the right thing in
    production and the wrong thing here.
    """
    monkeypatch.setattr(cli_main, "_service", lambda: ProjectService(factory, tenant_id="local"))
    monkeypatch.setattr(cli_main, "_check_database", _noop)
    monkeypatch.setattr(cli_main, "github_token", lambda: "test-token")

    from backend.db import session as db_session

    monkeypatch.setattr(db_session, "get_sessionmaker", lambda: factory)
    return factory


async def _noop() -> None:
    return None


def fake_probe(**overrides):
    """Replace the GitHub probe with a fixed answer."""
    defaults = {
        "status": RepoStatus.OK,
        "reference": "omnisense/api",
        "message": "omnisense/api — private",
        "node_id": "R_kgDOABCD1M",
        "full_name": "omnisense/api",
        "default_branch": "main",
        "is_private": True,
        "is_archived": False,
    }
    defaults.update(overrides)

    async def probe(reference: str, *, token, client=None, timeout_seconds=15.0) -> RepoProbe:
        return RepoProbe(**{**defaults, "reference": reference})

    return probe


class TestInit:
    def test_it_creates_a_project_and_attaches_a_repository(
        self, monkeypatch: pytest.MonkeyPatch, factory
    ) -> None:
        monkeypatch.setattr(cli_main, "probe_repository", fake_probe())

        result = runner.invoke(
            cli_main.app,
            ["init", "--name", "OmniSense", "--slug", "omnisense", "--repo", "omnisense/api"],
            input="the developer platform\n",
        )

        assert result.exit_code == 0, result.output
        assert "Created omnisense" in result.output
        assert "1 repository attached" in result.output

    def test_a_repository_that_cannot_be_read_is_refused_with_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The entire point of validating during the wizard: the problem is stated
        while the person who can fix it is still sitting there."""
        monkeypatch.setattr(
            cli_main,
            "probe_repository",
            fake_probe(
                status=RepoStatus.NOT_FOUND,
                message="omnisense/ghost not found, or your token cannot see it",
                fix="check the spelling, then the token's access",
            ),
        )

        result = runner.invoke(
            cli_main.app,
            ["init", "--name", "OmniSense", "--slug", "omnisense", "--repo", "omnisense/ghost"],
            input="\n",
        )

        assert "not found" in result.output
        assert "check the spelling" in result.output
        assert "created with no repositories" in result.output

    def test_a_duplicate_slug_stops_before_asking_anything_else(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nobody should type a description and then be told the project exists."""
        monkeypatch.setattr(cli_main, "probe_repository", fake_probe())
        runner.invoke(
            cli_main.app, ["init", "--name", "OmniSense", "--slug", "omnisense"], input="\n"
        )

        second = runner.invoke(
            cli_main.app, ["init", "--name", "OmniSense", "--slug", "omnisense"], input="\n"
        )

        assert second.exit_code == 1
        assert "already exists" in second.output
        assert "project show omnisense" in second.output
        # The description question must not have been reached.
        assert "What is it?" not in second.output

    def test_an_archived_repository_warns_but_still_attaches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli_main, "probe_repository", fake_probe(is_archived=True))

        result = runner.invoke(
            cli_main.app,
            ["init", "--name", "X", "--slug", "x", "--repo", "omnisense/old"],
            input="\n",
        )

        assert "archived" in result.output
        assert "1 repository attached" in result.output

    def test_the_source_is_keyed_on_the_node_id_not_the_name(
        self, monkeypatch: pytest.MonkeyPatch, factory
    ) -> None:
        """So a rename on GitHub keeps one row and one history."""
        monkeypatch.setattr(cli_main, "probe_repository", fake_probe())
        runner.invoke(
            cli_main.app,
            ["init", "--name", "X", "--slug", "x", "--repo", "omnisense/api"],
            input="\n",
        )

        import asyncio

        async def stored() -> SourceRow | None:
            async with factory() as session:
                from sqlalchemy import select

                return (await session.execute(select(SourceRow))).scalars().first()

        row = asyncio.run(stored())
        assert row is not None
        assert row.external_id == "R_kgDOABCD1M"
        assert row.name == "omnisense/api"

    def test_rerunning_after_a_rename_corrects_the_name(
        self, monkeypatch: pytest.MonkeyPatch, factory
    ) -> None:
        """Same node id, new name. It should update the row rather than leave a
        stale name or create a second one beside it."""
        monkeypatch.setattr(cli_main, "probe_repository", fake_probe())
        runner.invoke(
            cli_main.app,
            ["init", "--name", "X", "--slug", "x", "--repo", "omnisense/api"],
            input="\n",
        )

        monkeypatch.setattr(cli_main, "probe_repository", fake_probe(full_name="omnisense/backend"))
        result = runner.invoke(cli_main.app, ["project", "add-repo", "x", "omnisense/backend"])
        assert result.exit_code == 0

        import asyncio

        from sqlalchemy import select

        async def rows() -> list[SourceRow]:
            async with factory() as session:
                return list((await session.execute(select(SourceRow))).scalars().all())

        stored = asyncio.run(rows())
        assert len(stored) == 1, "a rename created a second source"
        assert stored[0].name == "omnisense/backend"


class TestProjectCommands:
    def test_list_says_what_to_do_when_there_is_nothing(self) -> None:
        result = runner.invoke(cli_main.app, ["project", "list"])
        assert result.exit_code == 0
        assert "omnisense init" in result.output

    def test_list_shows_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli_main, "probe_repository", fake_probe())
        runner.invoke(
            cli_main.app,
            ["init", "--name", "OmniSense", "--slug", "omnisense", "--repo", "omnisense/api"],
            input="\n",
        )

        result = runner.invoke(cli_main.app, ["project", "list"])
        assert "omnisense" in result.output
        assert "1 source" in result.output

    def test_show_reports_a_source_with_nothing_synced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero artifacts is the difference between "configured" and "working",
        and hiding it would make a half-finished setup look complete."""
        monkeypatch.setattr(cli_main, "probe_repository", fake_probe())
        runner.invoke(
            cli_main.app,
            ["init", "--name", "OmniSense", "--slug", "omnisense", "--repo", "omnisense/api"],
            input="\n",
        )

        result = runner.invoke(cli_main.app, ["project", "show", "omnisense"])
        assert "omnisense/api" in result.output
        assert "0 artifacts" in result.output

    def test_show_on_an_unknown_project_suggests_the_list(self) -> None:
        result = runner.invoke(cli_main.app, ["project", "show", "ghost"])
        assert result.exit_code == 1
        assert "No project called 'ghost'" in result.output
        assert "project list" in result.output

    def test_add_repo_to_an_unknown_project_fails_clearly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli_main, "probe_repository", fake_probe())
        result = runner.invoke(cli_main.app, ["project", "add-repo", "ghost", "omnisense/api"])
        assert result.exit_code == 1
        assert "No project called 'ghost'" in result.output

    def test_remove_repo_keeps_the_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli_main, "probe_repository", fake_probe())
        runner.invoke(
            cli_main.app,
            ["init", "--name", "OmniSense", "--slug", "omnisense", "--repo", "omnisense/api"],
            input="\n",
        )

        result = runner.invoke(
            cli_main.app, ["project", "remove-repo", "omnisense", "omnisense/api"]
        )
        assert result.exit_code == 0
        assert "artifacts are kept" in result.output

        after = runner.invoke(cli_main.app, ["project", "show", "omnisense"])
        assert "No sources attached" in after.output


class TestTokenResolution:
    def test_the_environment_beats_the_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`GITHUB_TOKEN=... omnisense init` is the obvious way to try a token
        without editing .env, and it should work."""
        monkeypatch.setenv("GITHUB_TOKEN", "from-environment")
        assert REAL_GITHUB_TOKEN() == "from-environment"

    def test_an_empty_environment_value_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """An exported-but-empty variable is how a shell reports "unset" after a
        typo, and treating it as a real token would send the user chasing a 401."""
        monkeypatch.setenv("GITHUB_TOKEN", "")
        monkeypatch.setattr(cli_main, "REPO_ROOT", tmp_path)
        assert REAL_GITHUB_TOKEN() is None
