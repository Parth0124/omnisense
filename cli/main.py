"""`omnisense` -- the command line.

Talks to the services directly rather than over HTTP. The CLI runs on the same
machine as the database, and routing it through the API would mean the wizard
could not run unless `make dev` was up -- which is precisely the state somebody
is in when they are still setting the thing up.

**Why `.env` is loaded here explicitly.** `backend/core/config.py` reads it for
the settings it owns, but connector credentials such as `GITHUB_TOKEN` are
deliberately not settings -- `docs/architecture.md` forbids `connectors/` from
importing config, so they reach a connector as constructor arguments. That leaves
`GITHUB_TOKEN` in the file and not in the environment, and the wizard needs it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

# Inlined rather than assigned to a name first. A module-level assignment before
# the imports makes every one of them "not at top of file"; a bare conditional
# does not. Same shape as `scripts/init_databases.py`, for the same reason: this
# runs as `python -m cli` from the repository root, which puts `cli/` on the path
# and not the root, so the first-party imports below would all fail without it.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from dotenv import dotenv_values

from backend.core.exceptions import ConflictError, NotFoundError, OmniSenseError
from cli.github_probe import probe_repository
from models.artifact import source_id
from models.enums import Platform
from models.orm.artifact import SourceRow
from models.project import normalize_slug
from services.artifact_sync import STREAMS
from services.catchup_service import MAX_ARTIFACTS, build_catchup_service, parse_since
from services.project_service import ProjectService, build_project_service

__all__ = ["app", "main"]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _quiet_service_logging() -> None:
    """Keep structured service logs out of the wizard's output.

    The services log at INFO through `structlog`, which is right for a worker and
    wrong here: `project.created project_id=prj_95e0... slug=omnisense` lands in
    the middle of a prompt, between the question and the answer, and reads as an
    error to anybody who has not seen it before.

    **Set through `LOG_LEVEL`, not by lowering the loggers directly.** Calling
    `setLevel` here does nothing that lasts: `configure_logging()` runs lazily,
    on the first log call, and sets the root level from `settings.app.log_level`
    -- so the levels are put back moments after being lowered, and the line
    prints anyway. Setting the environment variable before the settings are read
    means the configuration that eventually runs is already the quiet one.

    WARNING rather than silence: a genuine problem still prints, and `--verbose`
    leaves the level alone for when the question is "what is it actually doing".
    """
    os.environ["LOG_LEVEL"] = "WARNING"

    from backend.core.config import get_settings
    from backend.core.logging import configure_logging

    get_settings.cache_clear()
    # `force=True` because `configure_logging` is idempotent: something imported
    # above already triggered it at the settings' own INFO, and without the force
    # it returns early and the quieter level never applies. Idempotence is right
    # for its normal callers -- the API lifespan and each worker entrypoint, which
    # would otherwise stack handlers and duplicate every line -- and this is the
    # one caller that genuinely means "reconfigure".
    configure_logging(force=True)


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="OmniSense -- point it at your projects and ask what happened.",
)


@contextlib.contextmanager
def _quiet_import_warnings() -> Iterator[None]:
    """Silence third-party deprecation notices raised while importing.

    Importing the agent layer pulls in LangGraph, which warns that a LangChain
    default will change in a future release. That is addressed to whoever
    upgrades the dependency, and it lands above a briefing written for somebody
    who did not ask.

    **Why this replaces `showwarning` instead of adding a filter.**
    `langchain_core._api.deprecation` calls
    `warnings.filterwarnings("default", ...)` on its own categories at import
    time. A filter added later is prepended, so theirs wins over anything set
    beforehand -- including `-W ignore` and `catch_warnings(); simplefilter
    ("ignore")`, both of which were tried and both of which still printed it.
    Swapping the *sink* is a layer below the filters, which is the only place
    left that they cannot override.

    Scoped to one import, and restored in a `finally`. Warnings raised anywhere
    else -- and every warning the test suite and the workers see -- are
    untouched: suppressing where a warning would be read is housekeeping;
    suppressing where it would be caught is how a breaking upgrade lands
    silently.
    """
    import warnings

    original = warnings.showwarning
    warnings.showwarning = lambda *args, **kwargs: None
    try:
        yield
    finally:
        warnings.showwarning = original


@app.callback()
def _configure(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show service logs."),
) -> None:
    """Runs before every command."""
    if not verbose:
        _quiet_service_logging()


project_app = typer.Typer(no_args_is_help=True, help="Inspect and manage projects.")
app.add_typer(project_app, name="project")


def github_token() -> str | None:
    """The token from the environment, falling back to `.env`.

    The environment wins so a one-off run can override the file without editing
    it -- `GITHUB_TOKEN=... omnisense init` is the obvious thing to try when a
    token is being tested, and it should work.
    """
    from_env = os.environ.get("GITHUB_TOKEN")
    if from_env:
        return from_env
    value = dotenv_values(REPO_ROOT / ".env").get("GITHUB_TOKEN")
    return value or None


def _service() -> ProjectService:
    return build_project_service(tenant_id="local")


def _die(message: str, fix: str = "") -> None:
    typer.secho(f"\n{message}", fg=typer.colors.RED, bold=True)
    if fix:
        typer.echo(f"  {fix}")
    raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #


async def _check_database() -> None:
    """Fail early and clearly if the stack is not up.

    Every command below writes to PostgreSQL, and the failure without this is a
    connection traceback from four frames inside SQLAlchemy -- which says nothing
    about `make start` to somebody who has not run it yet.
    """
    from backend.db.session import check_postgres, dispose_engine

    try:
        reachable = await check_postgres()
    except Exception:
        reachable = False
    finally:
        await dispose_engine()
    if not reachable:
        _die(
            "The database is not answering.",
            "Start the stack first:  make start",
        )


async def _add_repository(
    service: ProjectService, slug: str, reference: str, token: str | None
) -> bool:
    """Validate one repository against GitHub and attach it. Returns whether it stuck.

    Validation happens *before* anything is written, which is the whole point of
    the wizard: a repository that cannot be read is rejected now, by the person
    who can fix it, rather than at the first sync as an empty result.
    """
    probe = await probe_repository(reference, token=token)

    if not probe.ok:
        typer.secho(f"  ✗ {probe.message}", fg=typer.colors.RED)
        if probe.fix:
            typer.echo(f"    {probe.fix}")
        return False

    if probe.is_archived:
        # Not refused. An archived repository is read-only, not invisible, and its
        # history is often exactly what somebody is asking about.
        typer.secho(
            f"  ! {probe.full_name} is archived — it will sync, but nothing new will arrive",
            fg=typer.colors.YELLOW,
        )

    from backend.db.session import get_sessionmaker

    identifier = source_id(Platform.GITHUB, probe.node_id or probe.full_name or reference)
    factory = get_sessionmaker()
    async with factory() as session:
        existing = await session.get(SourceRow, identifier)
        if existing is None:
            session.add(
                SourceRow(
                    id=identifier,
                    tenant_id="local",
                    platform=Platform.GITHUB,
                    # The node id, never the name: a rename keeps the node id and
                    # every artifact follows. Keyed on the name, a rename forks
                    # the repository's history in two.
                    external_id=probe.node_id or probe.full_name or reference,
                    name=probe.full_name or reference,
                    url=f"https://github.com/{probe.full_name}",
                    default_branch=probe.default_branch,
                )
            )
        else:
            # Re-running `init` after a rename should correct the name rather
            # than leave the old one or create a second row.
            existing.name = probe.full_name or existing.name
            existing.default_branch = probe.default_branch or existing.default_branch
        await session.commit()

    await service.attach_source(slug=slug, source_id=identifier)
    typer.secho(f"  ✓ {probe.message}", fg=typer.colors.GREEN)
    return True


@app.command()
def init(
    name: str = typer.Option(None, "--name", help="Project name. Prompted for if omitted."),
    slug: str = typer.Option(None, "--slug", help="Command-line handle. Derived from the name."),
    repo: list[str] = typer.Option(
        None, "--repo", help="Repository to add. Repeatable; prompted for if omitted."
    ),
) -> None:
    """Create a project and attach repositories to it, checking each one as you go."""

    async def run() -> None:
        await _check_database()
        token = github_token()

        typer.secho("\nOmniSense — new project\n", bold=True)
        if not token:
            typer.secho(
                "No GITHUB_TOKEN found. Public repositories will still validate; "
                "private ones cannot be checked.",
                fg=typer.colors.YELLOW,
            )
            typer.echo("  Set it in .env when you have one.\n")

        project_name = name or typer.prompt("Project name")
        resolved_slug = slug or normalize_slug(project_name)

        service = _service()

        # Checked before the next question, purely so nobody types a description
        # and is then told the project already exists. This is a courtesy, not
        # the rule: `create()` still lets the unique constraint decide, because
        # two `init` runs in different terminals would both pass a check here and
        # only one can win.
        try:
            existing = await service.get(resolved_slug)
        except NotFoundError:
            existing = None
        if existing is not None:
            _die(
                f"A project called {resolved_slug!r} already exists.",
                f"Look at it with:  omnisense project show {resolved_slug}\n"
                f"  Or add to it with:  omnisense project add-repo {resolved_slug} owner/name",
            )
            return

        description = typer.prompt(
            "What is it? (one line, read by the agents when they plan)",
            default="",
            show_default=False,
        )
        try:
            project = await service.create(
                name=project_name, slug=resolved_slug, description=description or None
            )
        except ConflictError as error:
            _die(str(error))
            return

        typer.secho(f"\nCreated {project.slug}\n", fg=typer.colors.GREEN, bold=True)

        typer.echo("Repositories (owner/name or a GitHub URL). Blank line when done.")
        added = 0
        for reference in repo or []:
            if await _add_repository(service, project.slug, reference, token):
                added += 1

        if not repo:
            while True:
                reference = typer.prompt("  repo", default="", show_default=False).strip()
                if not reference:
                    break
                if await _add_repository(service, project.slug, reference, token):
                    added += 1

        typer.echo("")
        if added:
            typer.secho(
                f"{project.slug}: {added} repositor{'y' if added == 1 else 'ies'} attached.",
                fg=typer.colors.GREEN,
                bold=True,
            )
            typer.echo("\nNothing has been synced yet — reading commits and pull requests")
            typer.echo("arrives in the next step.")
        else:
            typer.secho(f"{project.slug} created with no repositories.", fg=typer.colors.YELLOW)
            typer.echo(f"Add one later with:  omnisense project add-repo {project.slug} owner/name")

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# project
# --------------------------------------------------------------------------- #


@project_app.command("list")
def project_list(
    all_projects: bool = typer.Option(False, "--all", help="Include paused projects."),
) -> None:
    """Every project, with how much has been ingested for each."""

    async def run() -> None:
        await _check_database()
        service = _service()
        projects = await service.list_projects(include_inactive=all_projects)
        if not projects:
            typer.echo("No projects yet. Create one with:  omnisense init")
            return

        typer.echo("")
        for project in projects:
            sources = await service.sources(project.slug)
            total = sum(source.artifact_count for source in sources)
            paused = "" if project.is_active else "  (paused)"
            typer.secho(f"  {project.slug}{paused}", bold=True)
            typer.echo(f"    {project.name}")
            typer.echo(
                f"    {len(sources)} source{'' if len(sources) == 1 else 's'}, {total:,} artifacts"
            )
        typer.echo("")

    asyncio.run(run())


@project_app.command("show")
def project_show(slug: str) -> None:
    """One project and every source it owns."""

    async def run() -> None:
        await _check_database()
        service = _service()
        try:
            project = await service.get(slug)
            sources = await service.sources(slug)
        except NotFoundError:
            _die(f"No project called {slug!r}.", "See what exists with:  omnisense project list")
            return

        typer.echo("")
        typer.secho(f"  {project.slug}", bold=True)
        typer.echo(f"  {project.name}")
        if project.description:
            typer.echo(f"  {project.description}")
        typer.echo("")

        if not sources:
            typer.secho("  No sources attached.", fg=typer.colors.YELLOW)
            typer.echo(f"  Add one with:  omnisense project add-repo {slug} owner/name")
        else:
            for source in sources:
                # Zero is worth showing rather than hiding: it is the difference
                # between "configured" and "working", and the only way to see
                # that a repository was attached but never synced.
                count = f"{source.artifact_count:,} artifacts"
                typer.echo(f"    {source.name:<44} {count}")
        typer.echo("")

    asyncio.run(run())


@project_app.command("add-repo")
def project_add_repo(slug: str, repo: str) -> None:
    """Attach a repository to an existing project."""

    async def run() -> None:
        await _check_database()
        service = _service()
        try:
            await service.get(slug)
        except NotFoundError:
            _die(f"No project called {slug!r}.", "See what exists with:  omnisense project list")
            return
        if not await _add_repository(service, slug, repo, github_token()):
            raise typer.Exit(code=1)

    asyncio.run(run())


@project_app.command("remove-repo")
def project_remove_repo(slug: str, repo: str) -> None:
    """Detach a repository. Its history is kept."""

    async def run() -> None:
        await _check_database()
        service = _service()
        sources = await service.sources(slug)
        match = next((s for s in sources if s.name == repo or s.source_id == repo), None)
        if match is None:
            _die(f"{repo!r} is not attached to {slug!r}.")
            return
        await service.detach_source(source_id=match.source_id)
        typer.secho(f"Detached {match.name} from {slug}.", fg=typer.colors.GREEN)
        typer.echo("Its artifacts are kept — they simply stop answering project questions.")

    asyncio.run(run())


@app.command()
def sync(
    slug: str = typer.Argument(..., help="Which project to sync."),
    days: int = typer.Option(90, "--days", help="How far back a first sync reaches."),
    max_pages: int = typer.Option(20, "--max-pages", help="Page ceiling per stream."),
    stream: list[str] = typer.Option(
        [],
        "--stream",
        help=f"Sync only these streams (repeatable). One of: {', '.join(STREAMS)}.",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Forget where the last sync got to, so --days applies again.",
    ),
) -> None:
    """Read commits, pull requests, reviews and CI runs into the database.

    `--stream` exists because the four cost wildly different amounts. Reviews
    alone can want a request per pull request, so without a token -- 60 an hour --
    a full sync spends everything there and the cheap streams never run. Naming
    one makes the thing testable on a public repository.

    `--days` is the floor for a *first* sync only. After that each stream
    remembers where it got to, and that position outranks `--days` -- which is
    what stops a nightly sync re-reading a year every night, and also what makes
    a later `--days 730` silently do nothing. `--reset` clears the position so the
    wider window applies. Re-reading is safe: every write is an upsert, so it
    costs requests rather than duplicate rows.
    """
    chosen = tuple(stream) or STREAMS
    unknown = [name for name in chosen if name not in STREAMS]
    if unknown:
        _die(
            f"Unknown stream {unknown[0]!r}.",
            f"Pick from: {', '.join(STREAMS)}",
        )

    async def run() -> None:
        await _check_database()
        service = _service()
        try:
            await service.get(slug)
            source_ids = await service.resolve_source_ids(slug)
        except NotFoundError:
            _die(f"No project called {slug!r}.", "See what exists with:  omnisense project list")
            return

        if not source_ids:
            _die(
                f"{slug!r} has no repositories.",
                f"Add one with:  omnisense project add-repo {slug} owner/name",
            )
            return

        token = github_token()
        if not token:
            typer.secho(
                "No GITHUB_TOKEN — only public repositories will sync, and at a much "
                "lower rate limit (60 requests an hour rather than 5,000).",
                fg=typer.colors.YELLOW,
            )
            typer.echo("")

        from services.artifact_sync import build_artifact_sync

        syncer = build_artifact_sync(token=token)
        typer.secho(f"Syncing {slug}\n", bold=True)

        if reset:
            typer.secho(
                f"Re-reading the last {days} days from scratch — existing rows are "
                "updated in place, not duplicated.",
                fg=typer.colors.YELLOW,
            )
            typer.echo("")

        reports = await syncer.sync_project(
            source_ids,
            streams=chosen,
            max_pages=max_pages,
            backfill_days=days,
            reset=reset,
        )

        total = 0
        for report in reports:
            typer.secho(f"  {report.source_name}", bold=True)
            for stream in report.streams:
                if stream.error:
                    typer.secho(f"    {stream.stream:<15} {stream.error}", fg=typer.colors.RED)
                    continue
                # "fetched" and "written" differ when a payload was skipped as
                # unmappable -- a draft release, a pending review -- and the gap
                # is worth seeing rather than smoothing over.
                suffix = "" if stream.complete else "  (stopped at the page limit)"
                typer.echo(
                    f"    {stream.stream:<15} {stream.written:>6} stored"
                    f"  of {stream.fetched} read{suffix}"
                )
            total += report.written
            typer.echo("")

        typer.secho(f"{total:,} artifacts stored.", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"  {reports[-1].rate_limit if reports else ''}")
        # Two ways to stop early, and the advice is opposite for each: a rate
        # limit resumes on its own, an auth failure repeats forever until the
        # token changes. Telling someone with a bad token to "run it again" sends
        # them round the same loop.
        reasons = {r.stop_reason for r in reports if r.stop_reason}
        if "auth" in reasons:
            typer.secho(
                "\nStopped: GitHub refused the credentials. Running it again will "
                "not help — the token is missing, expired, or has no read access "
                "to that repository. Set GITHUB_TOKEN and try again.",
                fg=typer.colors.RED,
            )
        elif "quota" in reasons:
            typer.secho(
                "\nStopped early on the rate limit. Nothing is lost — run it again "
                "and it resumes where it left off.",
                fg=typer.colors.YELLOW,
            )

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# catchup
# --------------------------------------------------------------------------- #


@app.command()
def catchup(
    slug: str = typer.Argument(..., help="Which project to brief on."),
    since: str = typer.Option("2w", "--since", help="A duration (2w, 10d, 36h) or a date."),
    limit: int = typer.Option(
        MAX_ARTIFACTS, "--limit", help="Most artifacts to consider in one briefing."
    ),
    show_citations: bool = typer.Option(
        True, "--citations/--no-citations", help="Print what each paragraph rests on."
    ),
) -> None:
    """What happened while you were away.

    Every paragraph cites the artifacts it rests on, and those citations are
    checked against the database before printing -- a paragraph that cannot cite
    anything real is dropped rather than shown. That is not decoration: a
    narrative read off a list of commits is exactly the thing a language model
    writes fluently and wrongly, and the citation is the only part a reader can
    verify without going and looking.
    """

    async def run() -> None:
        await _check_database()

        try:
            start = parse_since(since)
        except ValueError as error:
            _die(str(error))
            return

        with _quiet_import_warnings():
            service = build_catchup_service(tenant_id="local")
        try:
            brief = await service.brief(slug, since=start, limit=limit)
        except NotFoundError:
            _die(f"No project called {slug!r}.", "See what exists with:  omnisense project list")
            return

        typer.echo("")
        typer.secho(
            f"{brief.project}  ·  {brief.since:%d %b %Y} to {brief.until:%d %b %Y}", bold=True
        )
        counted = f"{brief.considered} artifacts"
        if brief.omitted:
            # Never silently. A briefing that dropped three weeks reads exactly
            # like a briefing of a quiet three weeks.
            counted += f"  ({brief.omitted} older ones left out — raise --limit to include them)"
        typer.secho(f"  {counted}", dim=True)
        typer.echo("")

        if brief.is_empty:
            typer.echo(f"  {brief.headline}")
            typer.echo("")
            return

        typer.secho(f"  {brief.headline}", bold=True)
        typer.echo("")

        for phase in brief.phases:
            typer.secho(f"  {phase.period}  ·  {phase.label}", fg=typer.colors.CYAN, bold=True)
            for line in _wrap(phase.narrative, width=74):
                typer.echo(f"    {line}")
            if show_citations:
                for citation in phase.citations:
                    typer.secho(
                        f"      ↳ {citation.occurred_at:%d %b}  {citation.title[:58]}", dim=True
                    )
            typer.echo("")

        if brief.dropped_phases:
            # Surfaced rather than swallowed: a model that keeps inventing the
            # same phase is telling us the prompt is wrong, and a silent filter
            # would hide that from the only person who could fix it.
            typer.secho(
                f"  {len(brief.dropped_phases)} paragraph(s) dropped for citing nothing real: "
                + ", ".join(brief.dropped_phases),
                fg=typer.colors.YELLOW,
            )
            typer.echo("")

    asyncio.run(run())


def _wrap(text: str, *, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(text.split()), width=width) or [""]


@project_app.command("pause")
def project_pause(slug: str) -> None:
    """Stop a project syncing. Everything it has is kept."""

    async def run() -> None:
        await _check_database()
        try:
            await _service().set_active(slug=slug, is_active=False)
        except NotFoundError:
            _die(f"No project called {slug!r}.", "See what exists with:  omnisense project list")
            return
        typer.secho(f"Paused {slug}.", fg=typer.colors.GREEN)
        typer.echo(f"Its history is intact. Resume with:  omnisense project resume {slug}")

    asyncio.run(run())


@project_app.command("resume")
def project_resume(slug: str) -> None:
    """Start a paused project syncing again."""

    async def run() -> None:
        await _check_database()
        try:
            await _service().set_active(slug=slug, is_active=True)
        except NotFoundError:
            _die(f"No project called {slug!r}.", "See what exists with:  omnisense project list")
            return
        typer.secho(f"Resumed {slug}.", fg=typer.colors.GREEN)

    asyncio.run(run())


@project_app.command("delete")
def project_delete(
    slug: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Delete a project. Refused once it holds artifacts -- pause it instead."""

    async def run() -> None:
        await _check_database()
        service = _service()
        try:
            project = await service.get(slug)
            sources = await service.sources(slug)
        except NotFoundError:
            _die(f"No project called {slug!r}.", "See what exists with:  omnisense project list")
            return

        total = sum(source.artifact_count for source in sources)
        if not yes:
            typer.echo(f"\n  {project.slug} — {project.name}")
            typer.echo(
                f"  {len(sources)} source{'' if len(sources) == 1 else 's'}, {total:,} artifacts"
            )
            if sources:
                # Said before the prompt, because "the repositories are kept" is
                # the thing that decides whether this is frightening.
                typer.echo("\n  The repositories themselves are kept — only the grouping goes.")
            if not typer.confirm(f"\nDelete {slug}?"):
                typer.echo("Left alone.")
                return

        try:
            await service.delete(slug=slug)
        except ConflictError as error:
            _die(str(error))
            return
        typer.secho(f"Deleted {slug}.", fg=typer.colors.GREEN)

    asyncio.run(run())


def main() -> None:
    try:
        app()
    except OmniSenseError as error:
        _die(str(error))


if __name__ == "__main__":
    main()
