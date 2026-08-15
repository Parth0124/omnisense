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
from typing import Any

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
from models.artifact import WatchStatus, source_id
from models.enums import Platform
from models.orm.artifact import SourceRow
from models.project import normalize_slug
from services.artifact_sync import STREAMS
from services.blocking_service import build_blocking_service
from services.catchup_service import MAX_ARTIFACTS, build_catchup_service, parse_since
from services.discovery_service import DiscoveryService, build_discovery_service
from services.feature_service import FeatureService, build_feature_service
from services.identity_service import IdentityService, build_identity_service
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

people_app = typer.Typer(no_args_is_help=True, help="Who is who, across platforms.")
app.add_typer(people_app, name="people")

feature_app = typer.Typer(
    no_args_is_help=True, help="Versions, features, and what belongs to them."
)
app.add_typer(feature_app, name="feature")


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
    slug: str = typer.Argument(
        "", help="Optional. A single project; omit to sync everything you watch."
    ),
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

    With no argument it syncs **everything you have approved**, across every
    project and every repository with no project at all. That is the default
    because the questions this system answers are not project-shaped -- "what
    happened last week" spans every repository you touch, and having to name one
    is the interface getting in the way of its own premise.

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
        if slug:
            service = _service()
            try:
                await service.get(slug)
                source_ids = await service.resolve_source_ids(slug)
            except NotFoundError:
                _die(
                    f"No project called {slug!r}.",
                    "See what exists with:  omnisense project list",
                )
                return
            if not source_ids:
                _die(
                    f"{slug!r} has no repositories.",
                    f"Add one with:  omnisense project add-repo {slug} owner/name",
                )
                return
        else:
            # Everything approved, project or not. `included_source_ids` is the
            # only gate: a pending source has never been decided about and a
            # skipped one has been decided against, and reading from either would
            # make the review queue decorative.
            source_ids = await _discovery_service().included_source_ids()
            if not source_ids:
                _die(
                    "Nothing to sync — no source has been approved yet.",
                    "Find some with:  omnisense discover",
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
        label = slug or f"everything you watch ({len(source_ids)} source(s))"
        typer.secho(f"Syncing {label}\n", bold=True)

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


# --------------------------------------------------------------------------- #
# people
# --------------------------------------------------------------------------- #


def _identity_service() -> IdentityService:
    return build_identity_service(tenant_id="local")


@people_app.command("list")
def people_list() -> None:
    """Every human the system knows about, and the accounts behind them."""

    async def run() -> None:
        await _check_database()
        service = _identity_service()
        identities = await service.list_identities()
        unlinked = await service.unlinked()

        if not identities and not unlinked:
            typer.echo("\nNobody yet. Sync a repository first:  omnisense sync <project>")
            return

        typer.echo("")
        for identity in identities:
            marker = "  (needs review)" if identity.needs_review else ""
            typer.secho(f"  {identity.display_name}{marker}", bold=True)
            for account in identity.accounts:
                # The method is printed, always. A link is only as good as how it
                # was arrived at, and hiding that makes a coincidence look like a
                # fact -- which is the one failure this whole feature exists to
                # prevent.
                how = "confirmed" if account.is_confirmed else f"{account.method.value} guess"
                typer.echo(f"    {account.platform.value:<10} {account.handle or '?':<22} {how}")
            typer.echo("")

        if unlinked:
            typer.secho(f"  {len(unlinked)} account(s) not attached to anyone:", dim=True)
            for person in unlinked:
                typer.echo(f"    {person.platform.value:<10} {person.handle or person.external_id}")
            typer.echo("")
            typer.echo("  Give each one an identity:  omnisense people adopt")

    asyncio.run(run())


@people_app.command("adopt")
def people_adopt() -> None:
    """Give every unattached account an identity of its own.

    Deliberately not a merge. Two accounts are two people until there is a reason
    to think otherwise, and starting from that costs nothing but a `people link`
    later -- whereas starting from a guess means somebody else's work is already
    filed under your name before anyone looks.
    """

    async def run() -> None:
        await _check_database()
        created = await _identity_service().adopt_unlinked()
        if not created:
            typer.echo("\nEvery account already belongs to somebody.")
            return
        typer.secho(
            f"\nCreated {created} identit{'y' if created == 1 else 'ies'}.", fg=typer.colors.GREEN
        )
        typer.echo("  See them with:  omnisense people list")
        typer.echo("  Merge two:      omnisense people suggest")

    asyncio.run(run())


@people_app.command("suggest")
def people_suggest() -> None:
    """Accounts that might be the same human. Proposes only -- changes nothing."""

    async def run() -> None:
        await _check_database()
        suggestions = await _identity_service().suggest()

        if not suggestions:
            typer.echo("\nNothing to suggest — every account is either attached or unmatched.")
            return

        typer.echo("")
        typer.secho(f"  {len(suggestions)} possible match(es):", bold=True)
        typer.echo("")
        for suggestion in suggestions:
            typer.secho(
                f"    {suggestion.confidence:.0%}  {suggestion.identity_name}",
                fg=typer.colors.CYAN,
            )
            # The evidence itself, not just its name. Nobody can judge
            # "0.6, handle"; anybody can judge "both are `parth`" instantly.
            typer.echo(f"        matched on {suggestion.method.value}: {suggestion.evidence}")
            typer.echo(
                f"        omnisense people link {suggestion.person_id} {suggestion.identity_id}"
            )
            typer.echo("")

        typer.secho(
            "  Nothing was changed. Merging two people who are not the same is much "
            "worse\n  than leaving them apart, so every link is yours to make.",
            dim=True,
        )

    asyncio.run(run())


@people_app.command("link")
def people_link(
    person_id: str = typer.Argument(..., help="The account to attach."),
    identity_id: str = typer.Argument(..., help="The human to attach it to."),
) -> None:
    """Attach an account to a human. Recorded as confirmed, because you said so."""

    async def run() -> None:
        await _check_database()
        try:
            await _identity_service().link(person_id=person_id, identity=identity_id)
        except NotFoundError as error:
            _die(str(error))
            return
        except ConflictError as error:
            _die(str(error))
            return
        typer.secho("\nLinked.", fg=typer.colors.GREEN)

    asyncio.run(run())


@people_app.command("unlink")
def people_unlink(person_id: str = typer.Argument(..., help="The account to detach.")) -> None:
    """Detach an account. The human and their other accounts are untouched."""

    async def run() -> None:
        await _check_database()
        try:
            await _identity_service().unlink(person_id)
        except NotFoundError:
            _die(f"{person_id!r} is not attached to anybody.")
            return
        typer.secho("\nUnlinked.", fg=typer.colors.GREEN)

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #


def _feature_service() -> FeatureService:
    return build_feature_service(tenant_id="local")


@feature_app.command("version")
def feature_version(
    project: str = typer.Argument(..., help="Which project."),
    name: str = typer.Argument(..., help="v1, v1.1, 'launch'."),
    description: str = typer.Option("", "--description", help="What this release is."),
) -> None:
    """Declare a version. A version holds features; features hold the work."""

    async def run() -> None:
        await _check_database()
        try:
            await _feature_service().add_version(
                project=project, name=name, description=description or None
            )
        except (NotFoundError, ConflictError) as error:
            _die(str(error))
            return
        typer.secho(f"\nAdded version {name}.", fg=typer.colors.GREEN)
        typer.echo(
            f'  Now add features:  omnisense feature add {project} "<name>" --version {name}'
        )

    asyncio.run(run())


@feature_app.command("add")
def feature_add(
    project: str = typer.Argument(..., help="Which project."),
    name: str = typer.Argument(..., help='What the capability is, e.g. "image upload".'),
    version: str = typer.Option("", "--version", help="Which version it belongs to."),
    keyword: list[str] = typer.Option(
        [],
        "--keyword",
        help="Extra words the commits actually use. Repeatable.",
    ),
    description: str = typer.Option("", "--description"),
) -> None:
    """Declare a feature.

    `--keyword` matters more than it looks. A feature called "image upload" will
    not match a commit that says "implemented cloudinary service" -- and that
    commit is the feature. The keyword is where you put the word the work was
    actually named after.
    """

    async def run() -> None:
        await _check_database()
        try:
            await _feature_service().add_feature(
                project=project,
                name=name,
                version=version or None,
                keywords=keyword,
                description=description or None,
            )
        except (NotFoundError, ConflictError) as error:
            _die(str(error))
            return
        typer.secho(f"\nAdded feature {name!r}.", fg=typer.colors.GREEN)
        typer.echo(f"  Attach the work to it:  omnisense feature sort {project}")

    asyncio.run(run())


@feature_app.command("sort")
def feature_sort(project: str = typer.Argument(..., help="Which project.")) -> None:
    """Work out which artifacts belong to which features.

    Safe to run after every sync. Anything you have confirmed or rejected is left
    alone -- a correction that gets undone by the next pass is one nobody makes
    twice.
    """

    async def run() -> None:
        await _check_database()
        try:
            report = await _feature_service().sort(project)
        except NotFoundError as error:
            _die(str(error))
            return

        typer.echo("")
        if report.scanned == 0:
            typer.echo("  Nothing to sort — no artifacts yet.  omnisense sync " + project)
            return
        if not report.linked and not report.already_linked:
            typer.secho(
                f"  Scanned {report.scanned} artifacts and matched none of them.",
                fg=typer.colors.YELLOW,
            )
            typer.echo(
                "  The feature names probably do not appear in the commit messages.\n"
                "  Add the words that do:  omnisense feature add ... --keyword <word>"
            )
            return

        typer.secho(f"  {report.linked} newly attached", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"  {report.already_linked} already attached")
        if report.protected:
            typer.echo(f"  {report.protected} left alone — you had decided them")
        typer.echo(f"  {report.scanned} artifacts scanned")
        typer.echo("")
        typer.echo(f"  See the result:  omnisense feature list {project}")

    asyncio.run(run())


@feature_app.command("list")
def feature_list(project: str = typer.Argument(..., help="Which project.")) -> None:
    """Versions and features, with how much work sits in each."""

    async def run() -> None:
        await _check_database()
        service = _feature_service()
        try:
            versions = await service.versions(project)
            features = await service.features(project)
        except NotFoundError as error:
            _die(str(error))
            return

        if not versions and not features:
            typer.echo("\nNothing declared yet.")
            typer.echo(f"  omnisense feature version {project} v1")
            typer.echo(f'  omnisense feature add {project} "image upload" --version v1')
            return

        typer.echo("")
        for version in versions:
            typer.secho(
                f"  {version.name}  ({version.state.value})", fg=typer.colors.CYAN, bold=True
            )
            for feature in (f for f in features if f.version_name == version.name):
                _echo_feature(feature)
            typer.echo("")

        loose = [f for f in features if f.version_name is None]
        if loose:
            typer.secho("  (no version)", dim=True)
            for feature in loose:
                _echo_feature(feature)
            typer.echo("")

    asyncio.run(run())


def _short(identifier: str) -> str:
    """Enough of an id to type, in the shape git chose for the same problem."""
    return identifier[:12]


def _echo_feature(feature: Any) -> None:
    # The guessed count is printed beside the total, always. "12 artifacts" and
    # "12 artifacts, 12 guessed" are the same number meaning very different
    # things, and only one of them is worth trusting without a look.
    detail = f"{feature.artifact_count} artifact(s)"
    if feature.guessed_count:
        detail += f", {feature.guessed_count} guessed"
    typer.echo(f"    {feature.name:<28} {detail}")


@feature_app.command("show")
def feature_show(
    project: str = typer.Argument(..., help="Which project."),
    name: str = typer.Argument(..., help="Which feature."),
) -> None:
    """Everything attached to one feature, and why."""

    async def run() -> None:
        await _check_database()
        service = _feature_service()
        try:
            features = await service.features(project)
        except NotFoundError as error:
            _die(str(error))
            return

        match = next((f for f in features if f.name.casefold() == name.casefold()), None)
        if match is None:
            _die(
                f"No feature called {name!r} in {project}.",
                f"See what exists with:  omnisense feature list {project}",
            )
            return

        members = await service.members(match.id)
        typer.echo("")
        typer.secho(f"  {match.name}", bold=True)
        typer.echo("")
        if not members:
            typer.echo("    Nothing attached yet.  omnisense feature sort " + project)
            return

        for artifact, link in members:
            mark = "✓" if link.method.is_decided else " "
            reason = "you confirmed" if link.method.is_decided else (link.evidence or "")
            # The *kind* is printed because a commit and the CI run it triggered
            # now share a title -- two rows reading "Delete deploy.yml" look like
            # a duplication bug rather than two different things.
            #
            # The id prefix is printed because the hint below tells somebody to
            # pass one, and an instruction naming an argument the screen never
            # shows is an instruction nobody can follow.
            typer.echo(
                f"    {mark} {_short(artifact.id)}  {artifact.occurred_at:%d %b}"
                f"  {artifact.kind.value:<12} {(artifact.title or '')[:36]:<36}  {reason}"
            )
        typer.echo("")
        typer.secho(
            "    ✓ means you decided it. Everything else is a guess.\n"
            f'      omnisense feature reject {project} "{match.name}" <id>\n'
            "    A few characters of the id is enough, as long as it is unique.",
            dim=True,
        )

    asyncio.run(run())


def _decide(project: str, name: str, artifact: str, belongs: bool) -> None:
    async def run() -> None:
        await _check_database()
        service = _feature_service()
        try:
            features = await service.features(project)
        except NotFoundError as error:
            _die(str(error))
            return
        match = next((f for f in features if f.name.casefold() == name.casefold()), None)
        if match is None:
            _die(f"No feature called {name!r} in {project}.")
            return
        try:
            resolved = await service.resolve_artifact(artifact)
            await service.decide(feature=match.id, artifact=resolved, belongs=belongs)
        except (NotFoundError, ConflictError) as error:
            _die(str(error))
            return
        typer.secho(
            f"\n{'Confirmed' if belongs else 'Rejected'}. This will not be undone by the "
            "next sort.",
            fg=typer.colors.GREEN,
        )

    asyncio.run(run())


@feature_app.command("confirm")
def feature_confirm(
    project: str = typer.Argument(...),
    name: str = typer.Argument(..., help="Which feature."),
    artifact: str = typer.Argument(..., help="Which artifact."),
) -> None:
    """Say an artifact does belong to a feature."""
    _decide(project, name, artifact, belongs=True)


@feature_app.command("reject")
def feature_reject(
    project: str = typer.Argument(...),
    name: str = typer.Argument(..., help="Which feature."),
    artifact: str = typer.Argument(..., help="Which artifact."),
) -> None:
    """Say an artifact does not belong to a feature. Sticks across future sorts."""
    _decide(project, name, artifact, belongs=False)


@app.command()
def blocking(
    project: str = typer.Argument(..., help="Which project."),
    target: str = typer.Argument(..., help="A version or a feature name."),
) -> None:
    """What is standing in the way of a version or a feature.

    Each blocker says how sure it is. Two of the four rules work by *absence* --
    "no review exists", "nothing has happened" -- and absence means one thing when
    the data is being collected and nothing at all when it is not. The caveat is
    printed rather than left for the reader to know.
    """

    async def run() -> None:
        await _check_database()
        try:
            report = await build_blocking_service(tenant_id="local").blocking(project, target)
        except NotFoundError as error:
            _die(str(error), f"See what exists with:  omnisense feature list {project}")
            return

        typer.echo("")
        typer.secho(f"  {report.target}", bold=True)
        typer.secho(
            f"  {report.features_checked} feature(s) checked in {report.query_ms:.0f}ms", dim=True
        )
        typer.echo("")

        if report.is_clear:
            typer.secho("  Nothing is blocking it.", fg=typer.colors.GREEN)
            typer.echo("")
            return

        for blocker in report.blockers:
            colour = typer.colors.RED if blocker.confidence >= 0.9 else typer.colors.YELLOW
            typer.secho(f"    {blocker.feature} — {blocker.summary}", fg=colour)
            if blocker.since:
                typer.echo(f"      since {blocker.since:%d %b %Y}")
            if blocker.caveat:
                # Printed at the blocker, not in a footnote. A hedge somewhere
                # else on the page is a hedge nobody reads.
                typer.secho(f"      ({blocker.caveat})", dim=True)
            typer.echo("")

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #


def _discovery_service() -> DiscoveryService:
    return build_discovery_service(token=github_token(), tenant_id="local")


@app.command()
def discover(
    include_forks: bool = typer.Option(
        False, "--include-forks", help="Also propose repositories you forked."
    ),
) -> None:
    """Find everything your token can see. Reads nothing until you approve it.

    Nothing found here is synced. Every new repository lands as *pending* and
    stays inert until you say otherwise -- a token sees the tutorial you forked
    once, the repository somebody added you to for one review, and four archived
    services, and ingesting all of it costs budget and makes every later answer
    longer and worse.
    """

    async def run() -> None:
        await _check_database()
        if not github_token():
            _die(
                "Discovery needs a GitHub token.",
                "Add GITHUB_TOKEN to .env — without it GitHub only shows public repos you own.",
            )
            return

        typer.echo("\nLooking...")
        report = await _discovery_service().discover_github(include_forks=include_forks)

        if report.error:
            typer.secho(f"\n  Stopped: {report.error}", fg=typer.colors.YELLOW)
            typer.echo(f"  Kept what it found first — {report.new} new.")
        typer.echo("")
        typer.secho(f"  {report.found} repositories visible", bold=True)
        waiting = report.new + report.still_pending
        typer.echo(f"  {report.new} new")
        if waiting:
            typer.secho(f"  {waiting} waiting on you", fg=typer.colors.YELLOW)
        typer.echo(f"  {report.already_decided} already ingesting")
        if report.previously_excluded:
            # Printed so a returning repository does not look like discovery
            # silently missed it.
            typer.echo(f"  {report.previously_excluded} you had already excluded")
        typer.echo("")
        if waiting:
            typer.echo("  Review them:  omnisense watch review")

    asyncio.run(run())


watch_app = typer.Typer(no_args_is_help=True, help="Choose what gets ingested.")
app.add_typer(watch_app, name="watch")


def _echo_candidate(candidate: Any) -> None:
    marks = []
    if candidate.private:
        marks.append("private")
    if candidate.archived:
        marks.append("archived")
    if candidate.is_dormant:
        marks.append("dormant")
    when = f"{candidate.last_activity:%b %Y}" if candidate.last_activity else "never"
    suffix = f"  ({', '.join(marks)})" if marks else ""
    typer.echo(f"    {candidate.name:<44} {when:>9}{suffix}")


@watch_app.command("review")
def watch_review() -> None:
    """Everything waiting on a decision from you."""

    async def run() -> None:
        await _check_database()
        pending = await _discovery_service().candidates(WatchStatus.PENDING)

        if not pending:
            typer.echo("\nNothing waiting. Find more with:  omnisense discover")
            return

        typer.echo("")
        typer.secho(f"  {len(pending)} waiting on you", bold=True)
        typer.secho("  newest activity first", dim=True)
        typer.echo("")
        for candidate in pending:
            _echo_candidate(candidate)
        typer.echo("")
        typer.echo("    omnisense watch add <name>       ingest this one")
        typer.echo("    omnisense watch skip <name>      never ingest it, stop asking")
        typer.echo("    omnisense watch add --all        ingest everything above")

    asyncio.run(run())


@watch_app.command("list")
def watch_list() -> None:
    """What is being ingested, and what is not."""

    async def run() -> None:
        await _check_database()
        service = _discovery_service()
        included = await service.candidates(WatchStatus.INCLUDED)
        pending = await service.candidates(WatchStatus.PENDING)
        excluded = await service.candidates(WatchStatus.EXCLUDED)

        if not (included or pending or excluded):
            typer.echo("\nNothing yet.  omnisense discover")
            return

        typer.echo("")
        typer.secho(f"  Ingesting ({len(included)})", fg=typer.colors.GREEN, bold=True)
        for candidate in included:
            _echo_candidate(candidate)
        if pending:
            typer.echo("")
            typer.secho(f"  Waiting on you ({len(pending)})", fg=typer.colors.YELLOW, bold=True)
            for candidate in pending:
                _echo_candidate(candidate)
        if excluded:
            typer.echo("")
            typer.secho(f"  Skipped ({len(excluded)})", dim=True)
            for candidate in excluded:
                _echo_candidate(candidate)
        typer.echo("")

    asyncio.run(run())


def _decide_watch(name: str | None, every: bool, include: bool) -> None:
    async def run() -> None:
        await _check_database()
        service = _discovery_service()

        if every:
            count = await service.decide_all_pending(include=include)
            verb = "Ingesting" if include else "Skipping"
            typer.secho(
                f"\n{verb} {count} repositor{'y' if count == 1 else 'ies'}.", fg=typer.colors.GREEN
            )
            return

        if not name:
            _die("Name a repository, or pass --all.")
            return

        try:
            candidate = await service.decide(source=name, include=include)
        except NotFoundError:
            _die(
                f"No source matching {name!r}.",
                "See what exists with:  omnisense watch list",
            )
            return
        verb = "Ingesting" if include else "Skipping"
        typer.secho(f"\n{verb} {candidate.name}.", fg=typer.colors.GREEN)
        if include:
            typer.echo("  Pull it in with:  omnisense sync")

    asyncio.run(run())


@watch_app.command("add")
def watch_add(
    name: str = typer.Argument("", help="owner/repo, or the start of its id."),
    every: bool = typer.Option(False, "--all", help="Everything currently pending."),
) -> None:
    """Start ingesting a source."""
    _decide_watch(name or None, every, include=True)


@watch_app.command("skip")
def watch_skip(
    name: str = typer.Argument("", help="owner/repo, or the start of its id."),
    every: bool = typer.Option(False, "--all", help="Everything currently pending."),
) -> None:
    """Never ingest a source, and stop proposing it."""
    _decide_watch(name or None, every, include=False)


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
