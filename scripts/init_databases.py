"""Bring a fresh local stack to a usable state: schemas, indexes, collections, constraints.

`make init-db` is the first command a new engineer runs after `make up`, which
fixes two requirements that shape everything below.

**It must be idempotent.** Every step is a create-if-absent, so running it twice,
or running it after adding one new store to a half-built stack, converges instead
of failing. That is also what makes it safe to wire into a container entrypoint
later.

**It must never end in a traceback.** A store that is not up yet is the single
most likely outcome of running this on day one -- the OpenSearch container takes
the better part of a minute to go from "started" to "answering" -- and a wall of
Python frames tells a newcomer that the project is broken rather than that a
container is still booting. So each store is probed first, each step is isolated,
and the run finishes with a summary naming exactly what happened and what to do
about it. The exit code is still non-zero when something did not get set up:
silence on failure would be worse than the traceback.

The heavy lifting belongs to `backend/db/`, not here. `ensure_collection()` and
`ensure_index()` already encode the geometry checks and the create-race handling
(`backend/db/qdrant.py`, `backend/db/opensearch.py`); duplicating any of that in a
script would mean a bootstrap that disagrees with what the workers do at startup.
This module is the ordering, the reporting and the CLI around them.

Layer note: `scripts/` sits above the kernel (`docs/architecture.md` §6.1) and may
import `backend/`, `models/` and `services/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running this as `python scripts/init_databases.py` -- which is what
# `make init-db` and `scripts/README.md` both tell you to do -- puts `scripts/`
# on `sys.path`, not the repository root, so every first-party import below
# would fail with `ModuleNotFoundError: No module named 'backend'`. OmniSense is
# deliberately not an installed package (`pyproject.toml`), so there is no
# console-script entry point to inherit the right path from; the script has to
# put the root there itself. Written inline so it stays before the imports it
# enables.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import contextlib
import enum
import logging
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass

import typer
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from backend.core.config import get_settings
from backend.core.exceptions import ConfigurationError, OmniSenseError
from backend.core.logging import configure_logging
from backend.db.neo4j import check_neo4j, dispose_driver, read_session, write_session
from backend.db.opensearch import check_opensearch, dispose_opensearch, ensure_index
from backend.db.qdrant import check_qdrant, dispose_qdrant, ensure_collection
from backend.db.session import check_postgres, dispose_engine, get_engine
from models.orm.base import SCHEMA

__all__ = ["Outcome", "StepResult", "app", "cypher_statements", "main"]

REPO_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = REPO_ROOT / "migrations" / "alembic.ini"
ALEMBIC_SCRIPTS = REPO_ROOT / "migrations"
NEO4J_BOOTSTRAP = REPO_ROOT / "docker" / "local" / "neo4j" / "01-constraints.cypher"

VERSION_TABLE = "alembic_version"
"""Alembic's default revision table, restated because this module reads it directly.

It lives in `omnisense` rather than `public`; see `version_table_schema` in
`migrations/env.py`. Reading it is how this script can say "already at 0001"
instead of "ran the migration" on the second run.
"""

app = typer.Typer(
    add_completion=False,
    help="Create the schemas, indexes, collections and constraints a fresh stack needs.",
)


class Outcome(enum.StrEnum):
    """What one step actually did. Drives both the summary and the exit code."""

    CREATED = "created"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"
    UNREACHABLE = "unreachable"
    FAILED = "failed"

    @property
    def is_problem(self) -> bool:
        """Whether this outcome should make the process exit non-zero.

        `SKIPPED` is not a problem: the operator asked for it. `UNREACHABLE` is,
        even though it is the expected day-one result -- `make init-db` claims to
        have set the stack up, and it has not.
        """
        return self in (Outcome.UNREACHABLE, Outcome.FAILED)

    @property
    def colour(self) -> str:
        return {
            Outcome.CREATED: typer.colors.GREEN,
            Outcome.UNCHANGED: typer.colors.CYAN,
            Outcome.SKIPPED: typer.colors.BRIGHT_BLACK,
            Outcome.UNREACHABLE: typer.colors.YELLOW,
            Outcome.FAILED: typer.colors.RED,
        }[self]


@dataclass(frozen=True, slots=True)
class StepResult:
    """One store's bootstrap outcome, with a line a human can act on."""

    store: str
    outcome: Outcome
    detail: str

    def render(self) -> str:
        return f"  {self.store:<12} {self.outcome.value:<12} {self.detail}"


# --------------------------------------------------------------------------- #
# PostgreSQL
# --------------------------------------------------------------------------- #


def _alembic_config() -> Config:
    """Alembic config with absolute paths.

    `alembic.ini` sets `script_location = migrations`, which is resolved against
    the *current working directory*. That is correct for `make migrate`, which
    runs from the repository root, and wrong for this script, which anyone may
    run from anywhere. Overriding it with an absolute path removes the
    dependency on where you happened to be standing.
    """
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ALEMBIC_SCRIPTS))
    return config


async def _current_revision() -> str | None:
    """The revision the database believes it is at, or None if never migrated.

    Existence of the version table is checked with the inspector rather than by
    catching the error from selecting it. On PostgreSQL a failed statement
    poisons the whole transaction, so "just try the SELECT" would leave the
    connection unusable for the next question.
    """
    async with get_engine().connect() as connection:
        exists = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).has_table(VERSION_TABLE, schema=SCHEMA)
        )
        if not exists:
            return None
        result = await connection.execute(
            text(f'SELECT version_num FROM "{SCHEMA}".{VERSION_TABLE}')
        )
        revision: str | None = result.scalar()
        return revision


async def _migrate_postgres() -> tuple[Outcome, str]:
    """Run `alembic upgrade head`, reporting the revision it moved to.

    Alembic's `env.py` drives an async engine through `asyncio.run()`, which
    cannot be called from inside a running event loop. `to_thread` gives it a
    thread with no loop of its own, which is the whole reason the migration does
    not simply run inline here.
    """
    before = await _current_revision()
    await asyncio.to_thread(command.upgrade, _alembic_config(), "head")
    after = await _current_revision()

    if after is None:
        raise ConfigurationError(
            "alembic reported success but no revision was recorded in "
            f'"{SCHEMA}".{VERSION_TABLE}; the migration did not take effect.'
        )
    if before == after:
        return Outcome.UNCHANGED, f"already at revision {after}"
    return Outcome.CREATED, f"{before or 'empty database'} -> revision {after}"


# --------------------------------------------------------------------------- #
# Qdrant / OpenSearch
# --------------------------------------------------------------------------- #


async def _ensure_qdrant_collection() -> tuple[Outcome, str]:
    settings = get_settings()
    name = settings.qdrant.collection
    geometry = f"size={settings.embedding.dimensions} distance={settings.qdrant.distance.value}"
    if await ensure_collection():
        return Outcome.CREATED, f"collection {name!r} ({geometry})"
    # `ensure_collection()` verified the live geometry against this process's
    # configuration before returning False, so "unchanged" here means "unchanged
    # and correct", not merely "present".
    return Outcome.UNCHANGED, f"collection {name!r} already present ({geometry})"


async def _ensure_opensearch_index(replicas: int) -> tuple[Outcome, str]:
    name = get_settings().opensearch.signal_index
    if await ensure_index(number_of_replicas=replicas):
        return Outcome.CREATED, f"index {name!r} (replicas={replicas})"
    # An existing index is left exactly as it is -- see `ensure_index()`: a
    # mapping change is a reindex plus an alias swap, never a partial in-place
    # patch applied by a bootstrap script.
    return Outcome.UNCHANGED, f"index {name!r} already present, mapping left untouched"


# --------------------------------------------------------------------------- #
# Neo4j
# --------------------------------------------------------------------------- #


def cypher_statements(source: str) -> list[str]:
    """Split a `.cypher` file into individually executable statements.

    The Bolt driver runs one statement per call, so the file has to be split.
    Scanned character by character rather than `source.split(";")` because a `;`
    inside a string literal does not end a statement and a `//` inside one does
    not start a comment. Neither appears in the bootstrap file today, and both
    will the first time somebody stores a property containing a URL.
    """
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0
    length = len(source)

    while index < length:
        char = source[index]

        if quote is not None:
            current.append(char)
            if char == "\\" and index + 1 < length:
                current.append(source[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue

        if char in "'\"`":
            quote = char
            current.append(char)
            index += 1
            continue

        if source.startswith("//", index):
            newline = source.find("\n", index)
            index = length if newline == -1 else newline
            continue

        if source.startswith("/*", index):
            close = source.find("*/", index)
            index = length if close == -1 else close + 2
            continue

        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


async def _graph_schema_counts() -> tuple[int, int]:
    """How many constraints and indexes the graph currently has.

    Cheaper and more honest than trying to infer what changed from the Cypher:
    every statement in the bootstrap file is `IF NOT EXISTS`, so the server's
    own before/after counts are the only way to say whether anything was
    actually created.
    """
    async with read_session() as session:
        constraints = await session.run("SHOW CONSTRAINTS YIELD name RETURN count(name) AS n")
        constraint_count = (await constraints.single(strict=True))["n"]
        indexes = await session.run("SHOW INDEXES YIELD name RETURN count(name) AS n")
        index_count = (await indexes.single(strict=True))["n"]
    return int(constraint_count), int(index_count)


async def _apply_graph_constraints() -> tuple[Outcome, str]:
    """Apply `docker/local/neo4j/01-constraints.cypher` through the driver.

    Neo4j does not run this file on container start the way PostgreSQL runs
    `01-extensions.sql` (`docs/data-stores.md` §3.2), which is precisely why it
    is this script's job.

    Statements go through `write_session()` with autocommit rather than through
    `run_write()`. Schema commands are not data commands: Neo4j wants them
    outside a data transaction, and the managed-transaction retry that
    `run_write()` provides buys nothing for statements that are already
    `IF NOT EXISTS`.
    """
    if not NEO4J_BOOTSTRAP.is_file():
        raise ConfigurationError(
            f"Neo4j bootstrap file is missing: {NEO4J_BOOTSTRAP}",
            details={"path": str(NEO4J_BOOTSTRAP)},
        )

    statements = cypher_statements(NEO4J_BOOTSTRAP.read_text(encoding="utf-8"))
    if not statements:
        return Outcome.UNCHANGED, f"{NEO4J_BOOTSTRAP.name} contains no statements"

    constraints_before, indexes_before = await _graph_schema_counts()
    async with write_session() as session:
        for statement in statements:
            result = await session.run(statement)
            await result.consume()
    constraints_after, indexes_after = await _graph_schema_counts()

    detail = (
        f"{len(statements)} statements from {NEO4J_BOOTSTRAP.name}; "
        f"constraints {constraints_before} -> {constraints_after}, "
        f"indexes {indexes_before} -> {indexes_after}"
    )
    changed = constraints_after > constraints_before or indexes_after > indexes_before
    return (Outcome.CREATED if changed else Outcome.UNCHANGED), detail


# --------------------------------------------------------------------------- #
# Step runner
# --------------------------------------------------------------------------- #


async def _run_step(
    store: str,
    *,
    enabled: bool,
    probe: Callable[[], Awaitable[bool]],
    action: Callable[[], Awaitable[tuple[Outcome, str]]],
    hint: str,
) -> StepResult:
    """Run one store's bootstrap, converting every failure into a reportable line.

    The probe runs first so that the overwhelmingly common failure -- the
    container is not up yet -- is reported as `unreachable` with a hint, rather
    than as whatever transport exception the client happens to raise.

    The bare `except Exception` is deliberate here and would not be acceptable in
    a request path. This function's entire contract is that no step can abort the
    run, and an unexpected exception is exactly the case that contract exists
    for. The exception's class name is kept in the message so a genuine bug is
    still identifiable rather than flattened into "failed".
    """
    if not enabled:
        return StepResult(store, Outcome.SKIPPED, "disabled by flag")

    if not await probe():
        return StepResult(store, Outcome.UNREACHABLE, hint)

    try:
        outcome, detail = await action()
    except OmniSenseError as exc:
        return StepResult(store, Outcome.FAILED, exc.message)
    # Broad on purpose -- see the docstring.
    except Exception as exc:
        return StepResult(store, Outcome.FAILED, f"{type(exc).__name__}: {exc}")
    return StepResult(store, outcome, detail)


async def _bootstrap(
    *,
    postgres: bool,
    qdrant: bool,
    opensearch: bool,
    neo4j: bool,
    opensearch_replicas: int,
) -> list[StepResult]:
    """Run every enabled step in turn, printing each result as it lands.

    Sequential rather than gathered. The steps are seconds long, the output is
    meant to be read top to bottom, and interleaving four stores' progress lines
    would make a partial stack harder to diagnose, not easier.
    """
    settings = get_settings()
    results: list[StepResult] = []

    steps = (
        _run_step(
            "postgres",
            enabled=postgres,
            probe=check_postgres,
            action=_migrate_postgres,
            hint=(
                "not answering at the configured DATABASE_URL; "
                "start it with `make up`, or re-run with --no-postgres"
            ),
        ),
        _run_step(
            "qdrant",
            enabled=qdrant,
            probe=check_qdrant,
            action=_ensure_qdrant_collection,
            hint=(
                f"not answering at {settings.qdrant.url}; "
                "start it with `make up`, or re-run with --no-qdrant"
            ),
        ),
        _run_step(
            "opensearch",
            enabled=opensearch,
            probe=check_opensearch,
            action=lambda: _ensure_opensearch_index(opensearch_replicas),
            hint=(
                f"not answering at {settings.opensearch.url}; it takes ~60s to "
                "become healthy after `make up`, or re-run with --no-opensearch"
            ),
        ),
        _run_step(
            "neo4j",
            enabled=neo4j,
            probe=check_neo4j,
            action=_apply_graph_constraints,
            hint=(
                f"not answering at {settings.neo4j.uri}; "
                "start it with `make up`, or re-run with --no-neo4j"
            ),
        ),
    )

    for step in steps:
        result = await step
        typer.secho(result.render(), fg=result.outcome.colour)
        results.append(result)

    return results


NOISY_TRANSPORT_LOGGERS = ("opensearch",)
"""Third-party loggers muted for the duration of the run.

`opensearchpy` logs a failed request at WARNING with `exc_info=True`, once per
attempt. Under the console renderer configured by `backend/core/logging.py` that
one warning expands into a full rich traceback -- roughly eighty kilobytes of
frames for the single most likely outcome of running this script, "the container
is not up yet". The step line immediately below it already says `opensearch
unreachable` and what to do about it, so the traceback is not additional
information; it is the exact first impression this module exists to prevent.

Only WARNING and below are suppressed. A genuine error still prints, and every
failure is reported in the summary regardless, because `_run_step()` puts the
exception's type and message into the result line.

Add a logger here only after seeing it produce this specific failure mode --
muting one on suspicion trades a real diagnostic for tidiness.
"""


@contextlib.contextmanager
def _quiet_transport_logging() -> Iterator[None]:
    """Raise the noisy loggers to ERROR, then restore whatever they were on.

    Restored rather than left raised because this module is importable: a test
    or another script that calls `main()` must not inherit a permanently muted
    OpenSearch logger as a side effect.
    """
    previous = {name: logging.getLogger(name).level for name in NOISY_TRANSPORT_LOGGERS}
    for name in NOISY_TRANSPORT_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)


async def _dispose_everything() -> None:
    """Close every client this run may have opened.

    Unconditional and individually guarded: a client that was never built
    disposes to a no-op, and one that fails to close must not stop the others
    from closing. Skipping this leaves aiohttp sessions and Bolt connections to
    be torn down at interpreter exit, which surfaces as `ResourceWarning` noise
    after the summary -- the last thing the reader sees, and entirely spurious.
    """
    for close in (dispose_engine, dispose_qdrant, dispose_opensearch, dispose_driver):
        try:
            await close()
        # Broad on purpose: shutdown noise must not fail the run.
        except Exception as exc:
            typer.secho(f"  warning: {close.__name__} failed: {exc}", fg=typer.colors.YELLOW)


@app.command()
def main(
    postgres: bool = typer.Option(
        True, "--postgres/--no-postgres", help="Run `alembic upgrade head`."
    ),
    qdrant: bool = typer.Option(
        True, "--qdrant/--no-qdrant", help="Ensure the Qdrant collection exists."
    ),
    opensearch: bool = typer.Option(
        True, "--opensearch/--no-opensearch", help="Ensure the OpenSearch signal index exists."
    ),
    neo4j: bool = typer.Option(
        True, "--neo4j/--no-neo4j", help="Apply the Neo4j constraints and indexes."
    ),
    opensearch_replicas: int = typer.Option(
        1,
        "--opensearch-replicas",
        min=0,
        help=(
            "Replica count for a newly created index. The local single-node "
            "cluster cannot allocate a replica and sits yellow; pass 0 to keep "
            "it green. Ignored for an index that already exists."
        ),
    ),
) -> None:
    """Create everything a fresh OmniSense stack needs. Safe to re-run."""
    configure_logging()

    typer.secho("Initializing OmniSense datastores", bold=True)

    async def _run() -> list[StepResult]:
        try:
            return await _bootstrap(
                postgres=postgres,
                qdrant=qdrant,
                opensearch=opensearch,
                neo4j=neo4j,
                opensearch_replicas=opensearch_replicas,
            )
        finally:
            await _dispose_everything()

    with _quiet_transport_logging():
        results = asyncio.run(_run())

    problems = [result for result in results if result.outcome.is_problem]
    if not problems:
        # "Everything is ready" would be a lie when the operator turned steps
        # off, and this is the line somebody quotes back during an incident.
        skipped = sum(1 for result in results if result.outcome is Outcome.SKIPPED)
        done = len(results) - skipped
        message = f"\n{done} of {len(results)} datastores are ready"
        message += f" ({skipped} skipped by flag)." if skipped else "."
        typer.secho(message, fg=typer.colors.GREEN, bold=True)
        return

    typer.secho(
        f"\n{len(problems)} of {len(results)} steps did not complete:",
        fg=typer.colors.RED,
        bold=True,
    )
    for result in problems:
        typer.secho(f"  {result.store}: {result.detail}", fg=result.outcome.colour)
    typer.secho(
        "Nothing was left half-applied -- every step is idempotent, so fix the "
        "cause and run `make init-db` again.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
