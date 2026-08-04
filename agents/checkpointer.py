"""Durable checkpointing, so a thirty-minute investigation survives a restart.

`docs/agent-system.md` §7: a checkpoint is written after *every node returns*,
the thread key is the `investigation_id`, and resume is invoking the graph with
the same thread and a `None` input. The unit of lost work on a crash is one agent
step -- which is the entire argument for the graph being a graph rather than one
long generation.

Four decisions, each with a failure behind it.

**The saver is built here, not in `agents/graph.py`.** Building it in the graph
would put a Postgres driver import on the path of every topology test.
`psycopg` links against `libpq` at import time and raises `ImportError` where the
library is absent -- as it is in a plain CI container -- so a graph module that
imported it could not be unit tested at all. Every Postgres import in this file
is therefore *inside* a function.

**Its own schema.** LangGraph's tables are called `checkpoints`,
`checkpoint_blobs`, `checkpoint_writes` and `checkpoint_migrations`, which is
close enough to application table names to be dangerous. `migrations/env.py`
already excludes the `checkpoints` schema from Alembic's autogenerate sweep
precisely so it does not propose dropping tables it does not own; putting the
saver anywhere else would defeat that. The schema is selected with a libpq
`options=-csearch_path=...`, because the library emits unqualified DDL and has no
schema parameter.

**`DATABASE_URL` is a SQLAlchemy URL and libpq will not accept it.**
`postgresql+asyncpg://` names a driver, not a protocol. Handing it to psycopg
produces an obscure connection error at the first checkpoint write -- that is,
after the run has started and a node has already completed. `checkpoint_dsn()`
converts it once, and is a pure function so the conversion is testable without a
database.

**Disabling it is a supported mode, and it is not free.** `AGENT_CHECKPOINT_ENABLED=false`
compiles a graph with no saver: the run works, and nothing resumes. That is the
right default for a test and a deliberate trade for a short interactive run, so
the function returns `None` rather than a silent in-memory saver -- an in-memory
saver would claim durability across a process that has none, and the discovery
would come during an incident.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Final
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from langgraph.checkpoint.memory import InMemorySaver

from backend.core.config import PostgresSettings, Settings, get_settings
from backend.core.exceptions import ConfigurationError

__all__ = [
    "CHECKPOINT_SCHEMA",
    "checkpoint_dsn",
    "checkpointer_scope",
    "memory_checkpointer",
    "thread_config",
]

CHECKPOINT_SCHEMA: Final = "checkpoints"
"""The schema LangGraph's tables live in. Mirrors `migrations/env.py`.

Duplicated as a constant rather than imported from `migrations/`: Alembic's env
module is executed by the Alembic CLI with its own path setup and is not
importable from application code. The two must agree, and the migration that
creates the schema says so in its own docstring.
"""

_ASYNCPG_ONLY_PARAMS: Final = frozenset({"ssl", "server_settings", "command_timeout"})
"""Query parameters that mean something to asyncpg and nothing to libpq.

Dropped during conversion instead of forwarded, because libpq rejects unknown
keywords outright -- and it would do so at the first checkpoint write, which is
the worst possible moment to discover a URL problem.
"""


def memory_checkpointer() -> InMemorySaver:
    """A checkpointer that keeps everything in the process.

    For tests and for `agents/evaluation/harness.py`, which replays graphs
    offline. It has the full saver semantics -- writes after every node, resume
    from a thread id -- so a test exercising resume tests the real code path, not
    a simplified one. What it does not have is durability, which is exactly why
    production never gets it by default.
    """
    return InMemorySaver()


def checkpoint_dsn(
    settings: PostgresSettings | None = None,
    *,
    schema: str = CHECKPOINT_SCHEMA,
) -> str:
    """Convert `DATABASE_URL` into a libpq DSN pinned to the checkpoint schema.

    Three transformations, each covering a way the naive version breaks:

    - `postgresql+asyncpg` -> `postgresql`. The `+driver` suffix is SQLAlchemy's
      and libpq does not parse it.
    - asyncpg-only query parameters are dropped, because libpq errors on unknown
      keywords rather than ignoring them.
    - `options=-csearch_path=<schema>` is merged in, preserving any `options`
      already present. Merging rather than overwriting matters: a deployment that
      sets a statement timeout through `options` would otherwise lose it here and
      never know.
    """
    resolved = settings if settings is not None else get_settings().postgres
    parts = urlsplit(resolved.url)

    scheme = parts.scheme.split("+", 1)[0]
    if scheme not in ("postgres", "postgresql"):
        raise ConfigurationError(
            f"DATABASE_URL names {parts.scheme!r}, which is not PostgreSQL. The "
            "LangGraph checkpointer stores its tables in the same database as the "
            "application (docs/agent-system.md §7) and has no other backend here.",
            details={"scheme": parts.scheme},
        )

    query = [
        (key, value) for key, value in parse_qsl(parts.query) if key not in _ASYNCPG_ONLY_PARAMS
    ]
    search_path = f"-c search_path={schema}"
    merged = [(key, value) for key, value in query if key != "options"]
    existing = next((value for key, value in query if key == "options"), "")
    merged.append(("options", f"{existing} {search_path}".strip() if existing else search_path))

    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(merged, quote_via=quote), ""))


def thread_config(investigation_id: str, **extra: Any) -> dict[str, Any]:
    """The LangGraph config that binds a run to its checkpoint thread.

    One thread per investigation (§7), so a checkpoint namespace maps
    one-to-one onto a row in the investigations table and `resume(id)` needs
    nothing but the id. Anything else -- a per-attempt thread, say -- would make
    a resumed run start from scratch while still looking checkpointed.
    """
    if not investigation_id:
        raise ValueError(
            "an empty investigation_id would put every run on the same checkpoint "
            "thread, so a resume would load another investigation's state."
        )
    return {"configurable": {"thread_id": investigation_id, **extra}}


@asynccontextmanager
async def checkpointer_scope(
    settings: Settings | None = None,
    *,
    setup: bool = True,
) -> AsyncIterator[Any | None]:
    """Yield the configured checkpointer for the lifetime of the block.

    A context manager because the Postgres saver owns a connection, and a
    factory that returned one would leave every caller responsible for closing
    it -- which, for a long-running worker, means a leaked connection per
    investigation until the pool is exhausted.

    Yields `None` when `AGENT_CHECKPOINT_ENABLED` is false;
    `agents/graph.py` accepts that directly and compiles an ephemeral graph.

    `setup=True` runs LangGraph's own migrations. They are `CREATE TABLE IF NOT
    EXISTS`, so running them on every worker start is cheap and idempotent, and
    it is the only thing standing between a fresh deployment and a first
    checkpoint write that fails on a missing table. The schema itself is *not*
    created here: `migrations/versions/0001_initial_schema.py` owns it, and a
    worker that could create schemas is a worker with more grants than it needs.
    """
    resolved = settings if settings is not None else get_settings()
    if not resolved.agents.checkpoint_enabled:
        yield None
        return

    # Imported here, not at module scope: see the module docstring. `psycopg`
    # needs libpq present at import time, and the topology tests must run in a
    # process that has neither.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(checkpoint_dsn(resolved.postgres)) as saver:
        if setup:
            await saver.setup()
        yield saver
