"""Alembic migration environment for OmniSense.

Four decisions are encoded here, each of which is a silent failure if you get it
wrong.

**Where the URL comes from.** `alembic.ini` ships a blank `sqlalchemy.url` so no
credential is ever committed. The value is taken from
`backend.core.config.get_settings()` rather than from `os.environ` directly,
because that is the only module allowed to read the environment
(`docs/coding-standards.md` §2.9) and because it means `make migrate` picks up
the same `.env` as `make api`. Reading `DATABASE_URL` here as well would give
migrations a second, subtly different notion of "the database" -- for instance
one that skips the async-driver check in `PostgresSettings`, so `alembic upgrade`
would happily run against a psycopg2 URL that the application then refuses to
start with.

**Why the URL never goes through `config.set_main_option()`.** That writes into
the ConfigParser, which performs `%`-interpolation on read. A password containing
a literal `%` -- perfectly legal, and what a password generator will eventually
produce -- comes back mangled or raises `InterpolationSyntaxError`. Injecting it
straight into the keyword dict handed to `async_engine_from_config` skips the ini
layer entirely.

**Why the schemas are created here and not only in the revision.** Alembic calls
`_ensure_version_table()` *before* it runs the first migration. With
`version_table_schema="omnisense"` that is `CREATE TABLE omnisense.alembic_version`,
which fails on a fresh database because nothing has created the schema yet --
revision `0001` has not run. So the schemas are ensured first, from here. The
revision creates them too, and both statements are `IF NOT EXISTS`: the revision
must stand on its own for `alembic upgrade --sql` output that a DBA applies by
hand, and for a database bootstrapped without `docker/local/postgres/01-extensions.sql`.

**Why autogenerate is scoped to one schema.** `include_schemas=True` is required
for Alembic to see schema-qualified metadata at all, but it also makes reflection
sweep *every* schema on the server. `checkpoints` holds LangGraph's tables, which
no OmniSense metadata describes, so an unfiltered autogenerate would emit
`op.drop_table(..., schema="checkpoints")` and silently propose destroying the
orchestrator's state. `include_name` below confines comparison to the one schema
this project owns.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any

from alembic import context
from alembic.runtime.environment import NameFilterParentNames, NameFilterType
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from backend.core.config import get_settings

# Importing every mapping module is what populates `Base.metadata`. A module that
# is not imported here contributes no table, and autogenerate -- which compares
# metadata against the live database -- reads its absence as "this table was
# dropped" and writes a migration that drops it for real.
from models.orm import connector_account, investigation, report, run, signal  # noqa: F401
from models.orm.base import SCHEMA, Base

target_metadata = Base.metadata

CHECKPOINT_SCHEMA = "checkpoints"
"""LangGraph's checkpoint schema. Created here, never described by our metadata.

Kept out of `omnisense` so `pg_dump --schema=omnisense` captures application
state without orchestration scratch space (`models/orm/base.py`).
"""

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

database_url = get_settings().postgres.url


def include_name(
    name: str | None,
    type_: NameFilterType,
    parent_names: NameFilterParentNames,
) -> bool:
    """Restrict autogenerate to the schema this project owns.

    Only `type_="schema"` is filtered. Returning False for a schema excludes
    every object inside it from the comparison, which is exactly the intent: the
    `checkpoints` schema and PostgreSQL's own `public` are not ours to diff. The
    `None` case is the default schema, which under `include_schemas=True` is how
    reflection reports `public`.
    """
    if type_ == "schema":
        return name == SCHEMA
    return True


def _ensure_schemas() -> None:
    """Create the schemas the version table and the revisions live in.

    Idempotent, and deliberately runs before `context.run_migrations()`; see the
    module docstring for why the ordering is not negotiable.

    Skipped on any non-PostgreSQL backend. SQLite has no `CREATE SCHEMA` -- a
    schema there is an `ATTACH`ed database file, which the caller has already
    arranged if it is using one. That is what lets the migration round-trip be
    verified against SQLite without a container.
    """
    if context.get_context().dialect.name != "postgresql":
        return
    for schema in (SCHEMA, CHECKPOINT_SCHEMA):
        context.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


def _configure(**kwargs: Any) -> None:
    """Apply the options that must be identical in offline and online mode.

    Splitting them out is not tidiness. `version_table_schema` decides where
    Alembic looks for the revision it has already applied: if offline and online
    disagree, one of them reads an empty version table and re-runs every
    migration from scratch.
    """
    context.configure(
        target_metadata=target_metadata,
        # Without this, Alembic ignores the `schema="omnisense"` on every table
        # and compares our metadata against the default search_path, concluding
        # that all ten tables are missing.
        include_schemas=True,
        include_name=include_name,
        # Keep the revision table beside the tables it versions, so
        # `pg_dump --schema=omnisense` restores to a database Alembic recognizes
        # as already migrated. Left in `public`, a restore comes back with an
        # empty version table and the next `upgrade` tries to create everything
        # again.
        version_table_schema=SCHEMA,
        # Both default to False, which means a widened column or a changed
        # server default produces an empty migration and a schema that has
        # quietly drifted from the models.
        compare_type=True,
        compare_server_default=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database."""
    _configure(
        url=database_url,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        _ensure_schemas()
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    _configure(connection=connection)
    with context.begin_transaction():
        _ensure_schemas()
        context.run_migrations()


async def run_migrations_online() -> None:
    """Connect to the database and apply migrations."""
    # The programmatic entry point: `scripts/init_databases.py` and the
    # integration suite hand in a live connection so the migration runs inside a
    # transaction they control, rather than against a second engine that would
    # open its own pool against the same server.
    connection = config.attributes.get("connection")
    if connection is not None:
        do_run_migrations(connection)
        return

    configuration = dict(config.get_section(config.config_ini_section, {}))
    configuration["sqlalchemy.url"] = database_url

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        # A migration run is one connection used once. A pool would hold it open
        # after `upgrade` returns, which in a short-lived CLI process shows up as
        # an idle backend that outlives the command.
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
