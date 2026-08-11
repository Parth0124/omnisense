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

# Importing every mapping module is what populates `Base.metadata`. A module that
# is not imported here contributes no table, and autogenerate -- which compares
# metadata against the live database -- reads its absence as "this table was
# dropped" and writes a migration that drops it for real.
#
# Walked rather than listed. A hardcoded list fails in the most expensive
# direction available: adding a mapping module and forgetting this line does not
# error, it makes autogenerate emit `DROP TABLE` for every table the new module
# was supposed to add -- or, before the table exists, silently omit it and leave
# the migration empty. `tests/conftest.py` walks the same package for the same
# reason.
import importlib
import pkgutil
from logging.config import fileConfig
from typing import Any

from alembic import context
from alembic.runtime.environment import NameFilterParentNames, NameFilterType
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

import models.orm
from backend.core.config import get_settings
from models.orm.base import SCHEMA, Base, TolerantEnumType

for _module in pkgutil.iter_modules(models.orm.__path__):
    importlib.import_module(f"{models.orm.__name__}.{_module.name}")

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


VERSION_TABLE = "alembic_version"
"""Alembic's revision table. Never a candidate for autogenerate. See `include_name`."""


def include_name(
    name: str | None,
    type_: NameFilterType,
    parent_names: NameFilterParentNames,
) -> bool:
    """Restrict autogenerate to the schema this project owns.

    Only `type_="schema"` is filtered. Returning False for a schema excludes
    every object inside it from the comparison, which is exactly the intent: the
    `checkpoints` schema and PostgreSQL's own `public` are not ours to diff.

    **`None` is excluded, and `do_run_migrations` is what makes that safe.**
    Reflection reports the connection's *default* schema as `None` rather than by
    name. The database user here is called `omnisense` and the server default
    `search_path` is `"$user", public`, so `current_schema()` resolved to
    `omnisense` -- our own schema arrived as `None`, this function returned False
    for it, and autogenerate compared our metadata against nothing. It concluded
    every table was missing and generated a migration that recreated the entire
    database. That is not hypothetical: it is what the first autogenerate run on
    this repository produced.

    `None` -- the connection's default schema -- is excluded, and
    `do_run_migrations` is what makes that correct: it forces our schema to be
    non-default for the run, so reflection reports it by name and this comparison
    means what it says.
    """
    if type_ == "schema":
        return name == SCHEMA
    # `alembic_version` is alembic's own bookkeeping. It normally excludes this
    # itself, but that exclusion is keyed on the version table's schema, and ours
    # was reported under the default schema rather than by name -- so autogenerate
    # saw a table present in the database and absent from our metadata, and wrote
    # `op.drop_table("alembic_version")` into upgrade(). Running that would delete
    # the record of which migrations have been applied, and the next `upgrade`
    # would try to create every table again on top of the ones already there.
    return not (type_ == "table" and name == VERSION_TABLE)


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    """Render `TolerantEnumType` as the VARCHAR it actually is.

    Autogenerate renders a custom `TypeDecorator` by its Python path, producing
    `models.orm.base.TolerantEnumType(length=32)` in the migration -- which fails
    with `NameError: name 'models' is not defined`, because migrations import
    only `sqlalchemy` and `alembic`. Adding the import would work and would be
    worse: the migration would then pin itself to a class that can be renamed or
    deleted, and a migration that cannot run against an old checkout is not a
    migration.

    `TolerantEnumType` is a `TypeDecorator` over `String` and stores a plain
    unconstrained VARCHAR -- deliberately, so that adding an enum member needs no
    migration at all. Rendering it as `sa.String` is therefore not a
    simplification; it is what the column has always been. `0001` says the same
    thing by hand, with a local `_enum()` helper.

    Returning `False` for everything else means "use the default rendering".
    """
    if type_ == "type" and isinstance(obj, TolerantEnumType):
        return f"sa.String(length={obj.impl.length})"
    return False


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
        render_item=render_item,
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
    # Make our schema non-default *for the duration of this run*, then put the
    # dialect back.
    #
    # SQLAlchemy names the connection's default schema `None` everywhere it
    # reflects. The database user here is called `omnisense`, the server
    # `search_path` is `"$user", public`, and a schema called `omnisense` exists
    # -- so our schema was the default, and reflection described every existing
    # table and foreign key without a schema while our metadata described them
    # all with one. Autogenerate compared `artifacts` against
    # `omnisense.artifacts`, matched nothing, and produced a migration that
    # recreated the entire database and dropped `alembic_version` on the way.
    #
    # `SET search_path` cannot fix this from here: the dialect resolves
    # `default_schema_name` once, when the engine first connects, and reflection
    # uses that cached value rather than asking again.
    #
    # Restored in `finally` because this connection is not always ours --
    # `scripts/init_databases.py` and the integration suite hand in a live one
    # from the application's own engine, and a dialect left mutated would change
    # how every later query in that process resolves an unqualified name.
    dialect = connection.dialect
    previous_default = dialect.default_schema_name
    if previous_default == SCHEMA:
        dialect.default_schema_name = "public"
    try:
        _configure(connection=connection)
        with context.begin_transaction():
            _ensure_schemas()
            context.run_migrations()
    finally:
        dialect.default_schema_name = previous_default


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
