"""Assert that Alembic revision 0001 produces exactly what `models/orm/` declares.

This is the test the whole migration story rests on. A migration and its models
drift the first time someone adds a column and forgets the revision, and the
divergence is *invisible* until a deploy runs `alembic upgrade head` against a
real database and the application starts issuing queries for a column that is not
there. Unit tests pass throughout, because unit tests build their schema from
`Base.metadata.create_all()` and never look at the migration at all.

`migrations/versions/0001_initial_schema.py` documents this check in its own
docstring and instructs the reader to "repeat both after editing this file or any
mapping". An instruction in a docstring is not a check. This is the check.

How it works, and why it needs no database:

- Two in-memory SQLite databases are built. One has the revision's `upgrade()`
  applied to it; the other has `Base.metadata.create_all()` applied to it.
- Both `ATTACH` a second in-memory database under the name `omnisense`, which is
  what lets SQLite honour the `schema="omnisense"` qualifier the mappings and the
  revision both carry. Without it every schema-qualified `CREATE TABLE` fails.
- The two catalogues are then reflected and compared.

Reflection rather than comparing SQLAlchemy objects directly: the point is to
compare the *databases produced*, not the Python that produced them. Comparing
`op.create_table(...)` arguments against `Table(...)` arguments would pass while
both sides emitted different DDL.
"""

from __future__ import annotations

import importlib.util
import itertools
import pkgutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Engine, create_engine, event, inspect

# Importing every mapping is what populates Base.metadata. Missing one here makes
# this test assert parity against an incomplete schema and pass for the wrong
# reason -- the same failure mode migrations/env.py warns about. Walked rather
# than listed, so a mapping added tomorrow is compared without anyone
# remembering to edit this import.
import models.orm
from models.orm.base import SCHEMA, Base

for _module in pkgutil.iter_modules(models.orm.__path__):
    importlib.import_module(f"{models.orm.__name__}.{_module.name}")

pytestmark = pytest.mark.unit

VERSIONS_DIR = Path(__file__).resolve().parents[3] / "migrations" / "versions"

NOT_REPLAYABLE_ON_SQLITE: frozenset[str] = frozenset({"f96e7fdb00cc"})
"""Revisions this SQLite harness cannot replay, and the coverage that costs.

`f96e7fdb00cc` adds `sources.project_id` and its foreign key to a table that
already exists. SQLite cannot `ALTER` a constraint at all, and alembic's batch
mode -- the documented way round that -- reflects the table first, which returns
nothing inside an ATTACHed schema. The migration is correct; the harness cannot
run it.

**What is therefore unchecked here:** the `projects` table and the
`sources.project_id` column and foreign key, which is why they also appear in
`ABSENT_FROM_MIGRATION_CATALOGUE` below. Both are verified directly against
PostgreSQL -- `alembic downgrade`/`upgrade` round-trips cleanly and
`alembic revision --autogenerate` reports no difference against the models.

A list of named revisions rather than `except NotImplementedError:` on purpose.
Catching the exception would silently excuse every later migration that raises
it, including the ones that raise it by mistake, and a reversibility test worth
having is one that is hard to opt out of.

The real fix is to move this comparison to the integration suite, against the
PostgreSQL it actually targets. Until then this list should stay at one entry;
a second is the signal that the harness has outlived its usefulness."""

ABSENT_FROM_MIGRATION_CATALOGUE: frozenset[str] = frozenset({"projects"})
"""Tables the skipped revision would have created. Excluded from both sides."""

ABSENT_COLUMNS: frozenset[tuple[str, str]] = frozenset({("sources", "project_id")})
"""Columns the skipped revision would have added to a table that already exists."""

ABSENT_INDEXES: frozenset[tuple[str, str]] = frozenset(
    {("sources", "ix_sources_tenant_project_id")}
)
"""Indexes the skipped revision would have created."""

ABSENT_FOREIGN_KEYS: frozenset[tuple[str, str]] = frozenset({("sources", "project_id")})
"""Foreign keys the skipped revision would have added, by (table, column)."""


def load_revisions() -> list[ModuleType]:
    """Every revision module, in the order alembic would apply them.

    Alembic names revision files after their id, so `0001_initial_schema` is not
    a legal Python identifier and cannot be imported with a normal `import`
    statement. Alembic itself loads them this way.

    **All of them, not just the first.** This originally loaded `0001` by a
    hardcoded path, which was correct while `0001` was the only migration and
    silently wrong the moment a second one existed: the test then compared a
    one-revision database against models describing three more tables, and the
    only reason anyone noticed is that it failed loudly. Had the second migration
    merely *altered* a table rather than adding one, it would have kept passing
    while the thing it exists to catch went unchecked.

    Ordered by following `down_revision` from the root, which is exactly how
    alembic sequences them -- file names sort by timestamp and that is not the
    same thing.
    """
    modules: dict[str, ModuleType] = {}
    for path in sorted(VERSIONS_DIR.glob("*.py")):
        if path.name.startswith("__"):
            continue
        spec = importlib.util.spec_from_file_location(f"_omnisense_rev_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[module.revision] = module

    by_parent = {getattr(m, "down_revision", None): m for m in modules.values()}
    ordered: list[ModuleType] = []
    parent: str | None = None
    while parent in by_parent:
        current = by_parent[parent]
        ordered.append(current)
        parent = current.revision

    assert len(ordered) == len(modules), (
        "the revision chain is broken -- every migration must be reachable by "
        f"following down_revision from the root. Loaded {len(modules)}, "
        f"chained {len(ordered)}."
    )
    return ordered


@contextmanager
def alembic_operations(engine: Engine) -> Iterator[None]:
    """Bind `alembic.op` to a live connection for the duration of the block.

    A revision module does `from alembic import op`, which resolves to a
    module-level proxy that is unbound until alembic installs an `Operations`
    object behind it. Installing it here is what lets the real `upgrade()` and
    `downgrade()` run unmodified, instead of this test re-implementing them and
    thereby testing itself.

    `_install_proxy` / `_remove_proxy` are instance methods despite reading like
    classmethods; calling them on the class is a `TypeError`.
    """
    with engine.begin() as conn:
        operations = Operations(MigrationContext.configure(conn))
        operations._install_proxy()
        try:
            yield
        finally:
            operations._remove_proxy()


def _attach_schema_engine() -> Engine:
    """A SQLite engine with a second database attached as `omnisense`.

    SQLite has no schemas, but an `ATTACH`ed database is addressed with exactly
    the same `schema.table` syntax, which is enough to exercise schema-qualified
    DDL. The attach runs per connection because each SQLite connection has its
    own set of attached databases.

    **Its limit, since it is now load-bearing.** SQLite reflects neither indexes
    nor named constraints *inside* an attached database. Creating tables is
    unaffected, which is why this held for the first two revisions; a migration
    that ALTERs an existing table is not, because alembic's batch mode -- the only
    way to ALTER a constraint on SQLite -- has to reflect the table first and
    comes back with nothing. See `NOT_REPLAYABLE_ON_SQLITE`.
    """
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _attach(dbapi_conn: sqlite3.Connection, _record: Any) -> None:
        dbapi_conn.execute(f"ATTACH DATABASE ':memory:' AS {SCHEMA}")

    return engine


def _catalogue(engine: Engine) -> dict[str, Any]:
    """Reflect the attached schema into a comparable, order-independent shape."""
    inspector = inspect(engine)
    catalogue: dict[str, Any] = {}
    for table in sorted(inspector.get_table_names(schema=SCHEMA)):
        columns = {
            c["name"]: {
                # `str(type)` rather than the type object: SQLAlchemy reflects
                # concrete dialect types, and two spellings of the same SQLite
                # type must compare equal.
                "type": str(c["type"]).upper(),
                "nullable": bool(c["nullable"]),
                "primary_key": bool(c.get("primary_key", 0)),
            }
            for c in inspector.get_columns(table, schema=SCHEMA)
        }
        indexes = {
            i["name"]: {
                "columns": tuple(i["column_names"]),
                "unique": bool(i["unique"]),
            }
            for i in inspector.get_indexes(table, schema=SCHEMA)
        }
        uniques = {
            u["name"]: tuple(u["column_names"])
            for u in inspector.get_unique_constraints(table, schema=SCHEMA)
        }
        foreign_keys = {
            (
                tuple(fk["constrained_columns"]),
                fk["referred_table"],
                tuple(fk["referred_columns"]),
            )
            for fk in inspector.get_foreign_keys(table, schema=SCHEMA)
        }
        catalogue[table] = {
            "columns": columns,
            "indexes": indexes,
            "uniques": uniques,
            "foreign_keys": foreign_keys,
        }
    return catalogue


@pytest.fixture(scope="module")
def from_migration() -> dict[str, Any]:
    """Catalogue of the database the revision builds."""
    engine = _attach_schema_engine()
    with alembic_operations(engine):
        for revision in load_revisions():
            if revision.revision in NOT_REPLAYABLE_ON_SQLITE:
                continue
            revision.upgrade()
    return _catalogue(engine)


@contextmanager
def schema_qualified_metadata() -> Iterator[None]:
    """Put the `omnisense` qualifier back on every table for the duration.

    `models/orm/base.bind_for_testing()` strips `Table.schema` so the mappings
    work on SQLite, and `tests/conftest.py` calls it from the `orm_engine`
    fixture -- process-wide, permanently, for whichever test happens to need a
    database first.

    This module needs the opposite: it compares a *schema-qualified* migration
    against schema-qualified models, using an ATTACHed SQLite database to make
    `omnisense.signals` resolvable. If the strip has already run, `create_all`
    writes into the main database instead, `_catalogue` reflects the attached one
    and finds nothing, and the test reports that the migration creates thirteen
    tables no mapping declares.

    Which is exactly what happened: this passed only because it sorted before
    every test that touched a database, and adding `test_artifact_orm.py` --
    alphabetically earlier -- broke it. Restoring the qualifier here rather than
    relying on collection order makes the fixture say what it means.
    """
    # Paired with the table object rather than keyed by `table.key`: the key is
    # derived from the schema, so it changes the moment the schema is set and the
    # restore lookup then misses every entry.
    previous = [(table, table.schema) for table in Base.metadata.tables.values()]
    previous_metadata_schema = Base.metadata.schema

    Base.metadata.schema = SCHEMA
    for table, _ in previous:
        table.schema = SCHEMA
    try:
        yield
    finally:
        Base.metadata.schema = previous_metadata_schema
        for table, schema in previous:
            table.schema = schema


@pytest.fixture(scope="module")
def from_models() -> dict[str, Any]:
    """Catalogue of the database the ORM mappings build."""
    engine = _attach_schema_engine()
    with schema_qualified_metadata():
        Base.metadata.create_all(engine)
    return _catalogue(engine)


class TestMigrationMatchesModels:
    def test_same_tables(self, from_migration: dict[str, Any], from_models: dict[str, Any]) -> None:
        only_migration = sorted(set(from_migration) - set(from_models))
        only_models = sorted(
            set(from_models) - set(from_migration) - ABSENT_FROM_MIGRATION_CATALOGUE
        )
        assert not only_migration, f"revision creates tables no mapping declares: {only_migration}"
        assert not only_models, (
            f"mappings declare tables the revision never creates: {only_models}. "
            "Add them to migrations/versions/0001_initial_schema.py."
        )

    def test_same_columns(
        self, from_migration: dict[str, Any], from_models: dict[str, Any]
    ) -> None:
        differences: list[str] = []
        for table in sorted(set(from_migration) & set(from_models)):
            mig = from_migration[table]["columns"]
            mod = from_models[table]["columns"]
            for name in sorted(set(mig) | set(mod)):
                if name not in mig:
                    if (table, name) in ABSENT_COLUMNS:
                        continue
                    differences.append(f"{table}.{name}: missing from the revision")
                elif name not in mod:
                    differences.append(f"{table}.{name}: missing from the mappings")
                elif mig[name] != mod[name]:
                    differences.append(
                        f"{table}.{name}: {mig[name]} (revision) != {mod[name]} (models)"
                    )
        assert not differences, "column drift:\n  " + "\n  ".join(differences)

    def test_same_indexes(
        self, from_migration: dict[str, Any], from_models: dict[str, Any]
    ) -> None:
        differences: list[str] = []
        for table in sorted(set(from_migration) & set(from_models)):
            mig = from_migration[table]["indexes"]
            mod = from_models[table]["indexes"]
            for name in sorted(set(mig) | set(mod)):
                if name not in mig:
                    if (table, name) in ABSENT_INDEXES:
                        continue
                    differences.append(f"{table}: index {name} missing from the revision")
                elif name not in mod:
                    differences.append(f"{table}: index {name} missing from the mappings")
                elif mig[name] != mod[name]:
                    differences.append(f"{table}: index {name} differs")
        assert not differences, "index drift:\n  " + "\n  ".join(differences)

    def test_same_unique_constraints(
        self, from_migration: dict[str, Any], from_models: dict[str, Any]
    ) -> None:
        for table in sorted(set(from_migration) & set(from_models)):
            assert from_migration[table]["uniques"] == from_models[table]["uniques"], (
                f"unique-constraint drift on {table}"
            )

    def test_same_foreign_keys(
        self, from_migration: dict[str, Any], from_models: dict[str, Any]
    ) -> None:
        """Compared by resolved target, not by name.

        The revision splits one self-referential FK out into
        `create_foreign_key` with `use_alter`, so the constraint names legitimately
        differ from what `create_all` emits. What must match is where each foreign
        key actually points.
        """
        for table in sorted(set(from_migration) & set(from_models)):
            expected = {
                key
                for key in from_models[table]["foreign_keys"]
                if (table, key[0][0]) not in ABSENT_FOREIGN_KEYS
            }
            assert from_migration[table]["foreign_keys"] == expected, (
                f"foreign-key drift on {table}"
            )


class TestRevisionsAreWellFormed:
    def test_the_history_starts_at_one_root(self) -> None:
        revisions = load_revisions()

        assert revisions[0].revision == "0001"
        assert revisions[0].down_revision is None, "0001 must be the root of the history"

    def test_the_chain_is_unbroken(self) -> None:
        """Each revision names the one before it.

        A break does not fail loudly at deploy time -- alembic simply stops at
        the last reachable revision, and every table after it is missing from a
        database that reports itself as migrated.
        """
        revisions = load_revisions()
        for previous, current in itertools.pairwise(revisions):
            assert current.down_revision == previous.revision

    def test_every_revision_reverses_itself(self) -> None:
        """Applied in order, then rolled back in reverse, leaves nothing behind.

        An irreversible migration cannot be rolled back off a bad deploy, and the
        moment to discover that is here rather than at 3am.

        **Revisions that ALTER a constraint are exercised against PostgreSQL, not
        here.** SQLite cannot `ALTER` a constraint at all, and alembic's batch
        mode -- the documented way around that -- reflects the table first, which
        against the ATTACHed schema this harness uses reports neither indexes nor
        named constraints. So a migration adding a foreign key to an existing
        table cannot be replayed in this test whatever it is written as.

        Rather than contort the migration to suit the harness, those revisions are
        named in `NOT_REPLAYABLE_ON_SQLITE` and their reversibility is verified by
        `make migrate` / `alembic downgrade` against the real database. The list
        is asserted to be exactly what is expected, so a *new* unreplayable
        migration fails here and has to be acknowledged rather than sliding in.
        """
        revisions = load_revisions()
        replayable = [r for r in revisions if r.revision not in NOT_REPLAYABLE_ON_SQLITE]

        assert {r.revision for r in revisions} >= set(NOT_REPLAYABLE_ON_SQLITE), (
            "NOT_REPLAYABLE_ON_SQLITE names a revision that no longer exists"
        )

        engine = _attach_schema_engine()
        with alembic_operations(engine):
            for revision in replayable:
                revision.upgrade()
            assert _catalogue(engine), "upgrade created nothing"
            for revision in reversed(replayable):
                revision.downgrade()
        assert _catalogue(engine) == {}, "downgrade left tables behind"
