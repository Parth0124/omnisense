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
from collections.abc import Iterator
from contextlib import contextmanager
import sqlite3
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Engine, create_engine, event, inspect

from models.orm.base import SCHEMA, Base

# Importing every mapping is what populates Base.metadata. Missing one here makes
# this test assert parity against an incomplete schema and pass for the wrong
# reason -- the same failure mode migrations/env.py warns about.
from models.orm import (  # noqa: F401  (imported for the metadata side effect)
    connector_account,
    investigation,
    report,
    run,
    signal,
)

pytestmark = pytest.mark.unit

REVISION_PATH = (
    Path(__file__).resolve().parents[3] / "migrations" / "versions" / "0001_initial_schema.py"
)


def load_revision() -> ModuleType:
    """Import the revision by path.

    Alembic names revision files after their id, so `0001_initial_schema` is not
    a legal Python identifier and cannot be imported with a normal `import`
    statement. Alembic itself loads them this way.
    """
    spec = importlib.util.spec_from_file_location("_omnisense_revision_0001", REVISION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    revision = load_revision()

    engine = _attach_schema_engine()
    with alembic_operations(engine):
        revision.upgrade()
    return _catalogue(engine)


@pytest.fixture(scope="module")
def from_models() -> dict[str, Any]:
    """Catalogue of the database the ORM mappings build."""
    engine = _attach_schema_engine()
    Base.metadata.create_all(engine)
    return _catalogue(engine)


class TestMigrationMatchesModels:
    def test_same_tables(
        self, from_migration: dict[str, Any], from_models: dict[str, Any]
    ) -> None:
        only_migration = sorted(set(from_migration) - set(from_models))
        only_models = sorted(set(from_models) - set(from_migration))
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
                    differences.append(f"{table}.{name}: missing from the revision")
                elif name not in mod:
                    differences.append(f"{table}.{name}: missing from the mappings")
                elif mig[name] != mod[name]:
                    differences.append(f"{table}.{name}: {mig[name]} (revision) != {mod[name]} (models)")
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
            assert from_migration[table]["foreign_keys"] == from_models[table]["foreign_keys"], (
                f"foreign-key drift on {table}"
            )


class TestRevisionIsWellFormed:
    def test_is_the_base_revision(self) -> None:
        revision = load_revision()

        assert revision.revision == "0001"
        assert revision.down_revision is None, "0001 must be the root of the history"

    def test_downgrade_drops_everything_it_created(
        self, from_models: dict[str, Any]
    ) -> None:
        """An irreversible first migration cannot be rolled back off a bad deploy."""
        revision = load_revision()

        engine = _attach_schema_engine()
        with alembic_operations(engine):
            revision.upgrade()
            assert _catalogue(engine), "upgrade created nothing"
            revision.downgrade()
        assert _catalogue(engine) == {}, "downgrade left tables behind"
