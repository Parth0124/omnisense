"""SQLAlchemy declarative base, naming conventions and portable column types.

Two decisions here are worth more than they look.

**The naming convention.** Without one, Alembic autogenerates constraint names
from PostgreSQL's defaults, which differ between "created by `CREATE TABLE`" and
"created by `ALTER TABLE`". The result is a migration that cannot drop the
constraint it just created, discovered only on the downgrade path months later.
Naming every constraint deterministically makes migrations reversible.

**`JSONVariant` and `UUIDVariant`.** Production is PostgreSQL, but unit tests
must not require Docker (`docs/testing-strategy.md`). These types compile to
`JSONB`/`UUID` on PostgreSQL and to portable equivalents on SQLite, so the same
mappings can be exercised in-memory in milliseconds and against real PostgreSQL
in the integration suite. The variant is chosen by the dialect at DDL-compile
time, so nothing branches at runtime.
"""

from __future__ import annotations

import enum
from typing import Any

from sqlalchemy import JSON, MetaData, String, TypeDecorator
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase

__all__ = [
    "Base",
    "JSONVariant",
    "TolerantEnumType",
    "UUIDVariant",
    "metadata",
]


NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_N_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

SCHEMA = "omnisense"
"""Application tables live here; LangGraph checkpoints live in `checkpoints`.

Both schemas are created by `docker/local/postgres/01-extensions.sql` on first
container start. Keeping them apart means `pg_dump --schema=omnisense` captures
application state without dragging along orchestration scratch space.

Set to `None` when running against SQLite, which has no schema concept -- see
`bind_for_testing()`.
"""

metadata = MetaData(naming_convention=NAMING_CONVENTION, schema=SCHEMA)


JSONVariant = JSON().with_variant(postgresql.JSONB(astext_type=String()), "postgresql")
"""`JSONB` on PostgreSQL, plain `JSON` elsewhere.

JSONB is the right production choice -- it is binary, indexable with GIN, and
`btree_gin` is installed by the bootstrap script specifically to support
composite metadata filters. SQLite gets `JSON`, which is stored as text and is
adequate for tests.
"""

UUIDVariant = String(36).with_variant(postgresql.UUID(as_uuid=False), "postgresql")
"""Native `UUID` on PostgreSQL, `VARCHAR(36)` elsewhere.

`as_uuid=False` keeps the Python side a plain `str`. OmniSense identifiers are
prefixed strings (`sig_...`, `inv_...`) rather than bare UUIDs at the domain
level, and converting back and forth would invite exactly the confusion
`docs/api-reference.md` §3.2 warns about.
"""


class TolerantEnumType(TypeDecorator[Any]):
    """Stores a `TolerantStrEnum` as `VARCHAR`, reads it back as the enum member.

    Deliberately **not** `sqlalchemy.Enum`. Both the native PostgreSQL enum and
    `Enum(native_enum=False)` constrain the column -- a `CREATE TYPE` or a `CHECK`
    listing the members. Either one means adding a new `Platform` requires a
    migration before a single row can be written, which contradicts the design in
    `models/base.py`: `TolerantStrEnum` exists precisely so that adding a member
    is a backward-compatible change requiring no coordinated deploy.

    So the column is a plain `VARCHAR` with no constraint, and tolerance is
    applied on read: an unrecognized value from a newer writer degrades to
    `UNKNOWN` rather than raising, exactly as it does over Kafka.

    `cache_ok = True` lets SQLAlchemy cache compiled statements using this type;
    it is safe because the type is fully described by `enum_cls` and `length`.
    """

    impl = String
    cache_ok = True

    def __init__(self, enum_cls: type[enum.Enum], length: int = 32, **kwargs: Any) -> None:
        self.enum_cls = enum_cls
        super().__init__(length=length, **kwargs)

    def process_bind_param(self, value: Any, dialect: Dialect) -> str | None:
        if value is None:
            return None
        return value.value if isinstance(value, enum.Enum) else str(value)

    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        return self.enum_cls(value)


class Base(DeclarativeBase):
    """Declarative base for every OmniSense table.

    `models/orm/` is the only part of `models/` that knows a database exists.
    ORM types are persistence detail and must not appear in connector, service or
    agent signatures -- conversion happens at the service boundary, so swapping
    the storage layer never reaches the domain (`models/README.md`).
    """

    metadata = metadata

    type_annotation_map: dict[Any, Any] = {
        dict[str, Any]: JSONVariant,
        list[str]: JSONVariant,
    }

    def __repr__(self) -> str:
        pk = self.__mapper__.primary_key
        values = ", ".join(f"{c.name}={getattr(self, c.name)!r}" for c in pk)
        return f"<{type(self).__name__} {values}>"


def bind_for_testing() -> None:
    """Strip the schema qualifier so the mappings work on SQLite.

    SQLite has no schemas: a table declared in schema `omnisense` becomes
    `omnisense.signals`, which SQLite reads as a *database* qualifier and fails
    to resolve. Called by the unit-test fixture before `create_all`; never called
    in application code.
    """
    metadata.schema = None
    for table in metadata.tables.values():
        table.schema = None
