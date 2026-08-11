"""Shared pytest fixtures: the in-memory database every suite maps against.

`docs/testing-strategy.md` fixes the unit suite as "fast, isolated, no external
services", and `models/orm/base.py` was written to honour that -- `JSONVariant`
and `UUIDVariant` compile to portable types precisely so the mappings can be
exercised against SQLite in milliseconds instead of against a container.

The fixtures live here rather than in `tests/unit/models/test_orm.py` because a
database session is not an ORM-test concern: `services/`, `backend/api/` and the
graph nodes all need one the moment they are tested, and a second copy of this
setup is a second place for the `PRAGMA foreign_keys` line to be forgotten.

Nothing here touches `backend.core.config` or the process-wide engine in
`backend/db/session.py`. That engine is the application's; tests that need to
redirect *it* do so explicitly (see `tests/unit/backend/test_postgres.py`), and
mutating it from a globally-visible fixture would leak into every test that runs
after.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from pydantic_settings import BaseSettings
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import models.orm
from models.orm.base import Base, bind_for_testing

__all__ = ["db_session", "orm_engine"]

SQLITE_MEMORY_URL = "sqlite+aiosqlite://"
"""In-memory database, private to one connection.

The aiosqlite dialect answers this URL with a `StaticPool`, which hands out the
same connection for the life of the engine. That is required, not incidental:
under a pool that opens a second connection, the second one gets its own empty
`:memory:` database and every table `create_all` just made is invisible to it.
"""


@pytest.fixture(autouse=True, scope="session")
def _ignore_the_developers_env_file() -> Iterator[None]:
    """Stop every settings group from reading the repo's real `.env`.

    `backend/core/config.py` composes one `Settings` out of thirteen independent
    `BaseSettings` groups, each built by its own `default_factory` and each
    carrying `env_file` in its own `model_config`. So `Settings(_env_file=None)`
    suppresses the file for the outer object and for nothing inside it -- every
    group goes and reads `.env` regardless.

    That made the suite pass or fail depending on whether the person running it
    had a `.env`, and on what was in it: a test asserting the *declared default*
    of `LLM_MODEL_PLANNER` saw whatever the developer had configured instead.
    Nobody noticed because CI has no `.env` and neither did a fresh clone --
    it only appears the moment someone runs `make env`, which is the first
    instruction in the getting-started guide.

    Session-scoped and autouse because the failure is environmental: any test
    that reads settings is exposed, and remembering to opt in is exactly the
    thing that will not happen.
    """
    import backend.core.config as config

    patched: list[dict[str, object]] = []
    for name in dir(config):
        candidate = getattr(config, name)
        if (
            isinstance(candidate, type)
            and issubclass(candidate, BaseSettings)
            and candidate is not BaseSettings
        ):
            patched.append(candidate.model_config)
            candidate.model_config["env_file"] = None

    config.get_settings.cache_clear()
    try:
        yield
    finally:
        for model_config in patched:
            model_config["env_file"] = config._ENV_FILE
        config.get_settings.cache_clear()


def _import_every_orm_module() -> None:
    """Import all of `models/orm/` so `Base.metadata` describes every table.

    A mapped class registers itself with the metadata as a side effect of its
    module being imported, and `models/orm/__init__.py` deliberately re-exports
    nothing. Walking the package instead of listing the modules means a table
    added tomorrow is created -- and therefore tested -- without anyone
    remembering to edit this file. A hardcoded list fails silently: the new
    table simply never appears, and its tests fail with "no such table" pointing
    at the wrong place entirely.
    """
    for module in pkgutil.iter_modules(models.orm.__path__):
        importlib.import_module(f"{models.orm.__name__}.{module.name}")


@pytest.fixture
async def orm_engine() -> AsyncIterator[AsyncEngine]:
    """A fresh SQLite database with every ORM table created.

    Function-scoped on purpose. A per-test database costs a few milliseconds of
    `CREATE TABLE` and buys total isolation -- no truncation between tests, no
    ordering dependency, and a failing test cannot poison the next one.

    Two lines here are load-bearing.

    `bind_for_testing()` strips the `omnisense` schema qualifier. SQLite reads
    `omnisense.signals` as *database* `omnisense`, which is not attached, so
    without it `create_all` fails on the first table.

    `PRAGMA foreign_keys=ON` is off by default in every SQLite build, for
    backwards compatibility. With it off, `REFERENCES ... ON DELETE CASCADE` is
    parsed, stored in the schema, and then ignored at runtime: orphan rows insert
    happily and deleting a parent leaves its children behind. Every foreign-key
    assertion in the suite would then be measuring SQLite's indifference rather
    than the schema. It is set from a `connect` listener rather than executed
    once, because the pragma is per connection and lasts only as long as the one
    it was issued on.
    """
    _import_every_orm_module()
    bind_for_testing()

    engine = create_async_engine(SQLITE_MEMORY_URL)

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_foreign_keys(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    try:
        yield engine
    finally:
        # An undisposed aiosqlite connection does not fail the test that leaked
        # it; it resurfaces later as an unraisable exception from
        # `Connection.__del__`, blamed on whichever test triggered the GC.
        await engine.dispose()


@pytest.fixture
async def db_session(orm_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """An `AsyncSession` configured exactly as the application configures its own.

    `expire_on_commit=False` and `autoflush=False` are copied from
    `backend/db/session.py` rather than left at their defaults, because a test
    session that behaves differently from the production one tests something the
    application never does. `expire_on_commit=True` in particular would turn
    every post-commit attribute read below into a lazy refresh, which async
    SQLAlchemy answers with `MissingGreenlet` instead of a query.
    """
    factory = async_sessionmaker(
        bind=orm_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    async with factory() as session:
        yield session
