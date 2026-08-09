"""Fixtures for tests that need a real datastore.

Everything here exists to make one thing true: **an integration test either runs
against a real store or skips with a reason, and never fails because nothing is
running.** A suite that goes red on a laptop with no Docker is a suite people
learn to ignore, and the tests that matter most are the ones nobody runs.

So each store gets a probe that is *cheap and bounded*. Bounded matters more than
it sounds: the default connect timeout for several of these drivers is thirty
seconds, and six of them probed serially against a machine with nothing running
would take three minutes to decide the suite should be skipped.

**These tests write to the local development stores and clean up after
themselves.** They use a per-run namespace so a failed run leaves rows behind
that a later run does not trip over -- and so two people running the suite on a
shared store do not delete each other's fixtures. Anything that cannot be scoped
that way is marked `destructive` and excluded by default.

Run them with `make up` first, then:

    PYTHONPATH=. .venv/bin/python -m pytest tests/integration -v
"""

from __future__ import annotations

import asyncio
import os
import socket
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import closing
from typing import Any, Final

import pytest

pytestmark = pytest.mark.integration

PROBE_TIMEOUT_SECONDS: Final = 1.5
"""How long to wait deciding whether a store is reachable.

A TCP connect to a port that is not listening fails in milliseconds on a local
machine, so this only bites when something is listening but wedged -- exactly the
case where waiting thirty seconds per store helps nobody.
"""


def _port_open(host: str, port: int) -> bool:
    """Whether something is accepting connections. Not whether it is healthy.

    A TCP check rather than a protocol handshake, deliberately: this decides
    *skip or run*, and a store that accepts connections but is mid-startup should
    produce a real test failure with a real error message, not a silent skip that
    hides a broken environment.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(PROBE_TIMEOUT_SECONDS)
        return sock.connect_ex((host, port)) == 0


def _requires(host: str, port: int, name: str, hint: str) -> None:
    if not _port_open(host, port):
        pytest.skip(
            f"{name} is not reachable on {host}:{port}. Start it with `make up`"
            f" -- {hint}",
            allow_module_level=False,
        )


# --------------------------------------------------------------------------- #
# Per-run isolation
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def run_namespace() -> str:
    """A unique prefix for everything this run creates.

    Every id, key, index and node this suite writes carries it. Three things fall
    out of that: a crashed run leaves data a later run ignores rather than
    collides with, two developers can run the suite against one shared store, and
    cleanup is a prefix scan rather than a list of things to remember.
    """
    return f"it_{uuid.uuid4().hex[:12]}"


@pytest.fixture(scope="session")
def event_loop_policy() -> Any:
    """Session-scoped loop policy so async session fixtures are usable.

    Without it, `pytest-asyncio` builds a fresh loop per test and a
    session-scoped async fixture -- like a connection pool -- is bound to a loop
    that is already closed by the second test.
    """
    return asyncio.DefaultEventLoopPolicy()


# --------------------------------------------------------------------------- #
# PostgreSQL
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def postgres_available() -> None:
    _requires(
        os.getenv("POSTGRES_HOST", "localhost"),
        int(os.getenv("POSTGRES_PORT", "5432")),
        "PostgreSQL",
        "it is the commit point; nothing else in this suite is meaningful without it",
    )


@pytest.fixture
async def pg_sessionmaker(postgres_available: None) -> AsyncIterator[Any]:
    """A session factory against the local Postgres, disposed after each test.

    Per-test rather than per-session because a test that leaves a transaction
    open would otherwise block every subsequent one, and the resulting failure
    names the wrong test.
    """
    from backend.db.session import dispose_engine, get_sessionmaker

    yield get_sessionmaker()
    await dispose_engine()


# --------------------------------------------------------------------------- #
# Neo4j
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def neo4j_available() -> None:
    _requires(
        os.getenv("NEO4J_HOST", "localhost"),
        int(os.getenv("NEO4J_BOLT_PORT", "7687")),
        "Neo4j",
        "note this is the Bolt port 7687, not the HTTP console on 7474",
    )


@pytest.fixture
async def graph_writer(neo4j_available: None) -> AsyncIterator[Any]:
    from backend.db.neo4j import dispose_driver, write_session

    from graph.ingest.writer import GraphWriter, runner_from_session_factory

    yield GraphWriter(runner_from_session_factory(write_session))
    await dispose_driver()


@pytest.fixture
async def graph_client(neo4j_available: None) -> AsyncIterator[Any]:
    from backend.db.neo4j import dispose_driver, read_session

    from graph.client import GraphClient, read_runner_from_session_factory

    yield GraphClient(read_runner_from_session_factory(read_session))
    await dispose_driver()


@pytest.fixture
async def clean_graph(graph_writer: Any, run_namespace: str) -> AsyncIterator[str]:
    """Delete this run's nodes afterwards. Scoped by tenant, never a global wipe.

    `DETACH DELETE` restricted to `tenant_id = <namespace>`. An unscoped
    `MATCH (n) DETACH DELETE n` would be simpler and would also erase whatever a
    developer had in their local graph -- which is the kind of thing a test suite
    is only allowed to do once before nobody runs it again.
    """
    from backend.db.neo4j import run_write

    yield run_namespace
    await run_write(
        "MATCH (n {tenant_id: $tenant}) DETACH DELETE n", {"tenant": run_namespace}
    )


# --------------------------------------------------------------------------- #
# Redis / Qdrant / OpenSearch
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def redis_available() -> None:
    _requires(
        os.getenv("REDIS_HOST", "localhost"),
        int(os.getenv("REDIS_PORT", "6379")),
        "Redis",
        "used for the scratchpad, idempotency and timeline fan-out",
    )


@pytest.fixture(scope="session")
def qdrant_available() -> None:
    _requires(
        os.getenv("QDRANT_HOST", "localhost"),
        int(os.getenv("QDRANT_PORT", "6333")),
        "Qdrant",
        "vector search degrades to keyword-only without it",
    )


@pytest.fixture(scope="session")
def opensearch_available() -> None:
    _requires(
        os.getenv("OPENSEARCH_HOST", "localhost"),
        int(os.getenv("OPENSEARCH_PORT", "9200")),
        "OpenSearch",
        "keyword retrieval and the signals search endpoint need it",
    )


@pytest.fixture
async def redis_client(redis_available: None) -> AsyncIterator[Any]:
    from backend.db.redis import dispose_redis, get_redis

    yield get_redis()
    await dispose_redis()
