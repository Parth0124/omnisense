"""Check every dependency the stack needs, and say what to do about each failure.

`scripts/init_databases.py` *creates* things -- schemas, indexes, collections,
constraints. This script *verifies* them, plus the two dependencies that script
has no reason to touch: Redis, which nothing bootstraps because it needs no
schema, and the LLM provider, which lives outside the compose file entirely.

The split matters on a cold start. `init-db` reports "postgres created" the
moment the migration applies, which is true and is not the same as "the stack
works". A run can have every container healthy, every table migrated, and still
be unable to answer a single question because `LLM_API_KEY` is empty -- and the
first symptom of that is a failed investigation twenty seconds into a run, with
the real cause four layers down a traceback.

**Every check reports a fix, not just a verdict.** A red line saying `qdrant
unreachable` sends someone to read the compose file. A red line saying
`qdrant unreachable -> start it with make up` does not. The hint is the point of
the script; the check is just how it earns the right to print one.

**The LLM check spends real money.** It is one request capped at a handful of
tokens -- a fraction of a cent -- and it is the only way to distinguish a key
that is present from a key that works. A key with a typo, a revoked key and an
account out of credit all look identical to any offline check, and all three
produce the same "why did my investigation fail" question an hour later.

Exit code is 0 only when nothing is broken, so this is usable as a gate in a
script: `make start` runs it last and refuses to claim success without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running this as `python scripts/doctor.py` puts `scripts/` on `sys.path`, not
# the repository root, so every first-party import below would fail with
# `ModuleNotFoundError: No module named 'backend'`. OmniSense is deliberately not
# an installed package, so there is no console-script entry point to inherit the
# right path from. Same fix, and the same reason, as `scripts/init_databases.py`
# -- and it matters more here, because this is the script someone reaches for
# when things are already broken.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import contextlib
import enum
import logging
import os
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass

import typer

from backend.core.config import LLMProvider, get_settings
from backend.core.logging import configure_logging
from backend.db.neo4j import check_neo4j, dispose_driver
from backend.db.opensearch import check_opensearch, dispose_opensearch
from backend.db.qdrant import check_qdrant, dispose_qdrant
from backend.db.redis import check_redis, dispose_redis
from backend.db.session import check_postgres, dispose_engine
from services.events.producer import check_kafka, dispose_producer

__all__ = ["Check", "Status", "app", "main"]

REPO_ROOT = Path(__file__).resolve().parents[1]

app = typer.Typer(
    add_completion=False,
    help="Verify every dependency the stack needs and report what to fix.",
)


class Status(enum.StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"

    @property
    def is_problem(self) -> bool:
        """`WARN` is deliberately not a problem.

        A missing optional key should be visible without blocking a run that does
        not need it -- otherwise the first thing anyone learns is to stop reading
        the output.
        """
        return self is Status.FAIL

    @property
    def colour(self) -> str:
        return {
            Status.OK: typer.colors.GREEN,
            Status.WARN: typer.colors.YELLOW,
            Status.FAIL: typer.colors.RED,
        }[self]

    @property
    def mark(self) -> str:
        return {Status.OK: "OK  ", Status.WARN: "WARN", Status.FAIL: "FAIL"}[self]


@dataclass(frozen=True, slots=True)
class Check:
    """One dependency's verdict, and what to do if it is bad."""

    name: str
    status: Status
    detail: str
    fix: str = ""

    def render(self) -> str:
        line = f"  {self.status.mark}  {self.name:<14} {self.detail}"
        if self.fix and self.status is not Status.OK:
            line += f"\n        -> {self.fix}"
        return line


# --------------------------------------------------------------------------- #
# Configuration -- checked before anything is dialled
# --------------------------------------------------------------------------- #


def _check_secrets() -> Check:
    """The two generated secrets are still the template's placeholder.

    A warning rather than a failure, because `Settings` only rejects `change-me`
    outside local development and nothing in a step-0 stack needs either value.

    It is worth saying out loud anyway. `CREDENTIAL_ENCRYPTION_KEY` has to be a
    valid Fernet key, and the first thing that needs one is storing a connector
    credential -- so left alone, this surfaces at step 3 as an encryption error
    while onboarding a repo, which is nowhere near where the cause is.
    """
    security = get_settings().security
    placeholders = [
        name
        for name, value in (
            ("SECRET_KEY", security.secret_key),
            ("CREDENTIAL_ENCRYPTION_KEY", security.credential_encryption_key),
        )
        if value.get_secret_value() == "change-me"
    ]
    if not placeholders:
        return Check("secrets", Status.OK, "generated")
    return Check(
        "secrets",
        Status.WARN,
        f"still the template placeholder: {', '.join(placeholders)}",
        fix=(
            "fine for now, needed by step 3 -- "
            'SECRET_KEY: python -c "import secrets; print(secrets.token_urlsafe(48))" | '
            "CREDENTIAL_ENCRYPTION_KEY: python -c "
            '"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        ),
    )


def _check_env_file() -> Check:
    """`.env` exists at all.

    First because every other check reads its address from it. Without the file
    Pydantic falls back to defaults that point at localhost, which happen to be
    right locally -- so the run half-works and the missing file is discovered
    much later, on the first machine where a default is wrong.
    """
    if (REPO_ROOT / ".env").exists():
        return Check(".env", Status.OK, "present")
    return Check(
        ".env",
        Status.FAIL,
        "missing",
        fix="run `make env`, then put your OpenRouter key in LLM_API_KEY",
    )


def _check_llm_config() -> Check:
    """A provider and a key that belong together.

    The two ways to get this wrong are silent. `LLM_PROVIDER=anthropic` with only
    `LLM_API_KEY` set reads an empty `ANTHROPIC_API_KEY`; `LLM_PROVIDER=openai`
    with OpenRouter's default endpoint and a bare `claude-sonnet-5` gets a 400,
    because OpenRouter namespaces every model as `vendor/model`.
    """
    settings = get_settings()
    llm = settings.llm

    if llm.provider is LLMProvider.ANTHROPIC:
        if not (llm.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")):
            return Check(
                "llm config",
                Status.FAIL,
                "provider=anthropic but ANTHROPIC_API_KEY is empty",
                fix="set ANTHROPIC_API_KEY, or switch to LLM_PROVIDER=openai for OpenRouter",
            )
        return Check("llm config", Status.OK, f"anthropic, {llm.model_worker}")

    if not llm.api_key:
        return Check(
            "llm config",
            Status.FAIL,
            f"provider={llm.provider.value} but LLM_API_KEY is empty",
            fix="put your OpenRouter key in LLM_API_KEY in .env",
        )

    base = llm.base_url or "https://openrouter.ai/api/v1 (default)"
    unqualified = [
        name for name in (llm.model_planner, llm.model_worker, llm.model_fast) if "/" not in name
    ]
    if "openrouter" in base and unqualified:
        return Check(
            "llm config",
            Status.FAIL,
            f"OpenRouter needs vendor/model, got {sorted(set(unqualified))}",
            fix="e.g. LLM_MODEL_WORKER=anthropic/claude-sonnet-4.5 -- see openrouter.ai/models",
        )
    return Check("llm config", Status.OK, f"{llm.provider.value} -> {base}")


# --------------------------------------------------------------------------- #
# Datastores
# --------------------------------------------------------------------------- #


MUTED_WHILE_PROBING: dict[str, int] = {
    # `opensearchpy` logs a failed request at WARNING with `exc_info=True`, once
    # per attempt. Under the console renderer that one warning expands into a
    # full rich traceback -- roughly eighty kilobytes of frames for the single
    # most likely outcome of running this script, "the container is not up yet".
    "opensearch": logging.ERROR,
    # aiokafka reports a refused connection at ERROR, so it needs raising a
    # level higher than the others to stay quiet.
    "aiokafka": logging.CRITICAL,
}
"""Loggers silenced for the duration of the probes, and the level each needs.

The whole value of this script is a short readable report. A store that is not
running is its *expected* input, not an exceptional one, and the line below each
probe already says what is wrong and what to do -- so the library's own traceback
is not extra information, it is the exact thing that makes the report unreadable.

Add an entry only after seeing a logger produce this specific failure mode.
Muting one on suspicion trades a real diagnostic for tidiness.
"""


@contextlib.contextmanager
def _quiet_transports() -> Iterator[None]:
    """Mute the noisy clients, then put every level back.

    Restored rather than left raised because this module is importable: a test
    that calls `main()` must not inherit a permanently muted OpenSearch logger.
    """
    previous = {name: logging.getLogger(name).level for name in MUTED_WHILE_PROBING}
    for name, level in MUTED_WHILE_PROBING.items():
        logging.getLogger(name).setLevel(level)
    try:
        yield
    finally:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)


async def _probe(
    name: str,
    probe: Callable[[], Awaitable[bool]],
    *,
    reachable: str,
    fix: str,
) -> Check:
    """Run one probe, turning any exception into a FAIL rather than a traceback.

    A probe that raises is the ordinary case here -- the container is not up yet
    -- and the exception type is more useful in one line than as forty frames.
    """
    try:
        ok = await probe()
    except Exception as error:  # noqa: BLE001 -- an unreachable store is the expected case
        return Check(name, Status.FAIL, f"{type(error).__name__}: {error}"[:88], fix=fix)
    if ok:
        return Check(name, Status.OK, reachable)
    return Check(name, Status.FAIL, "not answering", fix=fix)


async def _check_migrations() -> Check:
    """Postgres has the tables, not merely a listening socket.

    Separate from the connectivity probe because "connected but empty" is its own
    failure with its own fix, and it is what a first run looks like if `make up`
    succeeded and `make init-db` was never run.
    """
    from sqlalchemy import text

    from backend.db.session import get_engine
    from models.orm.base import SCHEMA

    try:
        async with get_engine().connect() as conn:
            revision = (
                await conn.execute(
                    text(f"SELECT version_num FROM {SCHEMA}.alembic_version")  # noqa: S608
                )
            ).scalar_one_or_none()
            tables = (
                await conn.execute(
                    text("SELECT count(*) FROM information_schema.tables WHERE table_schema = :s"),
                    {"s": SCHEMA},
                )
            ).scalar_one()
    except Exception as error:  # noqa: BLE001 -- no schema yet is the expected first-run state
        return Check(
            "migrations",
            Status.FAIL,
            f"{type(error).__name__}"[:88],
            fix="run `make init-db`",
        )

    if revision is None:
        return Check("migrations", Status.FAIL, "no revision applied", fix="run `make init-db`")
    return Check("migrations", Status.OK, f"at {revision}, {tables} tables")


async def _check_embedding_call() -> Check:
    """One real embedding, and a check that its width matches the Qdrant collection.

    The width is the part worth automating. A collection's geometry is fixed when
    it is created, so a provider returning 768 where the collection expects 1536
    does not degrade -- every write is rejected, and the first time anyone notices
    is a backfill that stores nothing. Changing either side after ingestion has
    started means re-embedding the entire corpus, which is why this is checked on
    every run rather than assumed from configuration.

    A blank key is a warning rather than a failure: `.env.example` documents blank
    as "no embedding", which leaves ingestion and keyword search working and only
    turns off semantic retrieval. That is a legitimate way to run.
    """
    settings = get_settings()
    embedding = settings.embedding

    if not embedding.api_key:
        return Check(
            "embedding",
            Status.WARN,
            "EMBEDDING_API_KEY unset -- semantic search disabled",
            fix="fine for now; needed from step 4. OpenRouter has no embeddings API, "
            "so this wants an OpenAI key (or any OpenAI-compatible endpoint)",
        )

    from services.llm.embeddings import OpenAICompatibleEmbeddingProvider

    try:
        provider = OpenAICompatibleEmbeddingProvider(settings=embedding)
        vectors = await provider.embed(["connectivity probe"])
    except Exception as error:  # noqa: BLE001 -- every failure mode is worth reporting flat
        message = str(error)
        detail = f"{type(error).__name__}: {message}"[:88]
        if "401" in message or "403" in message:
            return Check(
                "embedding",
                Status.FAIL,
                detail,
                fix="key rejected -- note this is a separate OpenAI key, not OpenRouter",
            )
        if "429" in message or "quota" in message.lower() or "billing" in message.lower():
            return Check(
                "embedding",
                Status.FAIL,
                detail,
                fix="OpenAI account has no balance -- it bills separately from OpenRouter",
            )
        return Check(
            "embedding",
            Status.FAIL,
            detail,
            fix="check EMBEDDING_API_KEY, EMBEDDING_BASE_URL and EMBEDDING_MODEL",
        )

    width = len(vectors[0])
    if width != embedding.dimensions:
        return Check(
            "embedding",
            Status.FAIL,
            f"{embedding.model} returned {width}d, config and Qdrant expect "
            f"{embedding.dimensions}d",
            fix=(
                "these must match or every vector write is rejected. Either set "
                f"EMBEDDING_DIMENSIONS={width} and recreate the Qdrant collection "
                "(free now, means re-embedding everything once data exists), or "
                "pick a model of the configured width"
            ),
        )
    return Check("embedding", Status.OK, f"{embedding.model} returned {width}d, matches Qdrant")


async def _check_llm_call() -> Check:
    """One real request. The only check that proves the key works.

    Deliberately tiny and deliberately not cached: presence of a key, validity of
    a key and an account with credit are three different things that only a live
    call can tell apart. OpenRouter answers an out-of-credit account with 402,
    which is why that case gets its own hint rather than the generic one.
    """
    from agents.composition import build_llm_provider

    settings = get_settings()
    try:
        provider = build_llm_provider(settings.llm)
        reply = await provider.complete(
            prompt="Reply with the single word: ok",
            system="You are a connectivity probe. Answer in one word.",
            model=settings.llm.model_fast,
            max_tokens=16,
        )
    except Exception as error:  # noqa: BLE001 -- every failure mode here is worth reporting flat
        message = str(error)
        detail = f"{type(error).__name__}: {message}"[:88]
        if "402" in message or "credit" in message.lower():
            return Check(
                "llm call",
                Status.FAIL,
                detail,
                fix="OpenRouter account is out of credit -- top it up",
            )
        if "401" in message or "403" in message:
            return Check(
                "llm call",
                Status.FAIL,
                detail,
                fix="key rejected -- check LLM_API_KEY is correct and active",
            )
        if "404" in message or "400" in message:
            return Check(
                "llm call",
                Status.FAIL,
                detail,
                fix=f"model {settings.llm.model_fast!r} not accepted -- see openrouter.ai/models",
            )
        return Check("llm call", Status.FAIL, detail, fix="check LLM_API_KEY and LLM_BASE_URL")

    answered = reply.text.strip()[:20]
    spend = reply.input_tokens + reply.output_tokens
    return Check(
        "llm call",
        Status.OK,
        f"{reply.model} answered {answered!r} ({spend} tokens)",
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


async def _run(*, skip_llm: bool) -> list[Check]:
    settings = get_settings()
    checks: list[Check] = [_check_env_file(), _check_secrets(), _check_llm_config()]

    typer.secho("\nconfiguration", fg=typer.colors.BRIGHT_BLACK)
    for check in checks:
        typer.secho(check.render(), fg=check.status.colour)

    typer.secho("\ndatastores", fg=typer.colors.BRIGHT_BLACK)
    with _quiet_transports():
        store_checks = [
            await _probe(
                "postgres", check_postgres, reachable="reachable", fix="start it with `make up`"
            ),
            await _check_migrations(),
            await _probe(
                "redis", check_redis, reachable="reachable", fix="start it with `make up`"
            ),
            await _probe(
                "neo4j",
                check_neo4j,
                reachable=f"reachable at {settings.neo4j.uri}",
                fix="start it with `make up`",
            ),
            await _probe(
                "qdrant",
                check_qdrant,
                reachable=f"reachable at {settings.qdrant.url}",
                fix="start it with `make up`, then `make init-db`",
            ),
            await _probe(
                "opensearch",
                check_opensearch,
                reachable=f"reachable at {settings.opensearch.url}",
                fix="start it with `make up` -- it takes ~60s to go healthy",
            ),
            await _probe(
                "redpanda",
                check_kafka,
                reachable=f"reachable at {settings.kafka.bootstrap_servers}",
                fix="start it with `make up`",
            ),
        ]
    for check in store_checks:
        typer.secho(check.render(), fg=check.status.colour)
    checks.extend(store_checks)

    typer.secho("\nmodel providers", fg=typer.colors.BRIGHT_BLACK)
    if not skip_llm:
        embedded = await _check_embedding_call()
        typer.secho(embedded.render(), fg=embedded.status.colour)
        checks.append(embedded)

    if skip_llm:
        skipped = Check("llm call", Status.WARN, "skipped (--skip-llm)")
        typer.secho(skipped.render(), fg=skipped.status.colour)
        checks.append(skipped)
    elif any(c.name == "llm config" and c.status is Status.FAIL for c in checks):
        blocked = Check("llm call", Status.WARN, "not attempted -- fix llm config first")
        typer.secho(blocked.render(), fg=blocked.status.colour)
        checks.append(blocked)
    else:
        called = await _check_llm_call()
        typer.secho(called.render(), fg=called.status.colour)
        checks.append(called)

    return checks


async def _dispose() -> None:
    """Close every pool this script opened.

    Without it the event loop closes under live connections and asyncio prints
    'Task was destroyed but it is pending' after the summary -- which reads as a
    crash directly below a report that just said everything is fine.
    """
    for closer in (
        dispose_engine,
        dispose_redis,
        dispose_driver,
        dispose_qdrant,
        dispose_opensearch,
        # `check_kafka` constructs a producer to probe with. Left open, aiokafka
        # logs 'Unclosed AIOKafkaProducer' at ERROR *after* the summary line --
        # which reads as a crash directly beneath a report saying all is well.
        dispose_producer,
    ):
        with contextlib.suppress(Exception):
            await closer()


@app.command()
def main(
    skip_llm: bool = typer.Option(
        False, "--skip-llm", help="Skip the live model call (it costs a fraction of a cent)."
    ),
) -> None:
    """Check everything and print what to fix. Exits non-zero if anything is broken."""
    configure_logging()

    async def _go() -> list[Check]:
        try:
            return await _run(skip_llm=skip_llm)
        finally:
            await _dispose()

    checks = asyncio.run(_go())

    broken = [c for c in checks if c.status.is_problem]
    typer.echo("")
    if broken:
        typer.secho(
            f"{len(broken)} of {len(checks)} checks failed: {', '.join(c.name for c in broken)}",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    warned = [c for c in checks if c.status is Status.WARN]
    suffix = f" ({len(warned)} skipped)" if warned else ""
    typer.secho(f"all {len(checks)} checks passed{suffix}", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
