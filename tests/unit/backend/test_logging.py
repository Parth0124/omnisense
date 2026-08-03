"""Unit tests for `backend/core/logging.py`.

The redaction tests are the ones that matter. Every other property of this module
fails loudly -- a broken processor chain raises on the first log call. Redaction
fails *silently*: the line is emitted, it looks fine, and the credential is in a
log aggregator with 90-day retention. So the tests below assert on the negative
("the real value appears nowhere in the rendered output") rather than only on the
positive ("the field says redacted"), because those are different claims and only
the first one is the security property.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest
from structlog.types import EventDict

from backend.core.config import LogFormat, LogLevel, Settings
from backend.core.logging import (
    REDACTED,
    UNBOUND_CORRELATION_ID,
    bind_correlation_id,
    clear_correlation_id,
    configure_logging,
    correlation_scope,
    get_correlation_id,
    get_logger,
    redact_processor,
    reset_logging,
)

pytestmark = pytest.mark.unit


def _redact(event_dict: EventDict) -> EventDict:
    """Run the processor the way structlog would."""
    return redact_processor(None, "info", event_dict)


@pytest.fixture
def logging_state() -> Iterator[None]:
    """Reset structlog and the root logger around every test in this module.

    `configure_logging()` mutates process-global state. Without this fixture the
    first test that configures logging decides the behaviour of every test that
    runs after it, in any file.
    """
    reset_logging()
    clear_correlation_id()
    yield
    reset_logging()
    clear_correlation_id()


def _configure(
    capsys: pytest.CaptureFixture[str],
    log_format: LogFormat = LogFormat.JSON,
    level: LogLevel = LogLevel.INFO,
) -> None:
    settings = Settings(_env_file=None)
    settings.app.log_format = log_format
    settings.app.log_level = level
    configure_logging(settings, force=True)
    capsys.readouterr()  # discard anything emitted during configuration


class TestRedaction:
    def test_nested_api_key_is_redacted(self) -> None:
        """The case named in the specification, verbatim."""
        out = _redact({"event": "connector.auth", "auth": {"api_key": "sk-real"}})
        assert out["auth"] == {"api_key": REDACTED}
        assert "sk-real" not in json.dumps(out)

    def test_top_level_keys(self) -> None:
        out = _redact(
            {
                "password": "hunter2",
                "access_token": "at-1",
                "client_secret": "cs-1",
                "Authorization": "Bearer abc",
                "cookie": "session=1",
                "credential_encryption_key": "fernet",
                "api_key": "sk-1",
            }
        )
        assert set(out.values()) == {REDACTED}

    def test_deeply_nested_and_inside_lists(self) -> None:
        event = {
            "connector": {
                "slug": "reddit",
                "accounts": [
                    {"id": "acc_1", "credentials": {"refresh_token": "rt-real"}},
                    {"id": "acc_2", "credentials": {"refresh_token": "rt-also-real"}},
                ],
            }
        }
        rendered = json.dumps(_redact(event))
        assert "rt-real" not in rendered
        assert "rt-also-real" not in rendered
        # Non-sensitive siblings survive: a redactor that flattens the record is
        # useless for debugging and gets turned off.
        assert "acc_1" in rendered
        assert "reddit" in rendered

    def test_innocuous_keys_are_untouched(self) -> None:
        out = _redact(
            {
                "event": "signal.enrichment.completed",
                "keywords": ["pricing", "latency"],
                "signal_id": "sig_1",
                "duration_ms": 412,
                "monkey": "business",
            }
        )
        assert out["keywords"] == ["pricing", "latency"]
        assert out["signal_id"] == "sig_1"
        assert out["duration_ms"] == 412
        assert out["monkey"] == "business"

    def test_key_suffix_over_matches_on_purpose(self) -> None:
        """`*_key` is redacted even when it is not a credential.

        Asserted rather than tolerated: this is a deliberate trade documented on
        `_SENSITIVE_KEY_RE`, and if someone narrows the pattern later the test
        should make them justify it.
        """
        out = _redact({"idempotency_key": "idem-1", "point_id": "p1"})
        assert out["idempotency_key"] == REDACTED
        assert out["point_id"] == "p1"

    def test_tuples_stay_tuples(self) -> None:
        out = _redact({"pair": ("a", {"token": "t"})})
        assert isinstance(out["pair"], tuple)
        assert out["pair"] == ("a", {"token": REDACTED})

    def test_self_referential_structure_terminates(self) -> None:
        """A cycle must not hang the logging pipeline."""
        cyclic: dict[str, object] = {"name": "loop"}
        cyclic["self"] = cyclic
        rendered = json.dumps(_redact({"payload": cyclic}), default=str)
        assert "depth-limit" in rendered

    def test_processor_meta_survives(self) -> None:
        """`exc_info` and `_record` must reach the renderer intact."""
        record = logging.LogRecord("n", logging.INFO, "p", 1, "m", (), None)
        exc_info = (ValueError, ValueError("boom"), None)
        out = _redact({"exc_info": exc_info, "_record": record, "_from_structlog": False})
        assert out["exc_info"] is exc_info
        assert out["_record"] is record


class TestCorrelationId:
    def test_defaults_to_unbound(self, logging_state: None) -> None:
        assert get_correlation_id() == UNBOUND_CORRELATION_ID

    def test_bind_mints_when_not_supplied(self, logging_state: None) -> None:
        value = bind_correlation_id()
        assert value == get_correlation_id()
        assert len(value) == 32

    def test_bind_accepts_an_upstream_id(self, logging_state: None) -> None:
        """A worker must reuse the envelope's id, never mint its own."""
        bind_correlation_id("01J8ZQ2MZP0000000000000000")
        assert get_correlation_id() == "01J8ZQ2MZP0000000000000000"

    def test_scope_restores_the_previous_value(self, logging_state: None) -> None:
        bind_correlation_id("outer")
        with correlation_scope("inner") as value:
            assert value == "inner"
            assert get_correlation_id() == "inner"
        assert get_correlation_id() == "outer"

    async def test_inherited_by_child_tasks(self, logging_state: None) -> None:
        """Fan-out must not break the chain (`docs/observability.md` §2.3)."""
        import asyncio

        bind_correlation_id("fanout")

        async def child() -> str:
            return get_correlation_id()

        async with asyncio.TaskGroup() as group:
            task = group.create_task(child())
        assert task.result() == "fanout"


class TestConfiguredPipeline:
    def test_json_line_carries_the_required_fields(
        self, logging_state: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _configure(capsys)
        bind_correlation_id("corr-1")
        get_logger("tests.pipeline").info("signal.enrichment.completed", signal_id="sig_1")

        record = json.loads(capsys.readouterr().out.strip())
        assert record["event"] == "signal.enrichment.completed"
        assert record["level"] == "info"
        assert record["logger"] == "tests.pipeline"
        assert record["correlation_id"] == "corr-1"
        assert record["service"] == "omnisense-api"
        assert record["env"] == "local"
        assert record["signal_id"] == "sig_1"
        assert record["timestamp"].endswith("Z")

    def test_redaction_applies_to_real_output(
        self, logging_state: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End to end, not just the processor in isolation."""
        _configure(capsys)
        get_logger("tests.pipeline").info("connector.auth", auth={"api_key": "sk-real"})

        out = capsys.readouterr().out
        assert "sk-real" not in out
        assert REDACTED in out

    def test_stdlib_records_join_the_same_pipeline(
        self, logging_state: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A SQLAlchemy or uvicorn line must look like ours."""
        _configure(capsys)
        bind_correlation_id("corr-2")
        logging.getLogger("sqlalchemy.engine.Engine").warning("connection reset")

        record = json.loads(capsys.readouterr().out.strip())
        assert record["event"] == "connection reset"
        assert record["logger"] == "sqlalchemy.engine.Engine"
        assert record["correlation_id"] == "corr-2"
        assert record["level"] == "warning"

    def test_exception_becomes_error_type_and_message(
        self, logging_state: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _configure(capsys)
        try:
            raise ValueError("upstream said no")
        except ValueError:
            get_logger("tests.pipeline").exception("connector.fetch.failed")

        record = json.loads(capsys.readouterr().out.strip())
        assert record["error"] == {"type": "ValueError", "message": "upstream said no"}
        assert "ValueError" in record["exception"]

    def test_level_filtering_is_honoured(
        self, logging_state: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _configure(capsys, level=LogLevel.WARNING)
        logger = get_logger("tests.pipeline")
        logger.info("suppressed.event")
        logger.warning("emitted.event")

        out = capsys.readouterr().out
        assert "suppressed.event" not in out
        assert "emitted.event" in out

    def test_console_format_renders_without_json(
        self, logging_state: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _configure(capsys, log_format=LogFormat.CONSOLE)
        get_logger("tests.pipeline").info("connector.auth", auth={"api_key": "sk-real"})

        out = capsys.readouterr().out
        assert "connector.auth" in out
        assert "sk-real" not in out
        with pytest.raises(json.JSONDecodeError):
            json.loads(out.strip())

    def test_configure_is_idempotent(
        self, logging_state: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Two subsystems calling this must not duplicate every line."""
        _configure(capsys)
        settings = Settings(_env_file=None)
        settings.app.log_format = LogFormat.JSON
        configure_logging(settings)
        configure_logging(settings)

        get_logger("tests.pipeline").info("once.only")
        assert capsys.readouterr().out.count("once.only") == 1
        assert len(logging.getLogger().handlers) == 1

    def test_logger_created_before_configuration_still_picks_it_up(
        self, logging_state: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`logger = get_logger(__name__)` at module scope must stay lazy.

        Modules bind their logger at import time, long before the lifespan hook
        runs `configure_logging()`. If the proxy resolved eagerly, every module
        logger in the process would be stuck on structlog's defaults.
        """
        logger = get_logger("tests.lazy")
        _configure(capsys)
        logger.info("late.binding", signal_id="sig_1")

        record = json.loads(capsys.readouterr().out.strip())
        assert record["logger"] == "tests.lazy"
        assert record["service"] == "omnisense-api"
