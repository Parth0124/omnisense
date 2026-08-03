"""Unit tests for `backend/db/r2.py`.

Two things are worth testing without a bucket. The first is `raw_object_key()`:
it is a pure function whose output is written into `content.raw_ref` in
PostgreSQL and is therefore permanent -- a change to the template orphans every
raw payload already archived, so it is pinned here against the ADR-0006 template
and against the realized example in `docs/signal-model.md` §6.

The second is that a process with no R2 configured -- which is every local
process, since `.env.example` ships `R2_ENDPOINT_URL` blank -- gets `False` from
`check_r2()` rather than an exception. `/readyz` aggregates several dependencies
and must be able to report on the others.
"""

from __future__ import annotations

import hashlib
import io
import re
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import SecretStr

from backend.core.config import StorageSettings, get_settings
from backend.core.exceptions import ConfigurationError
from backend.db import r2

pytestmark = pytest.mark.unit

# ADR-0006: raw/{platform}/{yyyy}/{mm}/{dd}/{raw_sha256}.json
ADR_TEMPLATE = re.compile(r"^raw/([a-z0-9_]+)/(\d{4})/(\d{2})/(\d{2})/([0-9a-f]{64})\.json$")

DIGEST = hashlib.sha256(b"payload").hexdigest()


class TestRawObjectKey:
    def test_matches_the_adr_template(self) -> None:
        key = r2.raw_object_key("reddit", datetime(2026, 7, 28, 12, 0, tzinfo=UTC), DIGEST)
        match = ADR_TEMPLATE.match(key)
        assert match is not None, key
        assert match.groups() == ("reddit", "2026", "07", "28", DIGEST)

    def test_reproduces_the_signal_model_example(self) -> None:
        """`docs/signal-model.md` §6 shows a realized `raw_ref`. Reproduce it exactly."""
        digest = "ebad8169cc3aeee5890e6632a636c33c28f220f38f92a45cfc7182bdb9cd967e"
        key = r2.raw_object_key("reddit", datetime(2026, 7, 28, 3, 14, tzinfo=UTC), digest)
        assert key == f"raw/reddit/2026/07/28/{digest}.json"

    def test_month_and_day_are_zero_padded(self) -> None:
        """Unpadded components would sort wrong and split one day across two prefixes."""
        key = r2.raw_object_key("rss", datetime(2026, 1, 5, tzinfo=UTC), DIGEST)
        assert key == f"raw/rss/2026/01/05/{DIGEST}.json"

    def test_no_compression_suffix(self) -> None:
        """ADR-0006 is explicit: objects are stored as received, the key ends `.json`."""
        key = r2.raw_object_key("gdelt", datetime(2026, 3, 9, tzinfo=UTC), DIGEST)
        assert key.endswith(".json")
        assert ".zst" not in key

    def test_platform_enum_member_is_accepted(self) -> None:
        """`Platform` is a `StrEnum`, so members must format to their slug."""
        from models.enums import Platform

        key = r2.raw_object_key(Platform.REDDIT, datetime(2026, 7, 28, tzinfo=UTC), DIGEST)
        assert key.startswith("raw/reddit/2026/07/28/")

    def test_offset_datetime_is_converted_to_utc(self) -> None:
        """A +13:00 instant late in the day is the *previous* UTC day."""
        local = datetime(2026, 7, 29, 11, 0, tzinfo=timezone(timedelta(hours=13)))
        assert r2.raw_object_key("x", local, DIGEST).startswith("raw/x/2026/07/28/")

    def test_same_instant_in_two_zones_gives_one_key(self) -> None:
        """Content addressing only deduplicates if the key is zone-independent."""
        utc_moment = datetime(2026, 7, 28, 23, 30, tzinfo=UTC)
        elsewhere = utc_moment.astimezone(timezone(timedelta(hours=-7)))
        assert r2.raw_object_key("reddit", utc_moment, DIGEST) == r2.raw_object_key(
            "reddit", elsewhere, DIGEST
        )

    def test_naive_datetime_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            r2.raw_object_key("reddit", datetime(2026, 7, 28, 12, 0), DIGEST)

    def test_uppercase_digest_is_normalized(self) -> None:
        key = r2.raw_object_key("reddit", datetime(2026, 7, 28, tzinfo=UTC), DIGEST.upper())
        assert key.endswith(f"/{DIGEST}.json")

    @pytest.mark.parametrize("bad", ["", "not-a-digest", DIGEST[:63], DIGEST + "a", "g" * 64])
    def test_non_digest_is_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="64-character hex digest"):
            r2.raw_object_key("reddit", datetime(2026, 7, 28, tzinfo=UTC), bad)

    @pytest.mark.parametrize("bad", ["", "   ", "social/reddit"])
    def test_bad_platform_is_rejected(self, bad: str) -> None:
        """A slash would invent a prefix level and hide objects from a listing."""
        with pytest.raises(ValueError, match="non-empty slug"):
            r2.raw_object_key(bad, datetime(2026, 7, 28, tzinfo=UTC), DIGEST)


class TestUnconfiguredProcess:
    """No R2 endpoint is set locally (ADR-0006: there is no local object store)."""

    async def test_get_s3_raises_configuration_error(self) -> None:
        await r2.dispose_s3()
        with pytest.raises(ConfigurationError) as excinfo:
            await r2.get_s3()
        assert "R2_ENDPOINT_URL" in str(excinfo.value)

    async def test_check_r2_returns_false_and_does_not_raise(self) -> None:
        await r2.dispose_s3()
        assert await r2.check_r2() is False

    async def test_blank_credentials_count_as_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`SecretStr("")` is truthy, so blank must be rejected explicitly.

        `.env.example` ships `R2_ACCESS_KEY_ID=` empty; copying it to `.env` is
        the normal first step, and it must not produce a client that signs every
        request with an empty key.
        """
        settings = get_settings().model_copy(
            update={
                "storage": StorageSettings(
                    endpoint_url="https://acct.r2.cloudflarestorage.com",
                    access_key_id=SecretStr(""),
                    secret_access_key=SecretStr("   "),
                )
            }
        )
        monkeypatch.setattr(r2, "get_settings", lambda: settings)
        await r2.dispose_s3()

        with pytest.raises(ConfigurationError) as excinfo:
            await r2.get_s3()
        assert excinfo.value.details["missing"] == [
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
        ]


class _RecordingClient:
    """Stands in for the botocore client and records which thread called it."""

    def __init__(self) -> None:
        self.threads: list[int] = []

    def _record(self, **_: object) -> dict[str, object]:
        self.threads.append(threading.get_ident())
        return {}

    head_bucket = _record
    head_object = _record
    put_object = _record

    def get_object(self, **_: object) -> dict[str, object]:
        self.threads.append(threading.get_ident())
        return {"Body": io.BytesIO(b"stored bytes")}

    def close(self) -> None:
        return None


class TestBlockingCallsLeaveTheEventLoop:
    """boto3 is synchronous; the loop thread must never execute one of its calls.

    This is the single property `backend/db/r2.py` exists to guarantee. A
    regression here is invisible in tests that only assert return values -- the
    calls still succeed, they just stall every other coroutine on the worker
    while they do -- so it is asserted directly, by thread identity.
    """

    @pytest.fixture
    def client(self) -> Iterator[_RecordingClient]:
        fake = _RecordingClient()
        r2._client = fake
        try:
            yield fake
        finally:
            r2._client = None

    async def test_put_get_and_exists_all_run_off_the_loop_thread(
        self, client: _RecordingClient
    ) -> None:
        loop_thread = threading.get_ident()

        await r2.put_object("raw/x/2026/01/01/deadbeef.json", b"bytes")
        assert await r2.get_object("k") == b"stored bytes"
        assert await r2.object_exists("k") is True
        assert await r2.check_r2() is True

        assert len(client.threads) == 4
        assert loop_thread not in client.threads
