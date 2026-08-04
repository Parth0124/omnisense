"""Unit tests for `services/storage/`: the object store and the media fetcher.

Neither module can be exercised against a real bucket -- ADR-0006 records that
`docker-compose.yml` runs no object-storage container, which is the most concrete
gap in that decision. So R2 is replaced with an in-memory fake that speaks the
handful of botocore methods `backend/db/r2.py` actually calls, and HTTP is
replaced with `httpx.MockTransport`. Everything below runs with no network, no
credentials and no containers.

Four properties are load-bearing enough that a regression in any of them is a
production incident rather than a failing assertion, and each has a class here:

1. **The raw key template.** Its value is written into `content.raw_ref` in
   PostgreSQL and is permanent; changing the template orphans every payload
   already archived. Pinned against ADR-0006 character by character.
2. **Content addressing.** The same bytes must produce the same key on every
   worker, forever, and a repeated write must be a no-op -- that is the whole
   basis for idempotent replay in `docs/data-stores.md` §5.2.
3. **The media byte ceiling fires before the body is read.** Asserted by
   instrumenting the response stream and checking it was never consumed, because
   a version that downloads 40 GB and *then* rejects it passes every test that
   only looks at the exception type while still being the memory-exhaustion
   vector the ceiling exists to prevent.
4. **The content-type allowlist refuses what is not on it**, also before the
   body.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import re
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from botocore.exceptions import ClientError

from backend.core.exceptions import NotFoundError
from backend.db import r2
from models.enums import MediaKind
from models.signal import MediaRef
from services.storage import media as media_mod
from services.storage import object_store

pytestmark = pytest.mark.unit


# ADR-0006: raw/{platform}/{yyyy}/{mm}/{dd}/{raw_sha256}.json
ADR_RAW_TEMPLATE = re.compile(r"^raw/([a-z0-9_]+)/(\d{4})/(\d{2})/(\d{2})/([0-9a-f]{64})\.json$")
# ADR-0006: media/{signal_id}/{sha256}.{ext}
ADR_MEDIA_TEMPLATE = re.compile(r"^media/([A-Za-z0-9._-]+)/([0-9a-f]{64})\.([a-z0-9]+)$")
# ADR-0006: reports/{report_id}/{version}/report.{pdf|html|md}
ADR_REPORT_TEMPLATE = re.compile(r"^reports/([A-Za-z0-9._-]+)/(\d+)/report\.(pdf|html|md)$")

FETCHED_AT = datetime(2026, 7, 28, 3, 14, tzinfo=UTC)
PAYLOAD = b'{"id": "t3_1abcde", "title": "a post", "selftext": "some body text"}'

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeS3:
    """In-memory stand-in for the botocore S3 client.

    Only the four methods `backend/db/r2.py` calls, plus the presigner. Every
    call is recorded in order so a test can assert on the *shape* of the
    conversation -- "was a PUT issued at all?" is the question the idempotency
    tests need answered, and a store that only records final state cannot answer
    it.

    Keyword-only signatures throughout because that is how r2.py invokes them;
    `**kwargs` avoids restating botocore's PascalCase parameter names.
    """

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.presign_requests: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        self.calls.append(("put_object", key))
        self.objects[key] = {
            "Body": kwargs["Body"],
            "ContentType": kwargs.get("ContentType"),
            "Metadata": kwargs.get("Metadata", {}),
        }
        return {}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        self.calls.append(("get_object", key))
        if key not in self.objects:
            raise self._not_found("GetObject")
        return {"Body": io.BytesIO(self.objects[key]["Body"])}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        self.calls.append(("head_object", key))
        if key not in self.objects:
            raise self._not_found("HeadObject")
        return {"ContentLength": len(self.objects[key]["Body"])}

    def generate_presigned_url(self, **kwargs: Any) -> str:
        params: dict[str, Any] = kwargs["Params"]
        expires_in: int = kwargs["ExpiresIn"]
        self.presign_requests.append(
            {"method": kwargs["ClientMethod"], "params": params, "expires_in": expires_in}
        )
        return f"https://fake.r2.example/{params['Key']}?X-Amz-Expires={expires_in}"

    def close(self) -> None:
        return None

    @staticmethod
    def _not_found(operation: str) -> ClientError:
        """404 exactly as botocore reports it for a HEAD: a status, not a code.

        `head_object` has no response body to carry `NoSuchKey`, so r2.py has to
        recognize the bare `"404"`. Reproducing that here is the point of the
        fake -- a friendlier `NoSuchKey` would test a code path production never
        takes.
        """
        return ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, operation)


class RecordingStream(httpx.AsyncByteStream):
    """A response body that records whether anything ever read it.

    This is the instrument behind the "before the body" assertions. Asserting
    only that `MediaTooLargeError` was raised cannot distinguish a downloader
    that refuses on the header from one that buffers the whole file first and
    then complains -- and those two have completely different blast radii.
    """

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks
        self.consumed = False
        self.chunks_yielded = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.consumed = True
        for chunk in self._chunks:
            self.chunks_yielded += 1
            yield chunk

    async def aclose(self) -> None:
        return None


def make_client(handler: Any) -> httpx.AsyncClient:
    """An `httpx.AsyncClient` wired to a handler instead of a socket."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def responder(
    *,
    content_type: str | None = "image/png",
    stream: RecordingStream | None = None,
    content_length: str | None = None,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> Any:
    """Build a one-shot handler returning a streamed response."""
    hdrs = dict(headers or {})
    if content_type is not None:
        hdrs["content-type"] = content_type
    if content_length is not None:
        hdrs["content-length"] = content_length

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=hdrs, stream=stream or RecordingStream(PNG))

    return handle


@pytest.fixture
def s3() -> Iterator[FakeS3]:
    """Install the fake as the process-wide R2 client for one test.

    Assigning `r2._client` directly is the same seam
    `tests/unit/backend/db/test_r2.py` uses: `get_s3()` returns the singleton
    when one exists and never tries to build a real client, so no credentials and
    no `.env` are involved.
    """
    fake = FakeS3()
    r2._client = fake
    try:
        yield fake
    finally:
        r2._client = None


# --------------------------------------------------------------------------- #
# Key templates
# --------------------------------------------------------------------------- #


class TestKeyTemplatesMatchADR0006:
    """The templates are permanent: every stored ref resolves through them."""

    async def test_raw_key_matches_the_adr_template_exactly(self, s3: FakeS3) -> None:
        stored = await object_store.put_raw_payload("reddit", FETCHED_AT, PAYLOAD)

        match = ADR_RAW_TEMPLATE.match(stored.key)
        assert match is not None, stored.key
        assert match.groups() == (
            "reddit",
            "2026",
            "07",
            "28",
            hashlib.sha256(PAYLOAD).hexdigest(),
        )

    async def test_raw_key_keeps_the_json_suffix_when_compressed(self, s3: FakeS3) -> None:
        """ADR-0006: compression must not add `.zst`, or every `raw_ref` breaks."""
        for codec in object_store.Compression:
            stored = await object_store.put_raw_payload(
                "reddit", FETCHED_AT, PAYLOAD, compression=codec
            )
            assert stored.key.endswith(".json"), codec
            assert ".zst" not in stored.key
            assert ".gz" not in stored.key

    def test_media_key_matches_the_adr_template(self) -> None:
        digest = hashlib.sha256(PNG).hexdigest()
        key = object_store.media_object_key("sig_abc123", digest, "png")

        match = ADR_MEDIA_TEMPLATE.match(key)
        assert match is not None, key
        assert match.groups() == ("sig_abc123", digest, "png")

    def test_report_key_matches_the_adr_template(self) -> None:
        key = object_store.report_object_key("rep_9", 3, object_store.ReportFormat.PDF)

        match = ADR_REPORT_TEMPLATE.match(key)
        assert match is not None, key
        assert match.groups() == ("rep_9", "3", "pdf")

    @pytest.mark.parametrize("bad", ["", "  ", "a/b", "../etc", ".hidden"])
    def test_key_segments_that_would_escape_their_prefix_are_rejected(self, bad: str) -> None:
        """A slash invents a prefix level and hides the object from a listing."""
        digest = hashlib.sha256(PNG).hexdigest()
        with pytest.raises(ValueError, match="key segment"):
            object_store.media_object_key(bad, digest, "png")

    def test_report_version_must_start_at_one(self) -> None:
        with pytest.raises(ValueError, match="version must be >= 1"):
            object_store.report_object_key("rep_9", 0, object_store.ReportFormat.MD)


# --------------------------------------------------------------------------- #
# Content addressing
# --------------------------------------------------------------------------- #


class TestContentAddressing:
    """Identical bytes must always land on one key, and a repeat must do nothing.

    This is the property `docs/data-stores.md` §5.2 names as R2's idempotency
    mechanism. Without it, at-least-once delivery accumulates duplicate objects
    forever and replay stops converging.
    """

    async def test_same_bytes_produce_the_same_key(self, s3: FakeS3) -> None:
        first = await object_store.put_raw_payload("reddit", FETCHED_AT, PAYLOAD)
        second = await object_store.put_raw_payload("reddit", FETCHED_AT, PAYLOAD)
        assert first.key == second.key
        assert first.sha256 == second.sha256

    async def test_different_bytes_produce_different_keys(self, s3: FakeS3) -> None:
        first = await object_store.put_raw_payload("reddit", FETCHED_AT, PAYLOAD)
        second = await object_store.put_raw_payload("reddit", FETCHED_AT, PAYLOAD + b" ")
        assert first.key != second.key

    async def test_key_does_not_depend_on_the_codec(self, s3: FakeS3) -> None:
        """The digest is of the *original* bytes, so a codec change is invisible.

        A rollout that switches codec mid-flight must not fork the archive.
        """
        raw = await object_store.put_raw_payload(
            "reddit", FETCHED_AT, PAYLOAD, compression=object_store.Compression.NONE
        )
        gz = await object_store.put_raw_payload(
            "reddit", FETCHED_AT, PAYLOAD, compression=object_store.Compression.GZIP
        )
        assert raw.key == gz.key

    async def test_rewriting_the_same_payload_issues_no_put(self, s3: FakeS3) -> None:
        """The no-op is real, not merely harmless: no second PUT is sent."""
        first = await object_store.put_raw_payload("reddit", FETCHED_AT, PAYLOAD)
        assert first.already_present is False

        second = await object_store.put_raw_payload("reddit", FETCHED_AT, PAYLOAD)
        assert second.already_present is True
        assert second.stored_bytes is None

        puts = [call for call in s3.calls if call[0] == "put_object"]
        assert len(puts) == 1, s3.calls

    async def test_a_duplicate_write_costs_a_head_and_not_a_put(self, s3: FakeS3) -> None:
        """Class B HEAD instead of Class A PUT, and no eventually-consistent overwrite."""
        await object_store.put_raw_payload("reddit", FETCHED_AT, PAYLOAD)
        s3.calls.clear()

        await object_store.put_raw_payload("reddit", FETCHED_AT, PAYLOAD)
        assert [name for name, _ in s3.calls] == ["head_object"]

    async def test_report_artifact_rerender_is_idempotent(self, s3: FakeS3) -> None:
        """A retried render must not overwrite the version somebody already cited."""
        first = await object_store.put_report_artifact(
            "rep_1", 2, b"# findings", fmt=object_store.ReportFormat.MD
        )
        second = await object_store.put_report_artifact(
            "rep_1", 2, b"# findings", fmt=object_store.ReportFormat.MD
        )
        assert first.key == second.key
        assert second.already_present is True

    async def test_media_is_content_addressed_under_its_signal(self, s3: FakeS3) -> None:
        stored = await object_store.put_media(
            "sig_abc", PNG, content_type="image/png", extension="png"
        )
        assert stored.key == f"media/sig_abc/{hashlib.sha256(PNG).hexdigest()}.png"
        assert stored.compression is object_store.Compression.NONE

    @pytest.mark.parametrize(
        ("call", "message"),
        [
            ("raw", "raw_bytes is empty"),
            ("media", "data is empty"),
            ("report", "body is empty"),
        ],
    )
    async def test_empty_bodies_are_refused(self, s3: FakeS3, call: str, message: str) -> None:
        """An empty archive object looks exactly like a lost one on replay."""
        with pytest.raises(ValueError, match=message):
            if call == "raw":
                await object_store.put_raw_payload("reddit", FETCHED_AT, b"")
            elif call == "media":
                await object_store.put_media(
                    "sig_a", b"", content_type="image/png", extension="png"
                )
            else:
                await object_store.put_report_artifact(
                    "rep_1", 1, b"", fmt=object_store.ReportFormat.HTML
                )


# --------------------------------------------------------------------------- #
# Compression as a transport concern
# --------------------------------------------------------------------------- #


class TestCompressionIsTransportOnly:
    """Compression must change the bytes at rest and nothing a caller can see."""

    @pytest.mark.parametrize(
        "codec",
        [
            object_store.Compression.NONE,
            object_store.Compression.GZIP,
            object_store.Compression.ZSTD,
        ],
    )
    async def test_round_trip_is_transparent(
        self, s3: FakeS3, codec: object_store.Compression
    ) -> None:
        stored = await object_store.put_raw_payload(
            "reddit", FETCHED_AT, PAYLOAD, compression=codec
        )
        assert await object_store.get_raw_payload(stored.key) == PAYLOAD

    async def test_bytes_at_rest_really_are_compressed(self, s3: FakeS3) -> None:
        """Guards against a codec that silently degrades to a pass-through."""
        body = b'{"text": "' + b"repeat " * 500 + b'"}'
        stored = await object_store.put_raw_payload(
            "reddit", FETCHED_AT, body, compression=object_store.Compression.ZSTD
        )
        at_rest = s3.objects[stored.key]["Body"]

        assert at_rest != body
        assert len(at_rest) < len(body)
        assert stored.stored_bytes == len(at_rest)
        assert stored.size_bytes == len(body)

    async def test_uncompressed_objects_still_read(self, s3: FakeS3) -> None:
        """Objects written before compression existed must keep resolving."""
        stored = await object_store.put_raw_payload(
            "reddit", FETCHED_AT, PAYLOAD, compression=object_store.Compression.NONE
        )
        assert s3.objects[stored.key]["Body"] == PAYLOAD
        assert await object_store.get_raw_payload(stored.key) == PAYLOAD

    def test_gzip_output_is_deterministic(self) -> None:
        """`mtime=0`: the default stamps the clock into the header.

        Without it, compressing one payload twice yields two different byte
        strings, so two racing writers of identical content produce a genuine
        overwrite instead of the harmless one the design assumes.
        """
        once = object_store._compress(PAYLOAD, object_store.Compression.GZIP)
        twice = object_store._compress(PAYLOAD, object_store.Compression.GZIP)
        assert once == twice
        assert gzip.decompress(once) == PAYLOAD

    def test_codec_is_detected_from_the_bytes_not_the_key(self) -> None:
        assert (
            object_store.detect_compression(gzip.compress(PAYLOAD, mtime=0))
            is object_store.Compression.GZIP
        )
        assert object_store.detect_compression(PAYLOAD) is object_store.Compression.NONE
        assert object_store.detect_compression(b"") is object_store.Compression.NONE

    async def test_compressed_objects_declare_their_container(self, s3: FakeS3) -> None:
        """A direct reader outside this module must not be told these are JSON."""
        stored = await object_store.put_raw_payload(
            "reddit", FETCHED_AT, PAYLOAD, compression=object_store.Compression.ZSTD
        )
        assert s3.objects[stored.key]["ContentType"] == "application/zstd"
        assert s3.objects[stored.key]["Metadata"]["compression"] == "zstd"
        assert s3.objects[stored.key]["Metadata"]["content-sha256"] == stored.sha256


class TestReadIntegrity:
    """The key carries the expected digest, so a read can verify itself."""

    async def test_corrupted_bytes_are_refused(self, s3: FakeS3) -> None:
        """A replay from corrupt bytes yields confidently wrong Signals, silently."""
        stored = await object_store.put_raw_payload(
            "reddit", FETCHED_AT, PAYLOAD, compression=object_store.Compression.NONE
        )
        s3.objects[stored.key]["Body"] = b'{"tampered": true}'

        with pytest.raises(object_store.ObjectIntegrityError) as excinfo:
            await object_store.get_raw_payload(stored.key)
        assert excinfo.value.details["key"] == stored.key

    async def test_a_missing_object_is_not_found(self, s3: FakeS3) -> None:
        key = f"raw/reddit/2026/07/28/{'0' * 64}.json"
        with pytest.raises(NotFoundError):
            await object_store.get_raw_payload(key)

    @pytest.mark.parametrize(
        "bad",
        [
            "media/sig_a/" + "0" * 64 + ".png",
            "raw/reddit/2026/7/28/" + "0" * 64 + ".json",
            "raw/reddit/2026/07/28/nothex.json",
        ],
    )
    async def test_non_raw_keys_are_rejected(self, s3: FakeS3, bad: str) -> None:
        """Verification depends on parsing the key, so an unparseable one must fail."""
        with pytest.raises(ValueError, match="not a raw-payload key"):
            await object_store.get_raw_payload(bad)


# --------------------------------------------------------------------------- #
# Presigned URLs
# --------------------------------------------------------------------------- #


class TestPresignedUrls:
    """A presigned URL is a bearer credential; its limits are the security control."""

    async def test_signs_a_read_with_a_short_default_lifetime(self, s3: FakeS3) -> None:
        url = await object_store.presigned_url("reports/rep_1/1/report.pdf")

        assert url.startswith("https://fake.r2.example/")
        request = s3.presign_requests[0]
        assert request["method"] == "get_object"
        assert request["expires_in"] == object_store.DEFAULT_PRESIGN_TTL_SECONDS
        assert request["params"]["Key"] == "reports/rep_1/1/report.pdf"

    @pytest.mark.parametrize("ttl", [0, -1, object_store.MAX_PRESIGN_TTL_SECONDS + 1])
    async def test_unsignable_lifetimes_are_rejected(self, s3: FakeS3, ttl: int) -> None:
        """SigV4 caps expiry at seven days; a longer one fails at use, not at mint."""
        with pytest.raises(ValueError, match="expires_in_seconds"):
            await object_store.presigned_url("reports/rep_1/1/report.pdf", expires_in_seconds=ttl)

    async def test_download_filename_cannot_inject_a_header(self, s3: FakeS3) -> None:
        """The name can come from fetched content; a CR/LF in it is header injection."""
        await object_store.presigned_url(
            "reports/rep_1/1/report.pdf",
            download_filename='q1\r\nX-Injected: yes"',
        )
        disposition = s3.presign_requests[0]["params"]["ResponseContentDisposition"]
        assert "\r" not in disposition
        assert "\n" not in disposition
        assert disposition == 'attachment; filename="q1X-Injected: yes"'

    async def test_empty_key_is_rejected(self, s3: FakeS3) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await object_store.presigned_url("   ")


# --------------------------------------------------------------------------- #
# Media: the ceiling
# --------------------------------------------------------------------------- #


class TestMediaSizeCeiling:
    """The ceiling exists to stop an attacker-chosen URL exhausting worker memory.

    Every test here asserts *when* the refusal happens, not just that it happens.
    A downloader that buffers first and rejects second satisfies the exception
    type and none of the intent.
    """

    async def test_declared_size_over_the_ceiling_is_refused_before_the_body(self) -> None:
        stream = RecordingStream(b"x" * 4096)
        client = make_client(
            responder(content_type="image/png", content_length="999999999", stream=stream)
        )
        downloader = media_mod.MediaDownloader(client=client, max_bytes=1024)

        with pytest.raises(media_mod.MediaTooLargeError) as excinfo:
            await downloader.fetch("https://cdn.example.com/huge.png")

        assert stream.consumed is False, "the body was read before the ceiling was applied"
        assert excinfo.value.details["max_bytes"] == 1024
        await downloader.aclose()

    async def test_a_lying_content_length_is_still_bounded(self) -> None:
        """A hostile origin understates the header, or omits it under chunked encoding.

        The streaming counter is the gate that actually holds, and it must stop
        near the ceiling rather than after the whole body has arrived.
        """
        chunks = [b"x" * 65536 for _ in range(50)]
        stream = RecordingStream(*chunks)
        client = make_client(
            responder(content_type="image/webp", content_length="10", stream=stream)
        )
        downloader = media_mod.MediaDownloader(client=client, max_bytes=100_000)

        with pytest.raises(media_mod.MediaTooLargeError):
            await downloader.fetch("https://cdn.example.com/liar.webp")

        # 100_000 / 65_536 -> the ceiling is crossed on the second chunk.
        assert stream.chunks_yielded == 2, stream.chunks_yielded
        await downloader.aclose()

    async def test_a_body_without_content_length_under_the_ceiling_succeeds(self) -> None:
        stream = RecordingStream(PNG)
        client = make_client(responder(content_type="image/png", stream=stream))
        downloader = media_mod.MediaDownloader(client=client)

        result = await downloader.fetch("https://cdn.example.com/small.png")
        assert result.content == PNG
        assert result.size_bytes == len(PNG)
        assert result.sha256 == hashlib.sha256(PNG).hexdigest()
        await downloader.aclose()

    def test_a_non_positive_ceiling_is_rejected_at_construction(self) -> None:
        """`max_bytes=0` would disable the control while looking configured."""
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            media_mod.MediaDownloader(max_bytes=0)


class TestMediaContentTypeAllowlist:
    """Only types we can safely store and serve back are downloaded at all."""

    @pytest.mark.parametrize(
        "content_type",
        [
            "image/svg+xml",
            "text/html",
            "application/octet-stream",
            "application/zip",
            "",
        ],
    )
    async def test_disallowed_types_are_refused_before_the_body(self, content_type: str) -> None:
        """SVG and HTML are scriptable; octet-stream declines to say what it is."""
        stream = RecordingStream(b"<svg onload=alert(1)>")
        client = make_client(responder(content_type=content_type or None, stream=stream))
        downloader = media_mod.MediaDownloader(client=client)

        with pytest.raises(media_mod.MediaTypeNotAllowedError):
            await downloader.fetch("https://cdn.example.com/thing")

        assert stream.consumed is False, "the body was read before the type was checked"
        await downloader.aclose()

    async def test_content_type_parameters_do_not_defeat_the_check(self) -> None:
        stream = RecordingStream(JPEG)
        client = make_client(responder(content_type="IMAGE/JPEG; charset=binary", stream=stream))
        downloader = media_mod.MediaDownloader(client=client)

        result = await downloader.fetch("https://cdn.example.com/photo.jpg")
        assert result.content_type == "image/jpeg"
        assert result.kind is MediaKind.IMAGE
        await downloader.aclose()

    async def test_bytes_contradicting_the_declared_type_are_refused(self) -> None:
        """The declared type is attacker-controlled; the magic bytes are not."""
        stream = RecordingStream(b"<html><script>alert(1)</script></html>")
        client = make_client(responder(content_type="image/png", stream=stream))
        downloader = media_mod.MediaDownloader(client=client)

        with pytest.raises(media_mod.MediaTypeNotAllowedError, match="do not match"):
            await downloader.fetch("https://cdn.example.com/fake.png")
        await downloader.aclose()

    def test_the_allowlist_excludes_the_scriptable_types(self) -> None:
        """Pinned so a later 'just add SVG' cannot pass review by accident."""
        assert "image/svg+xml" not in media_mod.ALLOWED_CONTENT_TYPES
        assert "text/html" not in media_mod.ALLOWED_CONTENT_TYPES
        assert "application/octet-stream" not in media_mod.ALLOWED_CONTENT_TYPES

    @pytest.mark.parametrize(
        ("content_type", "expected"),
        [
            ("image/png", MediaKind.IMAGE),
            ("video/mp4", MediaKind.VIDEO),
            ("audio/mpeg", MediaKind.AUDIO),
            ("application/pdf", MediaKind.DOCUMENT),
            ("application/x-nonsense", MediaKind.UNKNOWN),
        ],
    )
    def test_kind_is_derived_from_the_top_level_type(
        self, content_type: str, expected: MediaKind
    ) -> None:
        assert media_mod.kind_for_content_type(content_type) is expected


class TestMediaUrlValidation:
    """The URL came out of a Reddit post. It is not a trusted input."""

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "data:image/png;base64,AAAA",
            "ftp://example.com/x.png",
            "https://user:pass@example.com/x.png",
            "http://localhost/x.png",
            "http://127.0.0.1/x.png",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5:6379/x.png",
            "http://[::1]/x.png",
        ],
    )
    async def test_unfetchable_urls_are_refused_without_a_request(self, url: str) -> None:
        """No socket is opened: the handler asserts if it is ever reached."""

        def handle(_request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"a request was issued for {url!r}")

        downloader = media_mod.MediaDownloader(client=make_client(handle))
        with pytest.raises(media_mod.MediaRejectedError):
            await downloader.fetch(url)
        await downloader.aclose()

    async def test_a_redirect_into_the_metadata_endpoint_is_refused(self) -> None:
        """The classic SSRF escalation: a public URL that redirects inward."""
        seen: list[str] = []

        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if request.url.host == "cdn.example.com":
                return httpx.Response(
                    302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
                )
            raise AssertionError("followed a redirect into a private address")

        downloader = media_mod.MediaDownloader(client=make_client(handle))
        with pytest.raises(media_mod.MediaRejectedError):
            await downloader.fetch("https://cdn.example.com/avatar.png")

        assert seen == ["https://cdn.example.com/avatar.png"]
        await downloader.aclose()

    async def test_a_legitimate_relative_redirect_is_followed(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/avatar":
                return httpx.Response(302, headers={"location": "/cdn/avatar.png"})
            return httpx.Response(
                200, headers={"content-type": "image/png"}, stream=RecordingStream(PNG)
            )

        downloader = media_mod.MediaDownloader(client=make_client(handle))
        result = await downloader.fetch("https://cdn.example.com/avatar")
        assert result.content == PNG
        await downloader.aclose()

    async def test_a_redirect_loop_terminates(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "https://cdn.example.com/loop"})

        downloader = media_mod.MediaDownloader(client=make_client(handle), max_redirects=2)
        with pytest.raises(media_mod.MediaFetchError, match="redirects"):
            await downloader.fetch("https://cdn.example.com/loop")
        await downloader.aclose()

    async def test_an_injected_client_cannot_disable_redirect_revalidation(self) -> None:
        """`follow_redirects=False` is restated per request, not left to the client.

        Otherwise a caller passing a client configured to follow redirects would
        silently bypass every per-hop address check.
        """

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.host == "cdn.example.com":
                return httpx.Response(302, headers={"location": "http://127.0.0.1/x.png"})
            raise AssertionError("httpx followed the redirect itself")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle), follow_redirects=True)
        downloader = media_mod.MediaDownloader(client=client)
        with pytest.raises(media_mod.MediaRejectedError):
            await downloader.fetch("https://cdn.example.com/avatar.png")
        await client.aclose()

    async def test_an_injected_client_is_not_closed_by_the_downloader(self) -> None:
        """It belongs to the caller; closing it breaks whoever else is sharing it."""
        client = make_client(responder())
        downloader = media_mod.MediaDownloader(client=client)
        await downloader.aclose()
        assert client.is_closed is False
        await client.aclose()


class TestMediaTransportFailures:
    async def test_an_error_status_does_not_read_the_body(self) -> None:
        """An unbounded error page from a hostile origin is still unbounded."""
        stream = RecordingStream(b"y" * 4096)
        client = make_client(responder(status=503, stream=stream))
        downloader = media_mod.MediaDownloader(client=client)

        with pytest.raises(media_mod.MediaFetchError) as excinfo:
            await downloader.fetch("https://cdn.example.com/x.png")

        assert stream.consumed is False
        assert excinfo.value.details["status"] == 503
        await downloader.aclose()

    async def test_a_transport_error_reports_only_the_exception_class(self) -> None:
        """httpx messages embed the URL, which is fetched content."""

        def handle(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out talking to https://cdn.example.com/secret")

        downloader = media_mod.MediaDownloader(client=make_client(handle))
        with pytest.raises(media_mod.MediaFetchError) as excinfo:
            await downloader.fetch("https://cdn.example.com/x.png")

        assert excinfo.value.details == {"error": "ConnectTimeout"}
        assert "cdn.example.com" not in str(excinfo.value)
        await downloader.aclose()

    async def test_an_empty_body_is_a_fetch_failure(self) -> None:
        downloader = media_mod.MediaDownloader(
            client=make_client(responder(stream=RecordingStream()))
        )
        with pytest.raises(media_mod.MediaFetchError, match="empty body"):
            await downloader.fetch("https://cdn.example.com/x.png")
        await downloader.aclose()

    async def test_a_redirect_without_a_location_fails_cleanly(self) -> None:
        def handle(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(302)

        downloader = media_mod.MediaDownloader(client=make_client(handle))
        with pytest.raises(media_mod.MediaFetchError, match="no Location"):
            await downloader.fetch("https://cdn.example.com/x.png")
        await downloader.aclose()


# --------------------------------------------------------------------------- #
# Media: archiving and Phase 2 hooks
# --------------------------------------------------------------------------- #


class TestMediaArchiving:
    async def test_archive_stores_the_bytes_and_returns_a_usable_ref(self, s3: FakeS3) -> None:
        client = make_client(responder(content_type="image/png", stream=RecordingStream(PNG)))
        downloader = media_mod.MediaDownloader(client=client)

        ref = await downloader.archive("https://cdn.example.com/a.png", signal_id="sig_xyz")

        assert isinstance(ref, MediaRef)
        assert ref.kind is MediaKind.IMAGE
        assert ref.source_url == "https://cdn.example.com/a.png"
        assert ref.object_key == f"media/sig_xyz/{hashlib.sha256(PNG).hexdigest()}.png"
        assert ref.mime_type == "image/png"
        assert ref.bytes == len(PNG)
        # Not guessed: filling these requires decoding the image (Phase 2).
        assert ref.width is None and ref.height is None
        assert s3.objects[ref.object_key]["Body"] == PNG
        await downloader.aclose()

    async def test_the_extension_comes_from_the_type_not_the_url(self, s3: FakeS3) -> None:
        """The URL is attacker-controlled; the key must stay deterministic."""
        client = make_client(responder(content_type="image/jpeg", stream=RecordingStream(JPEG)))
        downloader = media_mod.MediaDownloader(client=client)

        ref = await downloader.archive("https://cdn.example.com/photo.php?x=1", signal_id="sig_xyz")
        assert ref.object_key is not None and ref.object_key.endswith(".jpg")
        await downloader.aclose()


class TestPhaseTwoHooks:
    """The interfaces exist now; the providers do not, and say so."""

    async def test_the_shipped_transcriber_names_what_is_missing(self) -> None:
        with pytest.raises(NotImplementedError) as excinfo:
            await media_mod.UnavailableTranscriber().transcribe(b"\x00", content_type="audio/mpeg")
        message = str(excinfo.value)
        assert "Phase 2" in message
        assert "Modal" in message

    async def test_the_shipped_thumbnailer_names_what_is_missing(self) -> None:
        with pytest.raises(NotImplementedError) as excinfo:
            await media_mod.UnavailableThumbnailer().render(
                PNG, content_type="image/png", max_edge_px=256
            )
        message = str(excinfo.value)
        assert "Pixel" in message or "pixel" in message
        assert "Pillow" in message

    async def test_attach_transcript_works_with_an_injected_transcriber(self, s3: FakeS3) -> None:
        """Real code today: only the model is missing, and it is a parameter."""

        class FakeTranscriber:
            model = "fake-asr-1"

            async def transcribe(self, data: bytes, *, content_type: str) -> str:
                return "hello world"

        source = MediaRef(kind=MediaKind.AUDIO, mime_type="audio/mpeg", object_key="media/s/a.mp3")
        result = await media_mod.attach_transcript(
            source, b"\x00\x01", transcriber=FakeTranscriber(), signal_id="sig_t"
        )

        assert result.transcript_ref is not None
        assert result.transcript_ref.startswith("media/sig_t/")
        assert result.transcript_ref.endswith(".txt")
        assert s3.objects[result.transcript_ref]["Body"] == b"hello world"
        # The argument is untouched: it usually lives inside a Signal already.
        assert source.transcript_ref is None

    async def test_an_empty_transcript_is_refused(self, s3: FakeS3) -> None:
        """Empty text is indistinguishable from silence and would index as success."""

        class SilentTranscriber:
            model = "fake-asr-1"

            async def transcribe(self, data: bytes, *, content_type: str) -> str:
                return "   "

        source = MediaRef(kind=MediaKind.AUDIO, mime_type="audio/mpeg")
        with pytest.raises(ValueError, match="empty text"):
            await media_mod.attach_transcript(
                source, b"\x00", transcriber=SilentTranscriber(), signal_id="sig_t"
            )

    async def test_make_thumbnail_stores_a_separate_object(self, s3: FakeS3) -> None:
        class FakeThumbnailer:
            async def render(
                self, data: bytes, *, content_type: str, max_edge_px: int
            ) -> tuple[bytes, str]:
                return PNG, "image/png"

        source = MediaRef(
            kind=MediaKind.IMAGE, mime_type="image/jpeg", source_url="https://x/y.jpg"
        )
        thumb = await media_mod.make_thumbnail(
            source, JPEG, thumbnailer=FakeThumbnailer(), signal_id="sig_th"
        )

        assert thumb.object_key == f"media/sig_th/{hashlib.sha256(PNG).hexdigest()}.png"
        assert thumb.kind is MediaKind.IMAGE
        assert source.object_key is None

    async def test_a_thumbnailer_returning_an_unstorable_type_is_refused(self, s3: FakeS3) -> None:
        class BadThumbnailer:
            async def render(
                self, data: bytes, *, content_type: str, max_edge_px: int
            ) -> tuple[bytes, str]:
                return b"<svg/>", "image/svg+xml"

        source = MediaRef(kind=MediaKind.IMAGE, mime_type="image/jpeg")
        with pytest.raises(media_mod.MediaTypeNotAllowedError):
            await media_mod.make_thumbnail(
                source, JPEG, thumbnailer=BadThumbnailer(), signal_id="sig_th"
            )

    def test_the_placeholders_satisfy_the_protocols(self) -> None:
        """Runtime-checkable so a wiring mistake fails at construction, not at use."""
        assert isinstance(media_mod.UnavailableTranscriber(), media_mod.Transcriber)
        assert isinstance(media_mod.UnavailableThumbnailer(), media_mod.Thumbnailer)
