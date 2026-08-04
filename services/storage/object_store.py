"""The object-storage service layer: the only sanctioned way to reach R2.

`backend/db/r2.py` owns the *client* -- credentials, timeouts, and the thread hop
that keeps blocking `boto3` calls off the event loop. This module owns the
*policy*: which keys exist, what goes in them, and what a repeated write means.
ADR-0006 states the rule from both ends -- all access goes through this module,
and no module outside `backend/db/r2.py` imports `boto3` -- because that is what
keeps the provider swappable for any S3-compatible target.

Three decisions are encoded here and each has a failure mode behind it.

**Writes are content-addressed, so a repeat is a no-op.** The raw key embeds the
SHA-256 of the payload bytes, so re-`PUT`ting the same payload cannot produce a
second object. That is what makes step 1 of `docs/data-stores.md` §5.1 safe to
replay under at-least-once delivery. This module goes one step further than
"harmless" and makes the repeat *observable*: it `HEAD`s first and skips the
`PUT` entirely (`StoredObject.already_present`). Two reasons, and only the second
is about money. R2 is strongly consistent for **new** keys and eventually
consistent for **overwrites** (`docs/data-stores.md` §3.6); overwriting an object
with bytes identical to the ones already there buys nothing and exercises the
weak path for no reason. And a Class B `HEAD` is roughly a tenth the price of a
Class A `PUT`, so under replay -- where duplicates are the common case, not the
exception -- checking first is also cheaper.

**Compression is a transport concern and must never touch the key.** ADR-0006 is
explicit that `raw/reddit/2026/07/28/8f14e45f….json` keeps its `.json` suffix and
no `.zst` is appended, because `content.raw_ref` values already recorded in
PostgreSQL are permanent -- change the template and every one of them stops
resolving. So the codec lives in the *bytes*, not the name: everything written
here is compressed on the way in and decompressed on the way out by sniffing the
container's magic bytes, which is exactly what `file(1)` does and requires no
extra request to read object metadata. Both codecs we write (gzip, zstd) begin
with byte sequences that no JSON, PDF, HTML or Markdown document can begin with,
so the sniff is unambiguous for every object class this module stores.

**A presigned URL is a bearer credential.** Anyone holding the link can read the
object for as long as it lives, with no reference to the caller's permissions --
which is why the default lifetime here is minutes rather than the AWS default of
an hour, and why only *reads* can be presigned. A presigned `PUT` would let a
link-holder overwrite an object in an archive whose entire value is that it is
immutable.

**One deliberate deviation, called out because it is a deviation.**
`presigned_url()` calls a `botocore` method on the client returned by
`r2.get_s3()`, which the r2 module's docstring otherwise forbids. That rule
exists to stop blocking network I/O leaking back onto the event loop;
`generate_presigned_url` performs no I/O at all -- it is a local SigV4 signing
computation -- so it does not reintroduce what the rule protects against. It is
still wrapped in `asyncio.to_thread` for consistency with every other call into
that client, and it is the only such call in the codebase.

Layer note: **L2 service** (`docs/architecture.md` §6.1) -- may import `models/`,
`backend/`, and is imported by `workers/`, `agents/` and `backend/api/`.
"""

from __future__ import annotations

import asyncio
import enum
import gzip
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import zstandard
from botocore.exceptions import BotoCoreError, ClientError

from backend.core.config import get_settings
from backend.core.exceptions import ExternalServiceError
from backend.db import r2
from backend.db.r2 import RAW_KEY_PREFIX, raw_object_key
from models.enums import Platform

__all__ = [
    "DEFAULT_PRESIGN_TTL_SECONDS",
    "DEFAULT_RAW_COMPRESSION",
    "MAX_PRESIGN_TTL_SECONDS",
    "MEDIA_KEY_PREFIX",
    "RAW_KEY_PREFIX",
    "REPORT_KEY_PREFIX",
    "Compression",
    "ObjectIntegrityError",
    "ReportFormat",
    "StoredObject",
    "detect_compression",
    "get_raw_payload",
    "media_object_key",
    "presigned_url",
    "put_media",
    "put_raw_payload",
    "put_report_artifact",
    "raw_object_key",
    "report_object_key",
]

MEDIA_KEY_PREFIX: Final = "media"
"""First segment of the media key template (ADR-0006)."""

REPORT_KEY_PREFIX: Final = "reports"
"""First segment of the report-artifact key template (ADR-0006)."""

DEFAULT_PRESIGN_TTL_SECONDS: Final = 300
"""Five minutes. Short because the URL *is* the authorization.

Long enough to survive a slow client on a bad connection, short enough that a
link pasted into a chat or captured in a proxy log stops working before anyone
gets round to using it. Callers that need a shareable artifact link should be
minting one per request, not caching one for an hour.
"""

MAX_PRESIGN_TTL_SECONDS: Final = 7 * 24 * 60 * 60
"""SigV4's own ceiling. A longer expiry is rejected by the signing algorithm."""

_GZIP_LEVEL: Final = 6

_ZSTD_LEVEL: Final = 3
"""Level 3 is zstd's default: gzip-6 ratio at several times the throughput.

Raising it costs CPU on the ingestion hot path -- one compression per fetched
record -- to save storage on an archive that is already the cheapest byte in the
system. The ratio matters, the last few percent of it does not.
"""

_SLUG_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
"""Identifiers safe to interpolate into a key: no slash, no empty, no leading dot.

A slash silently invents a prefix level and hides the object from a listing of
the prefix it was supposed to be in; a leading dot produces keys some S3 tooling
declines to enumerate.
"""

_EXTENSION_RE: Final = re.compile(r"[a-z0-9]{1,8}")

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")

_RAW_KEY_RE: Final = re.compile(
    rf"^{RAW_KEY_PREFIX}/(?P<platform>[^/]+)/(?P<year>\d{{4}})/(?P<month>\d{{2}})/"
    r"(?P<day>\d{2})/(?P<digest>[0-9a-f]{64})\.json$"
)
"""The ADR-0006 raw template, parsed rather than merely built.

`get_raw_payload` uses the digest in the key to verify what it read, so the key
has to be machine-readable in both directions.
"""


class Compression(enum.StrEnum):
    """Transport codec applied to an object's bytes. Never part of its key."""

    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"


DEFAULT_RAW_COMPRESSION: Final = Compression.ZSTD
"""Raw payloads are compressed by default; connector JSON compresses 5-10x.

Retention is 400 days (`StorageSettings.raw_retention_days`) and the archive is
write-once/read-rarely, so storage volume -- not read latency -- is what this
prefix costs. zstd over gzip because it decompresses several times faster for a
better ratio, and the read path is the one a human is waiting on during an audit.
"""

_CODEC_CONTENT_TYPE: Final[dict[Compression, str]] = {
    Compression.GZIP: "application/gzip",
    Compression.ZSTD: "application/zstd",
}
"""What to declare as `Content-Type` once bytes have been compressed.

Deliberately *not* the payload's own type. `backend/db/r2.py` exposes no
`ContentEncoding` parameter, so declaring `application/json` over zstd bytes
would be a plain lie to anything reading the object outside this module -- an
`aws s3 cp`, a browser following a presigned URL, an auditor with the console
open. Naming the container instead tells such a reader exactly what to do.
"""

_MAGIC: Final[tuple[tuple[bytes, Compression], ...]] = (
    (b"\x28\xb5\x2f\xfd", Compression.ZSTD),
    (b"\x1f\x8b", Compression.GZIP),
)
"""Container signatures, longest first.

No JSON document (`{`, `[`, `"`, a digit, or whitespace), PDF (`%PDF-`), HTML
(`<`) or Markdown text can begin with either sequence, so sniffing cannot
misfire on an object this module stores uncompressed -- including one written
before compression was switched on.
"""


class ObjectIntegrityError(ExternalServiceError):
    """Bytes read back do not hash to the digest embedded in their own key.

    Its own class because the operational response is nothing like a generic R2
    failure: a retry will not help, and the object has to be re-derived from the
    source or restored from a backup copy. Silent corruption in the raw archive
    is the one failure that makes a replay produce *wrong* Signals rather than no
    Signals, so it is surfaced loudly rather than papered over.
    """

    code = "object_integrity_error"
    default_message = "A stored object does not match its content address."


@dataclass(frozen=True, slots=True)
class StoredObject:
    """The result of one write. Frozen -- callers record it, they do not edit it."""

    key: str
    sha256: str
    """Digest of the **original** bytes, before compression. This is the identity."""
    size_bytes: int
    """Size of the original bytes. What `MediaRef.bytes` and audits care about."""
    stored_bytes: int | None
    """Bytes actually written, or `None` when the write was skipped as a duplicate."""
    compression: Compression
    already_present: bool
    """True when an object was already at this key and the `PUT` was skipped."""


# --------------------------------------------------------------------------- #
# Key construction
# --------------------------------------------------------------------------- #


def media_object_key(signal_id: str, sha256: str, extension: str) -> str:
    """Build `media/{signal_id}/{sha256}.{ext}` (ADR-0006).

    Grouped by Signal rather than by date, unlike the raw prefix, because media
    is deleted with its Signal: erasure (`docs/security-and-privacy.md` §7) is a
    prefix delete on `media/{signal_id}/`, which is one operation instead of a
    scan. The raw prefix cannot do that -- it has no `signal_id` at write time --
    and pays for it with a date-partitioned sweep.

    The digest is of the media bytes, so the same image referenced by two Signals
    is stored twice, once under each. That is intentional: sharing one object
    between Signals would make the erasure above delete media still referenced by
    a Signal that was not erased.

    Args:
        signal_id: The owning `Signal.id`, e.g. `sig_1a2b…`.
        sha256: Hex SHA-256 of the media bytes.
        extension: Lowercase, no leading dot, e.g. `jpg`.

    Raises:
        ValueError: Any component would produce an unsafe or unparseable key.
    """
    slug = _require_slug(signal_id, "signal_id")
    digest = _require_digest(sha256)
    ext = extension.strip().lower().lstrip(".")
    if not _EXTENSION_RE.fullmatch(ext):
        raise ValueError(f"extension must be 1-8 lowercase alphanumerics, got {extension!r}")
    return f"{MEDIA_KEY_PREFIX}/{slug}/{digest}.{ext}"


class ReportFormat(enum.StrEnum):
    """The three artifact formats ADR-0006 fixes for the `reports/` prefix."""

    PDF = "pdf"
    HTML = "html"
    MD = "md"

    @property
    def content_type(self) -> str:
        return _REPORT_CONTENT_TYPE[self]


_REPORT_CONTENT_TYPE: Final[dict[ReportFormat, str]] = {
    ReportFormat.PDF: "application/pdf",
    ReportFormat.HTML: "text/html; charset=utf-8",
    ReportFormat.MD: "text/markdown; charset=utf-8",
}


def report_object_key(report_id: str, version: int, fmt: ReportFormat) -> str:
    """Build `reports/{report_id}/{version}/report.{pdf|html|md}` (ADR-0006).

    Versioned in the key rather than overwritten, so a report that has been cited
    or shared keeps resolving to the bytes that were actually reviewed. A revised
    report is a new version; it is never a rewrite of an old one.

    Raises:
        ValueError: `report_id` is unsafe as a key segment, or `version` < 1.
    """
    slug = _require_slug(report_id, "report_id")
    if version < 1:
        raise ValueError(f"version must be >= 1; versions start at 1, got {version}")
    return f"{REPORT_KEY_PREFIX}/{slug}/{version}/report.{fmt.value}"


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #


async def put_raw_payload(
    platform: Platform | str,
    fetched_at: datetime,
    raw_bytes: bytes,
    *,
    compression: Compression = DEFAULT_RAW_COMPRESSION,
) -> StoredObject:
    """Archive one fetched payload, returning its key and digest.

    Step 1 of `docs/data-stores.md` §5.1 and a hard precondition for everything
    after it: the payload exists in R2 before any store references it, so a
    PostgreSQL row can never point at an object that is not there.

    **Writing the same payload twice does nothing.** The key is derived from the
    SHA-256 of `raw_bytes`, so identical bytes always land on the same key; this
    checks for the object first and returns `already_present=True` without
    issuing the `PUT`. That is what makes a redelivered Kafka message, a retried
    connector page, or a re-run backfill converge instead of accumulating.

    Two racing writers can both miss the check and both `PUT`. That is harmless
    and not worth a lock: the bytes are identical, so the loser overwrites the
    winner with the same content. (Identical only if both chose the same codec --
    a mixed-codec rollout produces two different byte strings for one key, which
    still reads correctly because `get_raw_payload` sniffs the container rather
    than assuming one.)

    Args:
        platform: Platform slug, e.g. `"reddit"` or `Platform.REDDIT`.
        fetched_at: When the payload was retrieved. Must be timezone-aware.
        raw_bytes: The payload exactly as received, before any parsing.
        compression: Transport codec. Does not affect the key.

    Raises:
        ValueError: `raw_bytes` is empty, or a key component is invalid.
        ExternalServiceError: R2 rejected or could not serve the operation.
    """
    if not raw_bytes:
        # An empty archive object is indistinguishable from a lost one, and a
        # replay from it produces nothing while looking like it worked. A
        # connector that reached here with no bytes has a bug worth surfacing.
        raise ValueError("raw_bytes is empty; there is no payload to archive")

    digest = hashlib.sha256(raw_bytes).hexdigest()
    key = raw_object_key(platform, fetched_at, digest)
    return await _put_content_addressed(
        key=key,
        digest=digest,
        original=raw_bytes,
        payload_content_type="application/json",
        compression=compression,
    )


async def put_media(
    signal_id: str,
    data: bytes,
    *,
    content_type: str,
    extension: str,
) -> StoredObject:
    """Archive one media object under its owning Signal, content-addressed.

    Never compressed, and that is not an oversight: every format this platform
    accepts (JPEG, PNG, WebP, MP4, MP3, PDF) is already entropy-coded. A second
    pass costs CPU on every fetch, typically *grows* the object by the container
    overhead, and gains nothing.

    The size ceiling and the content-type allowlist are enforced upstream, in
    `services/storage/media.py`, before a single byte is downloaded. This
    function receives bytes that are already in memory and is therefore far too
    late to be the place that bounds them.

    Raises:
        ValueError: `data` is empty, or a key component is invalid.
        ExternalServiceError: R2 rejected or could not serve the operation.
    """
    if not data:
        raise ValueError("data is empty; there is no media to archive")

    digest = hashlib.sha256(data).hexdigest()
    key = media_object_key(signal_id, digest, extension)
    return await _put_content_addressed(
        key=key,
        digest=digest,
        original=data,
        payload_content_type=content_type,
        compression=Compression.NONE,
    )


async def put_report_artifact(
    report_id: str,
    version: int,
    body: bytes,
    *,
    fmt: ReportFormat,
    compression: Compression = Compression.NONE,
) -> StoredObject:
    """Store a rendered report artifact at its versioned key.

    Uncompressed by default, unlike the raw archive, because these bytes are
    served to users -- often straight from a presigned URL through the CDN. With
    no way to set `Content-Encoding` through `backend/db/r2.py`, a compressed
    artifact would arrive at a browser as an undecodable download. Compression is
    available for callers that will only ever read the artifact back through this
    module, and is off for the path that actually exists.

    Idempotent in the same way as the raw prefix: `(report_id, version)` names
    one immutable rendering, so a retried render finds its object already there
    and skips the write. A *changed* report is a new version, never a rewrite --
    see `report_object_key`.

    Raises:
        ValueError: `body` is empty, or a key component is invalid.
        ExternalServiceError: R2 rejected or could not serve the operation.
    """
    if not body:
        raise ValueError("body is empty; there is no artifact to store")

    digest = hashlib.sha256(body).hexdigest()
    key = report_object_key(report_id, version, fmt)
    return await _put_content_addressed(
        key=key,
        digest=digest,
        original=body,
        payload_content_type=fmt.content_type,
        compression=compression,
    )


async def _put_content_addressed(
    *,
    key: str,
    digest: str,
    original: bytes,
    payload_content_type: str,
    compression: Compression,
) -> StoredObject:
    """Shared write path: skip if present, compress, record what we did.

    The metadata written here is for readers that are not this module -- an
    operator with the R2 console open, a future migration script. It records the
    codec and the original digest and length so an object can be understood
    without guessing, but nothing on the read path depends on it, because reading
    metadata would cost an extra billed request per object.
    """
    if await r2.object_exists(key):
        return StoredObject(
            key=key,
            sha256=digest,
            size_bytes=len(original),
            stored_bytes=None,
            compression=compression,
            already_present=True,
        )

    body = _compress(original, compression)
    content_type = _CODEC_CONTENT_TYPE.get(compression, payload_content_type)
    await r2.put_object(
        key,
        body,
        content_type=content_type,
        metadata={
            "compression": compression.value,
            "content-sha256": digest,
            "content-length": str(len(original)),
        },
    )
    return StoredObject(
        key=key,
        sha256=digest,
        size_bytes=len(original),
        stored_bytes=len(body),
        compression=compression,
        already_present=False,
    )


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #


async def get_raw_payload(key: str) -> bytes:
    """Read an archived payload back, decompressed and verified.

    Transparent to whatever codec the object was written with -- including none,
    for objects predating compression -- because the container is detected from
    the bytes rather than from the key or from object metadata.

    The digest embedded in the key is checked against what came back. Content
    addressing makes that check almost free to design: the expected hash is
    already in the name, so verification costs one pass over bytes that are
    already in memory. It is worth doing because this archive is the input to
    every replay, and a replay from corrupted bytes produces confidently wrong
    Signals rather than a visible failure.

    Args:
        key: A `raw/…` key produced by `raw_object_key`.

    Raises:
        ValueError: `key` is not a raw-payload key.
        NotFoundError: No object at that key.
        ObjectIntegrityError: The bytes do not hash to the digest in the key.
        ExternalServiceError: R2 could not serve the read.
    """
    match = _RAW_KEY_RE.fullmatch(key)
    if match is None:
        raise ValueError(
            f"{key!r} is not a raw-payload key. Expected the ADR-0006 template "
            "raw/{platform}/{yyyy}/{mm}/{dd}/{sha256}.json"
        )

    payload = _decompress(await r2.get_object(key))

    actual = hashlib.sha256(payload).hexdigest()
    expected = match.group("digest")
    if actual != expected:
        raise ObjectIntegrityError(
            f"Object {key!r} does not match its content address.",
            # The bytes themselves are deliberately absent: `details` is
            # serialized into logs and HTTP responses, and this is untrusted
            # fetched content (`docs/security-and-privacy.md`).
            details={"key": key, "expected_sha256": expected, "actual_sha256": actual},
        )
    return payload


async def presigned_url(
    key: str,
    *,
    expires_in_seconds: int = DEFAULT_PRESIGN_TTL_SECONDS,
    download_filename: str | None = None,
) -> str:
    """Mint a time-limited, read-only URL for one object.

    Read-only by construction. Presigning a `PUT` would hand whoever holds the
    link the ability to overwrite an object in an archive whose only guarantee is
    that it is immutable, and links leak -- into chat, into browser history, into
    referrer headers.

    No existence check is performed. That would cost a billed request per link on
    a path where the object is nearly always there, and a link to a missing key
    fails at the moment of use with a perfectly clear 404.

    Args:
        key: Object key to sign.
        expires_in_seconds: Lifetime of the link. See `DEFAULT_PRESIGN_TTL_SECONDS`.
        download_filename: When set, forces a download with this name via
            `Content-Disposition` instead of letting the browser render it inline.

    Raises:
        ValueError: `key` is empty or the TTL is outside the signable range.
        ConfigurationError: R2 is not configured in this process.
        ExternalServiceError: The signer rejected the request.
    """
    if not key.strip():
        raise ValueError("key must be non-empty to presign")
    if not 1 <= expires_in_seconds <= MAX_PRESIGN_TTL_SECONDS:
        raise ValueError(
            f"expires_in_seconds must be between 1 and {MAX_PRESIGN_TTL_SECONDS} "
            f"(SigV4's own ceiling), got {expires_in_seconds}"
        )

    params: dict[str, Any] = {"Bucket": get_settings().storage.bucket, "Key": key}
    if download_filename is not None:
        params["ResponseContentDisposition"] = _content_disposition(download_filename)

    client = await r2.get_s3()
    try:
        # See the module docstring: this is the one sanctioned botocore call
        # outside `backend/db/r2.py`. It signs locally and opens no socket; the
        # thread hop is for consistency, not because it blocks.
        url = await asyncio.to_thread(
            client.generate_presigned_url,
            ClientMethod="get_object",
            Params=params,
            ExpiresIn=expires_in_seconds,
        )
    except (ClientError, BotoCoreError) as exc:
        raise ExternalServiceError(
            f"Failed to presign a URL for {key!r}.",
            details={"key": key},
            cause=exc,
        ) from exc
    return str(url)


def _content_disposition(filename: str) -> str:
    """Build a `Content-Disposition` value from an untrusted filename.

    The name can originate in fetched content -- a media URL's last path segment,
    a report title a user typed. Both `"` and any CR/LF have to go: the value is
    signed into a query parameter that a proxy or an S3-compatible server will
    reflect into a response header, and a bare newline there is header injection.
    """
    cleaned = "".join(c for c in filename if c.isprintable() and c not in '"\\')
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = "download"
    return f'attachment; filename="{cleaned}"'


# --------------------------------------------------------------------------- #
# Compression
# --------------------------------------------------------------------------- #


def _compress(data: bytes, codec: Compression) -> bytes:
    """Apply the transport codec. Deterministic: same bytes in, same bytes out.

    `mtime=0` on gzip is what makes that true. The default embeds the current
    clock in the header, so compressing one payload twice yields two different
    byte strings -- which would defeat ETag comparison, make a "did this write
    change anything?" check impossible, and turn two racing writers of identical
    content into a genuine overwrite instead of a harmless one.
    """
    if codec is Compression.NONE:
        return data
    if codec is Compression.GZIP:
        return gzip.compress(data, compresslevel=_GZIP_LEVEL, mtime=0)
    return zstandard.ZstdCompressor(level=_ZSTD_LEVEL).compress(data)


def detect_compression(data: bytes) -> Compression:
    """Identify the container from its leading bytes. See `_MAGIC` for why this is safe."""
    for magic, codec in _MAGIC:
        if data.startswith(magic):
            return codec
    return Compression.NONE


def _decompress(data: bytes) -> bytes:
    """Undo whatever codec the object was written with.

    Expansion is unbounded on purpose. The only writer to this bucket is this
    module, holding credentials no untrusted party has, so a decompression bomb
    would have to be planted by someone who already has write access to the
    authoritative archive -- at which point a memory ceiling here is not the
    control that matters. The ceiling that *does* matter guards the untrusted
    direction and lives in `services/storage/media.py`.
    """
    codec = detect_compression(data)
    if codec is Compression.NONE:
        return data
    if codec is Compression.GZIP:
        return gzip.decompress(data)
    return zstandard.ZstdDecompressor().decompress(data)


# --------------------------------------------------------------------------- #
# Key component validation
# --------------------------------------------------------------------------- #


def _require_slug(value: str, field: str) -> str:
    candidate = value.strip()
    if not _SLUG_RE.fullmatch(candidate):
        raise ValueError(
            f"{field} must be a non-empty key segment of [A-Za-z0-9._-] starting "
            f"with an alphanumeric, got {value!r}"
        )
    return candidate


def _require_digest(sha256: str) -> str:
    digest = sha256.strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"sha256 must be a 64-character hex digest, got {sha256!r}")
    return digest
