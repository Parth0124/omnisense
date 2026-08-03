"""Cloudflare R2 (S3-compatible) client bootstrap.

R2 holds the immutable raw payload of every fetched record, media, and rendered
report artifacts (ADR-0006, `docs/data-stores.md` §3.6). Together with PostgreSQL
it is one of the two **authoritative** stores: losing a raw payload means the
enrichment pipeline can never be replayed for that record, because sources delete
posts and close their API windows.

**Why every public function here is `async` and wraps a `to_thread` call.**
`boto3` is synchronous, and there is no async S3 client in the dependency set --
ADR-0006 chose `boto3` precisely because the S3 API is what makes the provider
swappable. So this module is the one place in OmniSense where blocking I/O
exists, and it is confined here on purpose. A blocking `put_object` awaited from
a request handler stalls *every* concurrent request on that worker, not just the
one uploading, and the symptom -- unrelated endpoints timing out whenever
somebody archives a large payload -- looks nothing like its cause.
`asyncio.to_thread` hands the blocking call to the default executor so the event
loop stays free. Two consequences follow and are load-bearing:

- **Nothing outside this module may call a `boto3` method.** A caller that
  reaches into `get_s3()` and calls `.get_object()` directly reintroduces exactly
  the blocking call this module exists to contain. ADR-0006 states the same rule
  from the other direction: all access goes through
  `services/storage/object_store.py`, and no module imports `boto3`.
- **Concurrency is bounded by the default executor**, not by the event loop.
  Thousands of simultaneous uploads queue in the thread pool rather than
  exhausting file descriptors, which is the desired failure mode -- but it does
  mean R2 throughput is a thread-pool property. `botocore` clients are
  thread-safe for API calls, so sharing the one singleton across those threads is
  correct (it is `boto3.resource`, not `boto3.client`, that is not thread-safe).

Layer note: **L1k kernel** (`docs/architecture.md` §6.1) -- importable by
`services/`, `agents/`, `workers/`, `backend/api/` and `scripts/`, never by
`connectors/`.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import SecretStr

from backend.core.config import get_settings
from backend.core.exceptions import (
    ConfigurationError,
    ExternalServiceError,
    NotFoundError,
)

__all__ = [
    "RAW_KEY_PREFIX",
    "check_r2",
    "dispose_s3",
    "get_object",
    "get_s3",
    "object_exists",
    "put_object",
    "raw_object_key",
]

RAW_KEY_PREFIX = "raw"
"""First path segment of the raw-payload key template (ADR-0006)."""

_HEALTH_TIMEOUT_SECONDS = 5.0
"""Wall-clock budget for the readiness probe.

The retry policy below is tuned for data operations, where three attempts against
a throttled bucket is right. Applied to a `head_bucket` during a network
partition, that same policy costs three connect timeouts back to back and turns
`/readyz` into a half-minute request the orchestrator gave up on long before.
"""

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

# `boto3.client("s3")` returns a class botocore generates at runtime from the
# service model; its methods exist only after that generation, so there is no
# static type for them without the separate `boto3-stubs` package, which is not a
# dependency. `Any` is the honest annotation here rather than a lie about a
# `BaseClient` that does not declare `put_object`
# (`docs/coding-standards.md` §2.2 rule 2).
_client: Any = None


def _reveal(secret: SecretStr | None) -> str:
    """Unwrap a `SecretStr` to a plain string, treating absent and blank alike.

    `SecretStr("")` is **truthy** -- it defines no `__bool__`, so a plain
    `if not settings.storage.access_key_id` is silently False for a blank
    credential. `.env.example` ships `R2_ACCESS_KEY_ID=` empty, so anyone who
    copies it to `.env` produces exactly that value, and the check below would
    wave it through into a client that then fails to sign every request.
    """
    return secret.get_secret_value().strip() if secret is not None else ""


def _build_client() -> Any:
    """Construct the S3 client. Blocking: reads botocore's bundled service model.

    Called only from `get_s3()`, inside a worker thread.
    """
    storage = get_settings().storage
    endpoint_url = (storage.endpoint_url or "").strip()
    access_key_id = _reveal(storage.access_key_id)
    secret_access_key = _reveal(storage.secret_access_key)

    # `.env.example` ships these blank -- ADR-0006 notes there is no local
    # object-storage container, so an unconfigured process is the normal local
    # state. Failing here with a named cause beats letting botocore fall through
    # to its ambient credential chain, which would silently talk to whatever AWS
    # account happens to be configured on the machine.
    missing = [
        name
        for name, value in (
            ("R2_ENDPOINT_URL", endpoint_url),
            ("R2_ACCESS_KEY_ID", access_key_id),
            ("R2_SECRET_ACCESS_KEY", secret_access_key),
        )
        if not value
    ]
    if missing:
        raise ConfigurationError(
            "Cloudflare R2 is not configured; missing " + ", ".join(missing) + ".",
            details={"missing": missing},
        )

    config = Config(
        region_name=storage.region,
        # R2 accepts only SigV4. Pinning it stops a future botocore default from
        # quietly producing signatures the endpoint rejects with a 403 that reads
        # like a credential problem.
        signature_version="s3v4",
        # Path-style addressing. The R2 endpoint is account-scoped
        # (`https://<account>.r2.cloudflarestorage.com`), so virtual-host
        # addressing would put the bucket in a subdomain of the account host and
        # depend on DNS Cloudflare does not guarantee for every bucket name.
        s3={"addressing_style": "path"},
        # ADR-0006: Class A/B operations are billed per request, so retries cost
        # money as well as latency. Three attempts in standard mode retries the
        # throttling and 5xx cases and nothing else.
        retries={"max_attempts": 3, "mode": "standard"},
        # botocore defaults both to 60s. A 60-second connect timeout means a
        # network partition parks a thread-pool thread for a minute per call, and
        # the pool is small -- that is how one dead dependency becomes a stalled
        # process.
        connect_timeout=10,
        read_timeout=60,
    )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=config,
    )


async def get_s3() -> Any:
    """Return the process-wide S3 client, creating it on first use.

    Async, and building in a thread, for the reason in the module docstring:
    `boto3.client()` opens no socket but does parse botocore's bundled service
    model off disk, which is tens of milliseconds of blocking file I/O. Paying
    that on the event loop once per process is survivable; doing it as a matter
    of style is how "async everywhere" erodes.

    Deliberately unsynchronized, exactly like `get_engine()` in `session.py`. Two
    coroutines racing on the very first call each build a client and one is
    discarded -- which costs a duplicate parse of the service model and nothing
    else, since a fresh botocore client holds no connections. Guarding that with
    a module-level `asyncio.Lock` would be strictly worse: the lock binds to the
    first event loop that awaits it and raises "bound to a different event loop"
    in any process that runs a second one -- every script that calls
    `asyncio.run()` twice, and the whole test suite.

    Raises:
        ConfigurationError: R2 endpoint or credentials are unset.
    """
    global _client
    if _client is None:
        _client = await asyncio.to_thread(_build_client)
    return _client


async def check_r2() -> bool:
    """Probe R2 for `/readyz`. Never raises.

    `head_bucket` rather than `list_buckets`, because it verifies the one thing
    that matters -- these credentials can reach *this* bucket -- and is a single
    cheap Class B request. It fails for an unconfigured process too, since
    `get_s3()` raises `ConfigurationError`, and that is correct: an unconfigured
    R2 is an unavailable R2.

    Returns a bool rather than raising because readiness aggregates several
    dependencies and one being down must not prevent reporting on the others
    (`docs/observability.md`). Note that per `docs/architecture.md` §7.3 an R2
    outage degrades rather than fails -- report artifacts are deferred and
    retried -- so there is deliberately no `require_r2()` counterpart to
    `require_postgres()`.

    On timeout the worker thread is abandoned rather than cancelled -- threads
    cannot be interrupted. That is acceptable and not a leak: the abandoned call
    is bounded by botocore's own connect/read timeouts and retry budget, so it
    exits on its own well inside a probe interval, and at most a couple can ever
    be in flight at once.
    """
    bucket = get_settings().storage.bucket
    try:
        client = await get_s3()
        await asyncio.wait_for(
            asyncio.to_thread(client.head_bucket, Bucket=bucket),
            timeout=_HEALTH_TIMEOUT_SECONDS,
        )
    except Exception:
        return False
    return True


async def put_object(
    key: str,
    body: bytes,
    *,
    content_type: str = "application/octet-stream",
    metadata: dict[str, str] | None = None,
) -> str:
    """Write `body` at `key` and return the key.

    Blocking call wrapped in `asyncio.to_thread` -- see the module docstring.

    No conditional-write guard, on purpose. Every key this platform writes is
    content-addressed (ADR-0006), so a repeated `PUT` of identical bytes to the
    same key is a genuine no-op and is what makes step 1 of `docs/data-stores.md`
    §5.1 safe to replay under at-least-once delivery. Adding an
    `IfNoneMatch: "*"` precondition would turn that intended no-op into an error
    the caller has to special-case.

    Raises:
        ExternalServiceError: R2 rejected or could not serve the write.
    """
    bucket = get_settings().storage.bucket
    client = await get_s3()
    kwargs: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": body,
        "ContentType": content_type,
    }
    if metadata:
        kwargs["Metadata"] = metadata

    try:
        await asyncio.to_thread(client.put_object, **kwargs)
    except (ClientError, BotoCoreError) as exc:
        raise ExternalServiceError(
            f"Failed to write object {key!r} to R2.",
            details={"bucket": bucket, "key": key},
            cause=exc,
        ) from exc
    return key


async def get_object(key: str) -> bytes:
    """Read the object at `key`.

    Blocking call wrapped in `asyncio.to_thread` -- see the module docstring. The
    body `read()` is inside the same thread hop: `StreamingBody.read()` is itself
    a blocking socket read, so returning the stream to the caller would move the
    blocking part back onto the event loop and undo the wrapping.

    Returns the whole body in memory, which is right for raw payloads and
    transcripts and wrong for large media. Streaming media belongs in
    `services/storage/media.py`, which owns that access pattern.

    Raises:
        NotFoundError: No object at that key.
        ExternalServiceError: R2 could not serve the read.
    """
    bucket = get_settings().storage.bucket
    client = await get_s3()

    def _read() -> bytes:
        response = client.get_object(Bucket=bucket, Key=key)
        data: bytes = response["Body"].read()
        return data

    try:
        return await asyncio.to_thread(_read)
    except ClientError as exc:
        if _error_code(exc) in ("NoSuchKey", "404", "NotFound"):
            raise NotFoundError.for_resource("R2 object", key) from exc
        raise ExternalServiceError(
            f"Failed to read object {key!r} from R2.",
            details={"bucket": bucket, "key": key},
            cause=exc,
        ) from exc
    except BotoCoreError as exc:
        raise ExternalServiceError(
            f"Failed to read object {key!r} from R2.",
            details={"bucket": bucket, "key": key},
            cause=exc,
        ) from exc


async def object_exists(key: str) -> bool:
    """Whether an object exists at `key`.

    Blocking call wrapped in `asyncio.to_thread` -- see the module docstring.

    Absence returns `False`; anything else raises. The distinction matters: a
    connector uses this to skip re-archiving a payload it already stored, and
    swallowing a permissions error as "absent" would make it re-`PUT` on every
    run and, worse, would report success for an archive that is not there.

    Raises:
        ExternalServiceError: R2 answered with something other than "absent".
    """
    bucket = get_settings().storage.bucket
    client = await get_s3()
    try:
        await asyncio.to_thread(client.head_object, Bucket=bucket, Key=key)
    except ClientError as exc:
        # `head_object` has no response body, so botocore can only report the
        # HTTP status: 404 arrives as error code "404", not "NoSuchKey".
        if _error_code(exc) in ("NoSuchKey", "404", "NotFound"):
            return False
        raise ExternalServiceError(
            f"Failed to stat object {key!r} in R2.",
            details={"bucket": bucket, "key": key},
            cause=exc,
        ) from exc
    except BotoCoreError as exc:
        raise ExternalServiceError(
            f"Failed to stat object {key!r} in R2.",
            details={"bucket": bucket, "key": key},
            cause=exc,
        ) from exc
    return True


async def dispose_s3() -> None:
    """Close the underlying connection pool and reset the singleton.

    Called from lifespan shutdown. `close()` is a blocking urllib3 pool shutdown,
    hence the thread hop here too.
    """
    global _client
    if _client is not None:
        client, _client = _client, None
        await asyncio.to_thread(client.close)


# --------------------------------------------------------------------------- #
# Key construction
# --------------------------------------------------------------------------- #


def raw_object_key(platform: str, fetched_at: datetime, sha256: str) -> str:
    """Build the raw-payload key: `raw/{platform}/{yyyy}/{mm}/{dd}/{sha256}.json`.

    The template is fixed by ADR-0006 and realized in `docs/signal-model.md` §6
    as `raw/reddit/2026/07/28/8f14e45f….json`. Two properties of it are
    load-bearing and are the reason this is a function rather than an f-string at
    each call site:

    - **Content-addressed, not id-addressed.** The `PUT` happens at step 1 of
      `docs/data-stores.md` §5.1, before enrichment and therefore before
      `signal.id` exists, so the digest of the payload bytes is the only
      identifier available. It is also what makes a replayed `PUT` a no-op.
    - **No `.zst` suffix.** Objects are stored as received; compression, if it is
      ever introduced, is a transport concern of
      `services/storage/object_store.py` and must not change the key, or every
      `raw_ref` already recorded in PostgreSQL stops resolving.

    `platform` is typed `str` rather than `models.enums.Platform` so this module
    stays free of a domain import; `Platform` is a `StrEnum`, so its members are
    accepted directly and format to their value.

    Args:
        platform: Platform slug, e.g. `"reddit"`.
        fetched_at: When the payload was retrieved. Must be timezone-aware; it is
            converted to UTC before the date is taken.
        sha256: Hex SHA-256 of the payload bytes.

    Raises:
        ValueError: `platform` is empty or contains a path separator, `sha256` is
            not a 64-character hex digest, or `fetched_at` is naive.
    """
    slug = str(platform).strip().lower()
    if not slug or "/" in slug:
        # A slash would silently invent a new prefix level, so a listing of
        # `raw/reddit/` would miss objects it should contain.
        raise ValueError(f"platform must be a non-empty slug without '/', got {platform!r}")

    # A naive datetime is almost always local time. Near midnight that lands the
    # object in the wrong day partition -- which no read path would notice,
    # because reads are by key, so the damage only surfaces years later when a
    # date-ranged retention or replay job silently skips it.
    if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
        raise ValueError(
            "fetched_at must be timezone-aware; a naive timestamp partitions the "
            "object under the wrong UTC date. Use models.base.utcnow()."
        )
    # Converted rather than assumed: a caller in a +13:00 zone and a caller in
    # UTC must produce the same key for the same bytes, or content addressing
    # stops deduplicating across deployments.
    utc = fetched_at.astimezone(UTC)

    # Case-folded rather than rejected: uppercase and lowercase hex are the same
    # digest, and normalizing guarantees one key per content. Rejecting a
    # non-digest is not optional though -- an object stored under a key that is
    # not its own hash breaks the idempotency the whole scheme rests on.
    digest = sha256.strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"sha256 must be a 64-character hex digest, got {sha256!r}")

    return f"{RAW_KEY_PREFIX}/{slug}/{utc.year:04d}/{utc.month:02d}/{utc.day:02d}/{digest}.json"


def _error_code(exc: ClientError) -> str:
    """Extract botocore's error code, which is nested and occasionally absent."""
    response: dict[str, Any] = getattr(exc, "response", {}) or {}
    error: dict[str, Any] = response.get("Error") or {}
    return str(error.get("Code", ""))
