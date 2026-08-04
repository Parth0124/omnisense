"""Media acquisition for `Signal.media`: bounded, allow-listed, archived.

Every URL this module fetches was written by somebody else. It came out of a
Reddit post, an RSS enclosure, a YouTube description -- content from the
untrusted zone in `docs/security-and-privacy.md` §2, which reaches this process
without ever passing through a control we operate. Everything below follows from
treating that URL as hostile rather than as data.

**The ceiling and the allowlist are checked before the body is read.** Not after
downloading and then discarding: `httpx` gives us the response headers first, and
that is where the decision is made. An unbounded `await response.aread()` on an
attacker-chosen URL is a one-line memory-exhaustion vector -- a 40 GB file behind
a link in a Reddit comment takes down the worker that fetched it and, with it,
every other record that worker was enriching. `Content-Length` is only the first
gate, because a server that wants to hurt us will lie about it or omit it
entirely under chunked encoding; the streaming loop therefore counts real bytes
and aborts the moment the ceiling is crossed, which bounds the damage to one
chunk regardless of what the header claimed.

**The allowlist is an allowlist, not a blocklist.** `image/svg+xml` is absent on
purpose even though it is an image: SVG is XML that can carry script, and an
archived SVG served back to a browser from our own origin is stored XSS. So is
`text/html`. So is `application/octet-stream`, which would let anything through
by declining to say what it is.

**Redirects are followed by hand, one hop at a time.** `follow_redirects` is
passed explicitly as `False` on every request -- not merely left at the client
default -- so that an injected client configured otherwise cannot silently
disable the check. Each `Location` is re-validated before it is followed, because
otherwise a public URL that redirects to `http://169.254.169.254/` turns the
allowlist into a cloud-metadata SSRF.

**Thumbnailing and transcription are Phase 2, and are injected, not imported.**
`MediaRef.transcript_ref` already exists in the Signal model, so the interface is
defined here now and the real code -- `attach_transcript`, `make_thumbnail` -- is
written and works the moment a provider is passed in. What does not exist is a
provider: no image decoder is in `requirements.txt` and no speech model is
deployed. The shipped defaults therefore raise `NotImplementedError` naming
exactly what is missing, rather than returning `None` and letting a caller record
a Signal whose transcript silently never arrives.

Layer note: **L2 service** (`docs/architecture.md` §6.1). Reaches R2 only through
`services/storage/object_store.py`, never through `backend/db/r2.py` directly.
"""

from __future__ import annotations

import hashlib
import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx

from backend.core.config import get_settings
from backend.core.exceptions import ExternalServiceError, ValidationError
from models.enums import MediaKind
from models.signal import MediaRef
from services.storage import object_store

__all__ = [
    "ALLOWED_CONTENT_TYPES",
    "DEFAULT_MAX_MEDIA_BYTES",
    "MAX_REDIRECTS",
    "DownloadedMedia",
    "MediaDownloader",
    "MediaFetchError",
    "MediaRejectedError",
    "MediaTooLargeError",
    "MediaTypeNotAllowedError",
    "Thumbnailer",
    "Transcriber",
    "UnavailableThumbnailer",
    "UnavailableTranscriber",
    "attach_transcript",
    "kind_for_content_type",
    "make_thumbnail",
]


DEFAULT_MAX_MEDIA_BYTES: Final = 25 * 1024 * 1024
"""25 MiB. A ceiling on one object *and* on the memory one download may hold.

The two are the same number because this module buffers the whole body before
writing it: `backend/db/r2.py` takes `bytes`, not a stream, so there is no
multipart upload path to hand a 200 MB video to. Raising this constant without
first adding streaming-to-R2 would not enable large media, it would just move the
memory-exhaustion vector from "unbounded" to "bounded at a number large enough to
still kill the worker" -- which is why the number and the reason for it live
together here.
"""

MAX_REDIRECTS: Final = 3
"""Hops allowed before a fetch is abandoned.

Three covers the real pattern (canonical URL -> CDN -> regional edge) and stops
a redirect loop from becoming an unbounded request chain.
"""

_CHUNK_BYTES: Final = 64 * 1024
"""Read granularity. Also the worst-case overshoot past the ceiling."""

CONTENT_TYPE_EXTENSIONS: Final[Mapping[str, str]] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/avif": "avif",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/quicktime": "mov",
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "audio/aac": "aac",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "application/pdf": "pdf",
    "text/plain": "txt",
}
"""Accepted types and the extension each gets in its object key.

The extension is fixed by this table rather than taken from the URL, because the
URL is attacker-controlled and the key must be deterministic: the same bytes with
the same declared type have to produce the same key on every worker, or content
addressing stops deduplicating.

`text/plain` is here for derived transcripts written by `attach_transcript`, not
for anything downloaded -- see `ALLOWED_CONTENT_TYPES`.
"""

ALLOWED_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    CONTENT_TYPE_EXTENSIONS.keys() - {"text/plain"}
)
"""What may be *downloaded*. Deliberately narrower than what may be stored.

Excluded and worth naming: `image/svg+xml` (XML with script, i.e. stored XSS when
served back from our origin), `text/html` (same), `application/octet-stream`
(refuses to declare what it is, which defeats the point of an allowlist), and
every archive format (a zip bomb is exactly the vector the byte ceiling exists to
stop, and we have no reason to fetch one).
"""

_MAGIC_PREFIXES: Final[Mapping[str, tuple[bytes, ...]]] = {
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/gif": (b"GIF87a", b"GIF89a"),
    "application/pdf": (b"%PDF-",),
}
"""Signatures for the types where the check is cheap and unambiguous.

The declared `Content-Type` is attacker-controlled, so a server can pass the
allowlist by claiming `image/png` and then send HTML. Verifying what actually
arrived closes that. Partial by design -- the container formats (MP4, WebM, WebP)
have offset-dependent signatures whose parsers are themselves attack surface, and
a wrong-but-inert video is a far smaller problem than a wrong-and-scriptable
image. A type absent from this table is stored on the strength of its declared
type alone.
"""

_REDIRECT_STATUSES: Final = frozenset({301, 302, 303, 307, 308})


# --------------------------------------------------------------------------- #
# Failure taxonomy
# --------------------------------------------------------------------------- #


class MediaRejectedError(ValidationError):
    """This media was refused by policy before or during download.

    A `ValidationError` (422) rather than a 5xx because nothing is broken: the
    third-party content violated a rule we set, and no retry will change that.
    Distinguishing it from `MediaFetchError` is what lets the caller count
    "refused" separately from "the internet was unreliable" -- a rising refusal
    rate means our allowlist is wrong, a rising fetch-error rate means something
    else entirely.
    """

    code = "media_rejected"
    default_message = "The media could not be accepted."


class MediaTooLargeError(MediaRejectedError):
    """The object exceeds the byte ceiling, declared or measured."""

    code = "media_too_large"
    default_message = "The media exceeds the maximum allowed size."


class MediaTypeNotAllowedError(MediaRejectedError):
    """The content type is outside the allowlist, or the bytes contradict it."""

    code = "media_type_not_allowed"
    default_message = "The media content type is not allowed."


class MediaFetchError(ExternalServiceError):
    """The media could not be retrieved: transport failure, timeout, or 4xx/5xx."""

    code = "media_fetch_failed"
    default_message = "The media could not be downloaded."


# --------------------------------------------------------------------------- #
# Downloaded media
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DownloadedMedia:
    """Bytes that passed every gate, plus what we learned on the way."""

    source_url: str
    content: bytes
    content_type: str
    """Normalized: lowercased, parameters stripped. Always in `ALLOWED_CONTENT_TYPES`."""
    sha256: str
    size_bytes: int
    kind: MediaKind

    @property
    def extension(self) -> str:
        return CONTENT_TYPE_EXTENSIONS[self.content_type]


def kind_for_content_type(content_type: str) -> MediaKind:
    """Map a MIME type onto the Signal model's coarse `MediaKind`.

    By top-level type rather than by table lookup, so a type added to the
    allowlist tomorrow classifies correctly without a second edit -- the
    place that decides *whether* to accept a type and the place that decides
    *what to call it* should not be able to drift apart.
    """
    top_level = content_type.split("/", 1)[0]
    if top_level == "image":
        return MediaKind.IMAGE
    if top_level == "video":
        return MediaKind.VIDEO
    if top_level == "audio":
        return MediaKind.AUDIO
    if content_type in ("application/pdf", "text/plain"):
        return MediaKind.DOCUMENT
    return MediaKind.UNKNOWN


# --------------------------------------------------------------------------- #
# The downloader
# --------------------------------------------------------------------------- #


class MediaDownloader:
    """Fetches media from untrusted URLs under a hard byte and type budget.

    Holds an `httpx.AsyncClient` so connections are pooled across a batch of
    media from one CDN. The client is a constructor argument for the same reason
    every provider in this codebase is: a test passes `httpx.MockTransport` and
    the whole class runs with no network and no fixtures.

    Not thread-safe and not intended to be; it is safe to drive concurrently from
    one event loop, which is what a worker does.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        max_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
        allowed_content_types: frozenset[str] = ALLOWED_CONTENT_TYPES,
        max_redirects: int = MAX_REDIRECTS,
        timeout_seconds: float | None = None,
    ) -> None:
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        if max_redirects < 0:
            raise ValueError(f"max_redirects must be non-negative, got {max_redirects}")

        self._max_bytes = max_bytes
        self._allowed = allowed_content_types
        self._max_redirects = max_redirects
        # `CONNECTOR_REQUEST_TIMEOUT_SECONDS` rather than a private knob: this is
        # an outbound fetch of third-party content on the ingestion path, which
        # is exactly what that setting bounds. Config is only ever read through
        # `get_settings()` (`docs/coding-standards.md` §2.9).
        self._timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else float(get_settings().connectors.request_timeout_seconds)
        )
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.AsyncClient:
        """The client, built on first use.

        Lazily, so constructing a downloader -- which the enrichment worker does
        per record -- never opens a connection pool that a record without media
        will not use.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                # A default `Accept` naming the allowlist lets a well-behaved
                # origin decline before it starts streaming, which is cheaper for
                # both ends than downloading and rejecting.
                headers={"accept": ", ".join(sorted(self._allowed)) + ", */*;q=0.1"},
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Release the connection pool. Only closes a client this instance built.

        An injected client belongs to the caller; closing it here would break the
        second downloader sharing it, and that failure surfaces as an unrelated
        "client has been closed" much later.
        """
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str) -> DownloadedMedia:
        """Download one media object, enforcing every limit before the body.

        Raises:
            MediaTypeNotAllowedError: The declared type is outside the allowlist,
                or the bytes contradict it.
            MediaTooLargeError: `Content-Length` or the measured body exceeds the
                ceiling.
            MediaFetchError: Transport failure, non-2xx status, or too many hops.
        """
        target = _validate_url(url)

        for _hop in range(self._max_redirects + 1):
            redirect = await self._attempt(target, url)
            if isinstance(redirect, DownloadedMedia):
                return redirect
            target = redirect

        raise MediaFetchError(
            f"Media URL exceeded {self._max_redirects} redirects.",
            details={"redirects": self._max_redirects},
        )

    async def _attempt(self, target: str, original_url: str) -> DownloadedMedia | str:
        """One hop. Returns the media, or the next URL to try.

        Split out so the redirect loop above reads as a loop rather than as a
        `while True` wrapped around a context manager with two exits.
        """
        try:
            # `follow_redirects=False` is restated per request so that an
            # injected client configured to follow them cannot bypass the
            # per-hop revalidation in `_validate_url`.
            async with self._http().stream(
                "GET", target, follow_redirects=False, timeout=self._timeout
            ) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise MediaFetchError(
                            "Media URL returned a redirect with no Location header.",
                            details={"status": response.status_code},
                        )
                    # Resolved against the current URL first: `Location` is
                    # frequently relative, and validating the relative form would
                    # check a string with no host in it at all.
                    return _validate_url(str(response.url.join(location)))

                if response.status_code >= 400:
                    # The body is not read. An error body from a hostile origin
                    # is unbounded content we have no use for.
                    raise MediaFetchError(
                        f"Media URL returned HTTP {response.status_code}.",
                        details={"status": response.status_code},
                    )

                content_type = _normalize_content_type(response.headers.get("content-type"))
                if content_type not in self._allowed:
                    raise MediaTypeNotAllowedError(
                        f"Content type {content_type!r} is not in the media allowlist.",
                        details={"content_type": content_type},
                    )

                declared = _parse_content_length(response.headers.get("content-length"))
                if declared is not None and declared > self._max_bytes:
                    # Nothing has been read yet, and nothing will be: this is the
                    # cheap gate, taken on the server's own word.
                    raise MediaTooLargeError(
                        "Media declares a size above the ceiling.",
                        details={"declared_bytes": declared, "max_bytes": self._max_bytes},
                    )

                body = await self._read_bounded(response)
                _verify_magic(body, content_type)

                return DownloadedMedia(
                    source_url=original_url,
                    content=body,
                    content_type=content_type,
                    sha256=hashlib.sha256(body).hexdigest(),
                    size_bytes=len(body),
                    kind=kind_for_content_type(content_type),
                )

        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            # Only the exception class, never `str(exc)`: httpx messages embed
            # the URL, and a URL from fetched content can carry a tracking
            # payload we should not be writing into logs. `MediaRejectedError`
            # and `MediaFetchError` raised inside the block are not `HTTPError`s,
            # so a policy refusal passes through with its own reason intact.
            raise MediaFetchError(
                "Media download failed at the transport layer.",
                details={"error": type(exc).__name__},
                cause=exc,
            ) from exc

    async def _read_bounded(self, response: httpx.Response) -> bytes:
        """Accumulate the body, aborting the instant it crosses the ceiling.

        The second gate, and the one that actually holds. `Content-Length` is
        advisory: it is absent under chunked transfer encoding and it is a free
        text field a malicious origin can simply understate. Counting what
        arrives caps memory at the ceiling plus one chunk no matter what any
        header said, and leaving the `async for` early closes the stream, so the
        remaining gigabytes are never pulled off the socket.
        """
        buffer = bytearray()
        async for chunk in response.aiter_bytes(_CHUNK_BYTES):
            buffer += chunk
            if len(buffer) > self._max_bytes:
                raise MediaTooLargeError(
                    "Media exceeded the byte ceiling while downloading.",
                    details={"max_bytes": self._max_bytes},
                )
        if not buffer:
            raise MediaFetchError("Media URL returned an empty body.")
        return bytes(buffer)

    async def archive(self, url: str, *, signal_id: str) -> MediaRef:
        """Fetch and store one media object, returning the `MediaRef` for a Signal.

        Both `source_url` and `object_key` end up on the ref, and the model says
        why: the source URL rots, and the archived copy is what a citation six
        months from now actually resolves against.

        `width`, `height` and `duration_s` are left unset. Filling them requires
        decoding the file, which is the Phase 2 work behind `make_thumbnail` --
        and guessing them from the URL or the container name would put a wrong
        number in a field a report layout trusts.
        """
        media = await self.fetch(url)
        stored = await object_store.put_media(
            signal_id,
            media.content,
            content_type=media.content_type,
            extension=media.extension,
        )
        return MediaRef(
            kind=media.kind,
            source_url=media.source_url,
            object_key=stored.key,
            mime_type=media.content_type,
            bytes=media.size_bytes,
        )


# --------------------------------------------------------------------------- #
# URL and header validation
# --------------------------------------------------------------------------- #


def _validate_url(url: str) -> str:
    """Reject a URL we must not fetch, before any socket is opened.

    Three classes of rejection, each with its own reason:

    - **Scheme.** Only `http`/`https`. `file://` reads the worker's filesystem,
      `data:` bypasses the size gate entirely by carrying the payload inline.
    - **Userinfo.** `https://user:pass@host/` is rejected outright: parsers
      disagree about where the host ends in that form, and any disagreement
      between this check and `httpx` makes the check worthless.
    - **Address.** Loopback, link-local, private and otherwise-reserved literal
      addresses. `http://169.254.169.254/` is the cloud metadata endpoint and the
      canonical SSRF target; `http://10.0.0.5:6379/` is the internal network.

    What this deliberately does **not** do is resolve DNS. Resolving here and
    connecting later is a TOCTOU window (DNS rebinding), and doing it properly
    means pinning the resolved address into the connection, which is a transport
    concern for a custom `httpx` transport rather than a string check. Stated
    plainly so nobody reads this function as complete SSRF protection: it stops
    the direct and redirect-based cases, not a hostile resolver.
    """
    parts = urlsplit(url.strip())

    if parts.scheme.lower() not in ("http", "https"):
        raise MediaRejectedError(
            f"Media URL scheme {parts.scheme!r} is not fetchable; expected http or https.",
            details={"scheme": parts.scheme},
        )
    if "@" in parts.netloc:
        raise MediaRejectedError("Media URL must not carry userinfo credentials.")

    host = (parts.hostname or "").strip().lower()
    if not host:
        raise MediaRejectedError("Media URL has no host.")
    if host == "localhost" or host.endswith(".localhost"):
        raise MediaRejectedError("Media URL points at the local host.", details={"host": host})

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # A name, not a literal. See the docstring on what is not checked.
        return url.strip()

    if not address.is_global or address.is_loopback or address.is_link_local:
        raise MediaRejectedError(
            "Media URL points at a non-public address.", details={"host": host}
        )
    return url.strip()


def _normalize_content_type(header: str | None) -> str:
    """Reduce a `Content-Type` header to a bare, comparable type.

    Parameters are dropped (`image/jpeg; charset=binary`), case is folded, and a
    missing header becomes the empty string rather than a default. There is no
    safe default: guessing `application/octet-stream` and then allowlisting it
    would let an origin through simply by saying nothing.
    """
    if not header:
        return ""
    return header.split(";", 1)[0].strip().lower()


def _parse_content_length(header: str | None) -> int | None:
    """Parse `Content-Length`, treating anything unparseable as absent.

    Absent is the safe reading: it routes the response to the streaming counter
    in `_read_bounded`, which enforces the same ceiling on measured bytes. A
    negative or non-numeric value is a broken or hostile origin, and pretending
    to have read a number from it would be worse than admitting we have none.
    """
    if header is None:
        return None
    try:
        value = int(header.strip())
    except ValueError:
        return None
    return value if value >= 0 else None


def _verify_magic(body: bytes, content_type: str) -> None:
    """Check the bytes against the declared type where a signature is known.

    Raises `MediaTypeNotAllowedError` on a mismatch, because "the server lied
    about what this is" is a type problem, not a size or transport one, and it is
    exactly the case where storing the object anyway would be dangerous.
    """
    signatures = _MAGIC_PREFIXES.get(content_type)
    if signatures is None:
        return
    if not any(body.startswith(signature) for signature in signatures):
        raise MediaTypeNotAllowedError(
            f"Media bytes do not match the declared content type {content_type!r}.",
            details={"content_type": content_type},
        )


# --------------------------------------------------------------------------- #
# Phase 2: derivation hooks
# --------------------------------------------------------------------------- #


@runtime_checkable
class Transcriber(Protocol):
    """Turns audio or video bytes into text.

    A protocol taking a constructor-injected implementation, exactly like
    `LLMProvider` and `EmbeddingProvider`: the model behind this is a deployment
    decision (ADR-0008 puts GPU inference on Modal), and a stage that constructed
    its own would be untestable and unswappable.

    `model` is required because the transcript ends up in the retrieval corpus
    and `docs/signal-model.md` §5.1 requires knowing which model produced any
    derived field before a result can be reproduced.
    """

    model: str

    async def transcribe(self, data: bytes, *, content_type: str) -> str:
        """Return the transcript text. Raises on failure; never returns a partial."""
        ...


@runtime_checkable
class Thumbnailer(Protocol):
    """Renders a small preview image from an image, video or document."""

    async def render(
        self, data: bytes, *, content_type: str, max_edge_px: int
    ) -> tuple[bytes, str]:
        """Return `(thumbnail bytes, content type of the thumbnail)`."""
        ...


class UnavailableTranscriber:
    """The shipped `Transcriber`. Raises, loudly, with the reason.

    Not a silent no-op returning `""`. An empty transcript is indistinguishable
    from "this video had no speech", which would put a media-only Signal into the
    corpus looking successfully enriched while being permanently unsearchable.
    """

    model = "unavailable"

    async def transcribe(self, data: bytes, *, content_type: str) -> str:
        raise NotImplementedError(
            "Speech-to-text is not implemented (Phase 2). Missing: a deployed ASR "
            "model -- ADR-0008 puts GPU inference on Modal, and infra/modal/ has "
            "no transcription endpoint yet -- plus an audio-duration guard, since "
            "billing here is per minute of audio and the input is untrusted. "
            "Pass a real Transcriber to attach_transcript() once one exists."
        )


class UnavailableThumbnailer:
    """The shipped `Thumbnailer`. Raises, loudly, with the reason."""

    async def render(
        self, data: bytes, *, content_type: str, max_edge_px: int
    ) -> tuple[bytes, str]:
        raise NotImplementedError(
            "Thumbnail rendering is not implemented (Phase 2). Missing: an image "
            "decoder -- Pillow is not in requirements.txt -- and, before one is "
            "added, a decoded-pixel ceiling: the byte ceiling in this module "
            "bounds the compressed file, not the bitmap, and a 20 KB PNG can "
            "declare 40000x40000 pixels. Video thumbnails additionally need a "
            "frame extractor. Pass a real Thumbnailer to make_thumbnail()."
        )


async def attach_transcript(
    media: MediaRef,
    data: bytes,
    *,
    transcriber: Transcriber,
    signal_id: str,
) -> MediaRef:
    """Transcribe media, store the text in R2, and return a ref pointing at it.

    Real code today: the only missing piece is the `Transcriber`, which is why it
    is a parameter. `MediaRef.transcript_ref` already exists on the Signal model
    and is described there as "what makes a video searchable by the text
    pipeline", so this is the function the enrichment pipeline will call.

    The transcript is stored content-addressed under the same `media/{signal_id}/`
    prefix as its source, so erasing a Signal erases its derivatives in the same
    prefix delete. Returns a revalidated copy rather than mutating the argument:
    a `MediaRef` handed in here is usually already inside a `Signal.media` list,
    and rewriting it in place would change an object other code is holding.
    `model_validate` rather than `model_copy` because only the former actually
    re-runs the field validators on the merged result.

    Raises:
        NotImplementedError: `transcriber` is the shipped placeholder.
        ValueError: The transcriber returned nothing usable.
    """
    text = await transcriber.transcribe(data, content_type=media.mime_type or "")
    if not text.strip():
        raise ValueError(
            "Transcriber returned empty text. An empty transcript is not a valid "
            "result -- it is indistinguishable from silence and would be indexed "
            "as though transcription had succeeded."
        )

    encoded = text.encode("utf-8")
    stored = await object_store.put_media(
        signal_id,
        encoded,
        content_type="text/plain",
        extension=CONTENT_TYPE_EXTENSIONS["text/plain"],
    )
    return MediaRef.model_validate({**media.model_dump(), "transcript_ref": stored.key})


async def make_thumbnail(
    media: MediaRef,
    data: bytes,
    *,
    thumbnailer: Thumbnailer,
    signal_id: str,
    max_edge_px: int = 512,
) -> MediaRef:
    """Render a preview for `media` and store it as its own media object.

    Returns a new `MediaRef` rather than mutating the source, because a thumbnail
    is a separate object with its own key and its own bytes -- `MediaRef` has no
    `thumbnail_ref` field, and adding one would be a Signal schema change
    (`docs/signal-model.md` §7), not a storage decision.

    Raises:
        NotImplementedError: `thumbnailer` is the shipped placeholder.
        MediaTypeNotAllowedError: The renderer produced a type we do not store.
    """
    thumb_bytes, thumb_type = await thumbnailer.render(
        data, content_type=media.mime_type or "", max_edge_px=max_edge_px
    )
    normalized = _normalize_content_type(thumb_type)
    if normalized not in CONTENT_TYPE_EXTENSIONS:
        raise MediaTypeNotAllowedError(
            f"Thumbnailer produced unstorable content type {thumb_type!r}.",
            details={"content_type": normalized},
        )

    stored = await object_store.put_media(
        signal_id,
        thumb_bytes,
        content_type=normalized,
        extension=CONTENT_TYPE_EXTENSIONS[normalized],
    )
    return MediaRef(
        kind=MediaKind.IMAGE,
        source_url=media.source_url,
        object_key=stored.key,
        mime_type=normalized,
        bytes=len(thumb_bytes),
    )
