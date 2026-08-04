"""Stage 1 -- Clean: provider bytes in, one UTF-8 plain-text body out.

`docs/signal-model.md` §5.1 gives this stage the shortest contract in the
pipeline and the harshest failure mode: *raw bytes + content type -> UTF-8 plain
text, boilerplate removed*, **fatal** on failure. Fatal because every later stage
reads `content.text`. A Signal whose body never got decoded is not a degraded
Signal, it is a Signal about nothing -- it would be embedded, indexed, retrieved
and eventually quoted in a report as evidence for a claim it does not contain.

Three decisions are encoded here and each is worth stating because none of them
is the obvious one.

**Encoding is decided before anything else, and it is decided sceptically.**
A mis-decoded body does not fail; it succeeds and is wrong. Mojibake still
detects as *a* language, still embeds to *a* vector, still matches *some* query.
Nothing downstream can notice, and the archived original in R2 is the only way
back. So the ladder in `decode_bytes` trusts the transport's declared charset
first, overrides it where honouring it is provably wrong, and treats a
successful `latin-1` decode as no evidence at all -- because `latin-1` decodes
every possible byte string, which makes "it decoded" a statement about the codec
rather than about the bytes.

**Markup handling is delegated, not reimplemented.** `connectors/normalize/html.py`
already extracts a readable body and collapses whitespace, and `services/` is
permitted to import `connectors/` (`docs/architecture.md` §6.1, row *services*).
Reimplementing it here would be worse than duplication: the connector-side
mapper (`connectors/normalize/mapper.py`) cleans body fields with that exact
function, and dedup layer 2 hashes the cleaned text
(`docs/signal-model.md` §4.2). Two cleaners that differ by one space would make a
re-fetched item look like a new one on every poll. Same function, same bytes,
same hash.

**Emoji survive.** They are not noise to be stripped: on social sources they
carry the polarity stage 5 is looking for, and a body reduced to "This update is"
has lost the entire observation. They are also the reason this module does not
apply any Unicode normalization of its own -- see the note on zero-width joiners
in `_clean_document`.

The PII hook (`docs/security-and-privacy.md` §6.1, `docs/signal-model.md` §9.7)
is `Redactor`, and it is **off by default**. That is deliberate. Redaction
rewrites `content.text`, and `content.text` feeds the layer-2 content hash and,
for connectors that reach rule 3, `native_id` itself -- so switching a redactor
on forks identity for every affected record and is a `pipeline_version` bump plus
a backfill, never a config flip. Shipping it on by default would make that
decision silently, at whatever moment someone deployed. `RegexRedactor` is a
deliberately conservative starting point, not a compliance control; the protocol
is the seam where a real detector (Presidio, a hosted classifier) plugs in.
"""

from __future__ import annotations

import codecs
import enum
import json
import re
from collections.abc import Callable
from typing import Any, Final, Protocol, runtime_checkable

from backend.core.exceptions import OmniSenseError
from connectors.normalize.html import collapse_whitespace, extract_readable
from models.enums import StageName
from services.signal_engine.pipeline import EnrichmentContext

__all__ = [
    "CARD_PLACEHOLDER",
    "EMAIL_PLACEHOLDER",
    "PHONE_PLACEHOLDER",
    "CleaningError",
    "CleaningStage",
    "ContentFamily",
    "Redactor",
    "RegexRedactor",
    "classify_content_type",
    "clean_text",
    "decode_bytes",
]


class CleaningError(OmniSenseError):
    """Stage 1 could not produce a body.

    A distinct class rather than a bare `ValueError` because the pipeline records
    only `type(exc).__name__` in `lineage.stages[]` -- deliberately, since a
    provider message can echo fetched content (`docs/security-and-privacy.md`).
    That name is the entire diagnostic a DLQ triager gets, so it has to mean
    something on its own.
    """

    status_code = 422
    code = "cleaning_failed"
    default_message = "The raw record could not be cleaned into a text body."


# --------------------------------------------------------------------------- #
# Content-type classification
# --------------------------------------------------------------------------- #


class ContentFamily(enum.StrEnum):
    """How stage 1 must treat a media type. A closed set of four behaviours.

    Deliberately *not* in `models/enums.py`. That module is the vocabulary shared
    across process boundaries; this never leaves stage 1, is never serialized and
    is never read by another package. Putting it there would grow the shared
    contract to describe a private dispatch table.

    - `MARKUP`  -- HTML/XML. Hand to the readable-text extractor.
    - `PLAIN`   -- already text. Collapse whitespace and nothing else: a declared
      `text/plain` body must never meet an HTML parser, or prose containing `<`
      or `&` gets silently rewritten.
    - `STRUCTURED` -- JSON. There is no document body to extract; the
      observation's text sits at a provider-specific path that only the field map
      in `connectors/normalize/mapper.py` knows. Stage 1 decodes and parses,
      stage 2 pulls the body out and cleans it with `clean_text`.
    - `BINARY`  -- PDF, office documents, media. Needs an extractor this
      deployment does not have.
    """

    MARKUP = "markup"
    PLAIN = "plain"
    STRUCTURED = "structured"
    BINARY = "binary"


_MARKUP_TYPES: Final[frozenset[str]] = frozenset(
    {"text/html", "application/xhtml+xml", "text/xml", "application/xml"}
)
_PLAIN_TYPES: Final[frozenset[str]] = frozenset(
    {"text/plain", "text/markdown", "text/x-markdown", "text/csv", "text/tab-separated-values"}
)
_STRUCTURED_TYPES: Final[frozenset[str]] = frozenset(
    {"application/json", "text/json", "application/x-ndjson"}
)
_BINARY_TYPES: Final[frozenset[str]] = frozenset(
    {
        "application/pdf",
        "application/zip",
        "application/gzip",
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.oasis.opendocument.text",
        "application/epub+zip",
    }
)
_BINARY_PREFIXES: Final[tuple[str, ...]] = ("image/", "audio/", "video/", "font/")


def classify_content_type(content_type: str | None) -> ContentFamily:
    """Map a media type onto the behaviour stage 1 owes it.

    Suffix rules (`+json`, `+xml`) come from RFC 6839 and matter in practice:
    feeds arrive as `application/atom+xml`, APIs as `application/vnd.api+json`,
    and a lookup table of exact strings would send both down the wrong path.

    An absent or unrecognized type resolves to `PLAIN` rather than to an error.
    The body is then whitespace-collapsed and left alone, which is the only
    treatment that cannot corrupt an unknown format -- and the NUL check in
    `_clean_document` still catches an actual binary that arrived unlabelled.
    """
    if not content_type:
        return ContentFamily.PLAIN
    essence = content_type.split(";", 1)[0].strip().lower()
    if not essence:
        return ContentFamily.PLAIN
    if essence in _BINARY_TYPES or essence.startswith(_BINARY_PREFIXES):
        return ContentFamily.BINARY
    if essence in _STRUCTURED_TYPES or essence.endswith("+json"):
        return ContentFamily.STRUCTURED
    if essence in _MARKUP_TYPES or essence.endswith("+xml"):
        return ContentFamily.MARKUP
    if essence in _PLAIN_TYPES:
        return ContentFamily.PLAIN
    # The two branches deliberately agree, and the redundancy is the point: the
    # first says "we know this is text", the second says "we have no idea what
    # this is, and leaving it alone is the only treatment that cannot corrupt it".
    # Collapsing them would lose the distinction the next person needs when a new
    # format has to be routed somewhere.
    return ContentFamily.PLAIN


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #

_CHARSET_PARAM = re.compile(r";\s*charset\s*=\s*\"?([A-Za-z0-9_.:+\-]+)\"?", re.IGNORECASE)
_META_CHARSET = re.compile(
    r"<meta[^>]+charset\s*=\s*[\"']?\s*([A-Za-z0-9_.:+\-]+)", re.IGNORECASE
)
_XML_ENCODING = re.compile(
    r"<\?xml[^>]+encoding\s*=\s*[\"']([A-Za-z0-9_.:+\-]+)[\"']", re.IGNORECASE
)

#: BOMs, longest first. UTF-32-LE begins with the UTF-16-LE BOM, so testing in
#: declaration order rather than length order would decode every UTF-32-LE
#: document as UTF-16 and produce NUL-separated garbage that still "succeeds".
_BOMS: Final[tuple[tuple[bytes, str], ...]] = (
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
)

#: Codecs that map every one of the 256 byte values to a character and therefore
#: **cannot fail**. A successful decode under one of these is not evidence that
#: the label was right -- it is a property of the codec. These are also exactly
#: the labels that get emitted by mistake: RFC 2616 made `ISO-8859-1` the default
#: for `text/*`, so a decade of servers announce it over UTF-8 bodies.
_NEVER_FAILS: Final[frozenset[str]] = frozenset(
    {
        "iso8859-1", "iso8859-2", "iso8859-4", "iso8859-5", "iso8859-9",
        "iso8859-13", "iso8859-15", "iso8859-16",
        "cp1250", "cp1251", "cp1252", "cp1253", "cp1254", "cp1257",
        "mac-roman", "cp437", "cp850",
    }
)

#: WHATWG Encoding Standard §4.2: `iso-8859-1`, `latin1` and `us-ascii` are all
#: *labels for* windows-1252, and every browser decodes them that way. Following
#: that is not pedantry -- a body honestly declared `iso-8859-1` but containing
#: 0x93/0x94 (the overwhelmingly common case, because authors type smart quotes)
#: decodes under true latin-1 into C1 control characters, which then travel into
#: the stored body, the embedding and every report that quotes it. cp1252 is a
#: superset for the byte ranges that matter, so this can only recover characters,
#: never lose them.
_LABEL_OVERRIDES: Final[dict[str, str]] = {"iso8859-1": "cp1252", "ascii": "cp1252"}

_MIN_SNIFF_HIGH_BYTES: Final = 16
"""How many non-ASCII bytes a body needs before statistical detection is asked.

Counted over *high* bytes rather than total length, because those are the only
bytes a charset detector learns anything from -- the ASCII range decodes
identically under every candidate codec. Asked with fewer, the detector still
answers, confidently: `b"Caf\\xe9"` comes back as a CJK codec that renders four
bytes of French as two ideographs, and an English article carrying two curly
quotes comes back as Baltic. Below the threshold `cp1252` is simply the better
prior for Latin-script content, and its worst case -- a handful of replacement
characters -- is legible and obviously wrong, which a wrong-script decode is not.
"""

_SNIFF_WINDOW: Final = 8192
"""How much of the body the high-byte count inspects. Bounded so a 40 MB
mislabelled document does not cost a full scan on the fatal path."""

_FALLBACK_CODEC: Final = "cp1252"
"""Terminal fallback, decoded with `errors="replace"` so it cannot raise.

`cp1252` rather than `latin-1` because a Windows-1252 body read as latin-1 turns
smart quotes and em dashes into C1 control characters, which survive into the
stored body and into every embedding of it. cp1252 is a strict superset of
latin-1 for the byte ranges that matter, so choosing it costs nothing and
recovers punctuation.
"""

try:  # pragma: no cover -- import-time branch, one side runs per environment
    from charset_normalizer import from_bytes as _cn_from_bytes

    _CHARSET_NORMALIZER: Callable[..., Any] | None = _cn_from_bytes
except ImportError:  # pragma: no cover
    _CHARSET_NORMALIZER = None


def decode_bytes(raw: bytes, *, content_type: str | None = None) -> str:
    """Decode provider bytes to `str`, in descending order of evidence.

    1. **The declared charset** from the transport's `Content-Type`. The provider
       is the only party that actually knows, so it is asked first.
    2. **A byte-order mark**, which overrides a declared *8-bit* charset. This
       inversion is not a contradiction of rule 1, it is the one case where the
       declaration is refutable: a UTF-16 document decodes cleanly under cp1252
       into text interleaved with NULs, so waiting for a `UnicodeDecodeError`
       that can never arrive would silently destroy the body.
    3. **A mislabelled body**, in three escalating forms: the document's own
       in-band declaration (`<meta charset>`, the XML prolog), a strict UTF-8
       attempt, and finally statistical detection via `charset_normalizer`.
    4. `cp1252` with replacement, which never raises. A body with three
       replacement characters is repairable by reprocessing from R2; a stage that
       raised here would quarantine the record instead.

    Deterministic at every rung -- stage 1 must "reproduce byte-identical output
    for the same input and version" (`docs/signal-model.md` §5.1), which is what
    lets `scripts/reindex.py` replay the corpus without drift.
    """
    if not raw:
        return ""

    declared = _normalize_codec(_charset_from_content_type(content_type))
    bom_codec = _bom_codec(raw)

    # Rule 2. Only a wide BOM can overrule the declaration, and only when the
    # declaration is not itself a wide codec. A UTF-8 BOM never needs to override
    # anything: every plausible declaration over UTF-8 bytes either succeeds or
    # raises honestly, which the ladder below already handles.
    if (
        bom_codec is not None
        and bom_codec.startswith(("utf-16", "utf-32"))
        and (declared is None or not declared.startswith(("utf-16", "utf-32")))
    ):
        decoded = _try_decode(raw, bom_codec)
        if decoded is not None:
            return _strip_bom(decoded)

    if declared is not None:
        # The classic lie: `ISO-8859-1` announced over a UTF-8 body. Valid
        # multi-byte UTF-8 does not occur by accident in genuine 8-bit text, so
        # its presence outweighs a label that costs the server nothing to emit.
        if declared in _NEVER_FAILS and _is_multibyte_utf8(raw):
            return _strip_bom(raw.decode("utf-8"))
        codec = "utf-8-sig" if declared == "utf-8" and raw.startswith(codecs.BOM_UTF8) else declared
        decoded = _try_decode(raw, codec)
        if decoded is not None:
            return _strip_bom(decoded)

    if bom_codec is not None:
        decoded = _try_decode(raw, bom_codec)
        if decoded is not None:
            return _strip_bom(decoded)

    in_band = _normalize_codec(_charset_from_document(raw))
    if in_band is not None and in_band not in _NEVER_FAILS:
        decoded = _try_decode(raw, in_band)
        if decoded is not None:
            return _strip_bom(decoded)

    decoded = _try_decode(raw, "utf-8")
    if decoded is not None:
        return _strip_bom(decoded)

    if in_band is not None:
        decoded = _try_decode(raw, in_band)
        if decoded is not None:
            return _strip_bom(decoded)

    sniffed = _sniff_codec(raw)
    if sniffed is not None:
        decoded = _try_decode(raw, sniffed)
        if decoded is not None:
            return _strip_bom(decoded)

    return _strip_bom(raw.decode(_FALLBACK_CODEC, errors="replace"))


def _charset_from_content_type(content_type: str | None) -> str | None:
    """Pull `charset=` out of a media type. `None` when absent or unparsable."""
    if not content_type:
        return None
    match = _CHARSET_PARAM.search(content_type)
    return match.group(1) if match else None


def _charset_from_document(raw: bytes) -> str | None:
    """Read the document's own declaration from its first 2 KiB.

    Decoded as ASCII with errors ignored on purpose: the declaration is by
    definition ASCII, and the point of reading it is that we do not yet know how
    to decode the rest. The window matches the HTML spec's pre-scan limit -- a
    `<meta charset>` further in than that arrives after the parser has already
    committed, so honouring it here would disagree with every browser.
    """
    head = raw[:2048].decode("ascii", errors="ignore")
    for pattern in (_XML_ENCODING, _META_CHARSET):
        match = pattern.search(head)
        if match:
            return match.group(1)
    return None


def _bom_codec(raw: bytes) -> str | None:
    """The codec a byte-order mark names, if the body carries one."""
    for mark, codec in _BOMS:
        if raw.startswith(mark):
            return codec
    return None


def _normalize_codec(name: str | None) -> str | None:
    """Canonicalize a charset label, or `None` if Python has no such codec.

    An unknown label is dropped rather than raised on: providers emit
    `charset=none`, `charset=utf8mb4` and worse, and none of those is a reason to
    quarantine an otherwise perfectly good record.
    """
    if not name:
        return None
    try:
        canonical = codecs.lookup(name).name
    except (LookupError, ValueError):
        return None
    return _LABEL_OVERRIDES.get(canonical, canonical)


def _try_decode(raw: bytes, codec: str) -> str | None:
    """Strict decode, or `None`. Strict is the point -- errors are the signal."""
    try:
        return raw.decode(codec)
    except (UnicodeDecodeError, LookupError, ValueError):
        return None


def _is_multibyte_utf8(raw: bytes) -> bool:
    """Whether `raw` is valid UTF-8 *and* actually uses a multi-byte sequence.

    The second half matters: pure ASCII is valid UTF-8 and valid latin-1 and
    decodes identically under both, so it is no evidence either way and must not
    be allowed to override a declaration.
    """
    if raw.isascii():
        return False
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _sniff_codec(raw: bytes) -> str | None:
    """Statistical detection, when `charset_normalizer` is installed.

    Last resort before replacement, and guarded by an import check so that a slim
    deployment degrades to `cp1252` instead of failing to import the whole
    signal engine.

    One input shape is deliberately *not* recovered here: UTF-16 with no BOM and
    no declaration. Its evidence is the interleaved NULs, and NULs are not high
    bytes, so the threshold above skips the detector and the body falls through to
    cp1252. Counting NULs as evidence instead would hand the detector every
    mislabelled PNG as well, and it would answer -- producing a plausible 8-bit
    string with the NULs decoded away, which is precisely what the guard in
    `_clean_document` uses to recognize binary. So that case is left to fail
    loudly as a NUL-bearing body (fatal, quarantined, original intact in R2)
    rather than quietly as a wrong-but-clean one.
    """
    if _CHARSET_NORMALIZER is None:
        return None
    if sum(1 for byte in raw[:_SNIFF_WINDOW] if byte > 0x7F) < _MIN_SNIFF_HIGH_BYTES:
        return None
    try:
        best = _CHARSET_NORMALIZER(raw).best()
    except Exception:  # noqa: BLE001 -- a detector must never fail the stage
        return None
    if best is None:
        return None
    return _normalize_codec(getattr(best, "encoding", None))


def _strip_bom(text: str) -> str:
    """Drop a leading U+FEFF the codec left behind.

    `utf-16-le` and friends do not consume the mark when the byte order is named
    explicitly, and a body starting with an invisible character breaks prefix
    comparisons, `startswith` checks and the language detector's first n-gram.
    """
    return text[1:] if text.startswith("\ufeff") else text


# --------------------------------------------------------------------------- #
# PII redaction hook
# --------------------------------------------------------------------------- #

EMAIL_PLACEHOLDER: Final = "[EMAIL]"
PHONE_PLACEHOLDER: Final = "[PHONE]"
CARD_PLACEHOLDER: Final = "[CARD]"


@runtime_checkable
class Redactor(Protocol):
    """Where a PII policy plugs into stage 1.

    `docs/security-and-privacy.md` §6.1 requires that self-typed PII ("call me at
    …") be replaced with typed placeholders **at cleaning time, before
    embedding** -- because once the text is a vector the PII is neither readable
    nor auditable nor deletable, and the vector is copied into Qdrant, into an
    OpenSearch document and into every report that quotes the passage.

    A protocol rather than a function so an implementation can carry state (a
    loaded NER model, a compiled ruleset) and can name itself: `name` is what
    would be written into `lineage` to record *which* policy produced a stored
    body, since two bodies redacted by different policies are not comparable.

    `redact` is synchronous. A redactor that needs to call a service does not
    belong in stage 1 at all -- it would put a network round trip on the fatal
    path of every record, and stage 1's determinism guarantee
    (`docs/signal-model.md` §5.1) forbids depending on a remote response.
    """

    name: str

    def redact(self, text: str) -> str:
        """Return `text` with detected PII replaced by typed placeholders."""
        ...


_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9](?:[A-Za-z0-9.\-]*[A-Za-z0-9])?\.[A-Za-z]{2,}"
)

#: Three narrow shapes rather than one permissive one. A general "run of digits
#: with separators" pattern matches dates, version strings, IP addresses, order
#: numbers and prices, and a redactor that eats `2026-07-31` out of a news body
#: has corrupted the evidence it was protecting.
_PHONE_RE = re.compile(
    r"(?<![\w+])(?:"
    r"\+\d{1,3}[\s.\-]?(?:\(?\d{1,4}\)?[\s.\-]?){1,4}\d{2,4}"  # international, `+` required
    r"|\(\d{2,4}\)\s?\d{2,4}[\s.\-]?\d{2,4}"                   # parenthesised trunk code
    r"|\b\d{3}[.\-]\d{3}[.\-]\d{4}\b"                          # NANP 415-555-0132
    r")(?!\w)"
)

_CARD_RE = re.compile(r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![\d\-])")


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum -- the difference between a card number and an order number.

    Without it the pattern below matches every 13-to-19-digit run in the corpus:
    tracking numbers, IMEIs, ISBN-13s with the hyphens removed. Luhn rejects
    roughly nine in ten of those for free and is the standard test, so a false
    positive here is at least a *plausible* card rather than an arbitrary number.
    """
    total = 0
    for index, character in enumerate(reversed(digits)):
        value = int(character)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


class RegexRedactor:
    """Conservative default redactor: emails, phone numbers, payment cards.

    **It under-redacts by design.** The alternative failure -- an aggressive
    pattern that eats dates, prices and product codes -- silently corrupts the
    body that every downstream claim is grounded in, and unlike a missed phone
    number it is undetectable after the fact because the original is only in R2.
    So the patterns demand explicit structure: a `+` country code, parentheses
    around a trunk code, a NANP 3-3-4 shape, a Luhn-valid card. `+1 5551234` is
    not matched, and that is the intended trade, not an oversight.

    This class is the *shape* of a policy, not a compliance control. A real
    deployment supplies a detector with a model behind it and keeps this as the
    fallback for when that detector is unavailable.
    """

    name = "regex/v1"

    def __init__(self, *, redact_cards: bool = True) -> None:
        self._redact_cards = redact_cards

    def redact(self, text: str) -> str:
        """Replace detected PII: emails, then cards, then phones.

        Cards before phones matters: a 16-digit card written as `4111 1111 1111 1111` would
        otherwise be partially consumed by the phone patterns, leaving half a
        card number in the body -- which is worse than leaving all of it, because
        it looks redacted.
        """
        if not text:
            return text
        redacted = _EMAIL_RE.sub(EMAIL_PLACEHOLDER, text)
        if self._redact_cards:
            redacted = _CARD_RE.sub(self._replace_card, redacted)
        return _PHONE_RE.sub(PHONE_PLACEHOLDER, redacted)

    @staticmethod
    def _replace_card(match: re.Match[str]) -> str:
        candidate = re.sub(r"[ \-]", "", match.group(0))
        if 13 <= len(candidate) <= 19 and _luhn_ok(candidate):
            return CARD_PLACEHOLDER
        return match.group(0)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def clean_text(
    source: str | bytes,
    *,
    content_type: str | None = None,
    url: str | None = None,
    redactor: Redactor | None = None,
) -> str:
    """Turn one document into the cleaned body `Content.text` promises.

    Exported rather than kept private to `CleaningStage` because stage 2 needs
    exactly this for the body field it pulls out of a JSON payload: the field map
    in `connectors/normalize/mapper.py` marks a field as HTML and cleans it, and
    a *second* implementation on the service side would give the same item two
    different cleaned bodies -- and therefore two different layer-2 content
    hashes -- depending on which side of the pipeline cleaned it.

    `url` is advisory and never fetched; it only helps the extractor resolve
    relative links and pick site heuristics.
    """
    family = classify_content_type(content_type)
    if family is ContentFamily.BINARY:
        # Checked before decoding: running a text codec over a PDF wastes the work
        # and, worse, produces a plausible-looking string that a later edit might
        # be tempted to keep.
        raise NotImplementedError(
            f"no text extractor for content type {content_type!r}. Cleaning a PDF, "
            "office document or media file needs a format-specific extractor "
            "(pdfminer.six / python-docx / a transcription service); none is "
            "installed and none is wired into the pipeline. Media bodies reach "
            "OmniSense through MediaRef.transcript_ref instead."
        )
    # A `str` caller has already decoded -- stage 2 passes body fields it pulled
    # out of a parsed payload, where the JSON decoder settled the encoding. Only
    # bytes go through the ladder, and only they can carry a charset to honour.
    text = source if isinstance(source, str) else decode_bytes(source, content_type=content_type)
    return _clean_document(text, family=family, url=url, redactor=redactor)


def _clean_document(
    text: str,
    *,
    family: ContentFamily,
    url: str | None,
    redactor: Redactor | None,
) -> str:
    """Strip markup, collapse whitespace, then redact -- in that order.

    Redaction runs last because it must see contiguous prose. Markup routinely
    splits an address across elements (`<span>user</span>@example.com`), and a
    detector run before extraction sees two fragments and matches neither.

    No Unicode normalization is applied on top of `collapse_whitespace`. NFKC
    would fold ligatures, full-width Latin and superscripts into ASCII, which is
    right for a dedup hash and wrong for a body a report later quotes verbatim --
    `connectors/dedup/hashing.py` applies it on the way into the hash, where the
    output is never shown to anyone.

    Known limitation, inherited deliberately rather than patched here:
    `collapse_whitespace` strips U+200C/U+200D, which are load-bearing in emoji
    ZWJ sequences and in Persian and Indic orthography. Compensating for it in
    this module would make the service-side cleaner disagree with the
    connector-side one and fork every content hash, so the fix belongs in
    `connectors/normalize/html.py` where both callers would pick it up.
    """
    if not text.strip():
        return ""
    if "\x00" in text:
        # A NUL never appears in decoded text. It means the body is binary and
        # arrived mislabelled (or unlabelled) -- extracting "text" from it would
        # store a screenful of control characters as an observation.
        raise CleaningError(
            "decoded body contains NUL bytes, so it is binary rather than text; "
            "the record's declared content type is wrong"
        )
    cleaned = (
        extract_readable(text, url=url)
        if family is ContentFamily.MARKUP
        else collapse_whitespace(text)
    )
    if redactor is not None:
        cleaned = redactor.redact(cleaned)
    return cleaned


class CleaningStage:
    """Stage 1. Satisfies `Stage`; **fatal** per `FATAL_STAGES`.

    Holds no client and performs no I/O, so it is trivially safe to share across
    concurrent records -- the only per-record state lives on the context, which
    is what `SignalPipeline` requires of a stage.

    Note what this stage does *not* do: it never decides that its own failure is
    fatal. It raises, and `SignalPipeline` consults `FATAL_STAGES`. Keeping that
    decision outside the stage is what stops a stage from promoting itself and
    taking down ingestion (`services/signal_engine/pipeline.py`).
    """

    name = StageName.CLEAN
    version = "1.0.0"

    def __init__(self, *, redactor: Redactor | None = None) -> None:
        self._redactor = redactor

    @property
    def model_id(self) -> str | None:
        """Always `None`: stage 1 is deterministic and calls no model.

        `docs/signal-model.md` §5.1 records a model id only for stages 4-6, whose
        output cannot be reproduced without knowing which model produced it.
        Recording a fake one here would suggest this stage needs a model to
        replay, which is precisely the property that makes reprocessing cheap.
        """
        return None

    @property
    def redactor_name(self) -> str | None:
        """Which PII policy is active, or `None` when redaction is off.

        Exposed so a worker can log it once at startup. "Was redaction on when
        this body was stored?" is otherwise unanswerable after the fact, and the
        answer changes what the stored text means.
        """
        return None if self._redactor is None else self._redactor.name

    async def apply(self, ctx: EnrichmentContext) -> None:
        """Populate `ctx.cleaned_text`, and `ctx.payload` for structured records.

        The `""` versus `None` distinction on `cleaned_text` is meaningful and
        stage 2 relies on it: `""` means "cleaned, and there is no body" -- legal
        for a media-only post, which `Content.text` explicitly permits -- while
        `None` means this stage never ran.
        """
        family = classify_content_type(ctx.content_type)

        if family is ContentFamily.STRUCTURED:
            self._apply_structured(ctx)
            return

        if ctx.raw_bytes is None:
            raise CleaningError(
                f"no raw bytes on the context for content type {ctx.content_type!r}; "
                "a text record cannot be cleaned from its parsed payload alone"
            )

        ctx.cleaned_text = clean_text(
            ctx.raw_bytes,
            content_type=ctx.content_type,
            redactor=self._redactor,
        )

    def _apply_structured(self, ctx: EnrichmentContext) -> None:
        """Decode and parse a JSON record; leave body extraction to stage 2.

        There is no document body in a provider JSON payload -- the observation's
        text sits at a path only the connector's field map knows -- so inventing
        one here (concatenating every string leaf, say) would put ids, URLs and
        timestamps into `content.text` and poison every embedding built from it.

        The parsed object is written to `ctx.payload` **only when the context has
        none**. The worker normally supplies the connector's verbatim payload,
        and re-parsing the bytes over the top of it would quietly discard
        whatever the connector had already resolved.
        """
        ctx.cleaned_text = ""
        if ctx.payload:
            return
        if ctx.raw_bytes is None:
            raise CleaningError(
                "structured record has neither a payload nor raw bytes; there is "
                "nothing for stage 2 to normalize"
            )
        text = decode_bytes(ctx.raw_bytes, content_type=ctx.content_type)
        if not text.strip():
            raise CleaningError("structured record decoded to an empty document")
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise CleaningError(
                f"record declared {ctx.content_type!r} but is not valid JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise CleaningError(
                f"structured record parsed to {type(parsed).__name__}, not an object; "
                "stage 2 maps fields out of a mapping"
            )
        ctx.payload = parsed
