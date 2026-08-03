"""Readable-text extraction and URL canonicalization.

Two jobs that look unrelated and are not: both exist to make a Signal's identity
and body *stable across re-fetches*.

**Extraction** turns a provider's markup into the cleaned body that
`Signal.content.text` promises (`docs/signal-model.md` §3.2) -- markup stripped,
boilerplate removed, whitespace collapsed, paragraph breaks kept. Paragraphs are
kept because the chunker (`retrieval/chunking/splitter.py`) splits on them; an
extractor that returned one 3,000-word line would push chunk boundaries into the
middle of sentences for every long article in the corpus. Extraction also feeds
dedup layer 2, which hashes the *cleaned* text, so a publisher that re-serializes
its own HTML between polls must not present as a new record.

**Canonicalization** is rule 2 of `native_id` derivation (`docs/signal-model.md`
§4.1): for a feed item with no guid, `native_id = sha256(canonical_url)`. That
makes this function part of the identity derivation, and identity derivation "is
not migratable in place" (§7). Hence the governing rule here:

    Two spellings of the same URL must canonicalize to the same string, and
    canonicalizing twice must change nothing.

Both failure directions are real but they are not equally bad. Failing to
collapse two spellings *forks* one item into two Signals -- annoying, caught
later by dedup layer 2, recoverable. Collapsing two genuinely different URLs
*merges* two items into one id -- one of them is silently never stored, and no
downstream layer can recover it. The parameter and host rules below are
deliberately conservative for that reason.

Nothing here raises. A connector that could not extract a body still has a valid
Signal (`docs/connector-spec.md` §11.2 step 6: "a failed article fetch degrades
to the summary rather than raising"), so every backend is wrapped and the module
degrades through trafilatura -> BeautifulSoup -> regex rather than failing.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from typing import Any, Final
from urllib.parse import (
    parse_qsl,
    quote,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

__all__ = [
    "DEFAULT_PORTS",
    "PARAGRAPH_SEPARATOR",
    "TRACKING_PARAMS",
    "TRACKING_PARAM_PREFIXES",
    "canonicalize_url",
    "collapse_whitespace",
    "extract_readable",
    "is_tracking_param",
    "looks_like_html",
]

PARAGRAPH_SEPARATOR: Final = "\n\n"
"""What a block boundary becomes in the extracted text. See the module docstring."""


# --------------------------------------------------------------------------- #
# Optional backends
# --------------------------------------------------------------------------- #
#
# Imported once at module load and held as module-level handles rather than
# imported inside the functions. Two reasons: an import inside a hot loop costs a
# `sys.modules` lookup per record, and holding the handle is what lets a test
# monkeypatch `_TRAFILATURA = None` to exercise the degraded path on a machine
# where trafilatura happens to be installed. Both paths ship, so both are tested.

try:  # pragma: no cover -- import-time branch, exercised by whichever env runs
    import trafilatura as _trafilatura_module

    _TRAFILATURA: Any | None = _trafilatura_module
except ImportError:  # pragma: no cover
    _TRAFILATURA = None

try:  # pragma: no cover
    from bs4 import BeautifulSoup as _BeautifulSoup

    _BS4: Any | None = _BeautifulSoup
except ImportError:  # pragma: no cover
    _BS4 = None


# --------------------------------------------------------------------------- #
# Whitespace
# --------------------------------------------------------------------------- #

_ZERO_WIDTH_AND_SPACES: Final[dict[int, str | None]] = {
    # Zero-width characters are invisible and survive every naive `strip()`, so
    # two bodies that render identically hash differently and defeat dedup
    # layer 2. Soft hyphen included: publishers inject it for justification.
    0x00AD: None,  # soft hyphen
    0x200B: None,  # zero-width space
    0x200C: None,  # zero-width non-joiner
    0x200D: None,  # zero-width joiner
    0x2060: None,  # word joiner
    0xFEFF: None,  # BOM / zero-width no-break space
    # Exotic spaces normalize to U+0020 so the collapse below can see them.
    0x00A0: " ",  # no-break space -- what `&nbsp;` decodes to
    0x2002: " ",
    0x2003: " ",
    0x2007: " ",
    0x2009: " ",
    0x202F: " ",
    0x205F: " ",
    0x3000: " ",
}

_HORIZONTAL_RUN = re.compile(r"[^\S\n]+")
_LINE_EDGES = re.compile(r"[^\S\n]*\n[^\S\n]*")
_BLANK_RUN = re.compile(r"\n{3,}")


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph breaks.

    Deliberately *not* NFKC-normalized. NFKC folds ligatures, full-width Latin
    and superscripts into ASCII, which is right for a dedup hash and wrong for a
    stored body: it would silently rewrite the text a report later quotes.
    `connectors/dedup/hashing.py` applies NFKC on its way into the hash, where
    the output is never shown to anyone.
    """
    if not text:
        return ""
    text = text.translate(_ZERO_WIDTH_AND_SPACES)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZONTAL_RUN.sub(" ", text)
    text = _LINE_EDGES.sub("\n", text)
    text = _BLANK_RUN.sub(PARAGRAPH_SEPARATOR, text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Readable-text extraction
# --------------------------------------------------------------------------- #

_TAG_HINT = re.compile(r"<\s*(?:!doctype|[a-zA-Z][a-zA-Z0-9-]*)(?:\s|/?>)")

#: Elements whose text is never part of the observation. `noscript` is on the
#: list for a reason that is easy to miss: tracking pixels are usually shipped
#: inside it, and `html.parser` surfaces that markup as *text*, so leaving it in
#: puts `<img src=...>` fragments into the body. `head` is on it because
#: `<title>` belongs to `Content.title`, which the field map reads from the
#: payload -- extracting it into the body too would duplicate the headline at the
#: top of every article and skew both the language detector and the chunker.
_DISCARD_TAGS: Final[frozenset[str]] = frozenset(
    {
        "head", "title", "meta", "link", "base",
        "script", "style", "noscript", "template", "svg", "canvas", "math",
        "nav", "header", "footer", "aside", "form", "button", "select",
        "textarea", "iframe", "object", "embed", "map", "area", "dialog",
    }
)

#: Whole class/id *tokens* that mark boilerplate. Matched token-wise (splitting
#: on `-`, `_`, whitespace) rather than by substring, because a substring match
#: on "ad" removes `class="headline"` and on "nav" removes `class="navigation-
#: free-article"`. The set is short on purpose: this path only runs when
#: trafilatura is absent or silent, and a fallback that over-removes turns a
#: recoverable "worse body" into an empty one.
_BOILERPLATE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "nav", "navbar", "navigation", "menu", "sidebar", "masthead",
        "breadcrumb", "breadcrumbs", "pagination", "paginator",
        "cookie", "cookies", "consent", "gdpr",
        "newsletter", "subscribe", "signup", "paywall",
        "advert", "adverts", "advertisement", "ads", "adsense", "sponsored",
        "promo", "promotion", "share", "sharing", "social",
        "related", "recirc", "recirculation", "trending",
        "comments", "disqus", "skip", "footer", "sitemap",
    }
)

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

#: Block-level elements, and what their boundary becomes. `li` and `tr` get a
#: single newline so a ten-item list is one paragraph rather than ten.
_BLOCK_BREAKS: Final[dict[str, str]] = {
    **dict.fromkeys(
        (
            "p", "div", "section", "article", "main", "blockquote", "pre",
            "figure", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6",
            "ul", "ol", "dl", "table", "hr",
        ),
        PARAGRAPH_SEPARATOR,
    ),
    **dict.fromkeys(("li", "tr", "dt", "dd", "td", "th"), "\n"),
}

_PIXEL_SRC = re.compile(
    r"(?:/(?:pixel|beacon|track(?:ing)?|impression|collect|analytics)\b|\.gif\?)",
    re.IGNORECASE,
)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE)

_SCRIPT_BLOCK = re.compile(
    r"<(head|script|style|noscript|template|nav|footer|aside|form)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_COMMENT_BLOCK = re.compile(r"<!--.*?-->", re.DOTALL)
_BLOCK_TAG = re.compile(
    r"</?\s*(?:p|div|section|article|br|li|tr|t[dh]|d[td]|h[1-6]"
    r"|blockquote|pre|table|ul|ol)\b[^>]*>",
    re.IGNORECASE,
)
_ANY_TAG = re.compile(r"<[^>]*>")


def looks_like_html(source: str) -> bool:
    """Whether this string is worth handing to an HTML parser.

    Feed summaries are frequently already plain text. Running a parser over them
    is not merely wasted work: an ampersand or a `<` in prose gets entity-decoded
    or swallowed as a malformed tag, which quietly edits the body.
    """
    return bool(_TAG_HINT.search(source))


def extract_readable(
    source: str | bytes,
    *,
    url: str | None = None,
    include_tables: bool = True,
) -> str:
    """Extract the readable body from an HTML document or fragment.

    `url` is advisory: trafilatura uses it to resolve relative links and to pick
    site-specific heuristics. It is never fetched -- this function performs no
    I/O, because normalization must produce the same output on replay when the
    page has changed or gone (`docs/connector-spec.md` §2.4).

    Returns `""` rather than raising when nothing readable is found. An empty
    body is legal (`Content.text` "may be empty for media-only posts"), and a
    connector that raised here would send a perfectly good feed entry to the DLQ
    because its linked article was a video player.
    """
    text = _as_text(source)
    if not text.strip():
        return ""
    if not looks_like_html(text):
        # Already plain text. See `looks_like_html`.
        return collapse_whitespace(text)

    for backend in (_extract_with_trafilatura, _extract_with_soup, _extract_with_regex):
        try:
            extracted = backend(text, url, include_tables)
        except Exception:  # noqa: BLE001 -- degrade to the next backend, never fail
            continue
        if extracted and extracted.strip():
            return collapse_whitespace(extracted)
    return ""


def _extract_with_trafilatura(source: str, url: str | None, include_tables: bool) -> str:
    """Primary backend: trafilatura's boilerplate model.

    `include_comments=False` matters for identity as much as for quality -- a
    comment thread grows between polls, so including it would change the cleaned
    text (and therefore the layer-2 content hash) on every re-fetch of an
    otherwise unchanged article.
    """
    if _TRAFILATURA is None:
        return ""
    result = _TRAFILATURA.extract(
        source,
        url=url,
        output_format="txt",
        include_comments=False,
        include_tables=include_tables,
        include_formatting=False,
        favor_precision=False,
    )
    if not result:
        return ""
    # trafilatura emits one block per line; this module's contract is one blank
    # line between blocks. Unifying the shape here rather than leaving each
    # backend to its own habit is what stops a deployment that lost trafilatura
    # from re-hashing its whole corpus under dedup layer 2 -- the *selection* of
    # blocks inevitably differs between backends, but the whitespace does not
    # have to.
    return PARAGRAPH_SEPARATOR.join(
        line.strip() for line in result.splitlines() if line.strip()
    )


def _extract_with_soup(source: str, url: str | None, include_tables: bool) -> str:
    """Fallback backend: strip the tree by hand.

    Runs when trafilatura is not installed, or when it returns nothing -- which
    it does for short fragments and comment-shaped pages, because its model is
    trained on articles. Those are exactly the payloads social connectors carry,
    so this path is not a rare degradation; it is the common one.
    """
    if _BS4 is None:
        return ""
    soup = _make_soup(source)
    if soup is None:
        return ""

    for element in soup.find_all(list(_DISCARD_TAGS)):
        element.decompose()
    for element in soup.find_all(_is_boilerplate):
        element.decompose()
    for element in soup.find_all(_is_tracking_pixel):
        element.decompose()
    if not include_tables:
        for element in soup.find_all("table"):
            element.decompose()

    for element in soup.find_all("br"):
        element.replace_with("\n")
    for name, separator in _BLOCK_BREAKS.items():
        for element in soup.find_all(name):
            # Appending inside the element rather than after it keeps the break
            # attached when a parent is later unwrapped by `get_text`.
            element.append(separator)

    return str(soup.get_text())


def _extract_with_regex(source: str, url: str | None, include_tables: bool) -> str:
    """Last-resort backend: no parser available at all.

    Exists so that a deployment missing lxml/bs4 degrades to a worse body rather
    than to an empty corpus. It is wrong on malformed markup in ways a parser is
    not, which is why it is last and not first.
    """
    text = _COMMENT_BLOCK.sub(" ", source)
    text = _SCRIPT_BLOCK.sub(" ", text)
    text = _BLOCK_TAG.sub(PARAGRAPH_SEPARATOR, text)
    # Inline tags collapse to nothing rather than to a space: `with <b>bold</b>.`
    # must render as "with bold." and a space would put one before the period.
    text = _ANY_TAG.sub("", text)
    return _unescape_entities(text)


def _make_soup(source: str) -> Any | None:
    """Parse with lxml when it is present, else the stdlib parser.

    lxml is preferred for its error recovery on real-world markup; `html.parser`
    is the guaranteed-present fallback so a missing wheel degrades instead of
    raising.
    """
    if _BS4 is None:
        return None
    for parser in ("lxml", "html.parser"):
        try:
            return _BS4(source, parser)
        except Exception:  # noqa: BLE001 -- missing parser or malformed input
            continue
    return None


def _is_boilerplate(tag: Any) -> bool:
    """Whether an element's class/id marks it as chrome rather than content."""
    attrs = getattr(tag, "attrs", None)
    if not attrs:
        return False
    values: list[str] = []
    class_value = attrs.get("class")
    if isinstance(class_value, str):
        values.append(class_value)
    elif isinstance(class_value, Iterable):
        values.extend(str(v) for v in class_value)
    for key in ("id", "role", "data-testid"):
        value = attrs.get(key)
        if isinstance(value, str):
            values.append(value)
    for value in values:
        for token in _TOKEN_SPLIT.split(value.lower()):
            if token and token in _BOILERPLATE_TOKENS:
                return True
    return False


def _is_tracking_pixel(tag: Any) -> bool:
    """Whether an element is a 1x1 beacon or a hidden tracker.

    Images carry no text, so this is not about the body directly -- it is about
    what `noscript` and hidden containers drag in with them, and about not
    leaving a third-party beacon URL sitting in an extracted body where it could
    be mistaken for a citation.
    """
    if getattr(tag, "name", None) not in {"img", "iframe", "image"}:
        return False
    attrs = getattr(tag, "attrs", {}) or {}
    if str(attrs.get("width", "")).strip() in {"0", "1"}:
        return True
    if str(attrs.get("height", "")).strip() in {"0", "1"}:
        return True
    style = str(attrs.get("style", ""))
    if style and _HIDDEN_STYLE.search(style):
        return True
    src = str(attrs.get("src", ""))
    return bool(src and _PIXEL_SRC.search(src))


def _unescape_entities(text: str) -> str:
    """Decode HTML entities without importing a parser."""
    from html import unescape

    return unescape(text)


def _as_text(source: str | bytes) -> str:
    """Decode provider bytes to text, guessing only when it must.

    UTF-8 first because everything modern is UTF-8; the declared charset second
    because legacy news feeds are still Windows-1252 and mojibake in the body is
    permanent -- the raw bytes are archived, but the cleaned text is what gets
    embedded and cited.
    """
    if isinstance(source, str):
        return source
    try:
        return source.decode("utf-8")
    except UnicodeDecodeError:
        pass
    head = source[:2048].decode("ascii", errors="ignore")
    match = re.search(r"charset=[\"']?([A-Za-z0-9_\-]+)", head)
    if match:
        try:
            return source.decode(match.group(1))
        except (LookupError, UnicodeDecodeError):
            pass
    return source.decode("cp1252", errors="replace")


# --------------------------------------------------------------------------- #
# URL canonicalization
# --------------------------------------------------------------------------- #

DEFAULT_PORTS: Final[dict[str, int]] = {
    "http": 80,
    "https": 443,
    "ws": 80,
    "wss": 443,
    "ftp": 21,
}
"""Ports that are implied by their scheme and must therefore be dropped.

`https://example.com:443/x` and `https://example.com/x` address the same
resource; keeping the port would give them two ids.
"""

TRACKING_PARAM_PREFIXES: Final[tuple[str, ...]] = (
    "utm_",       # Google Analytics / everything that copied it
    "pk_",        # Matomo
    "mtm_",       # Matomo 4
    "matomo_",
    "piwik_",
    "hsa_",       # HubSpot ads
    "vero_",
)

TRACKING_PARAMS: Final[frozenset[str]] = frozenset(
    {
        "fbclid", "gclid", "gclsrc", "dclid", "wbraid", "gbraid", "msclkid",
        "twclid", "ttclid", "igshid", "igsh", "yclid", "li_fat_id", "mkt_tok",
        "mc_cid", "mc_eid", "_hsenc", "_hsmi", "hsctatracking",
        "s_kwcid", "ef_id", "trk", "trkcampaign", "icid", "ncid", "spm", "scm",
        "ref_src", "ref_url", "guccounter", "guce_referrer", "guce_referrer_sig",
        "at_medium", "at_campaign", "_openstat", "wt_mc", "wt_zmc",
        "action_object_map", "action_type_map", "action_ref_map",
        "__twitter_impression", "cmpid", "cvid",
    }
)
"""Query parameters that identify the *referrer*, never the resource.

Notably absent: `ref`, `cid`, `source`, `id`. Each is a tracker on some sites and
a content selector on others, and the module docstring explains why the
asymmetry between forking and merging identity makes that a reason to keep them.
"""

_UNRESERVED: Final[frozenset[str]] = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_PCT_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_PATH_SAFE: Final = "/-._~!$&'()*+,;=:@"


def is_tracking_param(name: str) -> bool:
    """Whether a query parameter is pure attribution and safe to drop."""
    lowered = name.lower()
    if lowered in TRACKING_PARAMS:
        return True
    return lowered.startswith(TRACKING_PARAM_PREFIXES)


def canonicalize_url(
    url: str,
    *,
    base: str | None = None,
    resolver: Callable[[str], str | None] | None = None,
) -> str:
    """Return a stable canonical form of `url`, or `""` if it has no host.

    Applied, in order: optional redirect resolution, optional resolution against
    `base`, scheme and host lower-cased, userinfo dropped, default port dropped,
    dot segments removed, percent-escapes normalized, tracking parameters
    stripped, remaining parameters sorted, fragment dropped.

    `resolver` is a **pure callable**, not an HTTP client, and that is the whole
    point. Resolving a shortener at normalize time would make identity depend on
    what the network answered at that instant, so replaying the same payload
    after the redirect changed would mint a second Signal for the same item
    (`docs/connector-spec.md` §2.4: normalization "must not depend on database
    state, or the same payload will normalize differently on replay"). Connectors
    that follow redirects do it in `fetch()` and pass the captured mapping in
    here. Returning `None` from the resolver means "unresolved" and leaves the
    input untouched.

    Returns `""` for a relative reference with no `base`. That is not a silent
    failure -- an id derived from `/index.html` would collide across every feed
    that uses it, so the caller is expected to fall through to rule 3 of
    `docs/signal-model.md` §4.1 instead.
    """
    if not url or not url.strip():
        return ""
    candidate = _strip_control_characters(url.strip())

    if resolver is not None:
        resolved = resolver(candidate)
        if resolved and resolved.strip():
            candidate = _strip_control_characters(resolved.strip())
    if base:
        candidate = urljoin(base, candidate)

    parts = urlsplit(candidate)
    scheme = parts.scheme.lower()

    if not parts.netloc:
        if not scheme:
            # Relative reference. See the docstring: no stable identity exists.
            return ""
        # Opaque scheme (mailto:, urn:, data:). Nothing to normalize beyond the
        # scheme's case and the fragment, and guessing further would change what
        # the URI addresses.
        return f"{scheme}:{parts.path}" + (f"?{parts.query}" if parts.query else "")

    host = _canonical_host(parts.hostname or "")
    if not host:
        return ""
    port = _canonical_port(scheme, parts.port)
    netloc = f"{host}:{port}" if port is not None else host

    path = _normalize_percent_escapes(_remove_dot_segments(parts.path), _PATH_SAFE)
    if not path:
        path = "/"

    query = _canonical_query(parts.query)

    # The fragment is dropped unconditionally: it is never sent to the server, so
    # two URLs differing only in fragment are the same resource by definition.
    return urlunsplit((scheme, netloc, path, query, ""))


def _canonical_host(host: str) -> str:
    """Lower-case, de-root and IDNA-encode a host.

    The trailing dot in `example.com.` is the fully-qualified form of the same
    name. Unicode hosts are encoded to their punycode form so that `münchen.de`
    and `xn--mnchen-3ya.de` -- which resolve to one server -- get one id.

    `www.` is *not* stripped. It is a distinct DNS label that some hosts serve
    different content from, and a site that considers the two equivalent says so
    with a 301, which is what `resolver` is for.
    """
    host = host.strip().rstrip(".").lower()
    if not host:
        return ""
    if host.isascii():
        return host
    try:
        return host.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError):
        # An unencodable host (empty label, over-long label) still needs a
        # deterministic answer; NFKC at least folds the compatibility forms.
        return unicodedata.normalize("NFKC", host)


def _canonical_port(scheme: str, port: int | None) -> int | None:
    """Drop a port that the scheme already implies."""
    if port is None:
        return None
    if DEFAULT_PORTS.get(scheme) == port:
        return None
    return port


def _canonical_query(query: str) -> str:
    """Strip tracking parameters and re-serialize deterministically.

    Parameters are sorted. Order is semantically meaningful for a small number of
    APIs with repeated keys, but a provider re-ordering its own links between
    polls is far more common than an endpoint that cares -- and the first costs a
    forked identity on every record.

    Re-serialization also normalizes `+` to `%20` and a bare flag to `flag=`.
    Both are legal spellings of the same query, and picking one is what makes the
    function idempotent.
    """
    if not query:
        return ""
    pairs = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if not is_tracking_param(key)
    ]
    if not pairs:
        return ""
    pairs.sort()
    return urlencode(pairs, quote_via=quote, safe="")


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 §5.2.4. `/a/b/../c` and `/a/c` are the same resource."""
    if "." not in path:
        return path
    segments = path.split("/")
    output: list[str] = []
    for index, segment in enumerate(segments):
        if segment == ".":
            if index == len(segments) - 1:
                output.append("")
            continue
        if segment == "..":
            if len(output) > 1:
                output.pop()
            if index == len(segments) - 1:
                output.append("")
            continue
        output.append(segment)
    resolved = "/".join(output)
    if path.startswith("/") and not resolved.startswith("/"):
        resolved = "/" + resolved
    return resolved


def _normalize_percent_escapes(value: str, safe: str) -> str:
    """Normalize percent-encoding without changing what the URL means.

    RFC 3986 §6.2.2: escapes of unreserved characters are decoded (`%7E` -> `~`)
    and every remaining escape is upper-cased (`%2f` -> `%2F`). Reserved
    characters are deliberately *not* decoded -- `%2F` is an encoded slash and
    decoding it would move the resource to a different path.
    """
    if not value:
        return value
    out: list[str] = []
    cursor = 0
    for match in _PCT_ESCAPE.finditer(value):
        out.append(quote(value[cursor : match.start()], safe=safe))
        character = chr(int(match.group(1), 16))
        out.append(character if character in _UNRESERVED else f"%{match.group(1).upper()}")
        cursor = match.end()
    out.append(quote(value[cursor:], safe=safe))
    return "".join(out)


def _strip_control_characters(value: str) -> str:
    """Remove tabs, newlines and NULs a provider embedded in a URL.

    Feeds wrap long links across lines. A URL carrying a raw newline is not just
    ugly: it splits an HTTP request line, and it would give the wrapped and
    unwrapped spellings different ids.
    """
    return "".join(ch for ch in value if ch not in "\t\r\n\x00" and ord(ch) >= 0x20)
