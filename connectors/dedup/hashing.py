"""Content fingerprints and the near-duplicate *clustering* primitives.

`docs/signal-model.md` §4.2 defines three dedup layers. This module owns the
hashing for all three and the clustering for the third:

| Layer | Key                        | This module supplies            |
| ----- | -------------------------- | ------------------------------- |
| 1     | equal `Signal.id`          | `identity_key`                  |
| 2     | `sha256(cleaned text)`     | `canonicalize`, `content_sha256`|
| 3     | 64-bit SimHash, Hamming ≤3 | `simhash64`, `assign_clusters`  |

**Layer 3 clusters. It never drops.** `docs/signal-model.md` §4.3 is explicit:
the same press release on the wire, on the company blog, on X and in three
subreddits is *evidence of spread*. Deleting five of the six copies would destroy
the trend volume and collapse the `corroboration` component of `confidence`
(§3.5) to the value it would have had if the story had appeared once. So this
module deliberately exposes no function that removes, filters or drops a
near-duplicate -- only `assign_clusters`, which partitions its input and returns
every member it was given, and `elect_canonical`, which decides which member gets
indexed. There is a test asserting that no such removal function exists, because
the tempting one-liner is very easy to add later and impossible to notice.

Layers 1 and 2 *do* suppress, and that is not a contradiction: they collapse a
re-fetch or a byte-identical repost of the *same* observation, which adds no
information. Layer 3 collapses distinct observations of the same *event*, which
is the information.

Why banding, and why the threshold is 3
---------------------------------------
Comparing every new fingerprint against every stored one is O(n²) and n is the
whole corpus. Instead a 64-bit fingerprint is split into 4 disjoint 16-bit bands
and each band is a lookup key. Two fingerprints differing in at most 3 bits must
agree on at least one band -- 3 bits cannot be spread across 4 disjoint bands
without leaving one untouched. So banded lookup at threshold 3 has **no false
negatives**; it is an exact index, not an approximation, and candidate lookup
costs 4 set probes instead of n comparisons.

That pigeonhole bound is also why `CONNECTOR_SIMHASH_DISTANCE_THRESHOLD=3`
pairs with 4 bands: `threshold < bands` is a hard requirement, enforced in
`assign_clusters` rather than left as folklore. Widening the threshold to catch
the "related" 4-6 range that `docs/connector-spec.md` §7 records as a graph edge
requires widening the banding to match (8 bands of 8 bits), or the index starts
silently missing pairs it claims to find.

MinHash is offered alongside SimHash for the case SimHash handles badly:
documents over roughly 5k tokens, where a 64-bit fingerprint saturates and
everything long looks similar to everything else long
(`docs/connector-spec.md` §7).

Everything here is pure, synchronous and deterministic across processes and
machines. `hash()` is deliberately never used -- it is salted per interpreter by
`PYTHONHASHSEED`, so a fingerprint built on it would differ between two workers
computing it for the same text, which is worse than having no fingerprint at
all. `native_id` rule 3 in `docs/signal-model.md` §4.1 feeds `simhash64` into
identity derivation, so a change to any function here forks identity and is a
`schema_version` bump, not a refactor. The pinned-value tests exist to make that
impossible to do by accident.

This module imports nothing but the standard library (`docs/architecture.md`
§6.2 rule 2), so it is usable from a connector under test with no Redis, no
network and no services running.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

__all__ = [
    "DEFAULT_SHINGLE_SIZE",
    "MINHASH_LSH_BANDS",
    "MINHASH_PERMUTATIONS",
    "SIMHASH_BITS",
    "SIMHASH_DISTANCE_THRESHOLD",
    "SIMHASH_LSH_BANDS",
    "BandedIndex",
    "Cluster",
    "ClusterAssignment",
    "ClusterMember",
    "assign_clusters",
    "canonicalize",
    "cluster_id_for",
    "content_key",
    "content_sha256",
    "elect_canonical",
    "hamming",
    "identity_key",
    "is_near_duplicate",
    "jaccard",
    "minhash",
    "minhash_band_keys",
    "shingles",
    "simhash64",
    "simhash_band_keys",
    "tokenize",
]


SIMHASH_BITS = 64
"""Fingerprint width. Fixed at 64 because the threshold, the banding and the
`native_id` rule-3 derivation in `docs/signal-model.md` §4.1 are all stated in
terms of it; widening it is a `schema_version` change."""

SIMHASH_DISTANCE_THRESHOLD = 3
"""Default Hamming distance at which two fingerprints are near-duplicates.

Mirrors `CONNECTOR_SIMHASH_DISTANCE_THRESHOLD` in `.env.example`. A default
rather than a lookup: `connectors/` may not read settings (that would mean
importing `backend/`), so the runtime passes an override down when the operator
has tuned one. `docs/connector-spec.md` §9 records that 3 is a starting guess
awaiting a labelled corpus, and that the right value probably differs for a
200-character tweet and a 3,000-word article."""

SIMHASH_LSH_BANDS = 4
"""Bands the 64-bit fingerprint is split into: 4 x 16 bits.

Must stay strictly greater than the distance threshold, or the pigeonhole
guarantee in the module docstring fails and the index starts missing pairs
without any symptom other than a quietly lower duplicate rate."""

DEFAULT_SHINGLE_SIZE = 3
"""Token n-gram width. `docs/connector-spec.md` §7 fixes 3-gram shingles.

Shingling is what makes this a document fingerprint rather than a checksum:
hashing the whole string gives a value that changes completely on a one-word
edit, which is exactly the case near-duplicate detection exists to catch."""

MINHASH_PERMUTATIONS = 128
"""Signature length for MinHash. 128 puts the standard error of the Jaccard
estimate near 1/sqrt(128) ≈ 0.09, which is enough to separate "reworded" from
"unrelated" without making the signature bigger than the text it summarizes."""

MINHASH_LSH_BANDS = 16
"""Bands for MinHash LSH: 16 bands of 8 rows over a 128-element signature.

Unlike the SimHash banding this is genuinely probabilistic -- there is no
pigeonhole argument -- so it is a candidate *generator* whose output must always
be confirmed by `jaccard`."""


# --------------------------------------------------------------------------- #
# Canonicalization
# --------------------------------------------------------------------------- #

_ZERO_WIDTH_RE = re.compile(
    "[\u00ad\u200b-\u200f\u202a-\u202e\u2060\ufeff]"
)
"""Soft hyphens, zero-width spaces, bidi marks, word joiners and BOMs.

NFKC leaves every one of these in place, and they are the cheapest way to defeat
an exact-content hash: a scraper that injects a single U+200B per paragraph
produces text that is visually identical and byte-different on every fetch.

U+2028 and U+2029 are deliberately *not* here even though they are equally
invisible -- they are line separators, and deleting rather than collapsing them
would weld the last word of one line onto the first word of the next, inventing
a shingle that appears in no other copy. `_WHITESPACE_RE` handles them, since
Python's `\\s` matches both."""

_WHITESPACE_RE = re.compile(r"\s+")

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)

_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "gbraid",
        "wbraid",
        "msclkid",
        "yclid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "_hsenc",
        "_hsmi",
    }
)
"""Campaign parameters carrying no content.

`docs/connector-spec.md` §7 names `utm_*`, `fbclid` and `gclid`; the rest are the
same thing from other ad networks. Deliberately absent are `ref`, `source` and
`spm`, which look like tracking and are load-bearing on some sites -- stripping a
parameter that selects content would make two *different* pages hash alike, and
a false merge is far more expensive than a missed one."""

_TAIL_AFFORDANCE_RE = re.compile(
    r"\s*(?:[\[(]\s*)?(?:…|\.{3})?\s*"
    r"(?:read more|read the full (?:story|article)|continue reading|"
    r"see more|view more|view full article|full story|"
    r"the post .{0,160}? appeared first on .{0,80})"
    r"\s*[.…]*\s*(?:[\])])?\s*$",
    re.IGNORECASE,
)
"""Trailing "read more" affordances appended by feed generators.

The same wire story arrives from three aggregators with three different tails.
Left in, the tail is the only thing distinguishing the copies and layer 2 never
fires (`docs/connector-spec.md` §7)."""

_MAX_TAIL_STRIPS = 3
"""Affordances stack ("... Read more Continue reading"). Bounded rather than
`while True` so a pathological input cannot spin here."""


def canonicalize(text: str) -> str:
    """Normalize text to the form every hash in this module is taken over.

    Applied in this order for a reason: URLs are rewritten *before* the text is
    lowercased so the rewrite sees real hostnames, and the trailing-affordance
    strip runs *after* whitespace collapse so a tail broken across three lines by
    the feed generator still matches the pattern.

    Boilerplate removal is deliberately **not** repeated here. `content.text` is
    already the cleaned body by the time a Signal exists
    (`docs/signal-model.md` §3.2) -- markup stripped, boilerplate removed -- and
    re-running an extractor over cleaned prose would strip real sentences. What
    remains is the residue that survives cleaning and still differs between two
    copies of the same story: case, Unicode form, invisible characters, campaign
    parameters and feed-generator tails.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    normalized = _URL_RE.sub(lambda m: _canonical_url(m.group(0)), normalized)
    normalized = normalized.lower()
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    for _ in range(_MAX_TAIL_STRIPS):
        stripped = _TAIL_AFFORDANCE_RE.sub("", normalized).strip()
        if stripped == normalized:
            break
        normalized = stripped
    return normalized


def _canonical_url(url: str) -> str:
    """Lowercase scheme and host, drop campaign parameters, drop the fragment.

    Matches the URL canonicalization `docs/signal-model.md` §4.1 rule 2 uses to
    derive a `native_id`, so a feed without a guid and the body text that quotes
    its own permalink agree on what the link is.

    A URL that will not parse is returned untouched. Failing the whole hash
    because one link in a 3,000-word article is malformed would turn a cosmetic
    provider bug into a total loss of layer-2 dedup for that source.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(key)
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path,
            urlencode(kept),
            "",
        )
    )


def _is_tracking_param(key: str) -> bool:
    lowered = key.lower()
    return lowered in _TRACKING_PARAMS or lowered.startswith(_TRACKING_PARAM_PREFIXES)


# --------------------------------------------------------------------------- #
# Layer 1 and layer 2 keys
# --------------------------------------------------------------------------- #


def content_sha256(text: str) -> str:
    """SHA-256 of the **canonicalized** text -- layer 2 of `docs/signal-model.md` §4.2.

    Over the canonical form, not the raw string, because layer 2 exists to catch
    the same wire story republished by a dozen outlets. A provider that
    re-serializes its own HTML slightly differently between two polls, or an
    aggregator that appends its own "Read more", would otherwise present the same
    article as a new record forever.

    Empty text hashes like any other input; refusing to *use* the digest of an
    empty body is the caller's decision, and `BaseConnector.dedup_keys` already
    makes it -- an empty-bodied item must not suppress every later empty-bodied
    item.
    """
    return hashlib.sha256(canonicalize(text).encode("utf-8")).hexdigest()


def identity_key(connector_slug: str, signal_id: str) -> str:
    """Layer 1 Redis key: this exact item from this exact connector."""
    return f"os:dedup:id:{connector_slug}:{signal_id}"


def content_key(connector_slug: str, text: str) -> str:
    """Layer 2 Redis key: this exact cleaned body from this connector.

    Scoped by slug rather than global on purpose. A global content key would let
    one connector's poll suppress another connector's first sighting of the same
    story, and cross-*platform* recurrence is precisely the evidence layer 3
    exists to preserve rather than destroy (§4.3).
    """
    return f"os:dedup:sha:{connector_slug}:{content_sha256(text)}"


# --------------------------------------------------------------------------- #
# Tokenization and shingling
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_SHINGLE_SEP = "\x1f"
"""ASCII unit separator: a byte no token can contain.

Joining shingles with a space would make ("ab", "c") and ("a", "bc") the same
feature, which quietly inflates the similarity of unrelated text."""


def tokenize(text: str) -> tuple[str, ...]:
    """Split canonicalized text into word tokens.

    Word-character runs, which is right for space-delimited scripts and wrong for
    CJK -- an unsegmented Chinese sentence becomes one enormous token and its
    fingerprint degenerates to a checksum. Accepted for now because Phase 1
    sources are English-dominant; fixing it means a segmenter, which means a
    dependency and a model, and `docs/signal-model.md` §9 has not settled the
    multilingual story yet.
    """
    return tuple(_TOKEN_RE.findall(text))


def shingles(
    tokens: Sequence[str], size: int = DEFAULT_SHINGLE_SIZE
) -> tuple[str, ...]:
    """Overlapping token n-grams, duplicates preserved.

    Duplicates are preserved because SimHash weights features by frequency: a
    phrase repeated six times in a press release should pull the fingerprint
    toward itself six times as hard as one used once.

    Text shorter than `size` falls back to unigrams instead of producing nothing.
    Without the fallback every tweet, review title and one-line comment would
    fingerprint to zero and land in the same degenerate cluster -- the shortest
    content is where near-duplicate detection matters most, since a repost is
    usually the whole item.
    """
    if size < 1:
        raise ValueError(f"shingle size must be >= 1, got {size}")
    if not tokens:
        return ()
    if len(tokens) < size:
        return tuple(tokens)
    return tuple(
        _SHINGLE_SEP.join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)
    )


def _feature_hash64(feature: str) -> int:
    """Stable 64-bit hash of one shingle.

    BLAKE2b rather than the builtin `hash()`: `hash()` is salted per interpreter
    by `PYTHONHASHSEED`, so two workers would compute different fingerprints for
    the same text and no cross-process clustering would ever fire. The failure
    would be invisible in a single-process test suite.
    """
    return int.from_bytes(
        hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big"
    )


# --------------------------------------------------------------------------- #
# SimHash
# --------------------------------------------------------------------------- #


def simhash64(text: str, *, shingle_size: int = DEFAULT_SHINGLE_SIZE) -> int:
    """64-bit SimHash of the canonicalized text (`docs/connector-spec.md` §7).

    Charikar's construction, in full: every 3-gram shingle is hashed to 64 bits,
    each bit votes +weight or -weight according to whether it is set, votes are
    summed per position across all features, and the sign of each column becomes
    a bit. Weight is the shingle's frequency, so the fingerprint is dominated by
    what the document is actually about.

    The consequence that matters: a one-word edit disturbs only the three
    shingles containing that word, so it moves a handful of columns and usually
    flips no bits at all -- which is what makes Hamming distance meaningful here
    and meaningless for `content_sha256`.

    A column that sums to exactly zero yields a 0 bit. An arbitrary but *fixed*
    tie-break; anything non-deterministic here would make the fingerprint depend
    on iteration order.

    Returns 0 for text with no tokens at all. That value is a sentinel, not a
    fingerprint: `is_near_duplicate` refuses to compare it, because 0 sits within
    Hamming 3 of every sparse fingerprint and would drag unrelated items into a
    cluster of empty bodies.
    """
    weights = Counter(shingles(tokenize(canonicalize(text)), shingle_size))
    if not weights:
        return 0

    columns = [0] * SIMHASH_BITS
    for feature, weight in weights.items():
        digest = _feature_hash64(feature)
        for bit in range(SIMHASH_BITS):
            if (digest >> bit) & 1:
                columns[bit] += weight
            else:
                columns[bit] -= weight

    fingerprint = 0
    for bit, column in enumerate(columns):
        if column > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two 64-bit fingerprints.

    Range-checked rather than masked. A caller that passes a negative value or
    something wider than 64 bits has a bug -- most likely a fingerprint read back
    from a store that signed it, which Postgres `bigint` does -- and silently
    masking it would answer with a plausible distance for the wrong pair.
    """
    for name, value in (("a", a), ("b", b)):
        if not 0 <= value < (1 << SIMHASH_BITS):
            raise ValueError(
                f"{name}={value!r} is not a {SIMHASH_BITS}-bit unsigned fingerprint; "
                "a signed round-trip through a bigint column is the usual cause"
            )
    return (a ^ b).bit_count()


def is_near_duplicate(
    a: int, b: int, *, threshold: int = SIMHASH_DISTANCE_THRESHOLD
) -> bool:
    """Whether two fingerprints are within `threshold` bits.

    A *predicate*, and the only comparison this module offers. It answers "are
    these the same story" -- never "delete one of them". §4.3 keeps both.

    Either operand being the empty-text sentinel 0 answers `False`. Two
    media-only posts are not the same story merely because neither has text, and
    treating them as such would also pull in any real item whose fingerprint
    happens to have ≤3 bits set.
    """
    if not 0 <= threshold <= SIMHASH_BITS:
        raise ValueError(f"threshold must be within 0..{SIMHASH_BITS}, got {threshold}")
    if a == 0 or b == 0:
        return False
    return hamming(a, b) <= threshold


def simhash_band_keys(
    fingerprint: int, *, bands: int = SIMHASH_LSH_BANDS
) -> tuple[str, ...]:
    """Split a fingerprint into LSH band keys (`docs/connector-spec.md` §7 layer 3).

    The returned strings are usable directly as Redis set keys and as in-memory
    index keys, so the same banding serves a single-process test and a fleet of
    workers sharing one Redis without two implementations that can drift.

    Two fingerprints within Hamming `bands - 1` are guaranteed to share at least
    one key: that many differing bits cannot cover every disjoint band. This is
    why lookup is exact rather than approximate at the default threshold of 3.
    """
    width = _band_width(bands)
    mask = (1 << width) - 1
    hex_digits = width // 4
    return tuple(
        f"os:dedup:sim:{index}:{(fingerprint >> (index * width)) & mask:0{hex_digits}x}"
        for index in range(bands)
    )


def _band_width(bands: int) -> int:
    if bands < 1 or SIMHASH_BITS % bands != 0:
        raise ValueError(
            f"bands must divide {SIMHASH_BITS} evenly, got {bands}; "
            "unequal bands make the pigeonhole guarantee unstateable"
        )
    width = SIMHASH_BITS // bands
    if width % 4 != 0:
        raise ValueError(f"band width {width} is not a whole number of hex digits")
    return width


# --------------------------------------------------------------------------- #
# MinHash -- the long-document alternative
# --------------------------------------------------------------------------- #

_MINHASH_PRIME = (1 << 61) - 1
"""Mersenne prime for the universal hash family `(a*x + b) mod p`."""


@lru_cache(maxsize=8)
def _permutations(count: int) -> tuple[tuple[int, int], ...]:
    """Coefficients for `count` permutations.

    Derived from BLAKE2b of the permutation index rather than drawn from an RNG.
    A seeded `random.Random` would also be reproducible in one process, but it
    ties the signature to CPython's generator; deriving from a digest ties it to
    nothing, which is the property a fingerprint persisted for months needs.
    """
    coefficients = []
    for index in range(count):
        seed = index.to_bytes(4, "big")
        a = (
            int.from_bytes(hashlib.blake2b(b"a" + seed, digest_size=8).digest(), "big")
            % _MINHASH_PRIME
        )
        b = (
            int.from_bytes(hashlib.blake2b(b"b" + seed, digest_size=8).digest(), "big")
            % _MINHASH_PRIME
        )
        # a == 0 would collapse the permutation to the constant b.
        coefficients.append((a or 1, b))
    return tuple(coefficients)


def minhash(
    text: str,
    *,
    permutations: int = MINHASH_PERMUTATIONS,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> tuple[int, ...]:
    """MinHash signature of the canonicalized text.

    Estimates Jaccard similarity of the shingle *sets*, which is the right
    measure once a document is long enough that a 64-bit SimHash saturates --
    past roughly 5k tokens every long document has a middling distance to every
    other long document and the threshold stops separating anything
    (`docs/connector-spec.md` §7).

    Set semantics, so unlike SimHash this is insensitive to repetition: a phrase
    used once and a phrase used forty times contribute equally.

    Returns an empty signature for text with no tokens. `jaccard` reads that as
    "not comparable" rather than as "identical to every other empty document".
    """
    features = {
        _feature_hash64(shingle) % _MINHASH_PRIME
        for shingle in shingles(tokenize(canonicalize(text)), shingle_size)
    }
    if not features:
        return ()
    return tuple(
        min((a * feature + b) % _MINHASH_PRIME for feature in features)
        for a, b in _permutations(permutations)
    )


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    """Estimated Jaccard similarity from two MinHash signatures.

    The estimate is the fraction of positions that agree; each position agrees
    with probability exactly the true Jaccard similarity, so the mean is
    unbiased.

    Signatures of different lengths are a caller error, not a similarity of zero:
    comparing a 128-permutation signature against a 64-permutation one produces a
    number that looks like an answer and means nothing.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(
            f"signature lengths differ ({len(a)} vs {len(b)}); they are not comparable"
        )
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def minhash_band_keys(
    signature: Sequence[int], *, bands: int = MINHASH_LSH_BANDS
) -> tuple[str, ...]:
    """Split a MinHash signature into LSH band keys.

    Unlike `simhash_band_keys` this carries no pigeonhole guarantee -- it is a
    probabilistic candidate generator, and every candidate it returns must be
    confirmed with `jaccard` before it is treated as a near-duplicate.
    """
    if not signature:
        return ()
    if bands < 1 or len(signature) % bands != 0:
        raise ValueError(
            f"bands must divide the signature length {len(signature)} evenly, got {bands}"
        )
    rows = len(signature) // bands
    keys = []
    for index in range(bands):
        chunk = signature[index * rows : (index + 1) * rows]
        digest = hashlib.blake2b(
            b"".join(value.to_bytes(8, "big") for value in chunk), digest_size=8
        ).hexdigest()
        keys.append(f"os:dedup:mh:{index}:{digest}")
    return tuple(keys)


# --------------------------------------------------------------------------- #
# Candidate index
# --------------------------------------------------------------------------- #


class BandedIndex:
    """In-memory LSH index: band key -> member ids.

    The whole point of the class is the cost model. `candidates()` probes a fixed
    number of buckets regardless of how many members are indexed, which is what
    turns near-duplicate detection from O(n²) into O(n · bands). A linear scan
    would be simpler and would stop working somewhere around the first ten
    thousand signals, in production, silently, as a latency graph rather than an
    error.

    Deliberately not a `DedupStore`: this holds one run's worth of candidates,
    not a shared seen-set. The cross-process equivalent is Redis sets under the
    same keys `simhash_band_keys` returns, which is why those keys are already
    namespaced.
    """

    __slots__ = ("_buckets",)

    def __init__(self) -> None:
        self._buckets: dict[str, set[str]] = {}

    def add(self, member_id: str, band_keys: Iterable[str]) -> None:
        for key in band_keys:
            self._buckets.setdefault(key, set()).add(member_id)

    def candidates(
        self, band_keys: Iterable[str], *, exclude: str | None = None
    ) -> set[str]:
        """Member ids sharing at least one band. A superset of the true matches.

        Callers must confirm each candidate with `is_near_duplicate` (SimHash) or
        `jaccard` (MinHash): sharing a 16-bit band is evidence, not proof.
        """
        found: set[str] = set()
        for key in band_keys:
            found |= self._buckets.get(key, frozenset())
        if exclude is not None:
            found.discard(exclude)
        return found

    def __len__(self) -> int:
        """Number of occupied buckets, not members -- a member occupies several."""
        return len(self._buckets)


# --------------------------------------------------------------------------- #
# Clustering: assignment and canonical election
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ClusterMember:
    """One Signal's participation in near-duplicate clustering.

    Deliberately not a `Signal`. Clustering needs four scalars and would
    otherwise pull the entire enrichment model -- and every `Signal` in a
    candidate window -- into memory to compare fingerprints.
    """

    signal_id: str
    timestamp: datetime
    """Event time. First key of the canonical election, so it must be the
    source's publication time and never ingestion time: ordering by fetch time
    would elect whichever platform we happened to poll first."""

    fingerprint: int
    """64-bit SimHash of the cleaned text. 0 means "no text to fingerprint"."""

    confidence: float = 0.0
    platform: str | None = None
    """Used only to count independent sources for the corroboration term. Six
    copies from one platform are one platform's opinion (§4.3)."""

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValueError("signal_id must be non-empty")
        if self.timestamp.tzinfo is None:
            # Naive datetimes compare fine among themselves and raise TypeError
            # the moment one aware timestamp joins the cluster -- a crash in the
            # sort, far from the connector that produced the bad value.
            raise ValueError(
                f"timestamp for {self.signal_id!r} is naive; Signal.timestamp is "
                "timezone-aware UTC and the election orders by it"
            )
        if not 0 <= self.fingerprint < (1 << SIMHASH_BITS):
            raise ValueError(
                f"fingerprint for {self.signal_id!r} is not a {SIMHASH_BITS}-bit "
                "unsigned value"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence for {self.signal_id!r} must be within 0.0..1.0, "
                f"got {self.confidence!r}"
            )


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """What one member's `lineage` should record after clustering.

    `status` is absent on purpose. §4.3 says a non-canonical member becomes
    `status = duplicate`, but the canonical one keeps whatever the pipeline gave
    it -- `enriched`, `partial`, or still `raw` if enrichment has not run.
    Clustering cannot know which, so it reports `duplicate_of` and lets the
    caller derive the status it is entitled to set.
    """

    signal_id: str
    dedup_cluster_id: str
    duplicate_of: str | None
    """`None` for the canonical member. `models/lineage.py` rejects a
    `duplicate_of` without a `dedup_cluster_id`, which is why both travel
    together rather than being applied by two separate call sites."""

    @property
    def is_canonical(self) -> bool:
        return self.duplicate_of is None


def cluster_id_for(canonical_id: str) -> str:
    """Derive a cluster id from its canonical member.

    Deriving rather than generating keeps `scripts/reindex.py` able to rebuild
    every derived store from PostgreSQL and R2 with no coordination: the same
    cluster reconstructs to the same id on any machine, at any time.

    Keyed on the canonical member specifically, out of three bad options. Keying
    on the whole membership set re-ids the cluster every time it gains a member;
    keying on the smallest id re-ids it on an unrelated event. Keying on the
    canonical means the id changes exactly when §4.3 already requires a
    cluster-wide rewrite -- when a member with an earlier timestamp arrives and
    election is re-run -- so the id and the election never disagree.

    **The failure mode to know about:** that rewrite must reach every member. A
    late-arriving earlier copy changes `dedup_cluster_id` on all six Signals, not
    just the two that changed role. `assign_clusters` returns the complete
    assignment set for exactly that reason -- there is no per-member entry point
    that would let a caller update one and forget the rest.

    Hashed rather than embedding the id verbatim so a cluster id can never be
    mistaken for -- or joined against -- a Signal id.
    """
    if not canonical_id:
        raise ValueError("canonical_id must be non-empty")
    digest = hashlib.sha256(canonical_id.encode("utf-8")).hexdigest()
    return f"dc_{digest[:12]}"


def _election_key(member: ClusterMember) -> tuple[datetime, float, str]:
    """Earliest timestamp, then highest confidence, then smallest id (§4.3).

    Confidence is negated so a single `min()` expresses all three keys: ascending
    time, descending confidence, ascending id. The final key is what makes the
    result independent of input order -- without it two members published in the
    same second with equal confidence would elect whichever the iterator reached
    first, and the cluster's canonical member would change between two runs over
    identical data.
    """
    return (member.timestamp, -member.confidence, member.signal_id)


def elect_canonical(members: Sequence[ClusterMember]) -> ClusterMember:
    """Elect the member that gets indexed, per `docs/signal-model.md` §4.3.

    Earliest `timestamp` first because the point of the cluster is to name the
    original: whoever published the story before it was syndicated. Highest
    `confidence` breaks a same-instant tie toward the copy an agent should trust.
    The id tiebreak is not cosmetic -- it is what makes this a function of the
    set rather than of the list.
    """
    if not members:
        raise ValueError("cannot elect a canonical member from an empty cluster")
    return min(members, key=_election_key)


@dataclass(frozen=True, slots=True)
class Cluster:
    """A set of near-duplicate Signals and the one elected to represent them.

    Every member is retained. Only `canonical_id` is embedded into Qdrant and
    indexed into OpenSearch, so retrieval returns one hit for a press release
    that appeared in six places, while all six keep their own `id`, `platform`,
    `engagement` and `timestamp`, all six contribute `MENTIONS` edges, and all
    six count toward trend volume (§4.3).
    """

    cluster_id: str
    canonical_id: str
    members: tuple[ClusterMember, ...]
    """Ordered by the election key, so `members[0]` is always the canonical one
    and two runs over the same set produce byte-identical output."""

    @property
    def is_singleton(self) -> bool:
        """A Signal with no near-duplicate. Still a cluster, so callers have one
        code path rather than a nullable one."""
        return len(self.members) == 1

    def duplicates(self) -> tuple[ClusterMember, ...]:
        """Every member except the canonical. Not "the members to delete"."""
        return tuple(m for m in self.members if m.signal_id != self.canonical_id)

    def distinct_platforms(self) -> tuple[str, ...]:
        """Platforms represented, sorted, ignoring members that declare none.

        The input to the `corroboration` component of `confidence` (§3.5). Counted
        per platform, not per member, because §4.3 requires that one platform
        cannot inflate a cluster by itself -- three crossposts inside one
        subreddit are one source corroborating itself.
        """
        return tuple(sorted({m.platform for m in self.members if m.platform}))

    def assignments(self) -> tuple[ClusterAssignment, ...]:
        """Lineage updates for every member, canonical included.

        Returned for the whole cluster rather than per member so a caller cannot
        write `duplicate_of` on five Signals and leave the sixth pointing at a
        cluster id that no longer exists.
        """
        return tuple(
            ClusterAssignment(
                signal_id=member.signal_id,
                dedup_cluster_id=self.cluster_id,
                duplicate_of=(
                    None if member.signal_id == self.canonical_id else self.canonical_id
                ),
            )
            for member in self.members
        )


def assign_clusters(
    members: Sequence[ClusterMember],
    *,
    threshold: int = SIMHASH_DISTANCE_THRESHOLD,
    bands: int = SIMHASH_LSH_BANDS,
) -> tuple[Cluster, ...]:
    """Partition members into near-duplicate clusters. **Nothing is discarded.**

    A partition, in the strict sense: every input member appears in exactly one
    returned cluster, and a member with no near-duplicate comes back as a cluster
    of one. There is no filtered variant of this function and no `drop_*`
    companion -- §4.3 makes six copies of a press release the evidence, not the
    noise, and a function that returned "the survivors" would be used as one
    within a week.

    Clusters are transitive by construction: A near B and B near C puts all three
    together even when A and C are 5 bits apart. That is a real modelling choice
    with a real failure mode -- a long chain of small edits can walk a cluster
    away from where it started. It is the right default here because the
    alternative, requiring mutual similarity, splits genuine syndication chains
    where each outlet lightly rewrites the previous one. Chain drift is bounded
    in practice by the 7-day layer-3 TTL; runaway chains would show up as
    implausibly large clusters, which is worth alerting on rather than
    preventing here.

    Raises `ValueError` when `threshold >= bands`: the pigeonhole guarantee that
    makes banded lookup exact needs one band the differing bits cannot reach, and
    without it this function would silently return more clusters than there
    really are. Widening the threshold means widening the banding to match.
    """
    if threshold >= bands:
        raise ValueError(
            f"threshold {threshold} needs more than {bands} bands: {threshold} "
            f"differing bits can touch every one of {bands} disjoint bands, so "
            "banded lookup would miss real pairs. Use bands > threshold."
        )
    _band_width(bands)  # validates the banding before anything is indexed

    by_id: dict[str, ClusterMember] = {}
    for member in members:
        if member.signal_id in by_id:
            # Merging them would silently halve the cluster; keeping one would
            # silently discard a Signal. Both are worse than telling the caller.
            raise ValueError(
                f"duplicate signal_id {member.signal_id!r} in the input; each "
                "Signal may appear once"
            )
        by_id[member.signal_id] = member

    parent: dict[str, str] = {member_id: member_id for member_id in by_id}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # Attach deterministically (smaller id wins) so the union-find shape
            # does not depend on input order. Roots are internal, but a stable
            # structure makes a failing test reproducible.
            if right_root < left_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    index = BandedIndex()
    for member in by_id.values():
        if member.fingerprint == 0:
            # The empty-text sentinel shares every band with every other empty
            # fingerprint. Indexing it would collect every media-only post into
            # one cluster on the strength of having no text in common.
            continue
        keys = simhash_band_keys(member.fingerprint, bands=bands)
        for candidate_id in index.candidates(keys, exclude=member.signal_id):
            if is_near_duplicate(
                member.fingerprint, by_id[candidate_id].fingerprint, threshold=threshold
            ):
                union(member.signal_id, candidate_id)
        index.add(member.signal_id, keys)

    grouped: dict[str, list[ClusterMember]] = {}
    for member_id, member in by_id.items():
        grouped.setdefault(find(member_id), []).append(member)

    clusters = []
    for group in grouped.values():
        ordered = tuple(sorted(group, key=_election_key))
        canonical = ordered[0]
        clusters.append(
            Cluster(
                cluster_id=cluster_id_for(canonical.signal_id),
                canonical_id=canonical.signal_id,
                members=ordered,
            )
        )
    # Ordered by their canonical members so the output is a function of the input
    # set, not of dict insertion order.
    return tuple(sorted(clusters, key=lambda c: _election_key(c.members[0])))
