"""Candidate blocking: turn an O(n^2) comparison into an O(n * block) one.

Entity resolution compares mentions against canonical entities. Doing that
pairwise is quadratic, and the graph is expected to hold millions of nodes, so
the comparison set has to be cut down *before* the matcher ever runs. Blocking is
the cut: each record emits a handful of cheap keys, and only records that share
at least one key are ever scored (`docs/knowledge-graph.md` §6).

The asymmetry that shapes every decision in this module:

    A blocking key that fires too often costs CPU, and CPU is measurable.
    A blocking key that fails to fire is *invisible*.

If "Föö-Bär, Ltd." and "foo bar limited" never land in a common block they are
simply never compared, the matcher never gets a chance to say they are the same
company, and the graph carries two nodes for one company forever. Nothing raises,
no counter moves, and the only symptom is that a competitor query returns half
the evidence it should. Recall therefore wins every tradeoff here and precision is
delegated entirely to `graph/resolution/matcher.py`, which is allowed -- expected,
even -- to reject most of what blocking proposes.

Five key families, unioned (a pair is a candidate if it shares *any* key):

`EXACT`
    The full normalized name. Catches the common case at zero cost.
`PREFIX`
    First `NAME_PREFIX_LENGTH` characters of the normalized name with spaces
    removed. Catches suffix and spacing variation ("acmecloud" / "acme cloud").
`TOKEN_SET`
    Tokens sorted and rejoined. Catches word order ("Acme Cloud" / "Cloud Acme").
`ALIAS`
    One key per known surface. This is the alias-table lookup: it is the only key
    family that can bridge two names with no character overlap at all
    ("Big Blue" / "IBM"), provided somebody recorded the alias.
`PHONETIC`
    A metaphone-ish consonant skeleton. Catches transliteration and spelling
    drift ("Kolour" / "Colour", "Smith" / "Smyth") that no prefix or token key
    survives.
`EMBEDDING_LSH`
    Sign bits of the record embedding against fixed random hyperplanes, emitted
    in bands so that near vectors collide in at least one band. This is the
    stand-in for the "top-20 ANN neighbours" key in `docs/knowledge-graph.md` §6:
    it needs no vector store, no network and no index build, and it is
    deterministic because the hyperplanes are drawn from a seeded PRNG.
`IDENTIFIER`
    `domain`, `ticker`, platform handle. High precision rather than high recall;
    the matcher treats a hit here as a hard rule.

The one place recall is deliberately traded away is `max_block_size`. A block
holding ten thousand records reintroduces exactly the quadratic blow-up blocking
exists to prevent, and the offenders are always the low-precision families -- a
four-character prefix like "inte", or a phonetic code shared by every company
with "tech" in its name. Those blocks are skipped for pair generation, but the
skip is **recorded** in `BlockingIndex.oversized_blocks` rather than silently
applied, because an unrecorded recall loss is the exact failure this module is
built to avoid. High-precision families (`EXACT`, `ALIAS`, `IDENTIFIER`) are never
capped: a block of ten thousand records sharing an exact normalized name is not
noise, it is ten thousand records that genuinely need comparing.

This module holds no I/O and no configuration. It is pure, synchronous and
importable from a worker, a test or a script without a datastore running.
"""

from __future__ import annotations

import enum
import math
import random
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Final, NamedTuple, Self

from models.entity import Entity
from models.enums import EntityType

__all__ = [
    "DEFAULT_MAX_BLOCK_SIZE",
    "GENERIC_TOKENS",
    "IDENTIFIER_PROPERTIES",
    "LEGAL_SUFFIX_TOKENS",
    "LOW_PRECISION_KINDS",
    "NAME_PREFIX_LENGTH",
    "BlockKind",
    "BlockingIndex",
    "BlockingKey",
    "BlockingStats",
    "ResolutionRecord",
    "blocking_keys",
    "cosine_similarity",
    "name_tokens",
    "normalize_name",
    "pair_key",
    "phonetic_code",
]


# --------------------------------------------------------------------------- #
# Normalization vocabulary
# --------------------------------------------------------------------------- #


LEGAL_SUFFIX_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "llc",
        "lc",
        "llp",
        "lp",
        "ltd",
        "limited",
        "plc",
        "gmbh",
        "mbh",
        "ag",
        "sa",
        "sas",
        "sarl",
        "srl",
        "spa",
        "bv",
        "nv",
        "ab",
        "as",
        "oy",
        "oyj",
        "kk",
        "pty",
        "pte",
        "kft",
        "zrt",
        "doo",
        "dba",
    }
)
"""Corporate form tokens stripped before any key is computed.

"Acme Corp." and "Acme Corporation" are the same company written by two
extractors, and if the suffix survives normalization they land in different
`EXACT` and `TOKEN_SET` blocks. Stripping is done token-wise rather than by
string suffix so "Corp" is removed from "Acme Corp Holdings" as well.

Deliberately *not* in this set: `holdings`, `group`, `labs`, `ventures`,
`partners`. Those are part of a real name often enough ("Berkshire Hathaway"
versus "Berkshire Hathaway Energy") that stripping them merges distinct legal
entities, and a wrong merge is far more expensive to undo than a missed one is to
find.
"""

LEADING_ARTICLES: Final[frozenset[str]] = frozenset({"the"})
"""Leading articles dropped from the front of a name only.

"The Guardian" and "Guardian" are the same publication. Dropping `the` anywhere
would mangle "Ask The Doctor", so this applies to position zero exclusively.
"""

GENERIC_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "ai",
        "analytics",
        "app",
        "apps",
        "cloud",
        "data",
        "digital",
        "global",
        "group",
        "holdings",
        "international",
        "labs",
        "media",
        "network",
        "networks",
        "online",
        "platform",
        "services",
        "software",
        "solutions",
        "systems",
        "tech",
        "technologies",
        "technology",
        "ventures",
    }
)
"""Tokens too common to be worth a *per-token* phonetic key.

These stay in the name -- they are part of it, and `TOKEN_SET` still uses them.
They are excluded only from the per-token phonetic family, where "cloud" alone
would otherwise put every cloud vendor on the platform into one block and
guarantee that block is oversized and therefore skipped. Excluding them here is
what keeps the phonetic family usable at all.
"""

IDENTIFIER_PROPERTIES: Final[tuple[str, ...]] = ("domain", "ticker", "isin", "cik")
"""`Entity.properties` keys treated as strong identifiers.

Scalar, externally-assigned and near-unique. Platform handles are carried
separately (`handles`, a platform -> handle map) because they are unique only
within one platform: `@acme` on Reddit and `@acme` on X are not evidence of
anything on their own.
"""

NAME_PREFIX_LENGTH: Final = 4
"""Characters of the space-stripped normalized name used for the prefix key.

Four is the length `docs/knowledge-graph.md` §6 specifies. Spaces are removed
first, which the doc does not say: with spaces, "AC Milan" blocks on "ac m" and
"ACMilan" on "acmi", and the pair -- differing by exactly the whitespace this key
is supposed to be robust to -- never meets.
"""

MIN_PHONETIC_TOKEN_LENGTH: Final = 4
"""Shortest token that earns its own phonetic key.

Short tokens collapse to two- or three-character codes that collide with
everything, producing blocks that are pure cost. The whole-name phonetic key is
still emitted for short names, so a one-token name like "Nike" is not left
without phonetic coverage.
"""

DEFAULT_MAX_BLOCK_SIZE: Final = 200
"""Members above which a *low-precision* block stops generating pairs.

A block of size k costs k*(k-1)/2 comparisons. At 200 that is 19,900 -- already
the dominant cost of a resolution pass -- and blocks that large from a prefix or
phonetic key are, empirically, noise rather than signal. Set to `None` to disable
the cap when correctness matters more than latency (an offline backfill, a test).
Every skip is recorded; see `BlockingIndex.oversized_blocks`.
"""

LSH_BANDS: Final = 4
"""Independent bands of sign bits emitted per embedding.

Recall comes from the union: a pair blocks if it agrees on *any* band, so one
unlucky bit flip costs a band rather than the pair.
"""

LSH_BITS_PER_BAND: Final = 8
"""Bits per band -- one in 256 vectors shares a band by chance.

Wider bands mean fewer false candidates and lower recall; this is the knob to
turn if the LSH family dominates `BlockingStats.candidate_pairs`.
"""

LSH_SEED: Final = 0x0060_5E_45
"""Seed for the LSH hyperplanes.

Fixed and hard-coded rather than configurable: two workers resolving the same
corpus with different hyperplanes would build different blocks and therefore
different graphs, which is precisely the non-determinism
`graph/resolution/entity_resolution.py` exists to rule out. Changing this value
re-blocks the entire corpus, so it is a schema change, not a tuning knob.
"""


# --------------------------------------------------------------------------- #
# Name normalization
# --------------------------------------------------------------------------- #


_APOSTROPHE_TABLE: Final = str.maketrans(dict.fromkeys("'’‘ʼ`´", ""))  # noqa: RUF001
"""Apostrophe forms deleted outright by `normalize_name`, rather than spaced.

Scraped and syndicated copy is wildly inconsistent about which code point it
uses for the same possessive, so all six are folded to nothing.
"""


@lru_cache(maxsize=65_536)
def normalize_name(raw: str) -> str:
    """Case-fold, unaccent, de-punctuate and strip corporate form from a name.

    The output is the input to every string-shaped blocking key, so two surfaces
    that should block together must normalize identically. The steps, and what
    breaks if one is missing:

    1. **Unicode compatibility decomposition, combining marks dropped.** Without
       it "Föö" and "Foo" differ in the first character and share no key at all.
       NFKD also folds ligatures and full-width forms, which appear in scraped
       CJK-locale pages.
    2. **`&` expanded to `and`.** "AT&T" versus "AT and T" is common enough in
       news copy that leaving it to punctuation stripping ("at t") loses the
       token boundary entirely.
    3. **Apostrophes deleted, not spaced.** They are the one punctuation mark
       that sits *inside* a token: spacing "Zoë's Kitchen" yields the three
       tokens "zoe s kitchen", which shares no token key with "Zoes Kitchen"
       and adds a junk one-character token to every possessive name in the
       corpus. Covers the typographic variants too -- scraped copy uses U+2019
       far more often than U+0027.
    4. **Everything else non-alphanumeric becomes a space.** Punctuation is the
       single most common difference between two writings of the same name, and
       it carries no information a matcher can use.
    5. **Corporate form tokens removed.** See `LEGAL_SUFFIX_TOKENS`.
    6. **Leading article removed.** See `LEADING_ARTICLES`.

    Steps 5 and 6 are skipped when they would empty the string. An entity really
    named "The Company" must not normalize to `""`, because the empty string is a
    perfectly good dictionary key and every such entity would land in one block
    together and be compared against each other forever.
    """
    decomposed = unicodedata.normalize("NFKD", raw)
    stripped_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    expanded = stripped_marks.replace("&", " and ")
    folded = expanded.casefold()
    without_apostrophes = folded.translate(_APOSTROPHE_TABLE)
    cleaned = "".join(ch if ch.isalnum() else " " for ch in without_apostrophes)
    tokens = cleaned.split()
    if not tokens:
        return ""

    if len(tokens) > 1 and tokens[0] in LEADING_ARTICLES:
        tokens = tokens[1:]

    kept = [token for token in tokens if token not in LEGAL_SUFFIX_TOKENS]
    if kept:
        tokens = kept
    # else: the name consisted entirely of corporate-form tokens. Keep them --
    # a degenerate name is still better than the empty key.

    return " ".join(tokens)


@lru_cache(maxsize=65_536)
def name_tokens(raw: str) -> tuple[str, ...]:
    """Normalized tokens of `raw`, in original order.

    Order is preserved rather than sorted because the matcher's token alignment
    needs the original sequence; the sorted form is a blocking key, computed from
    this.
    """
    normalized = normalize_name(raw)
    return tuple(normalized.split()) if normalized else ()


# --------------------------------------------------------------------------- #
# Phonetic encoding
# --------------------------------------------------------------------------- #


_VOWELS: Final = frozenset("AEIOU")


@lru_cache(maxsize=65_536)
def phonetic_code(token: str) -> str:
    """A metaphone-ish consonant skeleton for one token. `""` when inapplicable.

    This is a reduced Metaphone: the transformations that matter for company and
    product names, without the full rule set. It is deliberately *not* an exact
    reimplementation of Lawrence Philips' algorithm, because the value here is
    only that two spellings of one sound produce one string -- consistency, not
    fidelity to a published table. Being self-contained also means blocking has
    no dependency that could change its output under us on a version bump, which
    would silently re-block the whole corpus.

    Returns `""` for tokens with no Latin letters. That guard is load-bearing:
    without it every CJK, Cyrillic or Arabic name encodes to the empty string,
    they all share the key `phon:`, and the single largest block in the index is
    "every non-Latin entity" -- which is then dropped as oversized, removing
    phonetic coverage from exactly the names that needed the most help.
    """
    word = "".join(ch for ch in token.upper() if "A" <= ch <= "Z")
    if not word:
        return ""

    # Initial-cluster exceptions: the leading letter is silent in all of these,
    # so "Knight"/"Night" and "Wright"/"Right" must lose it before the main pass.
    if word[:2] in {"AE", "GN", "KN", "PN", "WR"}:
        word = word[1:]
    elif word[0] == "X":
        word = "S" + word[1:]
    elif word[:2] == "WH":
        word = "W" + word[1:]
    if not word:
        return ""

    out: list[str] = []
    length = len(word)
    for index, ch in enumerate(word):
        prev = word[index - 1] if index > 0 else ""
        nxt = word[index + 1] if index + 1 < length else ""
        nxt2 = word[index + 2] if index + 2 < length else ""

        # Doubled letters sound once ("Bennett" -> BNT). CC is exempt because
        # "accept" genuinely carries two sounds.
        if ch == prev and ch != "C":
            continue

        if ch in _VOWELS:
            # Vowels survive only in first position; internal vowels are the
            # least stable part of any transliteration.
            if index == 0:
                out.append(ch)
            continue

        match ch:
            case "B":
                if not (index == length - 1 and prev == "M"):
                    out.append("B")
            case "C":
                if nxt == "H" or (nxt == "I" and nxt2 == "A"):
                    out.append("X")  # "chip", "special"
                elif nxt in {"I", "E", "Y"}:
                    out.append("S")
                else:
                    out.append("K")
            case "D":
                if nxt == "G" and nxt2 in {"E", "Y", "I"}:
                    out.append("J")
                else:
                    out.append("T")
            case "G":
                if nxt == "H" and not (nxt2 and nxt2 in _VOWELS):
                    continue  # silent, as in "night"
                if nxt == "N":
                    continue  # silent, as in "sign"
                out.append("J" if nxt in {"I", "E", "Y"} else "K")
            case "H":
                # Silent after a vowel unless another vowel follows.
                if prev in _VOWELS and nxt not in _VOWELS:
                    continue
                if prev in {"C", "S", "P", "T", "G"}:
                    continue  # already consumed by the digraph rules above
                out.append("H")
            case "K":
                if prev != "C":
                    out.append("K")
            case "P":
                out.append("F" if nxt == "H" else "P")
            case "Q":
                out.append("K")
            case "S":
                if nxt == "H" or (nxt == "I" and nxt2 in {"O", "A"}):
                    out.append("X")
                else:
                    out.append("S")
            case "T":
                if nxt == "I" and nxt2 in {"O", "A"}:
                    out.append("X")
                elif nxt == "H":
                    out.append("0")  # theta; a distinct sound from plain T
                elif not (nxt == "C" and nxt2 == "H"):
                    out.append("T")
            case "V":
                out.append("F")
            case "W" | "Y":
                if nxt in _VOWELS:
                    out.append(ch)
            case "X":
                out.append("KS")
            case "Z":
                out.append("S")
            case _:
                out.append(ch)

    return "".join(out)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ResolutionRecord:
    """One thing to be resolved: an incoming mention or an existing graph node.

    Deliberately a single type for both sides. Resolution compares mentions to
    entities, mentions to mentions (within one batch), and entities to entities
    (a backfill re-resolution), and a design with separate `Mention` and `Entity`
    record types needs three code paths, three sets of blocking keys and three
    chances for them to disagree about what a key means.

    Frozen because the resolver holds records in dictionaries keyed by id and
    reuses them across clustering passes; a record mutated mid-pass would make
    the run non-reproducible in a way no test would catch. Merging produces a
    *new* record (`dataclasses.replace`), never an edit.
    """

    id: str
    type: EntityType
    name: str
    aliases: tuple[str, ...] = ()
    identifiers: Mapping[str, str] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None
    context: frozenset[str] = frozenset()
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    mention_count: int = 0
    merged_from: tuple[str, ...] = ()

    # -- derived views -----------------------------------------------------

    @property
    def normalized(self) -> str:
        """The normalized canonical name. Cached in `normalize_name`, not here."""
        return normalize_name(self.name)

    @property
    def tokens(self) -> tuple[str, ...]:
        """Normalized tokens of the canonical name, in order."""
        return name_tokens(self.name)

    def surfaces(self) -> tuple[str, ...]:
        """Every string that should find this record, canonical name first.

        Deduplicated on the *normalized* form and returned in a stable order, so
        two records carrying the same aliases in different orders emit the same
        alias keys. Blocking output that depends on list order is a determinism
        bug that only shows up under concurrency.
        """
        seen: dict[str, str] = {}
        for surface in (self.name, *self.aliases):
            normalized = normalize_name(surface)
            if normalized and normalized not in seen:
                seen[normalized] = surface
        return tuple(seen.values())

    def normalized_surfaces(self) -> frozenset[str]:
        """The normalized form of every surface. The matcher's alias feature."""
        return frozenset(normalize_name(s) for s in (self.name, *self.aliases) if s.strip())

    # -- construction ------------------------------------------------------

    @classmethod
    def from_entity(cls, entity: Entity) -> Self:
        """Project a graph `Entity` onto a resolution record.

        `Entity.properties` is an open map (`models/entity.py`), so the fields
        resolution needs but the model does not declare -- embedding, source
        count, co-mention context, strong identifiers -- are read from it
        defensively. A property of the wrong shape is ignored rather than
        raising: these values arrive from the graph, where a different writer may
        have stored anything, and a resolution pass that dies on one malformed
        node fails an entire batch for one bad row.
        """
        props = entity.properties
        identifiers: dict[str, str] = {}
        for key in IDENTIFIER_PROPERTIES:
            value = props.get(key)
            if isinstance(value, str) and value.strip():
                identifiers[key] = value.strip().casefold()
        handles = props.get("handles")
        if isinstance(handles, Mapping):
            for platform, handle in handles.items():
                if isinstance(platform, str) and isinstance(handle, str) and handle.strip():
                    identifiers[f"handle:{platform.casefold()}"] = handle.strip().casefold()

        return cls(
            id=entity.id,
            type=entity.type,
            name=entity.canonical_name,
            aliases=tuple(entity.aliases),
            identifiers=identifiers,
            embedding=_coerce_vector(props.get("embedding")),
            context=frozenset(_coerce_str_sequence(props.get("context"))),
            first_seen=entity.first_seen,
            last_seen=entity.last_seen,
            mention_count=_coerce_count(props.get("source_count")),
            merged_from=tuple(entity.merged_from),
        )

    def to_entity(self) -> Entity:
        """Project back onto the graph `Entity` model.

        The inverse of `from_entity` for every field either model can hold.
        `description` and `resolution_confidence` are not round-tripped here --
        resolution does not own them (the first is LLM-generated, the second is
        set by the caller from the cluster's weakest link) and inventing values
        would overwrite better ones on write.
        """
        properties: dict[str, object] = {"source_count": self.mention_count}
        handles: dict[str, str] = {}
        for key, value in sorted(self.identifiers.items()):
            if key.startswith("handle:"):
                handles[key.removeprefix("handle:")] = value
            else:
                properties[key] = value
        if handles:
            properties["handles"] = handles
        if self.embedding is not None:
            properties["embedding"] = list(self.embedding)
        if self.context:
            properties["context"] = sorted(self.context)

        return Entity(
            id=self.id,
            type=self.type,
            canonical_name=self.name,
            aliases=list(self.aliases),
            merged_from=list(self.merged_from),
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            properties=properties,
        )


def _coerce_vector(value: object) -> tuple[float, ...] | None:
    """Read an embedding from an open property map, or `None`."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    return vector or None


def _coerce_str_sequence(value: object) -> tuple[str, ...]:
    """Read a list of ids from an open property map. Never raises."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _coerce_count(value: object) -> int:
    """Read a non-negative counter from an open property map. Never raises."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


class BlockKind(enum.StrEnum):
    """Key families, in descending precision.

    The kind travels with the key because two consumers need it: the block-size
    cap applies only to the low-precision families, and the matcher wants to know
    whether a pair was proposed by an identifier hit (a hard rule) or by a
    phonetic collision (barely evidence at all).
    """

    IDENTIFIER = "ident"
    EXACT = "exact"
    ALIAS = "alias"
    TOKEN_SET = "tokens"
    PREFIX = "prefix"
    PHONETIC = "phon"
    EMBEDDING_LSH = "lsh"


LOW_PRECISION_KINDS: Final[frozenset[BlockKind]] = frozenset(
    {BlockKind.PREFIX, BlockKind.PHONETIC, BlockKind.EMBEDDING_LSH}
)
"""Families subject to `max_block_size`.

These three are generative: they fire on a substring, a sound or a bucket of
vector space, so one popular value can pull in an unbounded number of records.
The other four fire on a value somebody actually wrote down, which bounds them
naturally.
"""


class BlockingKey(NamedTuple):
    """A (family, value) pair. Hashable, comparable, and cheap to build."""

    kind: BlockKind
    value: str

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.value}"


def blocking_keys(record: ResolutionRecord) -> tuple[BlockingKey, ...]:
    """Every key `record` belongs to, deduplicated and deterministically ordered.

    Type is *not* part of any key even though the matcher refuses to merge across
    `EntityType`. Two reasons: mentions frequently arrive typed `UNKNOWN`
    (`services/signal_engine/entities.py` degrades rather than drops), and a
    type-partitioned index would put those in a partition of their own where they
    can never meet the typed entity they refer to. Filtering on type after
    scoring costs one comparison; filtering before it costs recall permanently.
    """
    keys: list[BlockingKey] = []

    for key, value in sorted(record.identifiers.items()):
        if value:
            keys.append(BlockingKey(BlockKind.IDENTIFIER, f"{key}={value}"))

    normalized = record.normalized
    if normalized:
        keys.append(BlockingKey(BlockKind.EXACT, normalized))

        collapsed = normalized.replace(" ", "")
        if len(collapsed) >= 2:
            keys.append(BlockingKey(BlockKind.PREFIX, collapsed[:NAME_PREFIX_LENGTH]))

        tokens = record.tokens
        if len(tokens) > 1:
            keys.append(BlockingKey(BlockKind.TOKEN_SET, " ".join(sorted(tokens))))

        whole = "".join(phonetic_code(token) for token in tokens)
        if whole:
            keys.append(BlockingKey(BlockKind.PHONETIC, whole))
        for token in sorted(set(tokens)):
            if len(token) < MIN_PHONETIC_TOKEN_LENGTH or token in GENERIC_TOKENS:
                continue
            code = phonetic_code(token)
            if code and code != whole:
                keys.append(BlockingKey(BlockKind.PHONETIC, code))

    for surface in record.surfaces():
        normalized_surface = normalize_name(surface)
        if normalized_surface:
            keys.append(BlockingKey(BlockKind.ALIAS, normalized_surface))

    keys.extend(_lsh_keys(record.embedding))

    return tuple(dict.fromkeys(keys))


@lru_cache(maxsize=8)
def _hyperplanes(dimensions: int) -> tuple[tuple[float, ...], ...]:
    """`LSH_BANDS * LSH_BITS_PER_BAND` fixed random hyperplanes for `dimensions`.

    Drawn from a seeded `random.Random` rather than `numpy` so the values depend
    on nothing but the seed and CPython's Mersenne Twister -- reproducible across
    machines, processes and library versions. Cached per dimensionality because
    generating 32 * 1536 gaussians on every record would dominate the cost of
    blocking by two orders of magnitude.
    """
    # Not a cryptographic use: reproducibility is the entire requirement.
    rng = random.Random(LSH_SEED)
    return tuple(
        tuple(rng.gauss(0.0, 1.0) for _ in range(dimensions))
        for _ in range(LSH_BANDS * LSH_BITS_PER_BAND)
    )


def _lsh_keys(embedding: tuple[float, ...] | None) -> tuple[BlockingKey, ...]:
    """Random-hyperplane LSH keys, one per band.

    Signed random projection has the property that the probability two vectors
    agree on one bit is `1 - theta/pi`, so a run of `LSH_BITS_PER_BAND` bits
    agrees rarely by chance but almost always for near-identical vectors.
    Emitting `LSH_BANDS` independent bands and unioning them converts that into
    usable recall: a pair blocks if it agrees on *any* band, so one unlucky bit
    does not lose the pair.

    A zero vector is skipped. Its sign bits are all "not negative", which is a
    fixed bucket that every zero and every degenerate embedding shares -- the
    largest and least useful block the index could contain.
    """
    if not embedding:
        return ()
    if not any(embedding):
        return ()

    planes = _hyperplanes(len(embedding))
    keys: list[BlockingKey] = []
    for band in range(LSH_BANDS):
        bits = 0
        for offset in range(LSH_BITS_PER_BAND):
            plane = planes[band * LSH_BITS_PER_BAND + offset]
            projection = sum(a * b for a, b in zip(plane, embedding, strict=True))
            bits = (bits << 1) | (1 if projection >= 0.0 else 0)
        keys.append(BlockingKey(BlockKind.EMBEDDING_LSH, f"{band}:{bits:0{LSH_BITS_PER_BAND}b}"))
    return tuple(keys)


def pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    """Order-independent identity for an unordered pair.

    Every pair in this package is stored sorted. Without a canonical orientation
    `(a, b)` and `(b, a)` are two entries in every candidate set, review queue and
    `must_not_link` table, and a constraint written as one fails to block the
    other.
    """
    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BlockingStats:
    """What one index build cost and what it gave up.

    Exists so a resolution run can report its own recall risk. `skipped_pairs` is
    an upper bound on the comparisons the size cap removed, and a run where it
    dwarfs `candidate_pairs` is a run whose blocking configuration is wrong.
    """

    records: int
    keys: int
    blocks: int
    largest_block: int
    candidate_pairs: int
    skipped_blocks: int
    skipped_pairs: int


class BlockingIndex:
    """An inverted index from blocking key to record id.

    Build once per resolution pass, query many times. Two access patterns:

    `candidates_for(record)`
        The streaming case -- one new mention against a corpus already indexed.
        The record need not be in the index.
    `candidate_pairs()`
        The batch case -- every pair worth scoring, sorted, deduplicated.

    Not thread-safe for concurrent `add` and read: mutation while iterating would
    yield a different candidate set depending on timing, and this package's whole
    contract is that identical inputs produce identical output. Build fully, then
    read.
    """

    __slots__ = ("_by_key", "_max_block_size", "_records", "oversized_blocks")

    def __init__(self, *, max_block_size: int | None = DEFAULT_MAX_BLOCK_SIZE) -> None:
        self._by_key: dict[BlockingKey, list[str]] = {}
        self._records: dict[str, ResolutionRecord] = {}
        self._max_block_size = max_block_size
        # Low-precision blocks that exceeded the cap, and their sizes, refreshed
        # by every `candidate_pairs()` call. Read it: it is the only evidence
        # that blocking chose latency over recall, and a pass that never
        # inspects it is a pass whose misses are invisible -- the exact failure
        # mode this module's docstring opens with.
        self.oversized_blocks: dict[BlockingKey, int] = {}

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, record_id: object) -> bool:
        return record_id in self._records

    def add(self, record: ResolutionRecord) -> None:
        """Index one record. Re-adding the same id replaces its keys.

        Replacement rather than accumulation: a record re-indexed after its
        aliases changed must not keep blocking under the alias it lost, or an
        un-merge that removed an alias would leave the two entities still
        colliding and the next pass would re-merge them.
        """
        if record.id in self._records:
            self.remove(record.id)
        self._records[record.id] = record
        for key in blocking_keys(record):
            self._by_key.setdefault(key, []).append(record.id)

    def add_all(self, records: Iterable[ResolutionRecord]) -> None:
        """Index many records. Insertion order does not affect any output."""
        for record in records:
            self.add(record)

    def remove(self, record_id: str) -> None:
        """Drop a record and all of its keys. A no-op for an unknown id."""
        record = self._records.pop(record_id, None)
        if record is None:
            return
        for key in blocking_keys(record):
            bucket = self._by_key.get(key)
            if bucket is None:
                continue
            bucket[:] = [rid for rid in bucket if rid != record_id]
            if not bucket:
                del self._by_key[key]

    def get(self, record_id: str) -> ResolutionRecord | None:
        """The indexed record for `record_id`, or `None`."""
        return self._records.get(record_id)

    def records(self) -> tuple[ResolutionRecord, ...]:
        """Every indexed record, ordered by id so callers inherit determinism."""
        return tuple(self._records[rid] for rid in sorted(self._records))

    def block(self, key: BlockingKey) -> tuple[str, ...]:
        """Members of one block, sorted by id."""
        return tuple(sorted(self._by_key.get(key, ())))

    def candidates_for(self, record: ResolutionRecord) -> tuple[str, ...]:
        """Ids worth scoring against `record`, sorted, excluding `record.id`.

        The size cap does not apply here. Capping the batch path bounds a
        quadratic cost; capping this path would bound a *linear* one, throwing
        away recall to save nothing. A single mention against one oversized block
        is one scan, and that is affordable.
        """
        candidates: set[str] = set()
        for key in blocking_keys(record):
            candidates.update(self._by_key.get(key, ()))
        candidates.discard(record.id)
        return tuple(sorted(candidates))

    def shared_keys(self, left_id: str, right_id: str) -> tuple[BlockingKey, ...]:
        """Keys that put both records in one block. Empty if they never met.

        The matcher uses this to see whether a pair arrived via an
        `IDENTIFIER` hit, and a reviewer uses it to understand why a pair was
        ever proposed. Explaining a *proposal* is as necessary as explaining a
        merge -- an unexplained candidate looks like a bug in the index.
        """
        left = self._records.get(left_id)
        right = self._records.get(right_id)
        if left is None or right is None:
            return ()
        return tuple(sorted(set(blocking_keys(left)) & set(blocking_keys(right))))

    def candidate_pairs(self) -> tuple[tuple[str, str], ...]:
        """Every pair sharing at least one usable key, sorted and deduplicated.

        Sorted output is not cosmetic. The clustering pass in
        `entity_resolution.py` breaks linkage ties by pair order, so an unordered
        candidate set would make two workers with identical inputs build
        different clusters.
        """
        pairs: set[tuple[str, str]] = set()
        self.oversized_blocks.clear()

        for key, members in self._by_key.items():
            size = len(members)
            if size < 2:
                continue
            if (
                self._max_block_size is not None
                and key.kind in LOW_PRECISION_KINDS
                and size > self._max_block_size
            ):
                self.oversized_blocks[key] = size
                continue
            ordered = sorted(members)
            for i, left in enumerate(ordered):
                for right in ordered[i + 1 :]:
                    pairs.add((left, right))

        return tuple(sorted(pairs))

    def stats(self) -> BlockingStats:
        """Size and cost of the current index. Recomputes candidate pairs."""
        pairs = self.candidate_pairs()
        skipped_blocks = len(self.oversized_blocks)
        skipped_pairs = sum(size * (size - 1) // 2 for size in self.oversized_blocks.values())
        return BlockingStats(
            records=len(self._records),
            keys=len(self._by_key),
            blocks=sum(1 for members in self._by_key.values() if len(members) > 1),
            largest_block=max((len(m) for m in self._by_key.values()), default=0),
            candidate_pairs=len(pairs),
            skipped_blocks=skipped_blocks,
            skipped_pairs=skipped_pairs,
        )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Cosine of two vectors in `[-1, 1]`, or `None` when it is undefined.

    Lives here rather than in the matcher because the LSH keys above and the
    matcher's embedding feature must agree on what "similar vectors" means; two
    implementations would drift.

    `None` rather than `0.0` for a length mismatch or a zero vector. Zero reads
    as "measured, and they are unrelated", which is a claim this function cannot
    make -- and the matcher treats a missing feature very differently from a
    feature that scored zero.
    """
    if len(left) != len(right) or not left:
        return None
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return dot / math.sqrt(left_norm * right_norm)
