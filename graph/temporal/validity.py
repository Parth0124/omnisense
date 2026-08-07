"""Bitemporal edge validity: `[valid_from, valid_to)`, as-of predicates, merging.

Every edge in the knowledge graph is bitemporal (`docs/knowledge-graph.md` §5).
Two axes that this module refuses to conflate:

    valid time        `valid_from` / `valid_to`   when was this true in the world
    transaction time  `observed_at`               when did OmniSense learn it

**Why closing beats deleting.** When Datadog stops competing with New Relic the
edge is *closed* -- `valid_to` is set -- and never removed. "Datadog competed
with New Relic in 2024" does not become false in 2026; it stops being *current*.
A report written in 2024 cites that edge, and `docs/report-schema.md` requires a
citation to resolve for the life of the report. Deleting the edge would make that
report unreproducible: re-running the same investigation with the same `as_of`
would return a smaller graph and a different answer, and nothing in the system
would be able to say why. History is the product, so the delete path does not
exist here at all -- there is `close()`, and there is no `remove()`.

**Why the interval is half-open.** `[valid_from, valid_to)`. Two consecutive
facts about the same pair -- "competes on strength 0.9 until March, 0.4 after" --
must tile without overlapping, or an as-of query at exactly the boundary instant
returns both and the caller has to break the tie by guessing. Half-open intervals
tile; closed ones do not. `valid_to = None` means open-ended, and it is checked
explicitly (`IS NULL` in Cypher, `is None` here) rather than represented by a
sentinel like `9999-12-31`: a sentinel silently sorts, compares and *aggregates*
as a real date, so `max(valid_to)` over a company's edges reports the year 9999
and an "average relationship duration" dashboard reports eight millennia.
`Interval` rejects sentinel years outright for that reason.

**Out-of-order arrival is the normal case, not the edge case.** A connector
backfilling arXiv in August produces a paper from March that establishes a
`USES` edge with `valid_from` in March -- learned *after* an edge that started in
June. Kafka replays deliver a batch twice. A GDELT re-crawl surfaces a 2019
acquisition today. So every function here that combines assertions is a pure
function of the *set* of assertions, deterministically sorted by valid time
before anything else happens. Resolution is by **valid time, never by arrival
order**: the assertion that starts earlier is the one that gets closed, whichever
one showed up first. That is the property `tests/unit/graph/test_temporal.py`
pins with a permutation test, and it is what makes replaying a partition from the
beginning converge on the same graph rather than a differently-shaped one.

**Contradictions are recorded, never averaged away.** Two assertions that
overlap in valid time and disagree about the edge's properties are a real
conflict -- two extractors read the world differently, or one of them is wrong.
`merge_assertions()` resolves it the way `docs/knowledge-graph.md` §5 specifies
(close the earlier interval at the later one's `valid_from`, stamp
`superseded_by`) *and* returns a `Contradiction` describing what it did. A merge
that quietly picked a winner would make a systematically broken extractor look
exactly like a world that changes often, and the only place that difference is
visible is here.

Layer note: **L1** (`docs/architecture.md` §6.1). Imports `models/` and nothing
else in the repository -- in particular no driver and no session, so the interval
algebra is testable without a database and reusable by the writer, the query
builder and the retrieval path alike.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol, runtime_checkable

from models.base import utcnow

__all__ = [
    "AS_OF_PARAM",
    "MAX_REASONABLE_YEAR",
    "Assertion",
    "Contradiction",
    "ContradictionKind",
    "Interval",
    "IntervalSet",
    "Precision",
    "Resolution",
    "ResolvedInterval",
    "as_of",
    "as_of_cypher",
    "close",
    "coerce_instant",
    "covers",
    "merge_assertions",
    "open_interval",
]


AS_OF_PARAM: Final[str] = "as_of"
"""Canonical Cypher parameter name for the as-of instant.

Every template in `graph/queries/cypher.py` uses this spelling, so the fragment
emitted by `as_of_cypher()` can be dropped into any of them without the caller
having to remember which name that particular query chose. A query with two
differently-named as-of parameters is a query that reads the entity graph at one
instant and the mention graph at another -- a millisecond apart, invisible until
an edge closes in between.
"""

MAX_REASONABLE_YEAR: Final[int] = 2200
"""Above this, a `valid_to` is a sentinel wearing a date's clothes.

`9999-12-31`, `2999-01-01` and `3000-01-01` are the three that appear in real
extraction output. They are rejected rather than accepted-and-normalized because
accepting one means the *writer* still put it in Neo4j, where `r.valid_to IS
NULL` -- the predicate every as-of query uses -- is then false for an edge that
is conceptually open, and the edge vanishes from every current-view query while
remaining perfectly visible in the database. That failure is nearly impossible to
diagnose from the query side.
"""

_MIN_REASONABLE_YEAR: Final[int] = 1900
"""Below this, a date is an extraction artefact rather than a fact.

`0001-01-01` is what a date parser returns for an empty string, and a `valid_from`
in year 1 makes every as-of query in history match the edge.
"""


# --------------------------------------------------------------------------- #
# Instants
# --------------------------------------------------------------------------- #


def coerce_instant(value: Any, *, field_name: str = "instant") -> datetime:
    """Normalize anything datetime-shaped to a tz-aware UTC `datetime`.

    Accepts a `datetime` (aware, normalized to UTC) or an ISO-8601 string. A
    naive datetime is **rejected**, not assumed to be UTC: `models/base.py`
    `_require_utc` makes the same call for the same reason, and the reason is
    sharper here. Neo4j stores a naive datetime as a `LocalDateTime`, which
    compares against a zoned `$as_of` parameter by raising -- so a single naive
    `valid_from` written today turns every as-of query that happens to touch that
    edge into a server-side type error, months later, in a query that has nothing
    to do with the write that caused it.

    `neo4j.time.DateTime` is accepted through the `to_native()` duck-type rather
    than by importing the driver: this module is L1 and must stay importable
    without a Bolt driver on the path.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} is an empty string; a timestamp is required")
        # `fromisoformat` in 3.12 handles the trailing `Z` that every JSON API
        # emits; earlier versions did not, and a `.replace("Z", "+00:00")` shim
        # here would silently accept `2024Z01Z01`.
        try:
            value = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} {text!r} is not ISO-8601: {exc}") from exc
    elif not isinstance(value, datetime) and hasattr(value, "to_native"):
        # `neo4j.time.DateTime`, as returned by a driver record.
        value = value.to_native()

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime or ISO-8601 string, got {type(value)!r}")
    if value.tzinfo is None:
        raise ValueError(
            f"{field_name} is naive; temporal edges cross process and store "
            "boundaries where an implied timezone is unrecoverable. Attach UTC at "
            "the extraction boundary."
        )
    moment = value.astimezone(UTC)
    if moment.year > MAX_REASONABLE_YEAR:
        raise ValueError(
            f"{field_name} is {moment.isoformat()}, beyond year {MAX_REASONABLE_YEAR}. "
            "An open-ended interval is expressed as valid_to=None, never as a "
            "sentinel date -- a sentinel compares and aggregates as a real date."
        )
    if moment.year < _MIN_REASONABLE_YEAR:
        raise ValueError(
            f"{field_name} is {moment.isoformat()}, before year {_MIN_REASONABLE_YEAR}. "
            "This is almost always a date parser returning its zero value for an "
            "input it could not read; an edge dated year 1 matches every as-of query."
        )
    return moment


class Precision(enum.StrEnum):
    """How precisely a boundary instant is actually known.

    `docs/knowledge-graph.md` §5: extracted dates are frequently month- or
    quarter-precise -- "Acme acquired Globex in Q3 2024" carries no day. The
    instant stored is the *start* of the period, so an as-of query at day
    granularity would answer "was this true on 2024-08-15?" with the confidence of
    a day-precise fact when the underlying evidence cannot distinguish July from
    September. Storing the precision alongside is what lets a caller ask whether
    its answer sits inside the uncertainty window instead of finding out from a
    reader who noticed the report was a quarter out.

    Ordered coarsest-last by `rank`, so merging several assertions can take the
    coarsest precision as the result's precision -- the honest choice, since a
    merged interval is no more precise than its vaguest input.
    """

    EXACT = "exact"
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

    @property
    def rank(self) -> int:
        """Coarseness order. Higher is vaguer."""
        return _PRECISION_RANK[self]

    @property
    def window(self) -> timedelta:
        """Half-width of the uncertainty around a boundary at this precision.

        Approximate by construction -- a month is not 30 days and a quarter is not
        91 -- and that is acceptable because the value is used to decide whether to
        *warn*, never to shift a boundary. Rounding a boundary by the window would
        move a fact in time, which is the thing this whole module exists to
        prevent.
        """
        return _PRECISION_WINDOW[self]


_PRECISION_RANK: Final[dict[Precision, int]] = {
    Precision.EXACT: 0,
    Precision.DAY: 1,
    Precision.MONTH: 2,
    Precision.QUARTER: 3,
    Precision.YEAR: 4,
}

_PRECISION_WINDOW: Final[dict[Precision, timedelta]] = {
    Precision.EXACT: timedelta(0),
    Precision.DAY: timedelta(days=1),
    Precision.MONTH: timedelta(days=31),
    Precision.QUARTER: timedelta(days=92),
    Precision.YEAR: timedelta(days=366),
}


# --------------------------------------------------------------------------- #
# The interval
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True, order=False)
class Interval:
    """A half-open validity interval `[valid_from, valid_to)`.

    Frozen because an interval is shared by reference between an `Assertion`, a
    `ResolvedInterval` and whatever the caller keeps: closing one in place would
    close it everywhere, including inside the `Contradiction` record that exists
    to preserve what it looked like *before* it was closed.
    """

    valid_from: datetime
    valid_to: datetime | None = None
    precision: Precision = Precision.EXACT

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid_from", coerce_instant(self.valid_from, field_name="valid_from"))
        if self.valid_to is not None:
            object.__setattr__(self, "valid_to", coerce_instant(self.valid_to, field_name="valid_to"))
            if self.valid_to <= self.valid_from:
                # Equal endpoints are rejected as well as inverted ones. A
                # zero-length half-open interval contains no instant at all, so an
                # edge carrying one is invisible to every as-of query while
                # occupying a row, a `MERGE` key and a place in `evidence_count`.
                raise ValueError(
                    f"valid_to {self.valid_to.isoformat()} is not after valid_from "
                    f"{self.valid_from.isoformat()}; [from, to) is half-open, so an "
                    "interval that ends when it starts is never true."
                )

    # ------------------------------------------------------------ predicates --

    @property
    def is_open(self) -> bool:
        """Whether the fact is still believed true. `valid_to IS NULL` in Cypher."""
        return self.valid_to is None

    def contains(self, moment: datetime) -> bool:
        """Whether `moment` falls inside `[valid_from, valid_to)`."""
        return covers(self.valid_from, self.valid_to, moment)

    def overlaps(self, other: Interval) -> bool:
        """Whether the two intervals share at least one instant.

        Adjacency is deliberately *not* overlap: `[Jan, Mar)` and `[Mar, Jun)`
        tile and do not overlap, which is the entire point of half-open intervals.
        Getting this wrong reports a contradiction at every clean handover.
        """
        if other.valid_to is not None and other.valid_to <= self.valid_from:
            return False
        return not (self.valid_to is not None and self.valid_to <= other.valid_from)

    def meets(self, other: Interval) -> bool:
        """Whether `self` ends exactly where `other` begins."""
        return self.valid_to is not None and self.valid_to == other.valid_from

    def boundary_is_uncertain(self, moment: datetime) -> bool:
        """Whether `moment` sits inside the precision window of either boundary.

        The honest answer to "was this true on 2024-08-15?" for a quarter-precise
        edge starting in Q3 2024 is "the evidence cannot tell you". A caller that
        surfaces a claim in a report is expected to consult this and hedge the
        wording; `contains()` still answers the crisp question, because a
        three-valued `contains()` would infect every call site with `is True`
        comparisons and be got wrong.
        """
        window = self.precision.window
        if window <= timedelta(0):
            return False
        if abs(moment - self.valid_from) < window:
            return True
        return self.valid_to is not None and abs(moment - self.valid_to) < window

    # --------------------------------------------------------------- algebra --

    def close(self, at: datetime, *, precision: Precision | None = None) -> Interval:
        """Return a copy ending at `at`. Never mutates, never deletes.

        Closing an already-closed interval *earlier* is allowed -- a later, better
        source can shorten a relationship. Closing it later is refused: that is an
        attempt to resurrect a fact by extending its interval past a boundary
        something else was already stamped against, and it silently invalidates
        whatever `superseded_by` pointed at that boundary. Re-open by writing a new
        assertion for the new period instead, which leaves both periods visible.
        """
        moment = coerce_instant(at, field_name="close.at")
        if moment <= self.valid_from:
            raise ValueError(
                f"cannot close at {moment.isoformat()}, which is not after valid_from "
                f"{self.valid_from.isoformat()}. A fact that ends before it starts is "
                "a contradiction to record, not an interval to write."
            )
        if self.valid_to is not None and moment > self.valid_to:
            raise ValueError(
                f"cannot extend a closed interval: valid_to is "
                f"{self.valid_to.isoformat()} and close() was asked for "
                f"{moment.isoformat()}. Write a new assertion for the later period."
            )
        return Interval(
            valid_from=self.valid_from,
            valid_to=moment,
            precision=self.precision if precision is None else precision,
        )

    def to_cypher_properties(self) -> dict[str, Any]:
        """The three properties a writer sets on an edge.

        `valid_to` is present-and-`None` rather than omitted, so an `ON MATCH SET`
        that re-opens a previously closed edge actually clears the property. An
        omitted key leaves the old `valid_to` in place, which is how a reopened
        relationship stays invisible to the current view.
        """
        return {
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "valid_from_precision": self.precision.value,
        }

    def __str__(self) -> str:
        end = "open" if self.valid_to is None else self.valid_to.isoformat()
        return f"[{self.valid_from.isoformat()}, {end})"


def open_interval(valid_from: datetime, *, precision: Precision = Precision.EXACT) -> Interval:
    """An interval that is still true as far as we know. Sugar for the common case."""
    return Interval(valid_from=valid_from, valid_to=None, precision=precision)


def close(interval: Interval, at: datetime, *, precision: Precision | None = None) -> Interval:
    """Free-function form of `Interval.close`, for `functools.partial` call sites."""
    return interval.close(at, precision=precision)


# --------------------------------------------------------------------------- #
# As-of predicates
# --------------------------------------------------------------------------- #


def covers(valid_from: datetime, valid_to: datetime | None, moment: datetime) -> bool:
    """The as-of predicate in its lowest form, over three bare instants.

    Exists separately from `Interval.contains` because the values usually arrive
    as columns of a driver record or as fields of a Pydantic model, and forcing an
    `Interval` to be constructed -- with its validation -- just to ask a question
    would make the read path pay for the write path's invariants.

    Deliberately identical in shape to the Cypher in `as_of_cypher()`:
    `valid_from <= moment AND (valid_to IS NULL OR valid_to > moment)`. The two
    are the same predicate expressed in two languages, and a difference between
    them means a Python-side filter and a server-side filter disagree about the
    same edge -- which surfaces as a citation that resolves to nothing.
    """
    if valid_from > moment:
        return False
    return valid_to is None or valid_to > moment


@runtime_checkable
class HasValidity(Protocol):
    """Anything carrying the two validity columns.

    Structural rather than nominal so a driver record wrapper, a Pydantic edge
    model and an `Interval` all satisfy it without a common base class -- there is
    no base class they could share that does not violate the layer matrix.
    """

    @property
    def valid_from(self) -> datetime: ...

    @property
    def valid_to(self) -> datetime | None: ...


def as_of(moment: datetime | None = None) -> Any:
    """Build a predicate that answers "was this true at `moment`?".

    Returns a callable accepting an `Interval`, any object with `valid_from` /
    `valid_to`, or a `Mapping` (a driver record, `record.data()`). A single
    predicate object is what makes `filter(as_of(t), edges)` read correctly and,
    more importantly, what makes the instant get resolved **once**: `moment=None`
    resolves to `utcnow()` here, at construction, not per call. A predicate that
    re-read the clock per element would evaluate the first edge in a list at a
    different instant from the last, and an edge closing during the iteration
    would be both included and excluded in the same pass.

    The instant is never defaulted *inside* a Cypher template for the same reason
    `docs/knowledge-graph.md` §5 gives: a template that quietly means `datetime()`
    hides from the caller that they asked a question about the present.
    """
    instant = utcnow() if moment is None else coerce_instant(moment, field_name="as_of")

    def predicate(edge: HasValidity | Mapping[str, Any] | Interval) -> bool:
        valid_from, valid_to = _validity_of(edge)
        return covers(valid_from, valid_to, instant)

    predicate.instant = instant  # type: ignore[attr-defined]
    return predicate


def _validity_of(edge: HasValidity | Mapping[str, Any] | Interval) -> tuple[datetime, datetime | None]:
    """Pull `(valid_from, valid_to)` off whichever shape the caller had."""
    if isinstance(edge, Mapping):
        if "valid_from" not in edge:
            raise KeyError(
                "record has no 'valid_from'; an as-of filter over rows that do not "
                "carry validity would silently pass every row. RETURN r.valid_from "
                "and r.valid_to from the query."
            )
        raw_from: Any = edge["valid_from"]
        raw_to: Any = edge.get("valid_to")
    else:
        raw_from = edge.valid_from
        raw_to = edge.valid_to
    valid_from = coerce_instant(raw_from, field_name="valid_from")
    valid_to = None if raw_to is None else coerce_instant(raw_to, field_name="valid_to")
    return valid_from, valid_to


def as_of_cypher(alias: str = "r", *, param: str = AS_OF_PARAM) -> str:
    """The as-of fragment, verbatim from `docs/knowledge-graph.md` §5.

    Emitted here rather than written out in each template in
    `graph/queries/cypher.py` so there is exactly one copy of the predicate in the
    repository. The failure mode of a second copy is specific and quiet: someone
    writes `r.valid_to >= $as_of` instead of `>`, and every edge that closed at
    exactly the queried instant is returned by one query and not by another. Both
    look right in review.

    `alias` reaches the query *text*, so it is validated as an identifier. It is
    never a caller-supplied value -- it is the name the template gave its own
    relationship variable -- but a template built by string concatenation
    elsewhere could pass one, and Cypher injection through a relationship alias is
    exactly as effective as through a value.
    """
    _require_identifier(alias, "alias")
    _require_identifier(param, "param")
    return (
        f"{alias}.valid_from <= ${param}\n"
        f"  AND ({alias}.valid_to IS NULL OR {alias}.valid_to > ${param})"
    )


def _require_identifier(value: str, label: str) -> None:
    """Reject anything that is not a bare Cypher identifier."""
    if not value or not value.replace("_", "").isalnum() or value[0].isdigit():
        raise ValueError(
            f"{label} {value!r} is not a bare identifier; it is concatenated into "
            "Cypher text and cannot be parameterised."
        )


# --------------------------------------------------------------------------- #
# Assertions and merging
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Assertion:
    """One source's claim that an edge held over one interval.

    Not an edge. An edge is what the graph ends up storing; an assertion is what
    one extractor said, and several assertions from several sources are merged
    into the edge's interval set. Keeping them distinct is what makes the merge a
    pure function -- the alternative, mutating an edge as each new claim arrives,
    makes the result depend on arrival order, which is the exact bug this module
    is built to avoid.

    `properties` carries only the fields whose disagreement constitutes a
    contradiction -- `status` for `ACQUIRED`, `market` and `basis` for
    `COMPETES_WITH`. `confidence`, `evidence_count` and `source_signal_ids` are
    *not* in it: two sources agreeing on the fact with different confidence is
    corroboration, and treating it as a conflict would report a contradiction on
    every second signal.
    """

    source: str
    """Extractor or rule id -- `claude-sonnet-5`, `rule:acquisition_regex_v3`.

    Part of the sort key, so it must be stable across processes. Two assertions
    from the same source with the same interval and properties are duplicates and
    coalesce; that is what makes an at-least-once Kafka replay a no-op.
    """

    interval: Interval
    observed_at: datetime = field(default_factory=utcnow)
    confidence: float = 0.5
    properties: Mapping[str, Any] = field(default_factory=dict)
    signal_ids: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("Assertion.source is required; it is part of the merge sort key")
        object.__setattr__(
            self, "observed_at", coerce_instant(self.observed_at, field_name="observed_at")
        )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        object.__setattr__(self, "properties", dict(self.properties))
        object.__setattr__(self, "signal_ids", tuple(dict.fromkeys(self.signal_ids)))

    @property
    def fingerprint(self) -> tuple[tuple[str, str], ...]:
        """A hashable, order-independent digest of the contradiction-bearing fields.

        `repr()` of each value rather than the value itself: property values come
        out of an LLM extraction and are occasionally unhashable (a list of region
        ids). `repr` is stable within a process for the scalars and sequences that
        actually appear, and the digest is only ever compared to another digest
        built the same way in the same process.
        """
        return tuple(sorted((str(key), repr(value)) for key, value in self.properties.items()))

    @property
    def is_late(self) -> bool:
        """Whether this fact was learned after the period it describes had ended.

        Not a defect -- backfills and re-crawls produce these constantly -- but it
        is the case where an incremental "close the current edge" writer produces a
        different graph from a full replay, so it is worth being able to count.
        """
        return self.interval.valid_to is not None and self.observed_at > self.interval.valid_to

    def _sort_key(self) -> tuple[Any, ...]:
        """Total order by **valid time**, then by everything else, deterministically.

        Arrival order contributes nothing to the first two components. That is the
        whole design: a fact learned today about last year sorts where its *valid*
        time puts it, so a shuffled batch and an in-order batch produce identical
        interval sets. `observed_at` and `source` appear only as tie-breakers, to
        make the order total -- without them, two assertions identical in valid
        time would sort by list position and reintroduce the dependence on arrival
        order through the back door.
        """
        end = self.interval.valid_to
        return (
            self.interval.valid_from,
            # An open interval extends furthest, so it sorts last among equal starts.
            end is None,
            end if end is not None else self.interval.valid_from,
            self.observed_at,
            self.source,
            self.fingerprint,
        )


class ContradictionKind(enum.StrEnum):
    """What kind of disagreement was found."""

    OVERLAP = "overlap"
    """Two assertions disagree and share instants. Resolvable by truncation."""

    SIMULTANEOUS = "simultaneous"
    """Two assertions disagree and start at the same instant.

    Not resolvable by truncation -- closing the earlier one at the later one's
    `valid_from` would produce a zero-length interval. Something is wrong with the
    extraction, the entity resolution, or the world.
    """


class Resolution(enum.StrEnum):
    """What the merge actually did about it."""

    CLOSED_EARLIER = "closed_earlier"
    """The earlier interval was truncated at the later one's `valid_from`."""

    KEPT_HIGHER_CONFIDENCE = "kept_higher_confidence"
    """Neither could be truncated; the lower-confidence claim is not in the set."""

    UNRESOLVED = "unresolved"
    """Recorded and left alone. Nothing downstream should treat this as clean."""


@dataclass(frozen=True, slots=True)
class Contradiction:
    """A disagreement the merge found, and what it did about it.

    Returned rather than logged. A log line is invisible to the Critic, to
    `docs/api-reference.md` §4's confidence field and to the operator reading a
    weekly extraction-quality number; a value on the result is not. The rule from
    `docs/knowledge-graph.md` §5 -- close the older interval and record
    `superseded_by` -- is only half a rule without something that says *this
    happened*, because a graph where every second edge was superseded is a broken
    extractor and looks, from the graph alone, exactly like a volatile market.
    """

    kind: ContradictionKind
    earlier: Assertion
    later: Assertion
    resolution: Resolution
    at: datetime | None = None
    """Where the earlier interval was cut, when it was."""

    discarded_tail: datetime | None = None
    """The `valid_to` the earlier assertion claimed, when truncation dropped it.

    Set when the earlier assertion outlived the later one -- "true all of 2024"
    superseded by "different in March" leaves April-to-December unexplained.
    Truncation, not splitting, is deliberate: the later assertion says nothing
    about what happens when it ends, and inventing a resumption of the earlier
    fact would be the merge making up a fact nobody asserted. Recording the
    dropped endpoint keeps the loss auditable.
    """

    def describe(self) -> str:
        """One line for a log, a metric label or a Critic finding."""
        return (
            f"{self.kind.value}: {self.earlier.source} {self.earlier.interval} vs "
            f"{self.later.source} {self.later.interval} -> {self.resolution.value}"
        )


@dataclass(frozen=True, slots=True)
class ResolvedInterval:
    """One interval of the merged history, with everything that supports it."""

    interval: Interval
    properties: Mapping[str, Any] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    signal_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    observed_first: datetime | None = None
    observed_last: datetime | None = None
    superseded_by: str | None = None
    """Source of the assertion that truncated this one. `None` when nothing did.

    Mirrors the `superseded_by` property `docs/knowledge-graph.md` §5 requires on
    the stored edge, so the writer can persist it without recomputing why the
    interval ends where it does.
    """

    @property
    def valid_from(self) -> datetime:
        return self.interval.valid_from

    @property
    def valid_to(self) -> datetime | None:
        return self.interval.valid_to

    @property
    def evidence_count(self) -> int:
        """Distinct supporting signals. Not the number of assertions.

        Two extractors reading the same Reddit post are one piece of evidence.
        Counting assertions instead is how a rule and a model agreeing on one
        document becomes "corroborated by two sources" in a report.
        """
        return len(self.signal_ids)


@dataclass(frozen=True, slots=True)
class IntervalSet:
    """The merged history of one edge: disjoint intervals, plus what went wrong.

    Intervals are sorted, non-overlapping and half-open, so `at()` is a scan with
    at most one match. The invariant is worth stating because it is what lets the
    writer emit one row per interval and lets an as-of query return at most one
    answer per edge -- two overlapping rows would make "competitors as of March"
    list the same rival twice with different strengths.
    """

    intervals: tuple[ResolvedInterval, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()

    def at(self, moment: datetime) -> ResolvedInterval | None:
        """The interval covering `moment`, or `None`."""
        instant = coerce_instant(moment, field_name="at.moment")
        for resolved in self.intervals:
            if resolved.interval.contains(instant):
                return resolved
        return None

    def is_true_at(self, moment: datetime) -> bool:
        """Whether the edge held at `moment`."""
        return self.at(moment) is not None

    @property
    def current(self) -> ResolvedInterval | None:
        """The open interval, if the edge is still believed true.

        At most one: the merge closes every interval it supersedes, so two open
        intervals cannot survive it.
        """
        for resolved in reversed(self.intervals):
            if resolved.interval.is_open:
                return resolved
        return None

    @property
    def has_contradictions(self) -> bool:
        return bool(self.contradictions)

    @property
    def has_unresolved(self) -> bool:
        """Whether any conflict was recorded without being resolved.

        A caller writing this set to the graph should refuse, or write it and
        flag the edge for review -- but not treat it as ordinary history.
        """
        return any(c.resolution is Resolution.UNRESOLVED for c in self.contradictions)

    def spans(self) -> tuple[Interval, ...]:
        """Just the intervals, for the tests and for a compact log line."""
        return tuple(resolved.interval for resolved in self.intervals)


def merge_assertions(assertions: Iterable[Assertion]) -> IntervalSet:
    """Fold assertions about **one** edge into a disjoint interval set.

    The caller groups by `(edge_type, from_id, to_id)` first -- this function has
    no way to tell two edges apart and will happily merge assertions about
    unrelated relationships if handed both.

    Order-independent by construction: the input is sorted by valid time before
    anything is decided, so a batch replayed out of order, a backfill arriving
    months late and a live stream all converge on the same set. That is asserted
    over every permutation in `tests/unit/graph/test_temporal.py`.

    The four cases, in the order they are checked:

    1. **Same properties, overlapping or adjacent.** Coalesce. Two sources saying
       the same thing about overlapping periods is one fact with two sources, and
       leaving it as two intervals would double `evidence_count` and make an
       as-of query return the same edge twice.
    2. **Same properties, separated by a gap.** Two intervals. The gap is real
       information -- a relationship that lapsed and resumed -- and bridging it
       would assert a fact for a period in which nothing was observed.
    3. **Different properties, disjoint.** Two intervals, no conflict. This is
       ordinary change over time.
    4. **Different properties, overlapping.** A contradiction. Resolved per
       `docs/knowledge-graph.md` §5 by closing the earlier interval at the later
       one's `valid_from`, unless they start at the same instant, in which case
       there is nothing to close and the conflict is recorded as unresolved.

    Coalescing is checked against the *last* resolved interval only, which is
    sufficient because the sort guarantees every earlier interval starts no later
    and the walk maintains the "disjoint and sorted" invariant at every step.
    """
    ordered = sorted(assertions, key=Assertion._sort_key)
    if not ordered:
        return IntervalSet()

    resolved: list[_Accumulator] = []
    conflicts: list[Contradiction] = []

    for assertion in ordered:
        if not resolved:
            resolved.append(_Accumulator.of(assertion))
            continue

        last = resolved[-1]
        if last.fingerprint == assertion.fingerprint:
            if last.interval.overlaps(assertion.interval) or last.interval.meets(
                assertion.interval
            ):
                last.absorb(assertion)
            else:
                resolved.append(_Accumulator.of(assertion))
            continue

        if not last.interval.overlaps(assertion.interval):
            resolved.append(_Accumulator.of(assertion))
            continue

        # Case 4: genuine disagreement over shared instants.
        if assertion.interval.valid_from > last.interval.valid_from:
            dropped = last.interval.valid_to
            keeps_tail = dropped is None or (
                assertion.interval.valid_to is not None
                and dropped > assertion.interval.valid_to
            )
            conflicts.append(
                Contradiction(
                    kind=ContradictionKind.OVERLAP,
                    earlier=last.as_assertion(),
                    later=assertion,
                    resolution=Resolution.CLOSED_EARLIER,
                    at=assertion.interval.valid_from,
                    discarded_tail=dropped if keeps_tail else None,
                )
            )
            last.truncate_at(assertion.interval.valid_from, superseded_by=assertion.source)
            resolved.append(_Accumulator.of(assertion))
            continue

        # Same `valid_from`, different properties: truncation is impossible, so
        # one claim has to lose. Highest confidence wins; ties go to the more
        # recently observed, then to the lexicographically smaller source so the
        # outcome does not depend on which one the sort happened to place first.
        loser, winner = _rank_simultaneous(last.as_assertion(), assertion)
        conflicts.append(
            Contradiction(
                kind=ContradictionKind.SIMULTANEOUS,
                earlier=loser,
                later=winner,
                resolution=Resolution.KEPT_HIGHER_CONFIDENCE,
                at=assertion.interval.valid_from,
            )
        )
        if winner is assertion:
            resolved[-1] = _Accumulator.of(assertion)

    return IntervalSet(
        intervals=tuple(accumulator.finish() for accumulator in resolved),
        contradictions=tuple(conflicts),
    )


def _rank_simultaneous(left: Assertion, right: Assertion) -> tuple[Assertion, Assertion]:
    """`(loser, winner)` for two assertions that start at the same instant."""
    left_key = (left.confidence, left.observed_at, right.source)
    right_key = (right.confidence, right.observed_at, left.source)
    # The cross-referenced source in each key inverts the lexicographic tie-break
    # so that the *smaller* source name wins, which is the documented rule.
    if right_key > left_key:
        return left, right
    return right, left


@dataclass(slots=True)
class _Accumulator:
    """Mutable working state for one interval during the merge.

    Mutable, unlike everything else in this module, and deliberately private: the
    merge builds each interval incrementally, and doing that with frozen values
    would allocate a new object per absorbed assertion for no benefit. Nothing
    outside `merge_assertions` ever sees one.
    """

    interval: Interval
    fingerprint: tuple[tuple[str, str], ...]
    properties: dict[str, Any]
    sources: list[str]
    signal_ids: list[str]
    confidence: float
    observed_first: datetime
    observed_last: datetime
    superseded_by: str | None = None

    @classmethod
    def of(cls, assertion: Assertion) -> _Accumulator:
        return cls(
            interval=assertion.interval,
            fingerprint=assertion.fingerprint,
            properties=dict(assertion.properties),
            sources=[assertion.source],
            signal_ids=list(assertion.signal_ids),
            confidence=assertion.confidence,
            observed_first=assertion.observed_at,
            observed_last=assertion.observed_at,
        )

    def absorb(self, assertion: Assertion) -> None:
        """Fold an agreeing, overlapping or adjacent assertion into this interval."""
        end = self.interval.valid_to
        other_end = assertion.interval.valid_to
        merged_end = None if end is None or other_end is None else max(end, other_end)
        self.interval = Interval(
            valid_from=min(self.interval.valid_from, assertion.interval.valid_from),
            valid_to=merged_end,
            # The union is no more precise than its vaguest member. Taking the
            # finer precision would claim day-level confidence for a boundary that
            # came from "Q3 2024".
            precision=max(
                self.interval.precision,
                assertion.interval.precision,
                key=lambda p: p.rank,
            ),
        )
        if assertion.source not in self.sources:
            self.sources.append(assertion.source)
        for signal_id in assertion.signal_ids:
            if signal_id not in self.signal_ids:
                self.signal_ids.append(signal_id)
        # Max rather than mean: two sources agreeing does not make the fact less
        # certain than the more confident of them, and a mean lets a low-confidence
        # duplicate drag down a well-evidenced edge.
        self.confidence = max(self.confidence, assertion.confidence)
        self.observed_first = min(self.observed_first, assertion.observed_at)
        self.observed_last = max(self.observed_last, assertion.observed_at)

    def truncate_at(self, moment: datetime, *, superseded_by: str) -> None:
        self.interval = self.interval.close(moment)
        self.superseded_by = superseded_by

    def as_assertion(self) -> Assertion:
        """A snapshot of this interval as a single assertion, for a conflict record."""
        return Assertion(
            source=self.sources[0],
            interval=self.interval,
            observed_at=self.observed_first,
            confidence=self.confidence,
            properties=dict(self.properties),
            signal_ids=tuple(self.signal_ids),
        )

    def finish(self) -> ResolvedInterval:
        return ResolvedInterval(
            interval=self.interval,
            properties=dict(self.properties),
            sources=tuple(sorted(self.sources)),
            signal_ids=tuple(self.signal_ids),
            confidence=self.confidence,
            observed_first=self.observed_first,
            observed_last=self.observed_last,
            superseded_by=self.superseded_by,
        )
