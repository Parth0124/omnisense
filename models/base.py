"""Shared Pydantic configuration and primitive types for every OmniSense model.

`models/` is layer **L0** in the dependency matrix (`docs/architecture.md` §6.1):
it imports nothing else in this repository. Everything defined here is the
vocabulary that connectors, services, retrieval, agents and the API all agree on.

Two base classes exist because the Signal contract is deliberately asymmetric
(`docs/signal-model.md` §7):

    Producers validate strictly, consumers validate leniently.

A producer (the enrichment pipeline) must fail loudly on an unexpected field --
that is a bug in the code that built the object. A consumer (retrieval, agents,
the HTTP layer) must tolerate fields added by a newer producer, because during a
rolling deploy old and new versions read the same Kafka topic. Hence
`StrictModel` (`extra="forbid"`) and `LenientModel` (`extra="ignore"`).

`TolerantStrEnum` exists for the same reason: §7 requires that readers "map
unknown members to `unknown`, never raise", so that adding a new `Platform` is a
backward-compatible change rather than a coordinated redeploy.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

__all__ = [
    "LenientModel",
    "NonEmptyStr",
    "Score",
    "Sha256Hex",
    "StrictModel",
    "TolerantStrEnum",
    "UtcDatetime",
    "utcnow",
]


# --------------------------------------------------------------------------- #
# Primitive constrained types
# --------------------------------------------------------------------------- #


def _require_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalize everything else to UTC.

    Every timestamp in OmniSense crosses a process boundary -- connector to Kafka
    to worker to Postgres to Neo4j. A naive datetime is ambiguous the moment it
    leaves the process that created it, and `Signal.timestamp` in particular is
    the field trend and forecast agents key off exclusively
    (`docs/signal-model.md` §2, field 6). Guessing a timezone here would silently
    shift a trend by hours, so this raises instead.

    Runs as an `AfterValidator`, which matters: Pydantic parses `str -> datetime`
    first, so an ISO-8601 string is checked for tz-awareness *after* parsing.
    A `BeforeValidator` would see the raw string and the check would never fire.
    """
    if value.tzinfo is None:
        raise ValueError(
            "datetime must be timezone-aware; naive values are ambiguous across "
            "process boundaries. Attach tzinfo at the connector boundary, e.g. "
            "datetime.now(UTC), once you have confirmed the source's timezone."
        )
    return value.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]
"""A timezone-aware datetime, always normalized to UTC. Naive values are rejected."""

Score = Annotated[float, Field(ge=0.0, le=1.0)]
"""A normalized score in the closed interval [0.0, 1.0]."""

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
"""Lowercase hex SHA-256 digest, exactly 64 characters."""

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
"""A string guaranteed to hold at least one non-whitespace character."""


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Defined here so that no module reaches for the deprecated
    `datetime.utcnow()`, which returns a *naive* datetime and would be rejected
    by `UtcDatetime` at the next validation boundary.
    """
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Base models
# --------------------------------------------------------------------------- #


class StrictModel(BaseModel):
    """Base for models the pipeline *produces*.

    - `extra="forbid"` -- an unexpected key is a bug in the producing code.
    - `validate_assignment=True` -- the enrichment pipeline mutates a Signal
      stage by stage (`docs/signal-model.md` §5), so assignment is the most
      likely place to introduce an invalid value. Validating on assignment costs
      throughput; it is enabled deliberately because a malformed Signal
      propagates into five stores before anything notices.
    - `frozen=False` for the same reason: stages decorate the object in place.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        populate_by_name=True,
        ser_json_bytes="base64",
    )


class LenientModel(BaseModel):
    """Base for models the system *consumes*.

    Ignores unknown fields so a consumer running an older `pipeline_version` can
    still read messages written by a newer producer during a rolling deploy
    (`docs/signal-model.md` §7). `validate_assignment` is off: consumed objects
    are read-only projections and the validation cost buys nothing.
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=False,
        validate_default=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        populate_by_name=True,
        ser_json_bytes="base64",
    )


# --------------------------------------------------------------------------- #
# Forward-compatible enumerations
# --------------------------------------------------------------------------- #


class TolerantStrEnum(enum.StrEnum):
    """A string enum that degrades unrecognized values to its `UNKNOWN` member.

    `docs/signal-model.md` §7 lists "add a new enum member" as a change that is
    allowed *within* a schema version, on the condition that readers never raise
    on a member they do not know. Plain `StrEnum` raises `ValueError`, which
    would turn a routine connector addition into a coordinated redeploy of every
    consumer.

    Lookup order:
      1. exact value match (handled by `StrEnum` before `_missing_` is reached);
      2. case-insensitive value match -- providers are inconsistent about casing;
      3. the `UNKNOWN` member, if the subclass defines one;
      4. `None`, which lets `enum` raise as normal. A subclass that omits
         `UNKNOWN` is therefore strict by construction, which is the right
         behaviour for closed sets like `StageStatus`.

    The original value is not preserved -- callers that need it should keep it in
    the surrounding model's `metadata` before coercion.
    """

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            folded = value.strip().casefold()
            for member in cls:
                if member.value.casefold() == folded:
                    return member
        return cls.__members__.get("UNKNOWN")
