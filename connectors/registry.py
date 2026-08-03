"""Slug -> connector-class discovery, and the two gates that guard it.

The registry is a dictionary. Everything interesting about it is what it refuses
to put in that dictionary.

**Gate 1: declaration validity, at import time.** A connector's `ClassVar` block
is read by the scheduler *before* anything is instantiated
(`docs/connector-spec.md` §3), so a class whose `platform` disagrees with its
`category` is not a typo that shows up in a test -- it is a run that fetches four
thousand records and then has every one of them rejected by
`Signal._check_source_matches_platform`, after the quota is spent. That
mismatch is checked here, against the same `PLATFORM_CATEGORY` table `Signal`
uses at runtime, because an import-time failure costs a stack trace and a
runtime one costs a sync window.

**Gate 2: the legal review, at enable time.** `requires_tos_review = True` marks
every source with no viable official API for this use case -- Instagram, TikTok,
LinkedIn, Amazon reviews (`docs/connector-spec.md` §9). Implementing those means
scraping, and `enable()` refuses them. There is deliberately no override
parameter: an argument at a call site is not a legal review, and the moment one
exists it appears in a config file and the gate is decorative. Clearing the flag
on the class -- a diff, in a pull request, with a reviewer -- is the documented
path, which is the point.

Registration is explicit rather than discovered by walking the package. Import
side effects that scan directories make "which connectors exist here?" depend on
import order, and this file is the answer to that question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from models.enums import PLATFORM_CATEGORY, AuthType, Platform, SourceCategory
from connectors.base import BaseConnector
from connectors.exceptions import ConnectorConfigurationError

if TYPE_CHECKING:  # pragma: no cover -- typing only, keeps the import graph flat
    from connectors.protocol import Credentials, SyncContext

__all__ = [
    "ConnectorRegistration",
    "all",
    "by_category",
    "create",
    "disable",
    "enable",
    "enabled",
    "get",
    "is_enabled",
    "register",
    "registrations",
    "slugs",
    "unregister",
]

SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
"""Lower-case, underscore-separated, starting with a letter.

Not cosmetic. The slug is interpolated into Redis rate-limit keys
(`os:rl:{slug}:{account_id}`, `docs/connector-spec.md` §5.1) and into the Kafka
partition key. A slug containing a colon would split one bucket into two and
quietly double the effective request budget; one containing a space would make
the key un-typeable in `redis-cli` when someone is trying to find out why
ingestion stopped.
"""


@dataclass(frozen=True, slots=True)
class ConnectorRegistration:
    """One entry: the class plus the declaration that was validated for it.

    The declaration is copied out rather than read back off the class each time,
    so that `by_category()` and the scheduler's planning pass answer from
    validated data. Reading `cls.platform` live would mean a class mutated after
    registration silently answers differently from the class that passed the
    gate.
    """

    slug: str
    connector: type[BaseConnector]
    platform: Platform
    category: SourceCategory
    auth_type: AuthType
    version: str
    requires_tos_review: bool


_REGISTRY: dict[str, ConnectorRegistration] = {}
_ENABLED: set[str] = set()


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def register(connector: type[BaseConnector]) -> type[BaseConnector]:
    """Class decorator: validate a connector's declaration and record it.

    Returns the class unchanged, so `@register` sits above the class definition
    without altering it.

    Raises `ConnectorConfigurationError` -- a `PermanentError` -- on any invalid
    declaration. Raising at import means the process that would have run a
    misdeclared connector never starts, which is the cheapest possible place to
    find out.
    """
    if not isinstance(connector, type) or not issubclass(connector, BaseConnector):
        raise ConnectorConfigurationError(
            f"{connector!r} is not a BaseConnector subclass; the registry hands "
            "classes to a runtime that calls from_config/authenticate/fetch/normalize"
        )

    missing_methods = sorted(getattr(connector, "__abstractmethods__", ()))
    if missing_methods:
        raise ConnectorConfigurationError(
            f"{connector.__name__} does not implement {missing_methods}; it cannot "
            "be instantiated, so registering it would defer a TypeError to the "
            "first scheduled run",
            connector=getattr(connector, "slug", None),
        )

    slug = _validated_slug(connector)
    platform = _validated_enum(connector, "platform", Platform, slug)
    category = _validated_enum(connector, "category", SourceCategory, slug)
    auth_type = _validated_enum(connector, "auth_type", AuthType, slug)

    expected = PLATFORM_CATEGORY.get(platform, SourceCategory.UNKNOWN)
    if category is not expected:
        raise ConnectorConfigurationError(
            f"connector {slug!r} declares category {category.value!r} but platform "
            f"{platform.value!r} belongs to {expected.value!r}. This is the same "
            "disagreement Signal rejects per record at runtime "
            "(models/signal.py::_check_source_matches_platform); catching it at "
            "import costs a stack trace instead of a sync window",
            connector=slug,
        )

    incumbent = _REGISTRY.get(slug)
    if incumbent is not None:
        raise ConnectorConfigurationError(
            f"slug {slug!r} is already registered to "
            f"{incumbent.connector.__module__}.{incumbent.connector.__name__}; "
            f"{connector.__module__}.{connector.__name__} cannot claim it. Slugs key "
            "cursors, rate-limit buckets and credential rows, so a silent overwrite "
            "would point one connector's resume state at another connector's data",
            connector=slug,
        )

    _REGISTRY[slug] = ConnectorRegistration(
        slug=slug,
        connector=connector,
        platform=platform,
        category=category,
        auth_type=auth_type,
        version=str(getattr(connector, "version", "0.0.0")),
        requires_tos_review=bool(getattr(connector, "requires_tos_review", False)),
    )
    return connector


def unregister(slug: str) -> None:
    """Remove a registration, if present. Idempotent.

    Exists for test teardown and for replacing a class during development. It is
    idempotent because its callers are `finally` blocks, and a teardown helper
    that raises when the setup it was undoing never happened turns one failure
    into two.
    """
    key = _normalize(slug)
    _REGISTRY.pop(key, None)
    _ENABLED.discard(key)


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #


def get(slug: str) -> type[BaseConnector]:
    """Look up a connector class by slug.

    Raises `ConnectorConfigurationError` naming the known slugs, because the
    caller is nearly always an operator who typed one into
    `POST /api/v1/connectors/sync` or `scripts/sync_connector.py`, and a bare
    `KeyError: 'redit'` does not tell them they are one letter out.
    """
    registration = _REGISTRY.get(_normalize(slug))
    if registration is None:
        raise ConnectorConfigurationError(
            f"no connector registered under slug {slug!r}; known slugs: {slugs()}",
            connector=slug,
        )
    return registration.connector


def all() -> dict[str, type[BaseConnector]]:
    """Every registered connector, slug -> class, in slug order.

    Shadows the builtin inside this module deliberately: `registry.all()` is how
    `docs/connector-spec.md` §1.1 names the operation, and no code here calls the
    builtin.

    A copy: the registry is process-global mutable state, and handing out the
    live dict would let a caller's `.pop()` unregister a connector for every
    other caller in the process.
    """
    return {slug: entry.connector for slug, entry in sorted(_REGISTRY.items())}


def registrations() -> tuple[ConnectorRegistration, ...]:
    """Every validated declaration, in slug order.

    What the scheduler plans from: it needs `platform`, `auth_type` and
    `requires_tos_review` without instantiating anything.
    """
    return tuple(entry for _, entry in sorted(_REGISTRY.items()))


def slugs() -> tuple[str, ...]:
    """Registered slugs, sorted."""
    return tuple(sorted(_REGISTRY))


def by_category(category: SourceCategory) -> tuple[type[BaseConnector], ...]:
    """Every connector in one Design Doc §5 category, in slug order.

    Sorted rather than insertion-ordered so a caller that fans out over a
    category does so in the same order on every replica -- an unordered fan-out
    makes a shared provider quota get consumed in a different order per worker,
    which is how one account starves reproducibly-but-unexplainably.
    """
    return tuple(
        entry.connector
        for _, entry in sorted(_REGISTRY.items())
        if entry.category is category
    )


# --------------------------------------------------------------------------- #
# Enablement (gate 2)
# --------------------------------------------------------------------------- #


def enable(slug: str) -> None:
    """Mark a connector as runnable, refusing anything pending legal review.

    Raises `ConnectorConfigurationError` when `requires_tos_review` is set. See
    the module docstring for why there is no override argument: `docs/connector-
    spec.md` open question 7 records that "there is no defined owner or artifact
    for clearing `requires_tos_review`", and until there is, the flag is the only
    thing standing between a scraper and production traffic.
    """
    key = _normalize(slug)
    registration = _REGISTRY.get(key)
    if registration is None:
        raise ConnectorConfigurationError(
            f"cannot enable unknown connector {slug!r}; known slugs: {slugs()}",
            connector=slug,
        )
    if registration.requires_tos_review:
        raise ConnectorConfigurationError(
            f"connector {key!r} declares requires_tos_review=True and cannot be "
            "enabled. It has no viable official API for this use case, so running "
            "it means scraping, which needs a documented ToS/robots.txt review "
            "first (docs/connector-spec.md §9). Clear the flag on the class in a "
            "reviewed change once that review exists -- there is deliberately no "
            "runtime override",
            connector=key,
            details={"platform": registration.platform.value},
        )
    _ENABLED.add(key)


def disable(slug: str) -> None:
    """Stop a connector from being scheduled. Idempotent, and never refuses.

    Asymmetric with `enable()` on purpose: turning a source *off* must work even
    when its declaration is broken, because that is exactly when an operator
    needs it to.
    """
    _ENABLED.discard(_normalize(slug))


def is_enabled(slug: str) -> bool:
    """Whether this connector has passed the enablement gate."""
    return _normalize(slug) in _ENABLED


def enabled() -> tuple[str, ...]:
    """Enabled slugs, sorted."""
    return tuple(sorted(_ENABLED))


def create(slug: str, ctx: SyncContext, credentials: Credentials) -> BaseConnector:
    """Instantiate a connector through its own `from_config`.

    Re-checks `requires_tos_review` rather than trusting `enable()` to have been
    called. A gate that only guards one of two doors is decoration: without this,
    `create()` is a complete bypass, and it is the shorter call.

    Performs no I/O -- `from_config` is specified not to (`connectors/base.py`) --
    so the caller still has to `authenticate()`.
    """
    key = _normalize(slug)
    registration = _REGISTRY.get(key)
    if registration is None:
        raise ConnectorConfigurationError(
            f"no connector registered under slug {slug!r}; known slugs: {slugs()}",
            connector=slug,
        )
    if registration.requires_tos_review:
        raise ConnectorConfigurationError(
            f"connector {key!r} declares requires_tos_review=True and cannot be "
            "instantiated; see docs/connector-spec.md §9",
            connector=key,
        )
    return registration.connector.from_config(ctx, credentials)


# --------------------------------------------------------------------------- #
# Declaration validation
# --------------------------------------------------------------------------- #


def _validated_slug(connector: type[BaseConnector]) -> str:
    """Read and check `slug`.

    `BaseConnector` declares it as a bare `ClassVar[str]` annotation with no
    value, so a class that forgot it has no attribute at all rather than an empty
    one -- which is why this is a presence check and not a truthiness check.
    """
    slug = getattr(connector, "slug", None)
    if not isinstance(slug, str) or not slug:
        raise ConnectorConfigurationError(
            f"{connector.__name__} declares no slug; the registry, the cursor row, "
            "the rate-limit bucket and the credential row are all keyed by it"
        )
    # Checked verbatim, *not* after `_normalize`. Lookup is lenient because an
    # operator types the slug by hand; registration is strict because the class
    # keeps its own copy -- `BaseConnector.rate_limit_keys()` builds
    # `os:rl:{self.slug}` from it. Storing a normalized key for a class that
    # carries `"Reddit"` would give the registry and the limiter two different
    # buckets for one connector.
    if not SLUG_PATTERN.match(slug):
        raise ConnectorConfigurationError(
            f"slug {slug!r} must match {SLUG_PATTERN.pattern} -- it is interpolated "
            "into Redis keys and Kafka partition keys, and the class carries its own "
            "copy of it into both",
            connector=slug,
        )
    return slug


def _validated_enum[E: (Platform, SourceCategory, AuthType)](
    connector: type[BaseConnector], name: str, enum: type[E], slug: str
) -> E:
    """Read one enum-valued `ClassVar` and check it is really that enum member.

    `isinstance(value, enum)` rather than a truthiness check, because these are
    `StrEnum`s: a class that declared `platform = "reddit"` would pass every
    string test, compare equal to `Platform.REDDIT`, and then fail only where
    something calls `.value` on it.

    `UNKNOWN` is refused for all three. It is the *reader's* fallback for a value
    written by a newer producer (`models/base.py::TolerantStrEnum`), never a
    legitimate declaration -- a connector on platform `unknown` derives every
    Signal id under a name no connector owns.
    """
    value = getattr(connector, name, None)
    if value is None:
        raise ConnectorConfigurationError(
            f"connector {slug!r} declares no {name}; the scheduler reads the "
            "ClassVar block before instantiating anything",
            connector=slug,
        )
    if not isinstance(value, enum):
        raise ConnectorConfigurationError(
            f"connector {slug!r} declares {name}={value!r}, which is a "
            f"{type(value).__name__} and not a {enum.__name__} member",
            connector=slug,
        )
    if value.value == "unknown":
        raise ConnectorConfigurationError(
            f"connector {slug!r} declares {name}={enum.__name__}.UNKNOWN; that "
            "member exists so readers tolerate values from newer producers, not "
            "so a connector can decline to say what it is",
            connector=slug,
        )
    return value


def _normalize(slug: str) -> str:
    """Case-fold and trim a slug for lookup.

    Registered slugs are lower-case by construction, so this only ever helps the
    human who typed `--slug Reddit` on the command line.
    """
    return slug.strip().casefold()
