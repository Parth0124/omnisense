"""Unit tests for `connectors/registry.py`.

The registry is a dictionary; everything worth testing is what it refuses to put
in that dictionary. Two gates, and the tests below are organized around them:

**Declaration validity, at import time.** The scheduler reads a connector's
`ClassVar` block before instantiating anything (`docs/connector-spec.md` §3), so
a class whose `platform` disagrees with its `category` is not a typo caught by a
test -- it is a run that fetches four thousand records and has every one of them
rejected by `Signal._check_source_matches_platform`, after the quota is spent.

**The legal review, at enable time.** `requires_tos_review = True` marks sources
with no viable official API for this use case (§9). `enable()` must refuse them,
and so must `create()` -- a gate on one of two doors is decoration.

The registry is process-global mutable state, so every test here runs against a
snapshot that is restored afterwards. Without that, one test's connector is
visible to every later test's `all()`, and the suite passes or fails by ordering.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Self

import pytest

from connectors import registry
from connectors.base import BaseConnector
from connectors.exceptions import ConnectorConfigurationError, PermanentError
from connectors.protocol import Credentials, Cursor, FetchPage, RawRecord, SyncContext
from models.enums import AuthType, Platform, SourceCategory
from models.signal import Signal

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_registry() -> Iterator[None]:
    """Snapshot and restore the process-global registry around every test.

    `unregister()` alone is not enough: it would also have to know which
    connectors the *application* had registered before the test started, and a
    teardown that guesses at that either leaks test classes into later tests or
    deletes production registrations. Copying the dicts is the only version that
    is correct regardless of what ran first.

    The registry is also **emptied for the duration of the test**, not merely
    restored afterwards. `connectors/__init__.py` registers the four shipped
    connectors on import, so `rss` and `reddit` are already taken by the time any
    test runs -- and the tests below deliberately register doubles under exactly
    those slugs to exercise the duplicate-slug guard. Without the clear, every
    such test fails against the real registration instead of the condition it is
    trying to assert. Starting from empty is also what makes each test independent
    of whether some earlier module happened to import `connectors`.
    """
    saved_registry = dict(registry._REGISTRY)
    saved_enabled = set(registry._ENABLED)
    registry._REGISTRY.clear()
    registry._ENABLED.clear()
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved_registry)
        registry._ENABLED.clear()
        registry._ENABLED.update(saved_enabled)


class ConnectorMethods(BaseConnector):
    """The four abstract methods, and deliberately no declaration block.

    Split from `DemoConnector` so a test can build a connector that is complete
    in every way *except* the field under test. `BaseConnector` declares
    `slug`/`platform`/`category`/`auth_type` as bare annotations with no values,
    so "forgot to declare it" means the attribute does not exist -- which cannot
    be simulated by deleting one off a subclass that inherits it.
    """

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        self.authenticated = 0

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        self.authenticated += 1

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        yield FetchPage(records=[], cursor=cursor)

    async def normalize(self, record: RawRecord) -> Signal | None:
        return None


class DemoConnector(ConnectorMethods):
    """A complete, instantiable connector. Subclassed per test to vary one field."""

    slug = "demo"
    platform = Platform.RSS
    category = SourceCategory.NEWS
    auth_type = AuthType.NONE


def connector(name: str, **declaration: Any) -> type[BaseConnector]:
    """Build a connector class with the given declaration overrides."""
    return type(name, (DemoConnector,), declaration)


def ctx() -> SyncContext:
    return SyncContext(connector_slug="demo", account_id="acct_1", run_id="run_1")


class TestRegistration:
    """The happy path, and what the registry keeps once a class passes."""

    def test_returns_the_class_unchanged(self) -> None:
        """`@register` sits above a class definition; a decorator that returned a
        wrapper would break `issubclass` checks and `super()` in subclasses."""
        cls = connector("Rss", slug="rss")
        assert registry.register(cls) is cls

    def test_records_the_validated_declaration(self) -> None:
        """`by_category()` and the scheduler answer from this rather than reading
        the class live, so a class mutated after registration cannot answer
        differently from the class that passed the gate."""
        registry.register(connector("Rss", slug="rss", version="2.1.0"))
        entry = next(e for e in registry.registrations() if e.slug == "rss")
        assert (entry.platform, entry.category) == (Platform.RSS, SourceCategory.NEWS)
        assert (entry.auth_type, entry.version) == (AuthType.NONE, "2.1.0")
        assert entry.requires_tos_review is False

    def test_duplicate_slugs_are_an_error_not_an_overwrite(self) -> None:
        """Slugs key cursors, rate-limit buckets and credential rows. A silent
        overwrite would point one connector's resume state at another's data."""
        registry.register(connector("First", slug="rss"))
        with pytest.raises(ConnectorConfigurationError, match="already registered"):
            registry.register(connector("Second", slug="rss"))

    def test_the_duplicate_error_names_the_incumbent(self) -> None:
        registry.register(connector("First", slug="rss"))
        with pytest.raises(ConnectorConfigurationError, match="First"):
            registry.register(connector("Second", slug="rss"))

    def test_unregister_is_idempotent(self) -> None:
        """Its callers are `finally` blocks, and a teardown helper that raises
        when the setup it was undoing never happened turns one failure into two."""
        registry.register(connector("Rss", slug="rss"))
        registry.unregister("rss")
        registry.unregister("rss")
        assert "rss" not in registry.slugs()

    def test_registration_failures_are_permanent_errors(self) -> None:
        """The runtime routes by exception family (`docs/connector-spec.md` §6);
        a misdeclared connector must never be classified as retryable."""
        with pytest.raises(PermanentError):
            registry.register(connector("NoSlug", slug=""))


class TestDeclarationValidation:
    """Gate 1. Every failure here is cheaper at import than at 3am."""

    def test_platform_must_match_its_category(self) -> None:
        """The same disagreement `Signal` rejects per record at runtime. Catching
        it at import costs a stack trace; catching it at runtime costs a sync
        window and the quota spent filling it."""
        with pytest.raises(ConnectorConfigurationError, match="belongs to 'social'"):
            registry.register(
                connector("Wrong", slug="wrong", platform=Platform.REDDIT)
            )

    def test_a_missing_slug_is_refused(self) -> None:
        """`BaseConnector` declares `slug` as a bare annotation with no value, so
        a class that forgot it has no attribute at all -- which is why the check
        is for presence and not for truthiness."""

        class NoSlug(ConnectorMethods):
            platform = Platform.RSS
            category = SourceCategory.NEWS
            auth_type = AuthType.NONE

        with pytest.raises(ConnectorConfigurationError, match="declares no slug"):
            registry.register(NoSlug)

    def test_a_missing_platform_is_refused(self) -> None:
        """The scheduler reads the `ClassVar` block before instantiating
        anything, so an absent declaration is not discovered by running it."""

        class NoPlatform(ConnectorMethods):
            slug = "no_platform"
            category = SourceCategory.NEWS
            auth_type = AuthType.NONE

        with pytest.raises(ConnectorConfigurationError, match="declares no platform"):
            registry.register(NoPlatform)

    @pytest.mark.parametrize("slug", ["Reddit", "reddit:posts", "reddit posts", "9gag", "-rss"])
    def test_slugs_that_would_corrupt_a_redis_key_are_refused(self, slug: str) -> None:
        """The slug is interpolated into `os:rl:{slug}:{account_id}` and into the
        Kafka partition key. A colon splits one bucket into two and quietly
        doubles the effective request budget."""
        with pytest.raises(ConnectorConfigurationError, match="must match"):
            registry.register(connector("Bad", slug=slug))

    def test_a_platform_string_is_not_a_platform(self) -> None:
        """`Platform` is a `StrEnum`: `"rss"` passes every string test, compares
        equal to `Platform.RSS`, and fails only where something calls `.value`."""
        with pytest.raises(ConnectorConfigurationError, match="not a Platform member"):
            registry.register(connector("Stringly", slug="stringly", platform="rss"))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("platform", Platform.UNKNOWN),
            ("category", SourceCategory.UNKNOWN),
            ("auth_type", AuthType.UNKNOWN),
        ],
    )
    def test_unknown_enum_members_are_refused(self, field: str, value: Any) -> None:
        """`UNKNOWN` exists so readers tolerate a value written by a newer
        producer, not so a connector can decline to say what it is. A connector on
        platform `unknown` derives every Signal id under a name nobody owns."""
        with pytest.raises(ConnectorConfigurationError, match="UNKNOWN"):
            registry.register(connector("Vague", slug="vague", **{field: value}))

    def test_an_abstract_connector_is_refused(self) -> None:
        """Registering it would defer a `TypeError` to the first scheduled run,
        which is the worst possible place to discover a missing `normalize`."""

        class Half(BaseConnector):
            slug = "half"
            platform = Platform.RSS
            category = SourceCategory.NEWS
            auth_type = AuthType.NONE

        with pytest.raises(ConnectorConfigurationError, match="does not implement"):
            registry.register(Half)  # type: ignore[type-abstract]

    def test_a_non_connector_is_refused(self) -> None:
        with pytest.raises(ConnectorConfigurationError, match="not a BaseConnector"):
            registry.register(dict)  # type: ignore[arg-type]


class TestLookup:
    """Discovery: by slug, wholesale, and by category."""

    @pytest.fixture(autouse=True)
    def _catalogue(self) -> None:
        registry.register(connector("Rss", slug="rss"))
        registry.register(
            connector(
                "Reddit",
                slug="reddit",
                platform=Platform.REDDIT,
                category=SourceCategory.SOCIAL,
                auth_type=AuthType.OAUTH2,
            )
        )
        registry.register(connector("Gdelt", slug="gdelt", platform=Platform.GDELT))

    def test_get_returns_the_class(self) -> None:
        assert registry.get("rss").__name__ == "Rss"

    def test_an_unknown_slug_names_the_known_ones(self) -> None:
        """The caller is nearly always an operator who typed a slug into
        `POST /connectors/sync`; a bare `KeyError: 'redit'` does not tell them
        they are one letter out."""
        with pytest.raises(ConnectorConfigurationError, match="'reddit'"):
            registry.get("redit")

    def test_lookup_tolerates_the_case_an_operator_typed(self) -> None:
        assert registry.get("  RSS ") is registry.get("rss")

    def test_all_returns_a_copy(self) -> None:
        """The registry is process-global; handing out the live dict would let one
        caller's `.pop()` unregister a connector for everyone."""
        catalogue = registry.all()
        catalogue.pop("rss")
        assert "rss" in registry.all()

    def test_listings_are_sorted(self) -> None:
        """An unordered fan-out consumes a shared provider quota in a different
        order on every replica, which is how one account starves
        reproducibly-but-unexplainably."""
        assert registry.slugs() == ("gdelt", "reddit", "rss")
        assert list(registry.all()) == ["gdelt", "reddit", "rss"]

    def test_by_category_selects_and_orders(self) -> None:
        news = [cls.__name__ for cls in registry.by_category(SourceCategory.NEWS)]
        assert news == ["Gdelt", "Rss"]
        assert [c.__name__ for c in registry.by_category(SourceCategory.SOCIAL)] == ["Reddit"]

    def test_an_empty_category_is_empty_not_an_error(self) -> None:
        assert registry.by_category(SourceCategory.ENTERPRISE) == ()


class TestEnablementGate:
    """Gate 2: `docs/connector-spec.md` §9, the ToS review."""

    @pytest.fixture(autouse=True)
    def _catalogue(self) -> None:
        registry.register(connector("Rss", slug="rss"))
        registry.register(
            connector(
                "Instagram",
                slug="instagram",
                platform=Platform.INSTAGRAM,
                category=SourceCategory.SOCIAL,
                auth_type=AuthType.OAUTH2,
                requires_tos_review=True,
            )
        )

    def test_a_clean_connector_enables(self) -> None:
        registry.enable("rss")
        assert registry.is_enabled("rss") and registry.enabled() == ("rss",)

    def test_a_connector_pending_review_is_refused(self) -> None:
        """Instagram, TikTok, LinkedIn and Amazon reviews have no lawful
        public-search path; running them means scraping."""
        with pytest.raises(ConnectorConfigurationError) as caught:
            registry.enable("instagram")
        assert not registry.is_enabled("instagram")

        message = str(caught.value)
        assert "requires_tos_review" in message
        assert "review" in message and "docs/connector-spec.md §9" in message

    def test_there_is_no_runtime_override(self) -> None:
        """An argument at a call site is not a legal review, and the moment one
        exists it appears in a config file and the gate is decorative. Clearing
        the flag on the class -- a diff, in a pull request -- is the path."""
        import inspect

        assert list(inspect.signature(registry.enable).parameters) == ["slug"]

    def test_create_re_checks_the_gate(self) -> None:
        """A gate on one of two doors is decoration, and `create()` is the shorter
        call."""
        with pytest.raises(ConnectorConfigurationError, match="requires_tos_review"):
            registry.create("instagram", ctx(), Credentials(account_id="acct_1"))

    def test_enabling_an_unknown_slug_is_an_error(self) -> None:
        with pytest.raises(ConnectorConfigurationError, match="unknown connector"):
            registry.enable("nope")

    def test_disable_never_refuses(self) -> None:
        """Asymmetric with `enable()` on purpose: turning a source off must work
        even when its declaration is broken, because that is exactly when an
        operator needs it to."""
        registry.disable("instagram")
        registry.disable("never-registered")
        assert registry.enabled() == ()

    def test_unregister_clears_the_enabled_flag(self) -> None:
        """Otherwise a slug re-registered later would inherit an enablement
        decision that was made about a different class."""
        registry.enable("rss")
        registry.unregister("rss")
        registry.register(connector("RssAgain", slug="rss"))
        assert not registry.is_enabled("rss")


class TestInstantiation:
    """`create()` is discovery plus `from_config`, and nothing else."""

    @pytest.fixture(autouse=True)
    def _catalogue(self) -> None:
        registry.register(connector("Rss", slug="rss"))

    def test_creates_through_from_config(self) -> None:
        instance = registry.create("rss", ctx(), Credentials(account_id="acct_1"))
        assert isinstance(instance, BaseConnector)
        assert instance.ctx.run_id == "run_1"

    def test_does_not_authenticate(self) -> None:
        """`from_config` must not perform I/O (`connectors/base.py`), so the
        caller still has to `authenticate()`. A registry that authenticated would
        make construction fail for a reason that deserves a retry."""
        instance = registry.create("rss", ctx(), Credentials(account_id="acct_1"))
        assert getattr(instance, "authenticated", None) == 0

    def test_creating_an_unknown_slug_names_the_known_ones(self) -> None:
        with pytest.raises(ConnectorConfigurationError, match="known slugs"):
            registry.create("nope", ctx(), Credentials(account_id="acct_1"))
