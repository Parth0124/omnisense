"""Contract conformance, checked over every shipped connector at once.

Written as a parametrised sweep rather than per-connector assertions because
the point is coverage: a new connector added to `SHIPPED` is automatically
subject to every rule here, and cannot ship having quietly skipped one. The
alternative -- a test file per connector -- means the twenty-fifth connector is
correct only if whoever wrote it remembered to copy all of these.

Nothing here makes a network call. These are properties of the *declaration*,
and every one of them has a failure mode that only appears in production:
a missing rate limit gets an IP blocked, a wrong category breaks retrieval
filters, an unbounded burst against an undocumented limiter fails silently.
"""

from __future__ import annotations

import inspect

import pytest

from connectors import SHIPPED, registry
from connectors.base import BaseConnector
from models.enums import AuthType, Platform, SourceCategory

pytestmark = pytest.mark.unit

IDS = [connector.slug for connector in SHIPPED]


class TestRegistration:
    def test_every_shipped_connector_is_registered(self) -> None:
        """A connector nobody registered is one `registry.get` cannot resolve.

        The failure is total and silent: `POST /connectors/{slug}/sync` returns
        404 for a connector that demonstrably works, because nothing imported
        its module.
        """
        assert set(registry.slugs()) == {c.slug for c in SHIPPED}

    def test_slugs_are_unique(self) -> None:
        slugs = [connector.slug for connector in SHIPPED]
        assert len(slugs) == len(set(slugs))

    def test_platforms_are_unique(self) -> None:
        """Two connectors sharing a platform would collide on signal identity.

        `signal_id` is `uuid5(namespace, f"{platform}:{native_id}")`, so two
        connectors emitting the same native id under one platform produce the
        same Signal id -- and the second silently overwrites the first.
        """
        platforms = [connector.platform for connector in SHIPPED]
        assert len(platforms) == len(set(platforms))


@pytest.mark.parametrize("connector", SHIPPED, ids=IDS)
class TestDeclaration:
    def test_subclasses_base(self, connector: type[BaseConnector]) -> None:
        assert issubclass(connector, BaseConnector)

    def test_declares_the_full_identity(self, connector: type[BaseConnector]) -> None:
        assert isinstance(connector.slug, str) and connector.slug
        assert isinstance(connector.platform, Platform)
        assert isinstance(connector.category, SourceCategory)
        assert isinstance(connector.auth_type, AuthType)
        assert isinstance(connector.version, str) and connector.version

    def test_no_unknown_enum_members(self, connector: type[BaseConnector]) -> None:
        """`UNKNOWN` means "a value this build does not recognise".

        Right for a reader tolerating a newer producer; meaningless on a
        connector, which is the producer. A connector declaring
        `Platform.UNKNOWN` would emit Signals nothing can filter for.
        """
        assert connector.platform is not Platform.UNKNOWN
        assert connector.category is not SourceCategory.UNKNOWN
        assert connector.auth_type is not AuthType.UNKNOWN

    def test_slug_matches_platform(self, connector: type[BaseConnector]) -> None:
        """Kept in step so an operator reading either can predict the other."""
        assert connector.slug == connector.platform.value

    def test_declares_a_rate_limit(self, connector: type[BaseConnector]) -> None:
        """Every connector talks to somebody else's server.

        Without a declared budget the limiter has nothing to enforce, and the
        first symptom is an IP block applied by hand days later.
        """
        policy = connector.rate_limit
        assert policy.requests_per_minute > 0
        assert policy.burst >= 1
        assert policy.concurrency >= 1

    def test_rate_limit_is_not_absurd(self, connector: type[BaseConnector]) -> None:
        """A four-figure rate is a copied default, not a measured one.

        Every provider in this catalogue documents a limit well below this, so a
        value above it means nobody looked it up.
        """
        assert connector.rate_limit.requests_per_minute <= 600

    def test_undocumented_limits_are_serialised(
        self, connector: type[BaseConnector]
    ) -> None:
        """`docs/connector-spec.md` §9.5: where a limit is undocumented, serialise.

        Expressed here as a weaker, checkable rule -- a connector that allows
        real concurrency must also allow more than a trickle, because
        `concurrency > 1` with a very low rate is a configuration that cannot
        actually run concurrently and signals a copy-paste.
        """
        policy = connector.rate_limit
        if policy.concurrency > 1:
            assert policy.requests_per_minute >= 20

    def test_implements_the_four_abstract_methods(
        self, connector: type[BaseConnector]
    ) -> None:
        """Inherited stubs would raise at the first sync, per record, in a worker."""
        for name in ("from_config", "authenticate", "fetch", "normalize"):
            own = getattr(connector, name)
            base = getattr(BaseConnector, name)
            assert own is not base, f"{connector.slug} does not implement {name}"
            assert not getattr(own, "__isabstractmethod__", False)

    def test_run_is_not_overridden(self, connector: type[BaseConnector]) -> None:
        """`BaseConnector.run` is `@final` and holds the watermark guard.

        A connector overriding it would bypass the compounding-rewind fix, which
        is invisible until cursors have drifted for weeks.
        """
        assert connector.run is BaseConnector.run

    def test_fetch_is_an_async_generator(self, connector: type[BaseConnector]) -> None:
        """`run()` iterates it. A coroutine returning a list would raise at the
        first `async for`, inside the worker, per sync."""
        assert inspect.isasyncgenfunction(connector.fetch)

    def test_normalize_is_a_coroutine(self, connector: type[BaseConnector]) -> None:
        assert inspect.iscoroutinefunction(connector.normalize)

    def test_documents_itself(self, connector: type[BaseConnector]) -> None:
        """A connector's docstring is where its provider's quirks are recorded.

        The overlap window, the paging direction, the tier required -- none of
        that is inferable from the code, and the next person to touch it has no
        other source.
        """
        module = inspect.getmodule(connector)
        assert module is not None and module.__doc__
        assert len(module.__doc__) > 200, f"{connector.slug} has a stub module docstring"
        assert "TODO" not in module.__doc__

    def test_overlap_is_bounded(self, connector: type[BaseConnector]) -> None:
        """Overlap re-reads a window on every poll; dedup collapses it.

        Unbounded, it becomes a full re-crawl each cycle. A day is already
        generous for a provider whose index lags.
        """
        assert 0 <= connector.overlap_seconds <= 86_400


class TestGatedConnectors:
    """Connectors with no lawful open surface must say so."""

    TOS_GATED = {"linkedin", "instagram", "tiktok", "amazon", "google_reviews"}

    def test_the_gated_set_is_flagged(self) -> None:
        """`requires_tos_review` is what lets the catalogue distinguish a policy
        refusal from an outage -- otherwise an operator debugs a decision."""
        flagged = {c.slug for c in SHIPPED if c.requires_tos_review}
        assert flagged == self.TOS_GATED

    def test_gated_connectors_require_a_credential(self) -> None:
        """None of them may be reachable without one.

        `AuthType.NONE` on a gated platform would mean the connector believes it
        can collect anonymously -- which for these five would only be true of a
        scraper.
        """
        for connector in SHIPPED:
            if connector.requires_tos_review:
                assert connector.auth_type is not AuthType.NONE, (
                    f"{connector.slug} is ToS-gated but declares no auth; the only "
                    "way to reach it without a credential is by scraping"
                )
