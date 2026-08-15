"""Deciding which accounts are the same human, and refusing to guess quietly.

The asymmetry this file protects: a *missed* link makes an answer visibly
incomplete -- somebody notices their Slack activity is absent and says so. A
*wrong* link makes an answer confidently wrong -- another person's commits appear
under your name, the totals still add up, and nothing looks broken. So the tests
weight heavily towards "did it refuse to act", not "did it find the match".
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.core.exceptions import ConflictError, NotFoundError
from models.artifact import person_id
from models.enums import Platform
from models.identity import DEFAULT_CONFIDENCE, IdentityLink, LinkMethod, identity_id
from models.orm.artifact import PersonRow
from services.identity_service import SUGGESTION_FLOOR, IdentityService

pytestmark = pytest.mark.unit


@pytest.fixture
def factory(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=orm_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


@pytest.fixture
def service(factory) -> IdentityService:
    return IdentityService(factory, tenant_id="local")


async def add_person(
    factory,
    platform: Platform,
    external: str,
    *,
    handle: str | None = None,
    display_name: str | None = None,
    email: str | None = None,
    is_bot: bool = False,
) -> str:
    pid = person_id(platform, external)
    async with factory() as session:
        session.add(
            PersonRow(
                id=pid,
                tenant_id="local",
                platform=platform,
                external_id=external,
                handle=handle,
                display_name=display_name,
                email=email,
                is_bot=is_bot,
            )
        )
        await session.commit()
    return pid


class TestTheGuardOnConfidence:
    """`method` and `confidence` say the same thing, so they may not disagree."""

    def test_a_confirmed_link_must_carry_full_confidence(self) -> None:
        with pytest.raises(ValueError, match=r"confidence 1\.0"):
            IdentityLink(
                identity_id="i",
                person_id="p",
                platform=Platform.GITHUB,
                method=LinkMethod.CONFIRMED,
                confidence=0.6,
            )

    def test_full_confidence_is_reserved_for_confirmation(self) -> None:
        """A handle match at 1.0 is a guess wearing a confirmation's name, and no
        reader downstream could tell the difference."""
        with pytest.raises(ValueError, match="reserved for 'confirmed'"):
            IdentityLink(
                identity_id="i",
                person_id="p",
                platform=Platform.GITHUB,
                method=LinkMethod.HANDLE,
                confidence=1.0,
            )

    def test_every_method_has_a_default_worth(self) -> None:
        assert set(DEFAULT_CONFIDENCE) == set(LinkMethod)

    def test_the_defaults_are_spread_not_clustered(self) -> None:
        """These numbers are read by a person deciding whether to accept a
        suggestion. Three methods all scoring 0.8-something tells them nothing."""
        inferred = sorted(v for m, v in DEFAULT_CONFIDENCE.items() if not m.is_confirmed)
        assert all(b - a >= 0.2 for a, b in pairwise(inferred))


class TestSuggesting:
    async def test_a_matching_handle_across_platforms_is_suggested(self, factory, service) -> None:
        github = await add_person(factory, Platform.GITHUB, "U_gh", handle="parth0124")
        await service.create_identity(display_name="Parth", person_id=github)
        await add_person(factory, Platform.SLACK, "U_sl", handle="parth0124")

        suggestions = await service.suggest()

        assert len(suggestions) == 1
        assert suggestions[0].method is LinkMethod.HANDLE
        assert suggestions[0].evidence == "parth0124"

    async def test_case_and_spacing_do_not_hide_an_exact_match(self, factory, service) -> None:
        """GitHub treats logins case-insensitively itself, so a case-sensitive
        comparison reports no evidence where the evidence is exact."""
        github = await add_person(factory, Platform.GITHUB, "U_gh", handle="Parth0124")
        await service.create_identity(display_name="Parth", person_id=github)
        await add_person(factory, Platform.SLACK, "U_sl", handle=" parth0124 ")

        assert len(await service.suggest()) == 1

    async def test_email_beats_handle_when_both_match(self, factory, service) -> None:
        """Reporting the weaker signal beside the stronger makes the queue longer
        without making it more informative."""
        github = await add_person(
            factory, Platform.GITHUB, "U_gh", handle="p", email="p@example.com"
        )
        await service.create_identity(display_name="Parth", person_id=github)
        await add_person(factory, Platform.SLACK, "U_sl", handle="p", email="p@example.com")

        assert (await service.suggest())[0].method is LinkMethod.EMAIL

    async def test_two_accounts_on_the_same_platform_are_never_suggested(
        self, factory, service
    ) -> None:
        """Two GitHub logins are two people -- the platform's own uniqueness
        already settled it, and a shared display name says nothing."""
        first = await add_person(factory, Platform.GITHUB, "U_1", display_name="Alex Chen")
        await service.create_identity(display_name="Alex", person_id=first)
        await add_person(factory, Platform.GITHUB, "U_2", display_name="Alex Chen")

        assert await service.suggest() == []

    async def test_bots_are_left_out_of_matching_entirely(self, factory, service) -> None:
        """Fusing two unrelated `ci-bot` accounts is the likeliest wrong merge in
        the system, and it buys nothing."""
        github = await add_person(
            factory, Platform.GITHUB, "U_bot", handle="dependabot", is_bot=True
        )
        await service.create_identity(display_name="dependabot", person_id=github, is_bot=True)
        await add_person(factory, Platform.SLACK, "U_bot2", handle="dependabot", is_bot=True)

        assert await service.suggest() == []

    async def test_suggesting_writes_nothing(self, factory, service) -> None:
        """The line the whole design rests on: proposing and deciding are separate,
        and only a person does the second."""
        github = await add_person(factory, Platform.GITHUB, "U_gh", handle="parth")
        await service.create_identity(display_name="Parth", person_id=github)
        slack = await add_person(factory, Platform.SLACK, "U_sl", handle="parth")

        await service.suggest()

        assert slack in [p.id for p in await service.unlinked()]

    async def test_weak_evidence_stays_below_the_floor(self, factory, service) -> None:
        github = await add_person(factory, Platform.GITHUB, "U_gh", display_name="Alex Chen")
        await service.create_identity(display_name="Alex", person_id=github)
        await add_person(factory, Platform.SLACK, "U_sl", display_name="Alex Chen")

        suggestions = await service.suggest()
        assert all(s.confidence >= SUGGESTION_FLOOR for s in suggestions)

    async def test_nothing_in_common_produces_nothing(self, factory, service) -> None:
        github = await add_person(factory, Platform.GITHUB, "U_gh", handle="parth")
        await service.create_identity(display_name="Parth", person_id=github)
        await add_person(factory, Platform.SLACK, "U_sl", handle="someone-else")

        assert await service.suggest() == []


class TestLinking:
    async def test_an_account_cannot_be_moved_between_humans_silently(
        self, factory, service
    ) -> None:
        """Moving it would take its whole history with it, and "some of my commits
        vanished" is not a symptom anybody traces to a link command from last
        week."""
        first = await add_person(factory, Platform.GITHUB, "U_1", handle="a")
        second = await add_person(factory, Platform.SLACK, "U_2", handle="b")
        one = await service.create_identity(display_name="One", person_id=first)
        await service.create_identity(display_name="Two", person_id=second)

        with pytest.raises(ConflictError, match="already belongs"):
            await service.link(person_id=second, identity=one)

    async def test_relinking_to_the_same_human_is_a_no_op(self, factory, service) -> None:
        """Idempotent, because the obvious response to an unclear error is to run
        the command again."""
        person = await add_person(factory, Platform.GITHUB, "U_1", handle="a")
        identity = await service.create_identity(display_name="One", person_id=person)

        await service.link(person_id=person, identity=identity)

        assert len((await service.list_identities())[0].accounts) == 1

    async def test_a_link_records_how_it_was_decided(self, factory, service) -> None:
        first = await add_person(factory, Platform.GITHUB, "U_1", handle="a")
        second = await add_person(factory, Platform.SLACK, "U_2", handle="a")
        identity = await service.create_identity(display_name="One", person_id=first)

        await service.link(person_id=second, identity=identity, method=LinkMethod.HANDLE)

        view = (await service.list_identities())[0]
        methods = {a.method for a in view.accounts}
        assert methods == {LinkMethod.CONFIRMED, LinkMethod.HANDLE}
        assert view.needs_review is True

    async def test_unlinking_leaves_the_human_and_the_account_alone(self, factory, service) -> None:
        first = await add_person(factory, Platform.GITHUB, "U_1", handle="a")
        second = await add_person(factory, Platform.SLACK, "U_2", handle="a")
        identity = await service.create_identity(display_name="One", person_id=first)
        await service.link(person_id=second, identity=identity)

        await service.unlink(second)

        assert await service.people_for(identity) == [first]
        assert second in [p.id for p in await service.unlinked()]

    async def test_unlinking_something_unattached_is_reported(self, service) -> None:
        with pytest.raises(NotFoundError):
            await service.unlink("per_nobody")

    async def test_linking_an_unknown_account_is_reported(self, factory, service) -> None:
        person = await add_person(factory, Platform.GITHUB, "U_1", handle="a")
        identity = await service.create_identity(display_name="One", person_id=person)

        with pytest.raises(NotFoundError):
            await service.link(person_id="per_ghost", identity=identity)


class TestAdopting:
    async def test_every_unattached_account_becomes_its_own_human(self, factory, service) -> None:
        """Not a merge -- the opposite. Two accounts are two people until there is
        a reason to think otherwise."""
        await add_person(factory, Platform.GITHUB, "U_1", handle="a")
        await add_person(factory, Platform.SLACK, "U_2", handle="b")

        assert await service.adopt_unlinked() == 2
        assert len(await service.list_identities()) == 2
        assert await service.unlinked() == []

    async def test_adopting_twice_creates_nothing_the_second_time(self, factory, service) -> None:
        """The ids are derived from the seed account, so a re-run lands on the rows
        that already exist rather than beside them."""
        await add_person(factory, Platform.GITHUB, "U_1", handle="a")

        await service.adopt_unlinked()
        assert await service.adopt_unlinked() == 0
        assert len(await service.list_identities()) == 1

    async def test_a_bot_account_stays_a_bot(self, factory, service) -> None:
        await add_person(factory, Platform.GITHUB, "U_b", handle="dependabot", is_bot=True)
        await service.adopt_unlinked()

        assert (await service.list_identities())[0].is_bot is True

    async def test_creating_a_second_identity_for_one_account_is_refused(
        self, factory, service
    ) -> None:
        person = await add_person(factory, Platform.GITHUB, "U_1", handle="a")
        await service.create_identity(display_name="One", person_id=person)

        with pytest.raises(ConflictError, match="already belongs"):
            await service.create_identity(display_name="Duplicate", person_id=person)


class TestReading:
    async def test_a_human_reports_every_platform_they_appear_on(self, factory, service) -> None:
        """The join every cross-source question goes through."""
        first = await add_person(factory, Platform.GITHUB, "U_1", handle="a")
        second = await add_person(factory, Platform.SLACK, "U_2", handle="a")
        identity = await service.create_identity(display_name="One", person_id=first)
        await service.link(person_id=second, identity=identity)

        view = (await service.list_identities())[0]
        assert set(view.platforms) == {Platform.GITHUB, Platform.SLACK}
        assert set(await service.people_for(identity)) == {first, second}

    async def test_a_fully_confirmed_human_needs_no_review(self, factory, service) -> None:
        person = await add_person(factory, Platform.GITHUB, "U_1", handle="a")
        await service.create_identity(display_name="One", person_id=person)

        assert (await service.list_identities())[0].needs_review is False

    async def test_an_empty_database_returns_nothing_rather_than_failing(self, service) -> None:
        assert await service.list_identities() == []
        assert await service.unlinked() == []
        assert await service.suggest() == []


class TestIdentityIds:
    def test_the_same_seed_gives_the_same_id(self) -> None:
        assert identity_id("local", "per_1") == identity_id("local", "per_1")

    def test_tenants_do_not_collide(self) -> None:
        assert identity_id("a", "per_1") != identity_id("b", "per_1")

    def test_an_empty_seed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            identity_id("local", "")
