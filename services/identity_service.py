"""Work out which accounts are the same human -- and refuse to guess in silence.

Two operations, and the split between them is the whole design.

`suggest()` proposes. It compares accounts, scores the evidence, and returns
candidates. It writes nothing.

`link()` decides. It is called by a person, or by `adopt_unlinked()` for the
uncontroversial case of an account nobody else could plausibly be.

**Nothing merges two humans automatically, at any confidence.** That is not
caution for its own sake. A missed link makes an answer visibly incomplete -- a
person notices their Slack activity is absent and says so. A wrong link makes an
answer *confidently wrong*: somebody else's commits appear under your name, the
totals still add up, and nothing looks broken. The two failures are not
symmetric, and the cheap one is the one to prefer.

What the evidence is worth
--------------------------
| Signal            | Worth | Why not more                                    |
| ----------------- | ----- | ----------------------------------------------- |
| A person said so  | 1.00  | -- it is not evidence, it is a decision         |
| Same email        | 0.90  | shared team addresses exist; so do typos        |
| Same handle       | 0.60  | handles are reused across services by strangers |
| Same display name | 0.35  | "Alex Chen" is not an identifier                |

Bots are never suggested against humans. `dependabot` on GitHub and a `dependabot`
Slack app are arguably the same actor, but the merge buys nothing and the same
rule would happily fuse two unrelated `ci-bot` accounts.

Layer note: **L2 service.**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.exceptions import ConflictError, NotFoundError
from backend.core.logging import get_logger
from models.enums import Platform
from models.identity import DEFAULT_CONFIDENCE, LinkMethod, identity_id
from models.orm.artifact import PersonRow
from models.orm.identity import IdentityLinkRow, IdentityRow

__all__ = [
    "IdentityService",
    "IdentityView",
    "LinkedAccount",
    "Suggestion",
    "build_identity_service",
]

_log = get_logger(__name__)

DEFAULT_TENANT = "local"

SUGGESTION_FLOOR = 0.3
"""Below this, a suggestion is noise.

A review queue nobody finishes is the same as no review queue, and display-name
matches alone would fill one. The floor sits just under `DISPLAY_NAME` so that
signal still appears -- it is the weakest thing worth a person's glance, and the
first thing to drop if the queue gets long.
"""


@dataclass(frozen=True, slots=True)
class LinkedAccount:
    """One account, as seen from the human it belongs to."""

    person_id: str
    platform: Platform
    handle: str | None
    display_name: str | None
    method: LinkMethod
    confidence: float

    @property
    def is_confirmed(self) -> bool:
        return self.method.is_confirmed


@dataclass(frozen=True, slots=True)
class IdentityView:
    """A human and their accounts, for reading."""

    id: str
    display_name: str
    is_bot: bool
    accounts: tuple[LinkedAccount, ...] = ()

    @property
    def platforms(self) -> tuple[Platform, ...]:
        return tuple(dict.fromkeys(account.platform for account in self.accounts))

    @property
    def needs_review(self) -> bool:
        """Whether any account here was inferred rather than confirmed."""
        return any(not account.is_confirmed for account in self.accounts)


@dataclass(frozen=True, slots=True)
class Suggestion:
    """A proposed link. Never acted on without somebody saying so."""

    person_id: str
    identity_id: str
    identity_name: str
    method: LinkMethod
    confidence: float
    evidence: str
    """The matching value itself -- the shared email, the identical handle.

    Present because a person cannot judge "0.6, handle" but can judge
    "both are `parth`" in about a second.
    """


class IdentityService:
    """Groups platform accounts into humans."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id

    # ---------------------------------------------------------------- reads --

    async def list_identities(self) -> list[IdentityView]:
        """Every human, with their accounts. One query, not one per human."""
        async with self._session_factory() as session:
            identities = list(
                (
                    await session.execute(
                        select(IdentityRow)
                        .where(IdentityRow.tenant_id == self._tenant_id)
                        .order_by(IdentityRow.display_name)
                    )
                )
                .scalars()
                .all()
            )
            if not identities:
                return []

            rows = list(
                (
                    await session.execute(
                        select(IdentityLinkRow, PersonRow)
                        .join(PersonRow, PersonRow.id == IdentityLinkRow.person_id)
                        .where(
                            IdentityLinkRow.identity_id.in_([i.id for i in identities]),
                            IdentityLinkRow.tenant_id == self._tenant_id,
                        )
                    )
                ).all()
            )

        grouped: dict[str, list[LinkedAccount]] = {}
        for link, person in rows:
            grouped.setdefault(link.identity_id, []).append(
                LinkedAccount(
                    person_id=person.id,
                    platform=person.platform,
                    handle=person.handle,
                    display_name=person.display_name,
                    method=link.method,
                    confidence=link.confidence,
                )
            )

        return [
            IdentityView(
                id=identity.id,
                display_name=identity.display_name,
                is_bot=identity.is_bot,
                accounts=tuple(
                    sorted(grouped.get(identity.id, ()), key=lambda a: a.platform.value)
                ),
            )
            for identity in identities
        ]

    async def unlinked(self) -> list[PersonRow]:
        """Accounts belonging to no human yet. The work queue."""
        async with self._session_factory() as session:
            return list(
                (
                    await session.execute(
                        select(PersonRow)
                        .outerjoin(IdentityLinkRow, IdentityLinkRow.person_id == PersonRow.id)
                        .where(
                            PersonRow.tenant_id == self._tenant_id,
                            IdentityLinkRow.person_id.is_(None),
                        )
                        .order_by(PersonRow.platform, PersonRow.handle)
                    )
                )
                .scalars()
                .all()
            )

    # ----------------------------------------------------------- suggesting --

    async def suggest(self) -> list[Suggestion]:
        """Propose links for unlinked accounts. Writes nothing.

        Returns the *best* candidate per account rather than all of them. A queue
        offering three near-identical options per row is one a person abandons,
        and the second-best is one command away once the first is rejected.
        """
        async with self._session_factory() as session:
            unlinked = list(
                (
                    await session.execute(
                        select(PersonRow)
                        .outerjoin(IdentityLinkRow, IdentityLinkRow.person_id == PersonRow.id)
                        .where(
                            PersonRow.tenant_id == self._tenant_id,
                            IdentityLinkRow.person_id.is_(None),
                            # Bots are excluded on both sides. Fusing two
                            # unrelated `ci-bot` accounts is the likeliest wrong
                            # merge in the whole system and buys nothing.
                            PersonRow.is_bot.is_(False),
                        )
                    )
                )
                .scalars()
                .all()
            )
            if not unlinked:
                return []

            linked = list(
                (
                    await session.execute(
                        select(IdentityRow, PersonRow)
                        .join(IdentityLinkRow, IdentityLinkRow.identity_id == IdentityRow.id)
                        .join(PersonRow, PersonRow.id == IdentityLinkRow.person_id)
                        .where(
                            IdentityRow.tenant_id == self._tenant_id,
                            IdentityRow.is_bot.is_(False),
                        )
                    )
                ).all()
            )

        suggestions: list[Suggestion] = []
        for person in unlinked:
            best: Suggestion | None = None
            for identity, other in linked:
                # An account never matches another on the same platform: two
                # GitHub logins are two people, and the platform's own uniqueness
                # already settled it.
                if other.platform is person.platform:
                    continue
                scored = _score(person, identity, other)
                if scored and (best is None or scored.confidence > best.confidence):
                    best = scored
            if best and best.confidence >= SUGGESTION_FLOOR:
                suggestions.append(best)

        return sorted(suggestions, key=lambda s: s.confidence, reverse=True)

    # -------------------------------------------------------------- writing --

    async def create_identity(
        self, *, display_name: str, person_id: str, is_bot: bool = False
    ) -> str:
        """Start a human around one account, and attach it as confirmed.

        The seed account is confirmed rather than inferred: creating an identity
        *is* the assertion that this account is this human, so recording it as a
        guess would misstate what happened.
        """
        async with self._session_factory() as session, session.begin():
            person = await session.get(PersonRow, person_id)
            if person is None or person.tenant_id != self._tenant_id:
                raise NotFoundError.for_resource("person", person_id)

            existing = await session.get(IdentityLinkRow, person_id)
            if existing is not None:
                raise ConflictError(
                    f"{person.handle or person_id} already belongs to an identity.",
                    details={"person_id": person_id, "identity_id": existing.identity_id},
                )

            new_id = identity_id(self._tenant_id, person_id)
            if await session.get(IdentityRow, new_id) is None:
                session.add(
                    IdentityRow(
                        id=new_id,
                        tenant_id=self._tenant_id,
                        display_name=display_name,
                        primary_email=person.email,
                        is_bot=is_bot or person.is_bot,
                    )
                )
                # Flushed before the link is added, rather than left to the
                # commit. The factory sets `autoflush=False`, so without this both
                # rows reach the flush together and the link can be issued first
                # -- Postgres then rejects it against an identity that exists only
                # in the session. The failure is a foreign-key violation naming an
                # id that is very obviously about to be inserted, which reads as a
                # database fault rather than an ordering one.
                await session.flush()
            session.add(
                IdentityLinkRow(
                    person_id=person_id,
                    identity_id=new_id,
                    tenant_id=self._tenant_id,
                    platform=person.platform,
                    method=LinkMethod.CONFIRMED,
                    confidence=1.0,
                    confirmed_at=datetime.now(UTC),
                )
            )

        _log.info("identity.created", identity_id=new_id, person_id=person_id)
        return new_id

    async def link(
        self,
        *,
        person_id: str,
        identity: str,
        method: LinkMethod = LinkMethod.CONFIRMED,
        confirmed_by: str | None = None,
        note: str | None = None,
    ) -> None:
        """Attach an account to a human.

        Re-linking an account that is already attached elsewhere is refused
        rather than silently moved. Moving it would take its whole history with
        it, and "some of my commits vanished" is not a symptom anybody traces
        back to a link command they ran last week.
        """
        async with self._session_factory() as session, session.begin():
            person = await session.get(PersonRow, person_id)
            if person is None or person.tenant_id != self._tenant_id:
                raise NotFoundError.for_resource("person", person_id)
            target = await session.get(IdentityRow, identity)
            if target is None or target.tenant_id != self._tenant_id:
                raise NotFoundError.for_resource("identity", identity)

            existing = await session.get(IdentityLinkRow, person_id)
            if existing is not None:
                if existing.identity_id == identity:
                    return
                raise ConflictError(
                    f"{person.handle or person_id} already belongs to another identity. "
                    f"Detach it first:  omnisense people unlink {person_id}",
                    details={"person_id": person_id, "identity_id": existing.identity_id},
                )

            confirmed = method.is_confirmed
            session.add(
                IdentityLinkRow(
                    person_id=person_id,
                    identity_id=identity,
                    tenant_id=self._tenant_id,
                    platform=person.platform,
                    method=method,
                    confidence=DEFAULT_CONFIDENCE[method],
                    confirmed_at=datetime.now(UTC) if confirmed else None,
                    confirmed_by=confirmed_by if confirmed else None,
                    note=note,
                )
            )

        _log.info("identity.linked", identity_id=identity, person_id=person_id, method=method.value)

    async def unlink(self, person_id: str) -> None:
        """Detach an account. The human and every other account survive."""
        async with self._session_factory() as session, session.begin():
            # Fetched before deleting rather than relying on the statement's
            # `rowcount`: that attribute is `CursorResult`-only, and the async
            # `execute()` is typed as returning a plain `Result`. Reading it works
            # at runtime and does not type-check, which is exactly the kind of
            # thing that keeps working until the day a driver returns the other
            # shape.
            link = await session.get(IdentityLinkRow, person_id)
            if link is None or link.tenant_id != self._tenant_id:
                raise NotFoundError.for_resource("identity link", person_id)
            await session.execute(
                delete(IdentityLinkRow).where(IdentityLinkRow.person_id == person_id)
            )
        _log.info("identity.unlinked", person_id=person_id)

    async def adopt_unlinked(self) -> int:
        """Give every unattached account an identity of its own.

        Not a merge -- the opposite. Each account becomes a separate human,
        which is the honest default before anything is known: two accounts are
        two people until there is a reason to think otherwise.

        This is what makes the feature usable from the first run. Without it,
        every question about a person returns nothing until somebody has sat down
        and confirmed links one at a time.
        """
        created = 0
        for person in await self.unlinked():
            await self.create_identity(
                display_name=person.display_name or person.handle or person.external_id,
                person_id=person.id,
                is_bot=person.is_bot,
            )
            created += 1
        if created:
            _log.info("identity.adopted_unlinked", created=created)
        return created

    async def people_for(self, identity: str) -> list[str]:
        """Every `person_id` belonging to one human.

        The join every cross-source question goes through: an identity fans out
        to accounts, accounts to artifacts.
        """
        async with self._session_factory() as session:
            return list(
                (
                    await session.execute(
                        select(IdentityLinkRow.person_id).where(
                            IdentityLinkRow.identity_id == identity,
                            IdentityLinkRow.tenant_id == self._tenant_id,
                        )
                    )
                )
                .scalars()
                .all()
            )


def _score(person: PersonRow, identity: IdentityRow, other: PersonRow) -> Suggestion | None:
    """Best evidence that `person` is the same human as `other`, or nothing.

    Checked strongest first and returned on the first hit: a pair matching on both
    email and handle is an email match, and reporting the weaker signal beside it
    would make the queue longer without making it more informative.
    """
    if person.email and other.email and _same(person.email, other.email):
        method, evidence = LinkMethod.EMAIL, person.email
    elif person.handle and other.handle and _same(person.handle, other.handle):
        method, evidence = LinkMethod.HANDLE, person.handle
    elif (
        person.display_name
        and other.display_name
        and _same(person.display_name, other.display_name)
    ):
        method, evidence = LinkMethod.DISPLAY_NAME, person.display_name
    else:
        return None

    return Suggestion(
        person_id=person.id,
        identity_id=identity.id,
        identity_name=identity.display_name,
        method=method,
        confidence=DEFAULT_CONFIDENCE[method],
        evidence=evidence,
    )


def _same(left: str, right: str) -> bool:
    """Case- and space-insensitive equality.

    `Parth0124` and `parth0124` are one GitHub account written two ways -- the
    platform itself treats logins case-insensitively -- and a comparison that
    missed that would report no evidence where the evidence is exact.
    """
    return left.strip().casefold() == right.strip().casefold()


def build_identity_service(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    *,
    tenant_id: str = DEFAULT_TENANT,
) -> IdentityService:
    if session_factory is None:
        from backend.db.session import get_sessionmaker

        session_factory = get_sessionmaker()
    return IdentityService(session_factory, tenant_id=tenant_id)
