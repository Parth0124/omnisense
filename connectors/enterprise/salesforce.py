"""Salesforce REST + SOQL connector: Cases, Chatter feed items, Knowledge articles.

Phase 6 (`docs/connector-spec.md` §9.3). Four provider facts shape every decision
in this module: the credential is signed locally rather than stored, the column
that *looks* like the incremental key is the wrong one, the request budget
belongs to the customer rather than to us, and the data is somebody's CRM.

**JWT bearer, not the username-password flow.** Both are server-to-server and
only one is defensible. The password grant means storing a human's actual
password plus their security token -- the highest-value secret an org has -- in a
code path that only ever reads Cases; it breaks the moment that human rotates
their password or the org enforces MFA; and Salesforce now blocks the flow by
default, so an integration built on it works in a scratch org and fails in the
customer's. JWT bearer stores a private key instead: it is *ours*, rotatable
independently of any person, and the certificate half lives in the connected app
where the customer's admin can revoke it in one click. It also needs no consent
redirect and issues no refresh token, which is what lets a worker cold-start with
no human in the loop (`docs/connector-spec.md` §8.2, "JWT service credential ...
sign locally from the stored private key"). The assertion itself lives for three
minutes, so a captured request is worthless almost immediately -- and, as §8.2
warns, that same short window is why clock skew presents as `AuthError` and NTP
is a deployment requirement rather than a nicety.

**`SystemModstamp`, never `LastModifiedDate`.** They look interchangeable and
paging on the wrong one loses records silently. `LastModifiedDate` is *writable*:
a data load run with "Set Audit Fields upon Record Creation" can stamp an
imported record with a date from 2019, and a record whose modification date lands
below the watermark is never queried again. It also does not move for every
change -- system-driven updates (cascade, feed tracking, some platform-owned
fields) touch `SystemModstamp` alone. `SystemModstamp` is read-only, maintained
by the platform on every user *and* system modification, and indexed; it is the
field Salesforce's own replication API keys off. See `_soql`.

**The watermark is a modification stamp, not `Signal.timestamp`.** The two are
different clocks and mixing them is the second silent-loss bug here. `timestamp`
is event time -- when the case was opened, when the post was written -- because
that is what the trend and forecast agents read (`models/signal.py`). The cursor
tracks `SystemModstamp`, because that is the only thing that moves when a case is
*updated*. Filtering on `CreatedDate` with a modstamp watermark would re-read
history forever; committing a `CreatedDate` as the watermark would step over
every edit made to an old record.

**The API allocation is the customer's, and it is shared.** An org's 24-hour REST
allocation is spent by every tool the customer runs, not just by us
(`docs/connector-spec.md` §9.3: "never consume more than a configured share").
No per-minute limit is documented, so a per-minute bucket cannot express the real
constraint; instead `parse_rate_limit` reads the `Sforce-Limit-Info` header the
org actually sends and `_raise_if_over_allocation` stops the run once we have
consumed `params['max_api_usage_fraction']` of the org's allocation. Stopping is
a `QuotaError` -- a partial success -- so the pages already emitted stay emitted.

Identity is rule 1 of `docs/signal-model.md` §4.1: the record `Id`, verbatim, so
a DLQ entry names something a human can paste into their own org. The REST API
always returns the 18-character case-safe form, and the connector never accepts
an id from anywhere else -- the 15-character form of the same record is a
different string and would fork identity.

Two limits worth stating rather than discovering. SOQL returns only what the
*integration user* can see, so a service user with a restrictive profile makes
the connector quietly return a subset of the org; and a hard-deleted record
simply stops appearing, because `query` excludes the recycle bin and a Signal has
no tombstone. Neither is a bug this module can fix.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, Self
from urllib.parse import urlsplit

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from models.base import utcnow
from models.enums import AuthType, Platform, SourceCategory
from models.signal import Signal
from connectors.auth.token_store import InMemoryTokenStore, StoredToken, TokenStore
from connectors.base import BaseConnector
from connectors.exceptions import (
    AuthError,
    ConnectorConfigurationError,
    ConnectorError,
    NormalizationError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.normalize.mapper import FieldMap, FieldSpec, MappingContext, to_utc_datetime
from connectors.protocol import (
    Credentials,
    Cursor,
    DedupKeys,
    FetchPage,
    RateLimitHint,
    RateLimitPolicy,
    RawRecord,
    SyncContext,
)

__all__ = ["SalesforceConnector"]


# --------------------------------------------------------------------------- #
# Provider constants
# --------------------------------------------------------------------------- #

PRODUCTION_LOGIN_URL: Final = "https://login.salesforce.com"
SANDBOX_LOGIN_URL: Final = "https://test.salesforce.com"
TOKEN_PATH: Final = "/services/oauth2/token"

JWT_BEARER_GRANT: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"
"""RFC 7523 §2.1, which is how Salesforce spells the flow in `grant_type`."""

ASSERTION_LIFETIME_SECONDS: Final = 180
"""How far ahead the assertion's `exp` is set.

Salesforce refuses an assertion whose `exp` is more than three minutes past *its*
clock, so this is the ceiling rather than a choice. It is also the reason a
skewed worker clock fails as `invalid_grant`: the request is well-formed and the
signature verifies, and the assertion is simply expired on arrival.
"""

DEFAULT_ASSUMED_SESSION_MINUTES: Final = 60
"""How long a minted access token is treated as usable.

The JWT bearer response carries no `expires_in` -- the session lives as long as
the org's session-timeout setting, which is between 15 minutes and 24 hours and
is not readable from here. `StoredToken` with no expiry never refreshes, so the
connector would ride a dead session into a 401 on every run of a short-timeout
org. Assuming an hour errs in the cheap direction: guessing too short costs one
extra token mint, guessing too long costs a failed run.
"""

DEFAULT_API_VERSION: Final = "v60.0"
"""Pinned, because Salesforce retires versions on a rolling three-year schedule
and the REST API has no "current" alias -- the version is a path segment. An org
on a newer release overrides it with `params['api_version']`."""

_API_VERSION: Final = re.compile(r"^v\d{2,3}\.\d$")

DEFAULT_BATCH_SIZE: Final = 500
MIN_BATCH_SIZE: Final = 200
MAX_BATCH_SIZE: Final = 2000
"""`Sforce-Query-Options: batchSize=N` bounds, as the REST API documents them.

Below the floor and above the ceiling the header is ignored rather than refused,
which would leave the page arithmetic resting on a number the org never agreed
to.
"""

DEFAULT_LOOKBACK_HOURS: Final = 24
DEFAULT_MAX_RECORDS_PER_RUN: Final = 10_000
"""Ceiling on one run when the runtime sets no `ctx.max_records`.

A first sync of a large org is millions of rows. Without a bound the run holds a
worker for hours and spends the customer's whole daily allocation in one sitting;
with it, the cursor commits per page and the next run continues from there.
"""

DEFAULT_API_USAGE_FRACTION: Final = 0.5
"""Share of the org's 24-hour allocation this connector may consume. §9.3."""

DEFAULT_KNOWLEDGE_LANGUAGE: Final = "en_US"
_LANGUAGE: Final = re.compile(r"^[a-z]{2}(?:_[A-Z]{2})?$")

QUOTA_RETRY_AFTER_SECONDS: Final = 900.0
"""Above this a throttle becomes a `QuotaError` rather than a held worker (§5.2)."""

_MAX_ERROR_CODE_LENGTH: Final = 64
_MAX_WHERE_LENGTH: Final = 500

_SALESFORCE_HOSTS: Final = (".salesforce.com", ".force.com", ".salesforce.mil", ".cloudforce.com")
"""Host suffixes the access token may be sent to.

The token response *names* the host every subsequent request goes to, so an
`instance_url` from a mis-pointed login host would redirect a live bearer token
to whatever answered. Cheap to check, and the check is the only thing standing
between a configuration typo and a disclosed session.
"""

_LIMIT_INFO: Final = re.compile(r"api-usage=(\d+)\s*/\s*(\d+)")
"""`Sforce-Limit-Info: api-usage=127/15000` -- used and allocated, this org, 24h."""

_RATE_LIMIT_HEADERS: Final = frozenset({"sforce-limit-info", "retry-after"})
"""The only response headers that leave `fetch()`.

An allowlist rather than a redaction list: `FetchPage.raw_headers` travels with
the batch into code that may log it, and a Salesforce response also carries
session cookies (`docs/connector-spec.md` §1).
"""

_WHERE_ALLOWED: Final = re.compile(r"^[A-Za-z0-9_.,:'\"()<>=!%+\- ]+$")
"""Characters an operator-supplied `WHERE` fragment may contain.

Config, not user input -- but it is interpolated into SOQL, so the allowlist
excludes `;` and braces, and `_validated_where` refuses an odd number of quotes.
The failure this actually prevents is duller than injection: an unbalanced
apostrophe in `Origin = 'Customer's Portal'` is a `MALFORMED_QUERY` on every run
for a month, and here it is a configuration error before a request is spent.
"""

_TERMINAL_AUTH_CODES: Final = frozenset(
    {"INVALID_SESSION_ID", "INVALID_LOGIN", "INVALID_OPERATION_WITH_EXPIRED_PASSWORD"}
)
_QUOTA_CODES: Final = frozenset({"REQUEST_LIMIT_EXCEEDED", "TOTAL_REQUESTS_LIMIT_EXCEEDED"})
_CONFIGURATION_CODES: Final = frozenset(
    {
        "API_DISABLED_FOR_ORG",
        "INSUFFICIENT_ACCESS",
        "INSUFFICIENT_ACCESS_OR_READONLY",
        "INVALID_TYPE",
        "NOT_FOUND",
    }
)
"""403/404 codes that mean *this org or this user*, not *this token*.

Salesforce answers a revoked session and an object the integration user may not
read with statuses one band apart and bodies that differ only in `errorCode`.
Filing the second as an `AuthError` flags a working credential `needs_reauth` and
sends an operator to re-link something that was never broken.
"""


# --------------------------------------------------------------------------- #
# Field maps, one per supported sObject
# --------------------------------------------------------------------------- #


def _record_url(instance_url: str) -> Callable[[Any], str]:
    """Build the permalink for a record id.

    The bare record-id path rather than `/lightning/r/{Type}/{id}/view`: it
    resolves for objects that have no Lightning record page of their own (a
    `FeedItem` is rendered on its parent), it survives an org switching UI theme,
    and Salesforce redirects it to whichever surface the reader is entitled to.
    """

    def build(value: Any) -> str:
        record_id = _as_text(value)
        return f"{instance_url}/{record_id}" if record_id else ""

    return build


def _case_map(instance_url: str) -> FieldMap:
    """Support cases.

    `CreatedDate`, not `SystemModstamp`, is the Signal's timestamp: a case is an
    observation made when the customer reported the problem, and re-stamping it
    on every status change would drag the whole corpus forward in time each time
    an agent touched a queue.

    The author is the *contact*, not `CreatedById`. Web-to-case and email-to-case
    both create the record as an automation user, so keying the author on the
    creator would attribute a third of a support queue to a robot.
    """
    return FieldMap(
        platform=Platform.SALESFORCE,
        timestamp=FieldSpec.at("CreatedDate", required=True),
        item_id=FieldSpec.at("Id", required=True),
        url=FieldSpec.at("Id", transform=_record_url(instance_url)),
        title=FieldSpec.at("Subject"),
        text=FieldSpec.at("Description"),
        author_id=FieldSpec.at("ContactId"),
        author_display_name=FieldSpec.at("Contact.Name"),
        author_profile_url=FieldSpec.at("ContactId", transform=_record_url(instance_url)),
        metadata={
            "salesforce.object": FieldSpec.at("attributes.type"),
            "salesforce.case_number": FieldSpec.at("CaseNumber"),
            "salesforce.status": FieldSpec.at("Status"),
            "salesforce.priority": FieldSpec.at("Priority"),
            "salesforce.origin": FieldSpec.at("Origin"),
            "salesforce.type": FieldSpec.at("Type"),
            "salesforce.reason": FieldSpec.at("Reason"),
            "salesforce.is_closed": FieldSpec.at("IsClosed"),
            "salesforce.is_escalated": FieldSpec.at("IsEscalated"),
            "salesforce.account_id": FieldSpec.at("AccountId"),
            "salesforce.owner_id": FieldSpec.at("OwnerId"),
            "salesforce.last_modified_at": FieldSpec.at("LastModifiedDate"),
            "salesforce.system_modstamp": FieldSpec.at("SystemModstamp"),
        },
    )


def _feed_item_map(instance_url: str) -> FieldMap:
    """Chatter posts.

    `CommentCount` and `LikeCount` are raw platform counters and go to
    `Engagement.raw` verbatim; the normalized axes are percentiles within a
    cohort (`docs/signal-model.md` §3.4) and a connector holding one record
    cannot know one.
    """
    return FieldMap(
        platform=Platform.SALESFORCE,
        timestamp=FieldSpec.at("CreatedDate", required=True),
        item_id=FieldSpec.at("Id", required=True),
        url=FieldSpec.at("Id", transform=_record_url(instance_url)),
        title=FieldSpec.at("Title"),
        # `Body` is plain text with `@mentions` rendered as names, not markup, so
        # `text_is_html` stays false -- running the readability extractor over it
        # would strip nothing and risks mangling pasted code.
        text=FieldSpec.at("Body"),
        author_id=FieldSpec.at("CreatedById"),
        author_display_name=FieldSpec.at("CreatedBy.Name"),
        author_profile_url=FieldSpec.at("CreatedById", transform=_record_url(instance_url)),
        engagement={
            "comment_count": FieldSpec.at("CommentCount"),
            "like_count": FieldSpec.at("LikeCount"),
        },
        metadata={
            "salesforce.object": FieldSpec.at("attributes.type"),
            "salesforce.feed_type": FieldSpec.at("Type"),
            "salesforce.parent_id": FieldSpec.at("ParentId"),
            "salesforce.link_url": FieldSpec.at("LinkUrl"),
            "salesforce.visibility": FieldSpec.at("Visibility"),
            "salesforce.last_modified_at": FieldSpec.at("LastModifiedDate"),
            "salesforce.system_modstamp": FieldSpec.at("SystemModstamp"),
        },
    )


def _knowledge_map(instance_url: str) -> FieldMap:
    """Published Knowledge article versions.

    Identity is the *version* `Id`, not `KnowledgeArticleId`. A republished
    article is a new document, and a Signal is what was true when it was written
    (`models/signal.py`), so the previous version keeps its own id and its own
    citations rather than being overwritten by the edit. `KnowledgeArticleId`
    rides along in metadata so the graph layer can still group the versions.

    `truncated=True` because `Summary` is all a generic connector can reach: the
    article body lives in a rich-text field whose API name is chosen per article
    type when the org designs it (`Answer__c`, `Details__c`, anything), so there
    is no field this module could name that would exist in the next org. Saying
    so caps `content_integrity` (`docs/signal-model.md` §3.5) instead of letting
    a summary be trusted like a body.
    """
    return FieldMap(
        platform=Platform.SALESFORCE,
        # First publication is the event; `CreatedDate` covers a draft that was
        # published without the field being set.
        timestamp=FieldSpec.at("FirstPublishedDate", "CreatedDate", required=True),
        item_id=FieldSpec.at("Id", required=True),
        url=FieldSpec.at("Id", transform=_record_url(instance_url)),
        title=FieldSpec.at("Title"),
        text=FieldSpec.at("Summary"),
        engagement={"total_views": FieldSpec.at("ArticleTotalViewCount")},
        metadata={
            "salesforce.object": FieldSpec.at("attributes.type"),
            "salesforce.knowledge_article_id": FieldSpec.at("KnowledgeArticleId"),
            "salesforce.article_number": FieldSpec.at("ArticleNumber"),
            "salesforce.url_name": FieldSpec.at("UrlName"),
            "salesforce.publish_status": FieldSpec.at("PublishStatus"),
            "salesforce.language": FieldSpec.at("Language"),
            "salesforce.version_number": FieldSpec.at("VersionNumber"),
            "salesforce.last_published_at": FieldSpec.at("LastPublishedDate"),
            "salesforce.last_modified_at": FieldSpec.at("LastModifiedDate"),
            "salesforce.system_modstamp": FieldSpec.at("SystemModstamp"),
        },
        truncated=True,
    )


@dataclass(frozen=True, slots=True)
class _SObjectSpec:
    """One queryable object: what to SELECT, and how to map what comes back.

    `fields` doubles as the SOQL select list and as the set of paths the field map
    reads, which is why relationship fields are spelled exactly as SOQL spells
    them (`Contact.Name`) -- the REST response nests them under the same names, so
    one tuple cannot drift out of step with the map beside it.
    """

    api_name: str
    fields: tuple[str, ...]
    build_map: Callable[[str], FieldMap]
    requires_language_filter: bool = False


_CASE: Final = _SObjectSpec(
    api_name="Case",
    fields=(
        "Id",
        "CaseNumber",
        "Subject",
        "Description",
        "Status",
        "Priority",
        "Origin",
        "Type",
        "Reason",
        "IsClosed",
        "IsEscalated",
        "AccountId",
        "ContactId",
        # Name only. `Contact.Email` and `Contact.Phone` are deliberately absent:
        # nothing downstream reads them, and a field not fetched is a field that
        # cannot leak (`docs/security-and-privacy.md`, data minimisation).
        "Contact.Name",
        "OwnerId",
        "CreatedDate",
        "LastModifiedDate",
        "SystemModstamp",
    ),
    build_map=_case_map,
)

_FEED_ITEM: Final = _SObjectSpec(
    api_name="FeedItem",
    fields=(
        "Id",
        "Type",
        "Title",
        "Body",
        "LinkUrl",
        "ParentId",
        "CreatedById",
        "CreatedBy.Name",
        "CommentCount",
        "LikeCount",
        "Visibility",
        "CreatedDate",
        "LastModifiedDate",
        "SystemModstamp",
    ),
    build_map=_feed_item_map,
)

_KNOWLEDGE: Final = _SObjectSpec(
    api_name="Knowledge__kav",
    fields=(
        "Id",
        "KnowledgeArticleId",
        "ArticleNumber",
        "Title",
        "Summary",
        "UrlName",
        "PublishStatus",
        "Language",
        "VersionNumber",
        "ArticleTotalViewCount",
        "FirstPublishedDate",
        "LastPublishedDate",
        "CreatedDate",
        "LastModifiedDate",
        "SystemModstamp",
    ),
    build_map=_knowledge_map,
    requires_language_filter=True,
)

_SUPPORTED: Final[dict[str, _SObjectSpec]] = {
    "case": _CASE,
    "cases": _CASE,
    "feeditem": _FEED_ITEM,
    "feed": _FEED_ITEM,
    "chatter": _FEED_ITEM,
    "knowledge": _KNOWLEDGE,
    "knowledge__kav": _KNOWLEDGE,
    "article": _KNOWLEDGE,
}
"""Objects this connector knows how to read, plus the aliases operators type.

A closed set rather than "any sObject with a properties list in params": an
arbitrary object needs a select list, a timestamp field, an author decision and a
drop rule, and a connector that accepts one it has never seen produces Signals
nobody has looked at.
"""


@dataclass(frozen=True, slots=True)
class _Session:
    """What one minted assertion bought: a token and the host to spend it on."""

    access_token: str
    instance_url: str

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def __repr__(self) -> str:
        return f"_Session(instance_url={self.instance_url!r}, access_token=<redacted>)"

    __str__ = __repr__


# --------------------------------------------------------------------------- #
# The connector
# --------------------------------------------------------------------------- #


class SalesforceConnector(BaseConnector):
    """One sObject, walked forward by `SystemModstamp` through the REST query API."""

    slug: ClassVar[str] = "salesforce"
    platform: ClassVar[Platform] = Platform.SALESFORCE
    category: ClassVar[SourceCategory] = SourceCategory.ENTERPRISE
    auth_type: ClassVar[AuthType] = AuthType.OAUTH2
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=60, burst=5, concurrency=2
    )
    """No per-minute limit is documented, so this bucket is not the real budget.

    Salesforce publishes a 24-hour per-org request allocation and a ceiling of 25
    concurrent *long-running* synchronous requests, and nothing in between. A
    per-minute bucket cannot express a daily allocation, so the numbers here only
    stop one run from bursting; the allocation itself is enforced from the
    `Sforce-Limit-Info` header in `_raise_if_over_allocation`, and the long-term
    budget is the poll cadence in `workers/scheduler.py`. `concurrency=2` sits far
    under the documented 25 because the allocation is shared with every other tool
    the customer has pointed at the same org.
    """

    supports_incremental: ClassVar[bool] = True

    supports_backfill: ClassVar[bool] = True
    """SOQL reaches all of history; a backfill is a large `params['lookback_hours']`.

    That gives it a different `params_hash` and therefore its own cursor row,
    which is the separation §4.1 rule 5 requires between a historical crawl and
    the live watermark.
    """

    overlap_seconds: ClassVar[int] = 300
    """Declared rather than inherited, because the reason is specific.

    `SystemModstamp` is assigned when a row is written, but a row becomes visible
    to a query only when its *transaction commits*. A bulk load or a long trigger
    chain can therefore make a record with an older stamp appear after we have
    already queried past it. Resuming five minutes behind the watermark plus dedup
    is what catches those (`docs/connector-spec.md` §4.1 rule 3).
    """

    def __init__(
        self,
        ctx: SyncContext,
        credentials: Credentials,
        *,
        token_store: TokenStore | None = None,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        # Everything below raises `ConnectorConfigurationError` -- a
        # `PermanentError` -- before a socket exists. §6: configuration defects
        # fail fast and no cursor is ever created for one.
        self._spec = _validated_object(params)
        self._login_url = _validated_login_url(params)
        self._audience = _as_text(params.get("audience")) or self._login_url
        self._api_version = _validated_api_version(params)
        self._batch_size = _clamp(
            _as_int(params.get("batch_size"), DEFAULT_BATCH_SIZE), MIN_BATCH_SIZE, MAX_BATCH_SIZE
        )
        self._lookback_hours = max(1, _as_int(params.get("lookback_hours"), DEFAULT_LOOKBACK_HOURS))
        self._run_limit = max(1, _as_int(params.get("max_records"), DEFAULT_MAX_RECORDS_PER_RUN))
        self._usage_fraction = _validated_fraction(params)
        self._knowledge_language = _validated_language(params)
        self._where = _validated_where(params)

        self._token_store = token_store or InMemoryTokenStore()
        self._now = now
        self._client: httpx.AsyncClient | None = None
        self._private_key: rsa.RSAPrivateKey | None = None
        self._session: _Session | None = None
        self._field_map: FieldMap | None = None
        self._mapping = MappingContext(
            connector_slug=self.slug,
            connector_version=self.version,
            sync_run_id=ctx.run_id,
        )

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct and validate. No I/O: the PEM is not even parsed here.

        Key parsing is deferred to `authenticate()` on purpose. A malformed
        private key is a credential fault, and the stage whose failures are
        credential faults is authentication -- raising it from construction would
        file it as a scheduling defect and skip the `needs_reauth` flag the
        operator actually needs (`docs/connector-spec.md` §2.1).
        """
        connector = cls(ctx, credentials)
        if not _as_text(ctx.params.get("username")):
            raise ConnectorConfigurationError(
                "salesforce needs params['username']: the JWT bearer flow names the "
                "user to run as in the assertion's 'sub' claim, and there is no "
                "sensible default -- the integration user decides which records the "
                "connector can see at all",
                connector=cls.slug,
                account_id=ctx.account_id,
            )
        return connector

    # ------------------------------------------------------------ lifecycle --

    async def authenticate(self) -> None:
        """Mint or reuse an access token, and learn the org's API host. Idempotent.

        Idempotence is not a courtesy: the runtime calls this once per run and at
        most once more after a 401, and the second call has to actually replace
        the rejected token -- which it does, because `_access_denied` deletes the
        stored one first.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.ctx.request_timeout_seconds,
                headers={"User-Agent": self.ctx.user_agent, "Accept": "application/json"},
                # A redirect off the instance host would carry the bearer token to
                # whatever answered: httpx strips only *its own* auth across hosts,
                # and this Authorization header is ours.
                follow_redirects=False,
            )
        if self._private_key is None:
            self._private_key = self._load_private_key()

        token = await self._token()
        instance_url = _as_text(token.extra.get("instance_url"))
        if not instance_url:
            raise AuthError(
                "salesforce token response carried no instance_url; every REST call "
                "needs the org's own host and there is nothing to fall back to",
                connector=self.slug,
                account_id=self.ctx.account_id,
            )
        self._session = _Session(access_token=token.access_token, instance_url=instance_url)
        # Built here rather than at import: the permalink transform closes over the
        # org's host, which is not knowable until the token response names it.
        self._field_map = self._spec.build_map(instance_url)

    async def aclose(self) -> None:
        """Release the client. Idempotent: `run()` closes in a `finally`.

        The token survives in the store, so a second `run()` on the same instance
        rebuilds the connection pool without paying for another assertion.
        """
        client, self._client = self._client, None
        self._session = None
        if client is not None:
            await client.aclose()

    # ------------------------------------------------------------ token flow --

    async def _token(self) -> StoredToken:
        """Return a usable token, minting one under the store's lock if not.

        The double read is the whole mechanism, for the reason
        `connectors/auth/oauth.py` spells out: the optimistic read keeps the common
        case uncontended, and the second read *under* the lock is what makes the
        losers of a race use the winner's token instead of each minting their own.
        Salesforce counts concurrent logins per user, so a stampede here is not
        merely wasteful.
        """
        cached = await self._token_store.load(self.ctx.account_id)
        if cached is not None and not cached.needs_refresh(now=self._now()):
            return cached

        async with self._token_store.lock(self.ctx.account_id):
            cached = await self._token_store.load(self.ctx.account_id)
            if cached is not None and not cached.needs_refresh(now=self._now()):
                return cached
            minted = await self._mint()
            await self._token_store.save(self.ctx.account_id, minted)
            return minted

    async def _mint(self) -> StoredToken:
        """Sign one assertion and exchange it. Raises; never retries, never sleeps."""
        client = self._client
        key = self._private_key
        if client is None or key is None:  # pragma: no cover -- authenticate() builds both
            raise PermanentError(
                "salesforce token mint ran before authenticate() built its client",
                connector=self.slug,
            )

        issued = self._now()
        claims = {
            "iss": self._consumer_key(),
            "sub": _as_text(self.ctx.params.get("username")),
            # The token endpoint's own host, which is what Salesforce validates
            # against. A My Domain login URL with `aud` left at login.salesforce.com
            # is refused as `invalid_grant` -- indistinguishable from a wrong key,
            # and debugged as one for an afternoon.
            "aud": self._audience,
            "exp": int((issued + timedelta(seconds=ASSERTION_LIFETIME_SECONDS)).timestamp()),
            # Salesforce refuses a replayed assertion inside its validity window,
            # and two workers minting in the same second would otherwise build
            # byte-identical ones.
            "jti": uuid.uuid4().hex,
        }
        assertion = jwt.encode(claims, key, algorithm="RS256")

        await self.acquire_slot(self._login_url)
        try:
            response = await client.post(
                self._login_url + TOKEN_PATH,
                data={"grant_type": JWT_BEARER_GRANT, "assertion": assertion},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except httpx.TransportError as exc:
            raise TransientError(
                "salesforce token endpoint is unreachable",
                connector=self.slug,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise PermanentError(
                "salesforce token request could not be issued",
                connector=self.slug,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc

        if response.status_code != httpx.codes.OK:
            raise self._token_error(response)
        return self._stored_token(response, issued)

    def _stored_token(self, response: httpx.Response, issued: datetime) -> StoredToken:
        """Map the JWT bearer success body onto a `StoredToken`.

        `instance_url` goes into `extra` because it is not a secret and every
        later request needs it -- exactly the case `StoredToken.extra` documents.
        It is host-checked first: the response decides where the bearer token is
        sent next, so it does not get to name an arbitrary host.
        """
        payload = self._decode(response)
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise AuthError(
                "salesforce token response carried no access_token",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
            )
        instance_url = _validated_instance_url(
            _as_text(payload.get("instance_url")), connector=self.slug, account=self.ctx.account_id
        )
        scope = payload.get("scope")
        return StoredToken(
            access_token=access_token,
            token_type=_as_text(payload.get("token_type")) or "Bearer",
            # See DEFAULT_ASSUMED_SESSION_MINUTES: the flow reports no lifetime, so
            # this is an assumption, and it is the conservative direction.
            expires_at=issued + timedelta(minutes=DEFAULT_ASSUMED_SESSION_MINUTES),
            # No refresh token exists in this flow and none is wanted: the private
            # key mints a replacement whenever one is needed.
            refresh_token=None,
            scope=scope if isinstance(scope, str) else None,
            obtained_at=issued,
            extra={"instance_url": instance_url},
        )

    def _consumer_key(self) -> str:
        try:
            return self.credentials.require("client_id")
        except KeyError as exc:
            raise AuthError(
                "salesforce account has no 'client_id' secret; it is the connected "
                "app's consumer key and the assertion's 'iss' claim",
                connector=self.slug,
                account_id=self.ctx.account_id,
                cause=exc,
            ) from exc

    def _load_private_key(self) -> rsa.RSAPrivateKey:
        """Parse the PEM once per run, turning every failure into an `AuthError`.

        Parsed here rather than handed to `jwt.encode` as a string so the three
        ways an operator gets this wrong -- absent, unreadable, or the wrong key
        type -- each produce a message naming the problem. Left to PyJWT they
        surface as a bare `ValueError` or `TypeError` from inside a signing call,
        which the runtime records as a crash rather than as the credential fault
        it is. Nothing here renders the key or the underlying exception's message.
        """
        try:
            pem = self.credentials.require("private_key")
        except KeyError as exc:
            raise AuthError(
                "salesforce account has no 'private_key' secret; the JWT bearer flow "
                "signs its own assertion and there is nothing to sign with",
                connector=self.slug,
                account_id=self.ctx.account_id,
                cause=exc,
            ) from exc

        passphrase = self.credentials.secrets.get("private_key_passphrase")
        try:
            key = load_pem_private_key(
                pem.encode("utf-8"),
                password=passphrase.encode("utf-8") if passphrase else None,
            )
        except (ValueError, TypeError) as exc:
            raise AuthError(
                "salesforce private_key is not a readable PEM private key "
                f"({type(exc).__name__}); an encrypted key also needs "
                "'private_key_passphrase'",
                connector=self.slug,
                account_id=self.ctx.account_id,
                cause=exc,
            ) from exc

        if not isinstance(key, rsa.RSAPrivateKey):
            # An EC key signs ES256, and the connected app's certificate cannot
            # verify it. The provider reports that as `invalid_grant`, which reads
            # exactly like a revoked app.
            raise AuthError(
                f"salesforce private_key is a {type(key).__name__}, not an RSA key; "
                "the connected app's certificate verifies RS256 signatures",
                connector=self.slug,
                account_id=self.ctx.account_id,
            )
        return key

    # ---------------------------------------------------------------- fetch --

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Walk `SystemModstamp` forward, following the query locator.

        Genuinely oldest-first, not reconstructed: `ORDER BY SystemModstamp ASC`
        makes the result set a timeline and the locator preserves it, so every
        record on a page is older than every record on the next and each page's
        newest stamp is a legal watermark the moment the page is durable. That is
        also why stopping early on a page or record budget is safe here -- what
        the run did not reach is *newer* than what it committed.

        The locator is never persisted. It expires within minutes and an org keeps
        only a handful per user, so a token carried across a poll interval would be
        dead on arrival; ascending order means the watermark alone is a complete
        resume state (`docs/connector-spec.md` §4.1 rule 4 makes the token
        advisory, and here it is simply unused).
        """
        session = self._session
        if session is None:  # pragma: no cover -- run() authenticates first
            raise PermanentError(
                "salesforce fetch() ran before authenticate(); the six-stage order "
                "in BaseConnector.run() is what guarantees it does not",
                connector=self.slug,
                account_id=self.ctx.account_id,
            )

        budget = self._record_budget()
        if budget == 0:
            # A run given a zero record budget. Nothing was fetched, so there is
            # nothing to commit and no cursor to move.
            return

        since = cursor.watermark or self._now() - timedelta(hours=self._lookback_hours)
        url = f"{session.instance_url}/services/data/{self._api_version}/query"
        params: Mapping[str, str] | None = {"q": self._soql(since, budget)}
        newest: datetime | None = None
        pages = 0

        while True:
            await self.acquire_slot(url)
            body, headers = await self._request(url, params)

            rows = _rows(body, connector=self.slug)
            fingerprint = _fingerprint(url, params)
            records = [self._to_record(row, fingerprint) for row in rows]
            newest = _max_moment(newest, *(_modstamp(row) for row in rows))
            pages += 1

            next_path = _next_records_path(body)
            last = next_path is None or self._budget_reached(pages)
            yield FetchPage(
                records=records,
                # `page_token=None`: see the docstring. An empty page carries the
                # cursor unchanged, and `BaseConnector._guard_watermark` raises the
                # overlap-rewound value back to the watermark the run started from,
                # so a quiet org cannot walk its own cursor backwards.
                cursor=cursor.advanced_to(watermark=newest, page_token=None),
                raw_headers=headers,
            )
            # After the yield, so the page that revealed the wall is still durable.
            self._raise_if_over_allocation(headers)
            if last:
                return
            url = f"{session.instance_url}{next_path}"
            params = None

    def _soql(self, since: datetime, limit: int) -> str:
        """Build the incremental query.

        `>=` rather than `>`, and the bound floored to a whole second, because a
        SOQL dateTime literal has no sub-second component: a watermark of
        `14:02:11.400` can only be expressed as `14:02:11`, and `>` would skip
        every record sharing that second with the last one emitted. Re-reading the
        boundary second is the safe direction -- dedup collapses it.
        """
        filters = [f"SystemModstamp >= {_soql_datetime(since)}"]
        if self._spec.requires_language_filter:
            # Without this the query returns every draft, archived and translated
            # version of every article -- the same article many times over, each
            # with its own version id and therefore its own Signal.
            filters.append(
                f"PublishStatus = 'Online' AND Language = '{self._knowledge_language}'"
            )
        if self._where:
            filters.append(f"({self._where})")
        return (
            f"SELECT {', '.join(self._spec.fields)} FROM {self._spec.api_name} "
            f"WHERE {' AND '.join(filters)} "
            # The ordering *is* the pager. Without it the platform returns rows in
            # whatever order the index served them and no page boundary is a
            # watermark.
            f"ORDER BY SystemModstamp ASC LIMIT {limit}"
        )

    def _record_budget(self) -> int:
        """`LIMIT` for this run, applied before the request rather than after.

        `BaseConnector.run()` also enforces `ctx.max_records`, but only once a page
        has been fetched, normalized and hashed -- which on an object that answers
        in thousands means paying for records the run will discard.
        """
        if self.ctx.max_records is None:
            return self._run_limit
        return max(0, min(self.ctx.max_records, self._run_limit))

    def _budget_reached(self, pages: int) -> bool:
        return self.ctx.max_pages is not None and pages >= self.ctx.max_pages

    def _to_record(self, row: Mapping[str, Any], fingerprint: str) -> RawRecord:
        """Wrap one row verbatim, with the bytes that will be archived.

        Per-record provider bytes do not exist -- one response carried the whole
        page -- so they are synthesized here once, in a canonical encoding, and
        `lineage.raw_sha256` is taken over exactly the bytes the runtime PUTs to
        R2. Re-serializing later, on another json library version, would break the
        content-addressed key.
        """
        return RawRecord(
            native_id=self._record_identity(row),
            payload=row,
            fetched_at=self._now(),
            raw_bytes=json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
            ).encode("utf-8"),
            content_type="application/json",
            request_fingerprint=fingerprint,
        )

    def _record_identity(self, row: Mapping[str, Any]) -> str:
        """Rule 1: the record `Id`, verbatim.

        A row without one is filed under a digest of itself so the DLQ entry is at
        least attributable; it never becomes a Signal, because `item_id` is
        required in every field map above.
        """
        record_id = _as_text(row.get("Id"))
        if record_id:
            return record_id
        material = json.dumps(row, sort_keys=True, default=str).encode("utf-8")
        return f"unidentified:{hashlib.sha256(material).hexdigest()}"

    # ------------------------------------------------------------ normalize --

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map one row onto a Signal, or drop it.

        The drop rule reads the *mapped* Signal rather than re-testing payload
        paths, so it cannot drift out of step with the field map above. What it
        catches is real and common: a `TrackedChange` feed item, or a case opened
        by an automation with neither a subject nor a description. Those carry no
        observation at all, and emitting one would put an empty document into the
        embedding queue and the search index. They are dropped rather than filed in
        the DLQ, where they would bury genuine mapping bugs.
        """
        field_map = self._field_map
        if field_map is None:  # pragma: no cover -- run() authenticates first
            raise PermanentError(
                "salesforce normalize() ran before authenticate(); the field map "
                "closes over the org's host, which the token response names",
                connector=self.slug,
                account_id=self.ctx.account_id,
            )

        # The runtime keys the R2 object and the Kafka partition off
        # `RawRecord.native_id`, while every store keys off `Signal.id`, which is
        # derived from `Id`. A disagreement would give one record two identities,
        # so it is caught here instead of surfacing as duplicate rows months later.
        if _as_text(record.payload.get("Id")) != record.native_id:
            raise NormalizationError(
                "payload Id does not match the fetched record's native_id",
                native_id=record.native_id,
                connector=self.slug,
            )

        signal = field_map.to_signal(record, self._mapping)
        if not signal.content.text.strip() and not (signal.content.title or "").strip():
            return None
        return signal

    def dedup_keys(self, signal: Signal) -> DedupKeys:
        """Identity keyed on the record *and its revision*; no content layer.

        Both halves differ from the default, and both would otherwise silently
        defeat the point of this connector.

        The default identity key is `Signal.id`, which is derived from the record
        id alone. A CRM record is revised in place, so the second version of a case
        would be dropped as a duplicate of the first and everything after the first
        sync would be discarded -- the entire modstamp-driven design delivering
        nothing. Appending `SystemModstamp` keeps a genuine re-fetch (the overlap
        window, a replayed page) collapsing while letting a real edit through;
        `Signal.id` is unchanged, so the stores upsert and the newest version wins.

        The content layer is dropped entirely. It exists to catch one article
        syndicated across sources (`docs/signal-model.md` §4.2), and inside one
        org two support cases that both say "cannot log in" are two customers with
        the same problem, not a duplicate. Hashing the body here would delete the
        second one -- and volume is exactly the signal a support queue carries.
        """
        revision = _as_text(signal.metadata.get("salesforce.system_modstamp"))
        return DedupKeys(
            identity=f"os:dedup:id:{self.slug}:{signal.id}:{revision}",
            content=None,
            simhash=None,
        )

    # ------------------------------------------------------------ rate limit --

    def parse_rate_limit(self, headers: Mapping[str, str]) -> RateLimitHint | None:
        """Read `Sforce-Limit-Info`, which the inherited parser cannot see.

        Salesforce sends no `X-RateLimit-*` headers at all, so the base
        implementation returns `None` for every response and the shared bucket
        runs on a local guess forever. What it does send, on every REST response,
        is the org's 24-hour usage: `api-usage=127/15000`. Feeding that back is
        the only way the limiter learns that the customer's other tools have
        already spent the day's allocation.
        """
        lowered = {key.lower(): value for key, value in headers.items()}
        usage = _parse_limit_info(lowered.get("sforce-limit-info"))
        inherited = super().parse_rate_limit(headers)
        retry_after = inherited.retry_after_seconds if inherited is not None else None

        if usage is None:
            return inherited
        used, allocation = usage
        return RateLimitHint(
            remaining=max(0, allocation - used),
            limit=allocation,
            # The allocation is a rolling 24-hour window with no published reset
            # instant, so claiming one would be inventing it.
            reset_at=None,
            retry_after_seconds=retry_after,
        )

    def _raise_if_over_allocation(self, headers: Mapping[str, str]) -> None:
        """Stop once this connector has eaten its configured share of the org's day.

        `docs/connector-spec.md` §9.3 makes this an explicit obligation: the
        allocation belongs to the customer and their other integrations are
        spending it too. A `QuotaError` is a *partial success* -- the pages already
        yielded stay emitted and the cursor is committed, so the next run resumes
        rather than restarts.
        """
        usage = _parse_limit_info({k.lower(): v for k, v in headers.items()}.get(
            "sforce-limit-info"
        ))
        if usage is None:
            return
        used, allocation = usage
        if allocation <= 0 or used < allocation * self._usage_fraction:
            return
        raise QuotaError(
            "salesforce org API allocation is past this connector's configured share",
            connector=self.slug,
            account_id=self.ctx.account_id,
            details={
                "api_usage": used,
                "api_allocation": allocation,
                "max_fraction": self._usage_fraction,
            },
        )

    # --------------------------------------------------------------- request --

    async def _request(
        self, url: str, params: Mapping[str, str] | None
    ) -> tuple[Mapping[str, Any], dict[str, str]]:
        """Issue one authenticated GET and map every failure onto the taxonomy.

        No retry and no sleep (`docs/connector-spec.md` §1): a connector that
        retries privately makes the shared limiter's accounting wrong and hides the
        failure from metrics.
        """
        client, session = self._client, self._session
        if client is None or session is None:  # pragma: no cover
            raise PermanentError(
                "salesforce request issued without a session", connector=self.slug
            )

        try:
            response = await client.get(url, params=params, headers=session.headers())
        except httpx.TransportError as exc:
            raise TransientError(
                "salesforce is unreachable",
                connector=self.slug,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise PermanentError(
                "the salesforce query could not be issued",
                connector=self.slug,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc

        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise await self._query_error(response)
        return self._decode(response), _rate_limit_headers(response.headers)

    async def _query_error(self, response: httpx.Response) -> ConnectorError:
        """Classify a rejected query by `errorCode`, not by status alone.

        Salesforce puts the meaning in the body: 403 carries both "your daily
        allocation is gone" and "this user cannot read this object", and the status
        does not distinguish them. Only the code is read -- `message` is free text
        that can echo the query, and the query names the org's fields.
        """
        status = response.status_code
        code = _error_code(response)
        details: dict[str, Any] = {"salesforce_error": code} if code else {}

        if code in _QUOTA_CODES:
            return QuotaError(
                "salesforce org has exhausted its 24-hour API allocation",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if code in _CONFIGURATION_CODES:
            return ConnectorConfigurationError(
                f"salesforce refused the query on {self._spec.api_name}; the "
                "integration user's profile or the org's API settings, not the token",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if code == "INVALID_QUERY_LOCATOR":
            # The locator aged out mid-run, which only happens when the consumer
            # was slower than the org's locator lifetime. Transient because the
            # committed watermark makes the same query repeatable from scratch.
            return TransientError(
                "salesforce query locator expired mid-run",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if status == httpx.codes.UNAUTHORIZED or code in _TERMINAL_AUTH_CODES:
            # Drop the cached session so the runtime's one permitted
            # re-authentication mints a new token instead of replaying the rejected
            # one. Deleting is safe here in a way it never is for a refresh-token
            # grant: nothing is lost, because the private key can mint again.
            await self._token_store.delete(self.ctx.account_id)
            return AuthError(
                "salesforce rejected the session",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if status == httpx.codes.TOO_MANY_REQUESTS:
            hint = self.parse_rate_limit(response.headers)
            wait = hint.retry_after_seconds if hint is not None else None
            if wait is not None and wait > QUOTA_RETRY_AFTER_SECONDS:
                return QuotaError(
                    "salesforce asked for a long wait",
                    connector=self.slug,
                    account_id=self.ctx.account_id,
                    status_code=status,
                    retry_after_seconds=wait,
                    details=details,
                )
            return TransientError(
                "salesforce throttled the request",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            return TransientError(
                "salesforce returned a server error",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        return PermanentError(
            "salesforce rejected the SOQL query; it will be rejected identically "
            "on retry",
            connector=self.slug,
            account_id=self.ctx.account_id,
            status_code=status,
            details=details,
        )

    def _token_error(self, response: httpx.Response) -> ConnectorError:
        """Classify a rejected assertion.

        The `error` code is the only thing taken from the body.
        `error_description` is where Salesforce explains *why* the grant failed and
        it routinely quotes the request -- and the request is a form carrying a
        signed assertion (`docs/connector-spec.md` §1).
        """
        status = response.status_code
        code = _error_code(response)
        details: dict[str, Any] = {"oauth_error": code} if code else {}

        if status == httpx.codes.TOO_MANY_REQUESTS:
            return QuotaError(
                "salesforce token endpoint rate-limited the client",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            return TransientError(
                "salesforce token endpoint failed",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        # Everything left in the 4xx band is a credential or connected-app fault:
        # an unapproved user, a certificate that no longer matches the key, or a
        # clock far enough off that the assertion arrives expired. All terminal,
        # all needing a human -- and retrying a rejected grant in a loop is how an
        # integration earns an application-level ban rather than an account-level
        # one.
        return AuthError(
            "salesforce rejected the JWT assertion; check that the connected app "
            "pre-authorizes this user, that its certificate matches the stored "
            "private key, and that the worker's clock is within three minutes of "
            "Salesforce's",
            connector=self.slug,
            account_id=self.ctx.account_id,
            status_code=status,
            details=details,
        )

    def _decode(self, response: httpx.Response) -> Mapping[str, Any]:
        """Decode a JSON object body, or raise.

        The body is never quoted into the error: a Salesforce error page can echo
        the SOQL, and the SOQL names the customer's fields.
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise PermanentError(
                f"salesforce returned {len(response.content)} bytes of non-JSON "
                f"({response.headers.get('content-type', 'unknown')})",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
                cause=exc,
            ) from exc
        if not isinstance(payload, Mapping):
            raise PermanentError(
                f"salesforce returned a JSON {type(payload).__name__} where an object "
                "was expected; the response shape changed",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
            )
        return payload


# --------------------------------------------------------------------------- #
# Configuration validation -- all of it before any socket exists
# --------------------------------------------------------------------------- #


def _validated_object(params: Mapping[str, Any]) -> _SObjectSpec:
    raw = _as_text(params.get("object") or params.get("sobject")) or "Case"
    spec = _SUPPORTED.get(raw.casefold())
    if spec is None:
        raise ConnectorConfigurationError(
            f"salesforce object {raw!r} is not supported; this connector reads "
            f"{sorted({s.api_name for s in _SUPPORTED.values()})}. The name is "
            "interpolated into SOQL, so an unrecognised one is either a 400 or a "
            "query nobody designed",
            connector=SalesforceConnector.slug,
        )
    return spec


def _validated_login_url(params: Mapping[str, Any]) -> str:
    """Resolve the token host: production, sandbox, or the org's My Domain.

    A sandbox that authenticates against `login.salesforce.com` is refused with
    `invalid_grant`, which reads exactly like a bad key -- hence the explicit
    `sandbox` flag rather than leaving it to whoever remembers.
    """
    raw = _as_text(params.get("login_url"))
    if not raw:
        return SANDBOX_LOGIN_URL if _as_bool(params.get("sandbox")) else PRODUCTION_LOGIN_URL
    parts = urlsplit(raw)
    if parts.scheme != "https" or not parts.netloc:
        raise ConnectorConfigurationError(
            f"salesforce login_url must be an https origin, got {raw!r}; a signed "
            "assertion posted over plaintext is a credential disclosed to every hop",
            connector=SalesforceConnector.slug,
        )
    return f"https://{parts.netloc}"


def _validated_api_version(params: Mapping[str, Any]) -> str:
    version = _as_text(params.get("api_version")) or DEFAULT_API_VERSION
    if not _API_VERSION.match(version):
        raise ConnectorConfigurationError(
            f"salesforce api_version {version!r} must look like 'v60.0'; it is a path "
            "segment, so anything else is a 404 on every request of every run",
            connector=SalesforceConnector.slug,
        )
    return version


def _validated_language(params: Mapping[str, Any]) -> str:
    language = _as_text(params.get("knowledge_language")) or DEFAULT_KNOWLEDGE_LANGUAGE
    if not _LANGUAGE.match(language):
        raise ConnectorConfigurationError(
            f"salesforce knowledge_language {language!r} must be a Salesforce locale "
            "such as 'en_US'; it is quoted into SOQL",
            connector=SalesforceConnector.slug,
        )
    return language


def _validated_fraction(params: Mapping[str, Any]) -> float:
    raw = params.get("max_api_usage_fraction", DEFAULT_API_USAGE_FRACTION)
    try:
        fraction = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConnectorConfigurationError(
            f"salesforce max_api_usage_fraction must be a number, got {raw!r}",
            connector=SalesforceConnector.slug,
        ) from exc
    if not 0.0 < fraction <= 1.0:
        raise ConnectorConfigurationError(
            f"salesforce max_api_usage_fraction must be within (0, 1], got {fraction}; "
            "it is the share of the *customer's* daily allocation this connector may "
            "spend",
            connector=SalesforceConnector.slug,
        )
    return fraction


def _validated_where(params: Mapping[str, Any]) -> str:
    """Check an optional operator-supplied `WHERE` fragment. See `_WHERE_ALLOWED`."""
    where = _as_text(params.get("where"))
    if not where:
        return ""
    if len(where) > _MAX_WHERE_LENGTH:
        raise ConnectorConfigurationError(
            f"salesforce params['where'] is {len(where)} characters; the limit is "
            f"{_MAX_WHERE_LENGTH}",
            connector=SalesforceConnector.slug,
        )
    if not _WHERE_ALLOWED.match(where) or where.count("'") % 2:
        raise ConnectorConfigurationError(
            "salesforce params['where'] contains characters this connector will not "
            "interpolate into SOQL, or an unbalanced quote",
            connector=SalesforceConnector.slug,
        )
    return where


def _validated_instance_url(raw: str, *, connector: str, account: str) -> str:
    """Check the host the token response nominates. See `_SALESFORCE_HOSTS`."""
    parts = urlsplit(raw)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host.endswith(_SALESFORCE_HOSTS):
        raise AuthError(
            "salesforce token response named an instance_url this connector will not "
            "send a bearer token to; check params['login_url']",
            connector=connector,
            account_id=account,
            details={"instance_host": host},
        )
    return f"https://{host}"


# --------------------------------------------------------------------------- #
# Payload and header helpers
# --------------------------------------------------------------------------- #


def _rows(body: Mapping[str, Any], *, connector: str) -> list[Mapping[str, Any]]:
    records = body.get("records")
    if records is None:
        return []
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise PermanentError(
            "salesforce response has a non-list 'records'; the shape changed",
            connector=connector,
        )
    return [row for row in records if isinstance(row, Mapping)]


def _next_records_path(body: Mapping[str, Any]) -> str | None:
    """The locator path for the next page, or `None` when the query is done.

    `done` is checked as well as the presence of the URL because Salesforce sends
    both and they are the same fact stated twice; trusting only one of them means
    a shape change in the other goes unnoticed until a page is silently skipped.
    """
    if body.get("done") is True:
        return None
    path = body.get("nextRecordsUrl")
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    return path


def _modstamp(row: Mapping[str, Any]) -> datetime | None:
    """Parse `SystemModstamp`, or `None`.

    Never raises. A row with an unparseable stamp still has to reach `normalize`,
    which is the stage allowed to DLQ it with its identity attached; raising here
    would abort the page and block every well-formed row behind it. It is excluded
    from the watermark rather than defaulted, so it cannot poison the cursor.
    """
    raw = row.get("SystemModstamp")
    if raw is None:
        return None
    try:
        return to_utc_datetime(raw)
    except ValueError:
        return None


def _soql_datetime(moment: datetime) -> str:
    """Render a SOQL dateTime literal: UTC, whole seconds, unquoted."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_limit_info(header: str | None) -> tuple[int, int] | None:
    """`(used, allocation)` from `Sforce-Limit-Info`, or `None` when absent."""
    if not header:
        return None
    match = _LIMIT_INFO.search(header)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _error_code(response: httpx.Response) -> str | None:
    """The `errorCode` of a Salesforce error, and nothing else.

    Character-restricted and length-capped before it can reach a log line, for the
    same reason `connectors/auth/oauth.py` does it to `error`: a provider under
    load can put anything in a body field, including an echo of the request.
    Salesforce sends query errors as a JSON *array* and OAuth errors as an object,
    so both shapes are read.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if isinstance(payload, Mapping):
        candidates: list[Mapping[str, Any]] = [payload]
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        candidates = [item for item in payload if isinstance(item, Mapping)]
    else:
        return None
    for item in candidates:
        raw = item.get("errorCode") or item.get("error")
        if not isinstance(raw, str):
            continue
        code = raw.strip()[:_MAX_ERROR_CODE_LENGTH]
        if code and code.replace("_", "").replace("-", "").isalnum():
            return code
    return None


def _rate_limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() in _RATE_LIMIT_HEADERS}


def _fingerprint(url: str, params: Mapping[str, str] | None) -> str:
    """Hash of the exact request; there is no credential in it to omit.

    `lineage.request_fingerprint` is what makes a fetch reproducible -- it names
    the request that produced a record without naming who made it. The SOQL is
    part of the request and therefore part of the hash: the same record fetched
    under a different select list is a different observation.
    """
    query = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
    return hashlib.sha256(f"GET {url}?{query}".encode()).hexdigest()[:32]


def _max_moment(*moments: datetime | None) -> datetime | None:
    present = [moment for moment in moments if moment is not None]
    return max(present) if present else None


def _as_text(value: Any) -> str:
    """Render a scalar as a stripped string; `""` for anything unusable.

    Booleans are refused rather than stringified: `"True"` is never a record id or
    a hostname, and letting one through turns a type confusion into a
    plausible-looking value.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (Mapping, Sequence, set, frozenset)):
        return ""
    return str(value).strip()


def _as_int(value: Any, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    """Read a flag that may have arrived as an environment string.

    `bool("false")` is `True`, which would put a production connector on the
    sandbox login host -- or worse, the reverse.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
