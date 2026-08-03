"""Unit tests for `connectors/auth/`.

The auth layer fails in ways that are invisible until production, so these tests
target exactly those:

- a token refreshed *reactively* works fine in every single-worker test and
  stampedes the token endpoint at scale, so the pre-emptive margin is asserted on
  both sides of the boundary;
- a lock without a re-read under it serialises the stampede instead of collapsing
  it, and looks identical from one coroutine, so concurrency is asserted with ten;
- a retry on `invalid_grant` is what escalates an account ban to an application
  ban, so the call count is asserted, not just the exception type;
- a refresh response that omits `refresh_token` erases a working credential if
  you write `payload.get(...)` straight into the field, and the account only
  breaks an hour later;
- and every object here holds a secret, so `repr`, `str` and the structured log
  fields are checked against the literal token.

Everything runs against `respx` and `InMemoryTokenStore`. No network, no Redis,
no clock dependency -- `OAuth2Client` takes its clock as a parameter precisely so
expiry can be tested by moving a variable instead of by sleeping.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from connectors.auth.apikey import ApiKeyAuth, BasicAuth, BearerAuth, HeaderAuth, NoAuth
from connectors.auth.oauth import (
    ClientAuthMethod,
    OAuth2Client,
    OAuth2Config,
    OAuth2Grant,
)
from connectors.auth.token_store import (
    DEFAULT_REFRESH_MARGIN_SECONDS,
    InMemoryTokenStore,
    StoredToken,
    TokenStore,
)
from connectors.exceptions import (
    AuthError,
    ConnectorConfigurationError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.protocol import Credentials

pytestmark = pytest.mark.unit

T0 = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)
TOKEN_URL = "https://provider.example.com/oauth/token"
SECRET = "s3cret-value-nobody-should-see"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class Clock:
    """A movable clock.

    Injected rather than patched: `freezegun` would also freeze the event loop's
    view of time, and these tests need `asyncio` to keep scheduling while the
    provider's notion of "now" jumps an hour.
    """

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class YieldingStore(InMemoryTokenStore):
    """An in-memory store whose reads and writes suspend.

    Necessary for the single-flight test and for nothing else. `InMemoryTokenStore`
    never awaits internally, so ten gathered coroutines would run to completion
    one at a time and the first would have finished before the second started --
    the test would pass without the lock existing at all.
    """

    async def load(self, account_id: str) -> StoredToken | None:
        await asyncio.sleep(0)
        return await super().load(account_id)

    async def save(self, account_id: str, token: StoredToken) -> None:
        await asyncio.sleep(0)
        await super().save(account_id, token)


def token_response(
    *,
    access_token: str = "access-1",
    expires_in: int | None = 3600,
    refresh_token: str | None = None,
    token_type: str = "bearer",
    **extra: object,
) -> httpx.Response:
    body: dict[str, object] = {"access_token": access_token, "token_type": token_type}
    if expires_in is not None:
        body["expires_in"] = expires_in
    if refresh_token is not None:
        body["refresh_token"] = refresh_token
    body.update(extra)
    return httpx.Response(200, json=body)


def client_credentials_config(**overrides: object) -> OAuth2Config:
    defaults: dict[str, object] = {
        "token_url": TOKEN_URL,
        "client_id": "client-abc",
        "client_secret": SECRET,
        "grant": OAuth2Grant.CLIENT_CREDENTIALS,
    }
    defaults.update(overrides)
    return OAuth2Config(**defaults)  # type: ignore[arg-type]


def make_client(
    config: OAuth2Config | None = None,
    *,
    store: TokenStore | None = None,
    clock: Clock | None = None,
) -> OAuth2Client:
    return OAuth2Client(
        config or client_credentials_config(),
        account_id="acct_1",
        store=store or InMemoryTokenStore(),
        connector="demo",
        now=clock or Clock(),
    )


# --------------------------------------------------------------------------- #
# apikey.py
# --------------------------------------------------------------------------- #


class TestHeaderStrategies:
    """The three static schemes. Pure functions of a secret, so the tests are
    about shape and about what happens to malformed input."""

    def test_api_key_uses_the_provider_named_header(self) -> None:
        """There is no convention: NewsAPI wants `X-Api-Key`, others differ."""
        auth = ApiKeyAuth(key="k-123", header="X-Api-Key")
        assert auth.headers() == {"X-Api-Key": "k-123"}

    def test_api_key_supports_a_scheme_prefix(self) -> None:
        """Stored separately so the stored credential stays exactly what the
        provider's console displayed."""
        auth = ApiKeyAuth(key="k-123", header="Authorization", prefix="Token ")
        assert auth.headers() == {"Authorization": "Token k-123"}

    def test_bearer_renders_rfc6750(self) -> None:
        assert BearerAuth(token="t-1").headers() == {"Authorization": "Bearer t-1"}

    def test_basic_is_base64_of_user_colon_password(self) -> None:
        assert BasicAuth(username="alice", password="pw").headers() == {
            "Authorization": "Basic YWxpY2U6cHc="
        }

    def test_basic_encodes_as_utf8(self) -> None:
        """The historical ISO-8859-1 default does not raise on a non-ASCII
        password -- it encodes a different one, and the provider answers 401 with
        no hint as to why."""
        import base64

        header = BasicAuth(username="alice", password="pässwörd").headers()["Authorization"]
        decoded = base64.b64decode(header.removeprefix("Basic ")).decode("utf-8")
        assert decoded == "alice:pässwörd"

    def test_no_auth_attaches_nothing(self) -> None:
        """So a connector can hold a strategy unconditionally instead of
        branching on `is None` at every call site."""
        assert NoAuth().headers() == {}

    def test_headers_are_a_fresh_mapping_each_call(self) -> None:
        """A shared dict invites a caller to add `Content-Type` to it once and
        then send that header on every request the connector makes."""
        auth = ApiKeyAuth(key="k-123")
        assert auth.headers() is not auth.headers()

    def test_every_strategy_declares_its_auth_type(self) -> None:
        """The registry checks a connector's declared `auth_type` against what it
        was actually handed."""
        from models.enums import AuthType

        assert ApiKeyAuth(key="k").auth_type is AuthType.API_KEY
        assert BearerAuth(token="t").auth_type is AuthType.BEARER
        assert BasicAuth(username="u", password="p").auth_type is AuthType.BASIC
        assert NoAuth().auth_type is AuthType.NONE

    def test_all_strategies_are_header_auth(self) -> None:
        for auth in (ApiKeyAuth(key="k"), BearerAuth(token="t"), NoAuth()):
            assert isinstance(auth, HeaderAuth)


class TestCredentialValidation:
    """Rejecting bad material here, where nothing has been sent yet, rather than
    inside httpx's header encoder, whose exception names the header and then gets
    logged."""

    def test_trailing_newline_is_stripped_not_rejected(self) -> None:
        """The shape you get from `cat key.txt`. An operator's typo, not a
        security decision -- failing their sync over it helps nobody."""
        assert ApiKeyAuth(key="  k-123\n").headers() == {"X-API-Key": "k-123"}

    def test_embedded_newline_is_rejected(self) -> None:
        """Unlike surrounding whitespace, this changes the meaning of the
        request: one header becomes two."""
        with pytest.raises(ConnectorConfigurationError):
            ApiKeyAuth(key="k-123\r\nX-Admin: 1")

    def test_empty_secret_is_rejected(self) -> None:
        with pytest.raises(ConnectorConfigurationError, match="api key"):
            ApiKeyAuth(key="   ")

    def test_header_name_containing_a_colon_is_rejected(self) -> None:
        with pytest.raises(ConnectorConfigurationError):
            ApiKeyAuth(key="k", header="X-Key: oops")

    def test_basic_username_may_not_contain_a_colon(self) -> None:
        """RFC 7617 §2 makes the colon the field separator, so the server splits
        in the wrong place and reports a bad password."""
        with pytest.raises(ConnectorConfigurationError, match="':'"):
            BasicAuth(username="user:name", password="pw")

    def test_from_credentials_names_the_missing_key_and_the_account(self) -> None:
        """A bare `KeyError('api_key')` from inside an auth flow tells an
        operator running twelve connectors nothing about which one to fix."""
        creds = Credentials(account_id="acct_9", secrets={})
        with pytest.raises(KeyError) as caught:
            ApiKeyAuth.from_credentials(creds)
        assert "api_key" in str(caught.value) and "acct_9" in str(caught.value)

    def test_from_credentials_builds_a_working_strategy(self) -> None:
        creds = Credentials(account_id="acct_1", secrets={"api_key": "k-9"})
        assert ApiKeyAuth.from_credentials(creds).headers() == {"X-API-Key": "k-9"}


class TestFingerprints:
    """Rotation needs an answer to "which key did that succeed with", and the key
    itself may never be logged (`docs/connector-spec.md` §8.2)."""

    def test_fingerprint_is_stable_and_discriminating(self) -> None:
        assert ApiKeyAuth(key="k-1").fingerprint() == ApiKeyAuth(key="k-1").fingerprint()
        assert ApiKeyAuth(key="k-1").fingerprint() != ApiKeyAuth(key="k-2").fingerprint()

    def test_fingerprint_does_not_contain_the_secret(self) -> None:
        assert SECRET not in ApiKeyAuth(key=SECRET).fingerprint()

    def test_basic_fingerprints_the_username_only(self) -> None:
        """The inherited implementation digests `base64(user:password)`, and
        publishing a digest of a human-chosen password to an aggregator is a
        dictionary attack, not a fingerprint."""
        strong = BasicAuth(username="alice", password="pw-1")
        weak = BasicAuth(username="alice", password="pw-2")
        assert strong.fingerprint() == weak.fingerprint()
        assert strong.fingerprint() != BasicAuth(username="bob", password="pw-1").fingerprint()


# --------------------------------------------------------------------------- #
# token_store.py
# --------------------------------------------------------------------------- #


class TestStoredToken:
    def test_needs_refresh_inside_the_margin(self) -> None:
        token = StoredToken(access_token="a", expires_at=T0 + timedelta(seconds=299))
        assert token.needs_refresh(margin_seconds=300, now=T0) is True

    def test_does_not_need_refresh_outside_the_margin(self) -> None:
        token = StoredToken(access_token="a", expires_at=T0 + timedelta(seconds=301))
        assert token.needs_refresh(margin_seconds=300, now=T0) is False

    def test_a_token_without_an_expiry_never_needs_refresh(self) -> None:
        """Some providers omit `expires_in` entirely. Inventing a lifetime would
        mean discarding a good token on a fixed timer forever."""
        assert StoredToken(access_token="a").needs_refresh(now=T0) is False

    def test_bearer_scheme_is_normalised(self) -> None:
        """Providers return `bearer` lower-cased; a minority of resource servers
        compare the scheme literally."""
        token = StoredToken(access_token="a", token_type="bearer")
        assert token.authorization_header() == "Bearer a"

    def test_a_non_bearer_scheme_is_preserved(self) -> None:
        token = StoredToken(access_token="a", token_type="DPoP")
        assert token.authorization_header() == "DPoP a"

    def test_seconds_remaining_is_none_without_an_expiry(self) -> None:
        assert StoredToken(access_token="a").seconds_remaining(now=T0) is None


class TestInMemoryTokenStore:
    async def test_round_trips_a_token(self) -> None:
        store = InMemoryTokenStore()
        assert await store.load("acct_1") is None
        await store.save("acct_1", StoredToken(access_token="a"))
        loaded = await store.load("acct_1")
        assert loaded is not None and loaded.access_token == "a"

    async def test_delete_forgets_the_account(self) -> None:
        store = InMemoryTokenStore()
        await store.save("acct_1", StoredToken(access_token="a"))
        await store.delete("acct_1")
        assert await store.load("acct_1") is None

    async def test_delete_of_an_unknown_account_is_not_an_error(self) -> None:
        """De-authorising an account that never linked must not fail the run."""
        await InMemoryTokenStore().delete("nobody")

    async def test_lock_excludes_concurrent_holders_of_one_account(self) -> None:
        store = InMemoryTokenStore()
        order: list[str] = []

        async def hold(name: str) -> None:
            async with store.lock("acct_1"):
                order.append(f"{name}-in")
                await asyncio.sleep(0)
                order.append(f"{name}-out")

        await asyncio.gather(hold("a"), hold("b"))
        assert order in (
            ["a-in", "a-out", "b-in", "b-out"],
            ["b-in", "b-out", "a-in", "a-out"],
        )

    async def test_lock_is_per_account_not_global(self) -> None:
        """A global lock would serialise every connector account in the process
        behind whichever provider is slowest to answer."""
        store = InMemoryTokenStore()
        both_inside = asyncio.Event()
        entered = 0

        async def hold(account: str) -> None:
            nonlocal entered
            async with store.lock(account):
                entered += 1
                if entered == 2:
                    both_inside.set()
                await asyncio.wait_for(both_inside.wait(), timeout=1.0)

        await asyncio.gather(hold("acct_1"), hold("acct_2"))
        assert both_inside.is_set()


class TestTokenStoreBoundary:
    """The rule that lets this package be tested with two fakes and no services."""

    def test_token_store_imports_no_backend_and_no_cryptography(self) -> None:
        """`CREDENTIAL_ENCRYPTION_KEY` lives in `backend/core/config.py`, which
        this package may not import (`docs/architecture.md` §6.2 rule 2). The
        Fernet envelope is therefore applied by the `TokenStore` subclass in
        `services/connector_service.py`, not here."""
        import pathlib

        source = pathlib.Path("connectors/auth/token_store.py").read_text()
        offenders = [
            line
            for line in source.splitlines()
            if line.strip().startswith(("import ", "from "))
            and any(bad in line for bad in ("backend", "services", "cryptography"))
        ]
        assert not offenders, offenders

    def test_lock_is_abstract(self) -> None:
        """A defaulted per-process lock would be inherited by the Redis-backed
        store and guard nothing across replicas."""
        assert "lock" in TokenStore.__abstractmethods__

    def test_store_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            TokenStore()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# oauth.py -- configuration
# --------------------------------------------------------------------------- #


class TestOAuth2Config:
    def test_plaintext_token_url_is_rejected(self) -> None:
        """A client secret over http is disclosed to every hop, silently -- the
        request still succeeds."""
        with pytest.raises(ConnectorConfigurationError, match="https"):
            client_credentials_config(token_url="http://provider.example.com/token")

    def test_loopback_over_http_is_allowed(self) -> None:
        """So a locally mocked provider does not force a certificate on a
        developer."""
        assert client_credentials_config(token_url="http://localhost:8080/token")

    def test_client_credentials_requires_a_secret(self) -> None:
        with pytest.raises(ConnectorConfigurationError, match="client_secret"):
            OAuth2Config(token_url=TOKEN_URL, client_id="c", client_secret=None)

    def test_refresh_grant_does_not_require_a_secret_at_construction(self) -> None:
        """Public clients (PKCE) have no secret at all, and the refresh token may
        legitimately arrive later from the store."""
        assert OAuth2Config(
            token_url=TOKEN_URL, client_id="c", grant=OAuth2Grant.REFRESH_TOKEN
        )

    def test_from_credentials_reads_the_decrypted_secrets(self) -> None:
        creds = Credentials(
            account_id="acct_1",
            secrets={"client_id": "c-1", "client_secret": SECRET, "refresh_token": "r-1"},
        )
        config = OAuth2Config.from_credentials(creds, token_url=TOKEN_URL)
        assert config.client_id == "c-1" and config.refresh_token == "r-1"


# --------------------------------------------------------------------------- #
# oauth.py -- the grants
# --------------------------------------------------------------------------- #


class TestClientCredentialsGrant:
    @respx.mock
    async def test_mints_and_attaches_a_token(self) -> None:
        route = respx.post(TOKEN_URL).mock(return_value=token_response(access_token="a-1"))
        client = make_client()

        assert await client.headers() == {"Authorization": "Bearer a-1"}
        assert route.call_count == 1

    @respx.mock
    async def test_sends_the_client_credentials_grant_type(self) -> None:
        route = respx.post(TOKEN_URL).mock(return_value=token_response())
        await make_client(client_credentials_config(scope="read:all")).token()

        body = dict(httpx.QueryParams(route.calls.last.request.content.decode()))
        assert body["grant_type"] == "client_credentials"
        assert body["scope"] == "read:all"

    @respx.mock
    async def test_client_auth_defaults_to_basic(self) -> None:
        """RFC 6749 §2.3.1: servers MUST support Basic and MAY support the body
        form, so Basic is the safe default."""
        route = respx.post(TOKEN_URL).mock(return_value=token_response())
        await make_client().token()

        request = route.calls.last.request
        assert request.headers["Authorization"].startswith("Basic ")
        assert b"client_secret" not in request.content

    @respx.mock
    async def test_basic_client_auth_urlencodes_before_base64(self) -> None:
        """RFC 6749 §2.3.1 requires it and hand-rolled clients skip it. It bites
        only when the secret contains `:` or `+`, and then it looks exactly like
        a wrong password."""
        import base64

        route = respx.post(TOKEN_URL).mock(return_value=token_response())
        await make_client(client_credentials_config(client_secret="a+b:c")).token()

        header = route.calls.last.request.headers["Authorization"]
        decoded = base64.b64decode(header.removeprefix("Basic ")).decode()
        assert decoded == "client-abc:a%2Bb%3Ac"

    @respx.mock
    async def test_post_client_auth_puts_the_secret_in_the_body(self) -> None:
        """For the minority of providers that only accept that form."""
        route = respx.post(TOKEN_URL).mock(return_value=token_response())
        await make_client(
            client_credentials_config(client_auth=ClientAuthMethod.POST)
        ).token()

        request = route.calls.last.request
        assert "Authorization" not in request.headers
        body = dict(httpx.QueryParams(request.content.decode()))
        assert body["client_id"] == "client-abc" and body["client_secret"] == SECRET

    @respx.mock
    async def test_expires_in_becomes_an_absolute_expiry(self) -> None:
        """A relative lifetime means nothing once the value has sat in Postgres
        for an hour."""
        respx.post(TOKEN_URL).mock(return_value=token_response(expires_in=3600))
        clock = Clock()
        token = await make_client(clock=clock).token()

        assert token.expires_at == T0 + timedelta(seconds=3600)

    @respx.mock
    async def test_unrecognised_response_fields_are_kept(self) -> None:
        """Salesforce returns the org's API host in its token response and every
        later call needs it."""
        respx.post(TOKEN_URL).mock(
            return_value=token_response(instance_url="https://org.my.salesforce.com")
        )
        token = await make_client().token()
        assert token.extra["instance_url"] == "https://org.my.salesforce.com"

    @respx.mock
    async def test_an_id_token_is_dropped_rather_than_stored(self) -> None:
        """An OIDC `id_token` is a bearer credential describing a user. No
        server-to-server connector sends one, and keeping a credential we never
        use is liability with no upside (`docs/security-and-privacy.md` §6.1)."""
        respx.post(TOKEN_URL).mock(return_value=token_response(id_token="eyJ.header.sig"))
        token = await make_client().token()
        assert "id_token" not in token.extra

    @respx.mock
    async def test_extra_params_reach_the_body(self) -> None:
        route = respx.post(TOKEN_URL).mock(return_value=token_response())
        await make_client(
            client_credentials_config(extra_params={"duration": "permanent"})
        ).token()

        body = dict(httpx.QueryParams(route.calls.last.request.content.decode()))
        assert body["duration"] == "permanent"

    @respx.mock
    async def test_a_missing_access_token_is_permanent_not_transient(self) -> None:
        """A 200 with no token is a provider contract violation; retrying it
        forever would hide the breakage behind a flaky-provider counter."""
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={"token_type": "bearer"}))
        with pytest.raises(PermanentError):
            await make_client().token()

    @respx.mock
    async def test_a_non_json_body_is_permanent(self) -> None:
        """Usually an HTML error page from a proxy."""
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, text="<html>oops</html>"))
        with pytest.raises(PermanentError):
            await make_client().token()


class TestRefreshTokenGrant:
    @respx.mock
    async def test_sends_the_stored_refresh_token(self) -> None:
        route = respx.post(TOKEN_URL).mock(return_value=token_response())
        config = OAuth2Config(
            token_url=TOKEN_URL,
            client_id="c",
            client_secret=SECRET,
            grant=OAuth2Grant.REFRESH_TOKEN,
            refresh_token="r-seed",
        )
        await make_client(config).token()

        body = dict(httpx.QueryParams(route.calls.last.request.content.decode()))
        assert body["grant_type"] == "refresh_token" and body["refresh_token"] == "r-seed"

    @respx.mock
    async def test_a_rotated_refresh_token_replaces_the_old_one(self) -> None:
        respx.post(TOKEN_URL).mock(return_value=token_response(refresh_token="r-new"))
        store = InMemoryTokenStore()
        config = OAuth2Config(
            token_url=TOKEN_URL,
            client_id="c",
            client_secret=SECRET,
            grant=OAuth2Grant.REFRESH_TOKEN,
            refresh_token="r-seed",
        )
        await make_client(config, store=store).token()

        stored = await store.load("acct_1")
        assert stored is not None and stored.refresh_token == "r-new"

    @respx.mock
    async def test_an_omitted_refresh_token_does_not_erase_the_working_one(self) -> None:
        """RFC 6749 §6 lets a refresh response omit `refresh_token`, and then the
        old one is still valid. Writing `payload.get(...)` straight into the
        field bricks the account -- and it only breaks an hour later, when the
        access token expires and nothing is left to renew it."""
        respx.post(TOKEN_URL).mock(return_value=token_response(refresh_token=None))
        store = InMemoryTokenStore()
        config = OAuth2Config(
            token_url=TOKEN_URL,
            client_id="c",
            client_secret=SECRET,
            grant=OAuth2Grant.REFRESH_TOKEN,
            refresh_token="r-seed",
        )
        await make_client(config, store=store).token()

        stored = await store.load("acct_1")
        assert stored is not None and stored.refresh_token == "r-seed"

    @respx.mock
    async def test_no_refresh_token_anywhere_is_terminal_and_sends_nothing(self) -> None:
        """Never linked, or the token was lost. No request could help."""
        route = respx.post(TOKEN_URL).mock(return_value=token_response())
        config = OAuth2Config(
            token_url=TOKEN_URL, client_id="c", grant=OAuth2Grant.REFRESH_TOKEN
        )
        with pytest.raises(AuthError):
            await make_client(config).token()
        assert route.call_count == 0

    @respx.mock
    async def test_the_stored_refresh_token_beats_the_configured_seed(self) -> None:
        """A rotating provider invalidated the seed the moment it issued the
        first replacement; re-sending it would be an `invalid_grant`."""
        route = respx.post(TOKEN_URL).mock(return_value=token_response())
        store = InMemoryTokenStore()
        await store.save(
            "acct_1", StoredToken(access_token="old", refresh_token="r-current", expires_at=T0)
        )
        config = OAuth2Config(
            token_url=TOKEN_URL,
            client_id="c",
            client_secret=SECRET,
            grant=OAuth2Grant.REFRESH_TOKEN,
            refresh_token="r-seed",
        )
        await make_client(config, store=store).token()

        body = dict(httpx.QueryParams(route.calls.last.request.content.decode()))
        assert body["refresh_token"] == "r-current"


# --------------------------------------------------------------------------- #
# oauth.py -- the refresh window
# --------------------------------------------------------------------------- #


class TestPreemptiveRefresh:
    """The behaviour the whole module exists for. Refreshing on the 401 instead
    means every worker discovers expiry at the same instant and stampedes."""

    @respx.mock
    async def test_a_valid_token_is_reused_without_a_request(self) -> None:
        route = respx.post(TOKEN_URL).mock(return_value=token_response(expires_in=3600))
        clock = Clock()
        client = make_client(clock=clock)

        await client.token()
        clock.advance(1800)
        await client.token()

        assert route.call_count == 1

    @respx.mock
    async def test_no_refresh_just_outside_the_margin(self) -> None:
        """The boundary, from the safe side. 3600 - 300 = 3300 seconds in, one
        second early."""
        route = respx.post(TOKEN_URL).mock(return_value=token_response(expires_in=3600))
        clock = Clock()
        client = make_client(clock=clock)

        await client.token()
        clock.advance(3600 - DEFAULT_REFRESH_MARGIN_SECONDS - 1)
        await client.token()

        assert route.call_count == 1

    @respx.mock
    async def test_refresh_inside_the_margin_while_the_token_is_still_valid(self) -> None:
        """The whole point: the token has five minutes of life left and is
        replaced anyway, so nobody is ever waiting on a 401 to find out."""
        route = respx.post(TOKEN_URL).mock(
            side_effect=[
                token_response(access_token="a-1", expires_in=3600),
                token_response(access_token="a-2", expires_in=3600),
            ]
        )
        clock = Clock()
        client = make_client(clock=clock)

        await client.token()
        clock.advance(3600 - DEFAULT_REFRESH_MARGIN_SECONDS)
        assert (await client.headers())["Authorization"] == "Bearer a-2"
        assert route.call_count == 2

    @respx.mock
    async def test_the_margin_is_configurable(self) -> None:
        """A provider with clocks that drift, or one whose tokens live 60
        seconds, needs a different window."""
        route = respx.post(TOKEN_URL).mock(return_value=token_response(expires_in=3600))
        clock = Clock()
        client = make_client(client_credentials_config(refresh_margin_seconds=60), clock=clock)

        await client.token()
        clock.advance(3600 - 120)
        await client.token()

        assert route.call_count == 1, "a 60s margin must not refresh two minutes early"

    @respx.mock
    async def test_invalidate_forces_the_next_call_to_refresh(self) -> None:
        """What the runtime calls after a resource-server 401."""
        route = respx.post(TOKEN_URL).mock(
            side_effect=[
                token_response(access_token="a-1"),
                token_response(access_token="a-2"),
            ]
        )
        client = make_client()

        await client.token()
        await client.invalidate()
        assert (await client.token()).access_token == "a-2"
        assert route.call_count == 2

    @respx.mock
    async def test_invalidate_keeps_the_refresh_token(self) -> None:
        """Deleting the row would turn "one access token was revoked early" into
        "this account needs a human at the consent screen"."""
        respx.post(TOKEN_URL).mock(return_value=token_response(refresh_token="r-1"))
        store = InMemoryTokenStore()
        config = OAuth2Config(
            token_url=TOKEN_URL,
            client_id="c",
            client_secret=SECRET,
            grant=OAuth2Grant.REFRESH_TOKEN,
            refresh_token="r-seed",
        )
        client = make_client(config, store=store)

        await client.token()
        await client.invalidate()

        stored = await store.load("acct_1")
        assert stored is not None and stored.refresh_token == "r-1"

    async def test_invalidate_on_an_unlinked_account_is_a_no_op(self) -> None:
        await make_client().invalidate()


class TestSingleFlight:
    """The lock exists because the margin only spreads refreshes across time; the
    ones that still coincide have to be collapsed."""

    @respx.mock
    async def test_concurrent_callers_trigger_exactly_one_refresh(self) -> None:
        route = respx.post(TOKEN_URL).mock(return_value=token_response(access_token="a-1"))
        client = make_client(store=YieldingStore())

        tokens = await asyncio.gather(*(client.token() for _ in range(10)))

        assert route.call_count == 1, "the token endpoint was stampeded"
        assert {t.access_token for t in tokens} == {"a-1"}

    @respx.mock
    async def test_the_losers_of_the_race_read_the_winners_token(self) -> None:
        """Without the second read *under* the lock, the lock only serialises the
        stampede -- every worker still POSTs, just politely, one at a time."""
        route = respx.post(TOKEN_URL).mock(
            side_effect=[
                token_response(access_token=f"a-{i}") for i in range(1, 6)
            ]
        )
        client = make_client(store=YieldingStore())

        tokens = await asyncio.gather(*(client.token() for _ in range(5)))

        assert route.call_count == 1
        assert {t.access_token for t in tokens} == {"a-1"}

    @respx.mock
    async def test_two_accounts_do_not_share_one_token(self) -> None:
        """Single-flight collapses refreshes for *one* account. Collapsing across
        accounts would hand one tenant another tenant's credential -- the failure
        mode that turns a caching optimisation into a security incident."""
        route = respx.post(TOKEN_URL).mock(return_value=token_response())
        store = YieldingStore()
        first = OAuth2Client(
            client_credentials_config(), account_id="acct_1", store=store, now=Clock()
        )
        second = OAuth2Client(
            client_credentials_config(), account_id="acct_2", store=store, now=Clock()
        )

        await asyncio.gather(first.token(), second.token())
        assert route.call_count == 2

    @respx.mock
    async def test_a_failed_refresh_releases_the_lock(self) -> None:
        """Otherwise one bad credential deadlocks every other coroutine that
        needs a token for the account, and the run hangs instead of failing."""
        respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(401, json={"error": "invalid_grant"})
        )
        client = make_client(store=YieldingStore())

        with pytest.raises(AuthError):
            await client.token()
        with pytest.raises(AuthError):
            await asyncio.wait_for(client.token(), timeout=1.0)


# --------------------------------------------------------------------------- #
# oauth.py -- failure taxonomy
# --------------------------------------------------------------------------- #


class TestFailuresAreClassifiedAndTerminal:
    @respx.mock
    async def test_401_raises_auth_error_and_does_not_retry(self) -> None:
        """A client that loops on rejected credentials earns an application-level
        ban rather than an account-level one."""
        route = respx.post(TOKEN_URL).mock(return_value=httpx.Response(401, json={}))

        with pytest.raises(AuthError):
            await make_client().token()
        assert route.call_count == 1, "a rejected grant must not be retried"

    @respx.mock
    async def test_invalid_grant_raises_auth_error(self) -> None:
        """A revoked refresh token stays revoked."""
        respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(400, json={"error": "invalid_grant"})
        )
        with pytest.raises(AuthError) as caught:
            await make_client().token()
        assert caught.value.details["oauth_error"] == "invalid_grant"
        assert caught.value.retryable is False

    @respx.mock
    async def test_auth_error_names_the_account_so_it_can_be_flagged(self) -> None:
        """`AuthError` without an account halts the run and tells nobody which
        row to mark `needs_reauth`."""
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(401, json={}))
        with pytest.raises(AuthError) as caught:
            await make_client().token()
        assert caught.value.account_id == "acct_1" and caught.value.connector == "demo"

    @respx.mock
    async def test_invalid_scope_is_a_configuration_error_not_a_reauth(self) -> None:
        """Re-linking the account cannot fix a scope the operator never asked
        for; it would fail again identically."""
        respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(400, json={"error": "invalid_scope"})
        )
        with pytest.raises(ConnectorConfigurationError):
            await make_client().token()

    @respx.mock
    async def test_429_is_a_quota_error_carrying_retry_after(self) -> None:
        """A quota wall is not a bad credential: flagging the account
        `needs_reauth` would send an operator to fix nothing."""
        respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(429, headers={"Retry-After": "90"}, json={})
        )
        with pytest.raises(QuotaError) as caught:
            await make_client().token()
        assert caught.value.retry_after_seconds == 90.0

    @respx.mock
    async def test_5xx_is_transient_and_still_not_retried_here(self) -> None:
        """Retryable, but the runtime owns the backoff
        (`docs/connector-spec.md` §1)."""
        route = respx.post(TOKEN_URL).mock(return_value=httpx.Response(503, json={}))
        with pytest.raises(TransientError):
            await make_client().token()
        assert route.call_count == 1

    @respx.mock
    async def test_a_timeout_is_transient(self) -> None:
        route = respx.post(TOKEN_URL).mock(side_effect=httpx.ConnectTimeout("too slow"))
        with pytest.raises(TransientError) as caught:
            await make_client().token()
        assert caught.value.retryable is True
        assert route.call_count == 1

    @respx.mock
    async def test_a_connection_error_is_transient(self) -> None:
        respx.post(TOKEN_URL).mock(side_effect=httpx.ConnectError("no route"))
        with pytest.raises(TransientError):
            await make_client().token()

    @respx.mock
    async def test_a_200_carrying_an_error_body_is_still_an_auth_failure(self) -> None:
        """A real minority of providers answer a rejected grant with 200. Letting
        it through fails later on "no access_token" as a `PermanentError`, which
        leaves the account unflagged and quietly failing every run."""
        respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"error": "invalid_grant"})
        )
        with pytest.raises(AuthError):
            await make_client().token()

    @respx.mock
    async def test_a_terminal_error_code_wins_over_an_odd_status(self) -> None:
        """The status a provider attaches to `invalid_grant` is not reliable, and
        a gateway in front of the token endpoint can rewrite it. Classifying that
        as `PermanentError` would leave the account unflagged and failing."""
        respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(418, json={"error": "invalid_client"})
        )
        with pytest.raises(AuthError):
            await make_client().token()

    @respx.mock
    async def test_an_unexpected_4xx_is_permanent(self) -> None:
        """A 404 on the token endpoint means the URL is wrong, which no retry
        and no re-authorisation fixes."""
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(404, json={}))
        with pytest.raises(PermanentError):
            await make_client().token()

    @respx.mock
    async def test_a_hostile_error_code_is_not_propagated(self) -> None:
        """Providers under load have been known to echo the offending request
        into `error`, and the request is a form carrying the client secret."""
        respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(400, json={"error": f"client_secret={SECRET}"})
        )
        with pytest.raises(AuthError) as caught:
            await make_client().token()
        assert SECRET not in repr(caught.value.details)


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


class TestNothingRendersASecret:
    """`docs/security-and-privacy.md` §4.2: plaintext credentials exist only in
    memory. Every object below holds one, so every rendering path is checked."""

    def test_stored_token_repr_and_str(self) -> None:
        token = StoredToken(access_token=SECRET, refresh_token=SECRET + "-r")
        for rendered in (repr(token), str(token), f"{token}", f"{token!r}", f"{token!s}"):
            assert SECRET not in rendered
        assert "<redacted>" in repr(token)

    def test_stored_token_log_fields_carry_no_material(self) -> None:
        token = StoredToken(
            access_token=SECRET, refresh_token=SECRET, expires_at=T0, scope="read"
        )
        fields = token.to_log_fields()
        assert SECRET not in repr(fields)
        assert fields["refreshable"] is True and fields["scope"] == "read"

    def test_log_field_names_survive_the_structlog_redactor(self) -> None:
        """`backend/core/logging.py` blanks any key matching `token`, so
        `token_type` and `has_refresh_token` would arrive as `***redacted***` and
        say nothing. These fields are not secret and are worth reading."""
        import re

        sensitive = re.compile(
            r"password|passwd|secret|token|credential|authorization|cookie"
            r"|api[_-]?key|_key$|^key$",
            re.IGNORECASE,
        )
        fields = StoredToken(access_token=SECRET).to_log_fields()
        assert not [key for key in fields if sensitive.search(key)]

    def test_strategy_reprs(self) -> None:
        strategies: list[HeaderAuth] = [
            ApiKeyAuth(key=SECRET),
            BearerAuth(token=SECRET),
            BasicAuth(username="alice", password=SECRET),
        ]
        for strategy in strategies:
            assert SECRET not in repr(strategy)
            assert SECRET not in str(strategy)
            assert "<redacted>" in repr(strategy)

    def test_dataclass_default_repr_is_actually_overridden(self) -> None:
        """`@dataclass` generates a `__repr__` that prints every field. Losing
        the override to a refactor is the exact way this leaks, so the check is
        that the base64 body is absent too, not just the plaintext."""
        auth = BasicAuth(username="alice", password=SECRET)
        assert auth.headers()["Authorization"].removeprefix("Basic ") not in repr(auth)

    def test_oauth_config_repr(self) -> None:
        config = client_credentials_config(client_secret=SECRET, refresh_token=SECRET)
        assert SECRET not in repr(config) and SECRET not in str(config)
        assert "client-abc" in repr(config), "non-secret fields stay readable"

    def test_oauth_client_repr(self) -> None:
        assert SECRET not in repr(make_client())

    @respx.mock
    async def test_a_failing_client_never_puts_the_token_in_an_error(self) -> None:
        """`ConnectorError.to_log_fields()` is what actually reaches the
        aggregator."""
        respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(400, text=f"denied for client_secret={SECRET}")
        )
        with pytest.raises(AuthError) as caught:
            await make_client().token()
        assert SECRET not in repr(caught.value)
        assert SECRET not in repr(caught.value.to_log_fields())


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


class TestClientLifecycle:
    @respx.mock
    async def test_aclose_does_not_close_an_injected_client(self) -> None:
        """Closing a lent client tears the connection pool out from under the
        connector that lent it."""
        respx.post(TOKEN_URL).mock(return_value=token_response())
        async with httpx.AsyncClient() as http:
            client = OAuth2Client(
                client_credentials_config(),
                account_id="acct_1",
                store=InMemoryTokenStore(),
                client=http,
                now=Clock(),
            )
            await client.token()
            await client.aclose()
            assert not http.is_closed

    @respx.mock
    async def test_aclose_closes_a_client_it_opened(self) -> None:
        respx.post(TOKEN_URL).mock(return_value=token_response())
        client = make_client()
        await client.token()
        await client.aclose()
        # Reaching into the private attribute is the point: ownership of the
        # client is the thing under test and there is no public way to observe it.
        assert client._client.is_closed

    @respx.mock
    async def test_the_user_agent_is_sent(self) -> None:
        """Providers ban unidentified clients, and `docs/connector-spec.md`
        expects every outbound call to be attributable to us."""
        route = respx.post(TOKEN_URL).mock(return_value=token_response())
        client = OAuth2Client(
            client_credentials_config(),
            account_id="acct_1",
            store=InMemoryTokenStore(),
            user_agent="omnisense/test",
            now=Clock(),
        )
        await client.token()
        assert route.calls.last.request.headers["User-Agent"] == "omnisense/test"
