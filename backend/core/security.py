"""Cryptographic primitives: signing, verification, hashing, redaction.

Everything in this module is small, pure and boring, which is the intent.
Security code that is interesting is security code that is wrong. There is no
policy here -- no decision about who may do what, no scope vocabulary, no token
lifetime. Those live in `backend/api/deps.py`, which is where an auditor looks
for them. This module answers only "is this signature valid" and "is this the
right secret".

**Nothing here raises a domain exception.** Every function returns a value or a
bool, and the caller decides what an invalid signature *means*. That separation
matters because the right response differs by call site: an API request gets a
bare 401 with no reason (telling an attacker which half of a forgery succeeded is
a gift), while a worker verifying a webhook signature wants the reason in its
logs.

**Constant-time comparison everywhere it matters.** `==` on bytes returns as soon
as it finds a difference, and the time that takes leaks how many leading bytes
matched. Over enough requests that is sufficient to construct a valid signature
one byte at a time. Every comparison of a secret in this module goes through
`hmac.compare_digest`, and the two places where it would be tempting not to --
the API key check and the JWS check -- are exactly the two that are reachable by
an unauthenticated caller.

Layer note: **L1k kernel**. Imports the standard library and
`backend/core/config.py`. Nothing above L1k may reimplement any of this.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping
from typing import Any, Final

__all__ = [
    "API_KEY_ITERATIONS",
    "API_KEY_PREFIX",
    "SUPPORTED_JWT_ALGORITHMS",
    "JwsError",
    "b64url_decode",
    "b64url_encode",
    "constant_time_equals",
    "decode_jws",
    "encode_jws",
    "generate_api_key",
    "hash_api_key",
    "redact",
    "verify_api_key",
]

SUPPORTED_JWT_ALGORITHMS: Final[Mapping[str, Any]] = {
    "HS256": hashlib.sha256,
    "HS384": hashlib.sha384,
    "HS512": hashlib.sha512,
}
"""The algorithms this system will verify. Symmetric only, and deliberately so.

`none` is absent, and its absence is the point. The `alg: none` attack works
because a verifier reads the algorithm out of the *attacker-supplied header* and
honours it; a token declaring `none` then verifies with no signature at all. A
closed table means an unrecognised algorithm has no entry and verification fails
before any signature work happens.

RS256 is also absent. It is not unsafe, but supporting both symmetric and
asymmetric families in one verifier is where the confused-deputy variant lives --
an attacker signs with the *public* key as an HMAC secret, and a verifier that
picks its algorithm from the header accepts it. One family, chosen by
configuration and checked against the header, removes the class.
"""

API_KEY_PREFIX: Final = "osk_"
"""Prefix on every issued API key.

Not decoration: it makes keys greppable in a leaked file, recognisable to secret
scanners, and distinguishable from a JWT at a glance in a support ticket.
"""

API_KEY_ITERATIONS: Final = 210_000
"""PBKDF2-HMAC-SHA256 iterations for API key storage.

The OWASP 2023 floor for this construction. High enough that a leaked hash table
is expensive to attack offline, low enough that verifying one key on a hot path
costs single-digit milliseconds. Stored alongside the hash so the cost can be
raised later without invalidating existing keys -- a bare hash with the iteration
count only in code cannot be upgraded without forcing every key to be reissued.
"""

_API_KEY_BYTES: Final = 32


class JwsError(ValueError):
    """A token failed verification. Carries a reason for the *log*, not the caller.

    A `ValueError` rather than a kernel exception because this module is not
    allowed to decide the HTTP consequence. `backend/api/deps.py` catches it and
    raises a bare `UnauthenticatedError` with nothing in `details`; a worker
    might log the reason and continue. The reason travels on the exception so
    the caller can choose, rather than being baked into a status code here.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- #
# base64url
# --------------------------------------------------------------------------- #


def b64url_encode(raw: bytes) -> str:
    """base64url without padding, as JWS requires."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def b64url_decode(segment: str) -> bytes:
    """Decode one base64url segment, restoring the padding JWS strips.

    Raises `binascii.Error` on malformed input, which every caller here converts
    into a `JwsError`. Left raising rather than returning `None` so a decode
    failure cannot be accidentally treated as empty bytes -- which would compare
    equal to an empty signature.
    """
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded)


def constant_time_equals(left: str | bytes, right: str | bytes) -> bool:
    """Compare two secrets without leaking where they differ.

    Accepts either type and encodes strings to UTF-8 first, because
    `hmac.compare_digest` refuses mixed types and a caller working around that
    with `str(...)` would silently fall back to `==` semantics somewhere.
    """
    left_bytes = left.encode("utf-8") if isinstance(left, str) else left
    right_bytes = right.encode("utf-8") if isinstance(right, str) else right
    return hmac.compare_digest(left_bytes, right_bytes)


# --------------------------------------------------------------------------- #
# JWS / JWT
# --------------------------------------------------------------------------- #


def encode_jws(claims: Mapping[str, Any], *, secret: str, algorithm: str = "HS256") -> str:
    """Sign a claims object into a compact JWS.

    `separators` is pinned so the encoding is byte-stable. It has to be: the
    signature covers the *serialised* header and payload, so a JSON encoder that
    varied its whitespace between versions would produce tokens that fail to
    verify against themselves.
    """
    digest = SUPPORTED_JWT_ALGORITHMS.get(algorithm.upper())
    if digest is None:
        raise JwsError(f"unsupported algorithm {algorithm!r}")

    header_segment = b64url_encode(
        json.dumps({"alg": algorithm.upper(), "typ": "JWT"}, separators=(",", ":")).encode()
    )
    payload_segment = b64url_encode(
        json.dumps(dict(claims), separators=(",", ":"), sort_keys=True).encode()
    )
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, digest).digest()
    return f"{header_segment}.{payload_segment}.{b64url_encode(signature)}"


def decode_jws(
    token: str,
    *,
    secret: str,
    algorithm: str,
    now: float | None = None,
    leeway_seconds: float = 0.0,
) -> dict[str, Any]:
    """Verify a compact JWS and return its claims. Raises `JwsError` otherwise.

    **The signature is verified before any claim is read.** An unverified payload
    is attacker-typed JSON; deciding "this token is for tenant X" from it before
    proving it was signed is precisely how a verifier is talked into trusting an
    unsigned claim. The only thing read first is the header's algorithm, and that
    is compared against the *configured* value rather than trusted.

    **`exp` is required, not optional.** A token with no expiry is a permanent
    credential, and this system has no revocation list -- expiry *is* the
    revocation path, so a token without one can never be withdrawn.
    """
    configured = algorithm.upper()
    digest = SUPPORTED_JWT_ALGORITHMS.get(configured)
    if digest is None:
        raise JwsError(f"unsupported configured algorithm {algorithm!r}")

    parts = token.split(".")
    if len(parts) != 3:
        raise JwsError("malformed_jws")
    header_segment, payload_segment, signature_segment = parts

    try:
        header = json.loads(b64url_decode(header_segment))
        signature = b64url_decode(signature_segment)
    except (binascii.Error, UnicodeDecodeError, ValueError) as error:
        raise JwsError("undecodable_segment") from error

    if not isinstance(header, dict) or header.get("alg") != configured:
        # Compared against the configured algorithm, never merely "is it in the
        # table". A deployment configured for HS256 must reject an HS512 token
        # even though HS512 is supported -- otherwise the header, which the
        # attacker writes, chooses the verification.
        raise JwsError("algorithm_mismatch")

    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    expected = hmac.new(secret.encode("utf-8"), signing_input, digest).digest()
    if not hmac.compare_digest(expected, signature):
        raise JwsError("bad_signature")

    try:
        claims = json.loads(b64url_decode(payload_segment))
    except (binascii.Error, UnicodeDecodeError, ValueError) as error:
        raise JwsError("undecodable_claims") from error
    if not isinstance(claims, dict):
        raise JwsError("claims_not_an_object")

    import time

    moment = now if now is not None else time.time()
    expiry = claims.get("exp")
    if not isinstance(expiry, (int, float)) or isinstance(expiry, bool):
        raise JwsError("expired_or_no_exp")
    if moment >= float(expiry) + leeway_seconds:
        raise JwsError("expired_or_no_exp")

    not_before = claims.get("nbf")
    if (
        isinstance(not_before, (int, float))
        and not isinstance(not_before, bool)
        and moment < float(not_before) - leeway_seconds
    ):
        raise JwsError("not_yet_valid")

    return claims


# --------------------------------------------------------------------------- #
# API keys
# --------------------------------------------------------------------------- #


def generate_api_key() -> str:
    """A new API key. Shown to the user once and never recoverable afterwards.

    `secrets.token_urlsafe`, not `uuid4` and certainly not `random`. A uuid4 has
    122 bits of entropy but a recognisable structure, and `random` is seeded
    predictably enough that keys generated close together are related.
    """
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(_API_KEY_BYTES)}"


def hash_api_key(key: str, *, salt: bytes | None = None, iterations: int | None = None) -> str:
    """Hash a key for storage as `pbkdf2_sha256$iterations$salt$hash`.

    Slow hashing rather than SHA-256, and the reason is the threat model. An API
    key is high-entropy, so a fast hash is *usually* fine -- but keys get written
    into scripts, truncated, reused and occasionally chosen by humans through a
    "bring your own key" path nobody remembered. PBKDF2 makes the leaked-database
    case expensive regardless of how the key was actually produced.

    The iteration count is stored in the encoded value so the cost can be raised
    later without invalidating existing keys. A bare hash with the count only in
    code cannot be upgraded without reissuing every key.
    """
    resolved_salt = salt if salt is not None else secrets.token_bytes(16)
    rounds = iterations if iterations is not None else API_KEY_ITERATIONS
    derived = hashlib.pbkdf2_hmac("sha256", key.encode("utf-8"), resolved_salt, rounds)
    return (
        f"pbkdf2_sha256${rounds}${base64.b64encode(resolved_salt).decode()}"
        f"${base64.b64encode(derived).decode()}"
    )


def verify_api_key(key: str, encoded: str) -> bool:
    """Check a presented key against a stored hash. Never raises.

    Returns `False` for a malformed stored value rather than raising, because the
    alternative is a 500 on the authentication path -- which tells an
    unauthenticated caller that they found a row with a corrupt hash, and takes
    the endpoint down for everyone whose record is fine.
    """
    try:
        scheme, rounds, salt_b64, hash_b64 = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            key.encode("utf-8"),
            base64.b64decode(salt_b64),
            int(rounds),
        )
        return hmac.compare_digest(derived, base64.b64decode(hash_b64))
    except (ValueError, binascii.Error, TypeError):
        return False


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def redact(value: str | None, *, keep: int = 4) -> str:
    """Render a secret safe to log: `osk_1234…` with the body removed.

    Keeping a prefix is a deliberate trade. Full redaction (`***`) makes two
    different leaked keys indistinguishable in a log, so nobody can tell whether
    an incident involves one credential or forty. Four characters is enough to
    correlate and far too few to reconstruct.

    Short values are redacted entirely -- keeping four characters of a six
    character secret is not redaction.
    """
    if not value:
        return "***"
    if len(value) <= keep * 2:
        return "***"
    return f"{value[:keep]}…({len(value)} chars)"
