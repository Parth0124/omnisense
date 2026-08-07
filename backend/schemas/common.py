"""The wire vocabulary every endpoint shares: envelopes, pages, problem documents.

These are **DTOs, not domain models**. Nothing here is persisted, nothing here is
passed to a service, and nothing in `models/` is ever returned from a handler
directly. The separation is the point, and it is not ceremony:

`models/signal.py` is the pipeline's contract and grows fields whenever the
pipeline needs one -- `embeddings[]`, `lineage.confidence_components`,
`author.follower_count`. `docs/api-reference.md` §4.7 documents a *subset* of
that, and the subset was chosen deliberately: embeddings are megabytes nobody
asked for, and `author.follower_count` is personal data that §6.1 of
`docs/security-and-privacy.md` keeps out of responses. If a handler returned an
ORM row or a `SignalView`, every future pipeline field would ship to every client
the day it was added, and the first anyone would notice is a privacy review.

So the rule this module exists to enforce: **a response is built by naming its
fields.** A field that appears on the wire appears in a class here, and a field
that does not appear here cannot reach a client by accident.

Two further decisions worth stating.

**Requests forbid unknown fields** (`docs/api-reference.md` §3.2: "Unknown fields
in a request body are rejected (`422 validation_error`), not ignored, so typos
surface immediately"). The alternative -- ignoring them -- turns
`{"max_step": 10}` into a silent acceptance of the default 50, and the caller
discovers the typo when the bill arrives. `RequestModel` is where that is set,
once, rather than on each of the three request bodies.

**Responses forbid unknown fields too**, which is a different bet. We author
every response object here, so an unexpected key means *our* bug -- a handler
passing a field the schema does not declare -- and it is far better caught by a
`ValidationError` in a test than by a client parsing a key we never meant to
publish. Client-side tolerance runs the other way (§1: "clients must ignore
unknown fields"), and nothing here constrains that.
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DEFAULT_PAGE_LIMIT",
    "MAX_PAGE_LIMIT",
    "PROBLEM_MEDIA_TYPE",
    "Page",
    "PageInfo",
    "ProblemDocument",
    "RequestModel",
    "ResponseModel",
    "problem_responses",
]

PROBLEM_MEDIA_TYPE: Final = "application/problem+json"
"""Mirrors `backend/api/errors.py::PROBLEM_CONTENT_TYPE`.

Restated rather than imported to keep `backend/schemas/` free of any dependency
on `backend/api/`: the schemas describe the wire and are imported *by* the API,
and importing back the other way would make the pair mutually recursive the first
time `errors.py` wants to reference `ProblemDocument` for its own OpenAPI entry.
`tests/unit/backend/test_routes.py` asserts the two strings agree.
"""

DEFAULT_PAGE_LIMIT: Final = 50
MAX_PAGE_LIMIT: Final = 200
"""Page bounds from `docs/api-reference.md` §3.4.

Deliberately duplicated from `services/signal_service.py`, which states the same
two numbers for the same reason -- they are part of a published HTTP contract.
The duplication is asserted away in `tests/unit/backend/test_routes.py`: the two
modules must agree, and a test is the only thing that keeps them agreeing without
`backend/schemas/` importing a service.

§3.4 is explicit that a `limit` above the maximum is **rejected**, never silently
clamped. Clamping makes a client believe it has reached the end of a collection
when it has only reached the end of a page.
"""


class RequestModel(BaseModel):
    """Base for every request body. Unknown fields are rejected.

    `str_strip_whitespace` is on because a query of `"  "` and a query of `""`
    are the same mistake, and only one of them would be caught by a `min_length`
    without stripping first.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
        validate_default=True,
    )


class ResponseModel(BaseModel):
    """Base for every response body.

    `populate_by_name` matters more here than it looks. Several documented wire
    fields are Python keywords -- `from` and `to` on a time window, `from` and
    `to` on a graph edge -- so they are declared as `from_` / `to_` with an
    `alias`. Without this flag the constructor would only accept the alias, and
    every handler would build responses using the very keyword that does not
    parse.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_default=True,
    )


class PageInfo(ResponseModel):
    """The `page` object of §3.4. Cursor-based; deliberately without a total.

    §3.4 declines to return a total count and this respects that. `COUNT(*)` over
    a filtered slice of a continuously-written table is unbounded work *and*
    immediately stale, and a number that is already wrong when it renders is
    worse than an absent one because the UI will do arithmetic on it.
    """

    limit: int = Field(ge=1, le=MAX_PAGE_LIMIT)
    next_cursor: str | None = Field(
        default=None,
        description="Opaque resume token. Null exactly when `has_more` is false.",
    )
    has_more: bool = False

    @classmethod
    def of(cls, *, limit: int, next_cursor: str | None) -> PageInfo:
        """Build a page envelope, deriving `has_more` from the cursor.

        A factory rather than two constructor arguments because the invariant
        "`next_cursor` is null exactly when `has_more` is false" (§3.4) is stated
        in the contract and is trivially violable by a caller passing both by
        hand. Deriving one from the other makes the pair unable to disagree.
        """
        return cls(limit=limit, next_cursor=next_cursor, has_more=next_cursor is not None)


class Page[ItemT](ResponseModel):
    """The collection envelope of §3.4: `{"items": [...], "page": {...}}`.

    Generic so that `Page[SignalItem]` and `Page[StepItem]` produce distinct
    OpenAPI schemas rather than a single `items: [any]`. A shared envelope is
    what lets a client write one pagination loop for every collection in the API
    instead of one per endpoint.
    """

    items: list[ItemT]
    page: PageInfo


class ProblemDocument(BaseModel):
    """RFC 7807 problem document, exactly as `backend/api/errors.py` emits it.

    Declared for the OpenAPI schema, not for construction: the handlers in
    `errors.py` build these as plain dicts from `OmniSenseError.to_problem()`,
    and nothing here should become a second place that decides what an error
    looks like.

    `extra="allow"` for that reason. `to_problem()` attaches a `details` object
    whose keys are per-error -- `{"resource": ..., "id": ...}` for a 404,
    `{"errors": [...]}` for a validation failure -- and a model that forbade them
    would document a shape the API does not actually produce.

    Note one divergence from `docs/api-reference.md` §3.3, which is the built
    code's and not this module's: the doc specifies top-level `code` and
    `request_id` members, while `to_problem()` encodes the code into `type`
    (`https://omnisense.dev/errors/<code>`) and `title`, and returns the
    correlation id in the `X-Request-ID` header only. Clients branch on `type`.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="Stable per error class: https://omnisense.dev/errors/<code>.")
    title: str = Field(description="The error code, spaced. Does not vary with occurrence.")
    status: int = Field(description="Mirrors the HTTP status.")
    detail: str = Field(description="Occurrence-specific and safe to show a user.")
    instance: str | None = Field(default=None, description="Path of the failing request.")
    details: dict[str, Any] | None = Field(
        default=None, description="Structured, non-sensitive context. Never secrets or content."
    )


def problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """Build the OpenAPI `responses=` entry for a route's documented failures.

    Every non-2xx response in this API is `application/problem+json` (§3.3), so
    the entry is identical for every status and every route. Writing it out per
    route is how one endpoint ends up documenting `application/json` for its 404
    and quietly teaching a client to parse the wrong shape.
    """
    described = {
        400: "Malformed JSON, bad content type, or an unparseable cursor.",
        401: "Missing, expired or unparseable credential.",
        403: "Authenticated, but out of scope or the wrong tenant.",
        404: "Unknown resource, or a resource belonging to another tenant.",
        409: "The request conflicts with the current state of the resource.",
        422: "Well-formed request that failed a validation rule.",
        429: "Inbound rate limit or plan quota exhausted.",
        500: "Unhandled failure.",
        501: "A documented capability whose backing store does not exist yet.",
        502: "A downstream dependency failed.",
        503: "Not ready to accept work; retry with backoff.",
        504: "Server-side deadline exceeded.",
    }
    return {
        status: {
            "description": described.get(status, "Error."),
            "model": ProblemDocument,
            # An empty media-type entry, not an inline schema. FastAPI fills the
            # schema in from `model` for whichever media types are named here;
            # inlining `model_json_schema()` instead would embed a `$defs` block
            # that resolves against the model's own namespace rather than the
            # document's `#/components/schemas`, producing dangling `$ref`s in
            # the published OpenAPI.
            "content": {PROBLEM_MEDIA_TYPE: {}},
        }
        for status in statuses
    }
