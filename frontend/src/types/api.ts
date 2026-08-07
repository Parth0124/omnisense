/**
 * The wire vocabulary every endpoint shares: envelopes, pages and problem documents.
 *
 * These types are hand-mirrored from `backend/schemas/common.py` and
 * `docs/api-reference.md` §3. There is no generator (`docs/architecture.md` §6.2 rule 6
 * makes the HTTP surface the only contract between the two halves of the system, and
 * `docs/frontend.md` §6 open question 4 records that the drift risk is accepted for now),
 * so anything declared here is a claim about the backend that a reviewer has to check.
 * Keep the field order aligned with the Python so the two can be diffed.
 *
 * The one idea worth internalising before reading further is the encoding of *open*
 * enumerations. §1 of the API reference makes "adding a new enum member to a response
 * field" a backward-compatible change and requires clients to tolerate members they have
 * never heard of. A plain TypeScript union does the opposite: the day the backend starts
 * returning `platform: "mastodon"`, every exhaustive `switch` silently loses its default
 * and every `zod` enum throws at the boundary, taking down a page over a value it only
 * needed to display. `Open<T>` below is the encoding that satisfies both halves — known
 * members still autocomplete, unknown members still parse.
 */

/**
 * An enumeration that is closed *today* and open *forever*.
 *
 * `string & {}` is not a typo and not a no-op. It is a string type TypeScript refuses to
 * collapse into the literal union next to it, which is what keeps editor completion
 * listing the known members while the type as a whole still accepts an unknown one.
 * Written as bare `| string` the union widens to `string`, every known member disappears
 * from completion, and the next person deletes the literals as dead weight.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export type Open<T extends string> = T | (string & {});

/**
 * The `page` object of §3.4. Cursor-based, and deliberately without a total.
 *
 * There is no `total` here because the backend refuses to compute one: `COUNT(*)` over a
 * filtered slice of a continuously-written table is unbounded work and immediately stale
 * (`backend/schemas/common.py::PageInfo`). A UI that wants "showing 50 of 3,187" has to
 * do without. The alternative — rendering a number that was already wrong when it was
 * serialised — is worse, because the page then does arithmetic on it.
 *
 * `next_cursor` is null exactly when `has_more` is false; the backend derives one from
 * the other in `PageInfo.of`, so a client may branch on either.
 */
export interface PageInfo {
  limit: number;
  next_cursor: string | null;
  has_more: boolean;
}

/** The collection envelope of §3.4: `{"items": [...], "page": {...}}`. */
export interface Page<T> {
  items: T[];
  page: PageInfo;
}

/**
 * The stable, machine-readable error slugs of §3.3.
 *
 * Branch on this, never on `title` or `detail` — `detail` is occurrence-specific and is
 * the only part of a problem document that is safe to show a user verbatim.
 */
export type ApiErrorCode = Open<
  | 'validation_error'
  | 'malformed_request'
  | 'unauthenticated'
  | 'permission_denied'
  | 'not_found'
  | 'conflict'
  | 'idempotency_key_reuse'
  | 'sync_already_running'
  | 'report_not_ready'
  | 'rate_limited'
  | 'quota_exceeded'
  | 'connector_auth_failed'
  | 'upstream_unavailable'
  | 'timeout'
  | 'internal_error'
  // Not in the §3.3 catalogue. `http_error` is what `backend/api/errors.py` emits for a
  // router-level 404/405; the last two never reach the server at all and are minted by
  // `lib/api/client.ts` so that *every* failure a component sees has the same shape.
  | 'http_error'
  | 'network_error'
  | 'malformed_response'
>;

/** One field-level failure inside a `validation_error` problem document. */
export interface ValidationIssue {
  /** Dotted path to the offending field, e.g. `body.scope.time_window.from`. */
  location: string;
  /** The rule that rejected it, e.g. `must be timezone-aware`. */
  rule: string;
}

/**
 * RFC 7807 problem document, as `backend/core/exceptions.py::to_problem` actually emits it.
 *
 * Mirrored from the *code*, not from `docs/api-reference.md` §3.3, because the two
 * disagree and `backend/schemas/common.py::ProblemDocument` documents the disagreement:
 * the doc specifies top-level `code` and `request_id` members, while the built handler
 * encodes the code into `type` (`https://omnisense.dev/errors/<code>`) and `title`, and
 * returns the correlation id in the `X-Request-ID` header only.
 *
 * `ApiError` in `lib/api/client.ts` recovers `code` from `type` and `request_id` from the
 * header, so callers see the shape §3.3 promised no matter which side moves first.
 */
export interface ProblemDocument {
  /** `https://omnisense.dev/errors/<code>`. Stable per error class. */
  type: string;
  /** The error code with underscores spaced out. Does not vary with occurrence. */
  title: string;
  status: number;
  /** Occurrence-specific and safe to show a user. Never contains internals or secrets. */
  detail: string;
  /** Path of the failing request. */
  instance?: string | null;
  /**
   * Structured, non-sensitive context. Per-error: `{"errors": [...]}` for a validation
   * failure, `{"resource", "id"}` for a 404. Never secrets, never fetched content.
   */
  details?: ({ errors?: ValidationIssue[] } & Record<string, unknown>) | null;
  /** Seconds to wait. Present on `429` and on `report_not_ready` (§3.3). */
  retry_after?: number | null;
}

/** Sort direction for the collection endpoints that accept one (§4.7). */
export type SortOrder = 'asc' | 'desc';

/**
 * Half-open time bound, `[from, to)`, as it appears in query strings and request bodies.
 *
 * Half-open rather than closed because consecutive daily windows must tile without
 * overlap — a Signal sitting exactly on midnight belongs to exactly one bucket, and the
 * dashboard's volume chart is built from those buckets. The convention is inherited from
 * `services/signal_service.py`; restating it as a closed interval anywhere in this app
 * double-counts one signal per boundary per series.
 */
export interface TimeWindow {
  /** RFC 3339 in UTC with an explicit offset. Naive datetimes are rejected with 422. */
  from?: string | null;
  to?: string | null;
}

/**
 * Token accounting, shared by investigations and single-agent runs (§4.2, §4.6).
 *
 * `tool_calls` is nullable and the nullability is load-bearing: nothing in `models/orm/`
 * records a tool-call count yet, so the backend reports `null` rather than `0`
 * (`backend/schemas/investigation.py` module docstring). "Nobody called a tool" and "we
 * do not measure tool calls" are different claims, and a UI cannot tell them apart once
 * the second has been rendered as the first.
 */
export interface Usage {
  input_tokens: number;
  output_tokens: number;
  tool_calls: number | null;
}
