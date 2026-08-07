/**
 * The one place this application knows how to talk to the OmniSense API.
 *
 * `docs/frontend.md` §5 states the rule plainly: base URL, credential and error
 * normalisation live here and nowhere else, and **no component calls `fetch` directly**.
 * That is not tidiness. Each of those three concerns has exactly one correct answer and
 * several plausible wrong ones, and a second implementation is how a page ends up calling
 * the API without a correlation id, or rendering a raw `TypeError: Failed to fetch` to a
 * user, or — worst — parsing a `problem+json` body as if it were the success shape.
 *
 * What this module guarantees to every caller above it:
 *
 * **Every failure is an `ApiError`.** A 4xx, a 5xx, a DNS failure, a timeout and a body
 * that is not JSON all arrive as the same class with the same fields. Components branch
 * on `error.code`; they never see a `Response`, a `SyntaxError` or a `DOMException`.
 *
 * **Every request carries a correlation id.** `X-Request-ID` is generated per request and
 * echoed by `backend/middleware/request_id.py`, which then binds it to every log line the
 * request produces. Without it, "the report page failed for a customer at 14:03" is
 * unjoinable to anything in the backend logs, which is the difference between a
 * ten-minute investigation and an afternoon.
 *
 * **Repeatable query parameters are repeated, not joined.** `platform=reddit&platform=rss`
 * ORs within a parameter and ANDs across parameters (§4.7). Comma-joining produces a
 * single unknown platform named `reddit,rss` and a 422 that reads like a client bug in
 * the wrong place.
 *
 * The authentication seam is deliberately unfinished, and says so. `docs/frontend.md` §6
 * open question 5 records that token storage, refresh before `ACCESS_TOKEN_TTL_SECONDS`
 * and mid-stream expiry are all undecided. `setAuthTokenProvider` is where that decision
 * lands; guessing at it here — reaching into `localStorage`, say — would bake an
 * unreviewed storage choice for a bearer token into the bottom of the app.
 */

import type { ApiErrorCode, ProblemDocument, ValidationIssue } from '@/types/api';

/**
 * Version prefix, applied once. Handler modules never hardcode it server-side
 * (`docs/api-reference.md` §1) and neither does any resource module here.
 */
export const API_VERSION_PREFIX = '/api/v1';

/**
 * Same-origin path that `next.config.mjs` rewrites onto the backend.
 *
 * Used in the browser when no absolute base URL is configured: a same-origin request needs
 * no CORS preflight, which removes a round trip from every call and removes CORS
 * configuration from the list of things that can break a deployment.
 */
export const API_PROXY_PREFIX = '/api/backend';

/** Media type of an RFC 7807 problem document (`backend/api/errors.py`). */
const PROBLEM_MEDIA_TYPE = 'application/problem+json';

/** Prefix `backend/core/exceptions.py::to_problem` puts in front of every error code. */
const ERROR_TYPE_PREFIX = 'https://omnisense.dev/errors/';

/**
 * Default per-request deadline.
 *
 * Every endpoint this client calls is either a fast read or an explicitly asynchronous
 * `202` — nothing here waits on an investigation. A request still open after 30s has met
 * a server that will not answer, and holding the connection open only delays the retry.
 * The SSE stream does not come through here; it has no deadline by design.
 */
const DEFAULT_TIMEOUT_MS = 30_000;

// --------------------------------------------------------------------------- //
// Errors
// --------------------------------------------------------------------------- //

/**
 * Every failure this client can produce, in one shape.
 *
 * Extends `Error` so it survives being thrown through TanStack Query, React error
 * boundaries and `console.error` without losing its stack. The fields are flattened out
 * of the problem document rather than left nested, because the alternative —
 * `error.problem?.details?.errors?.[0]?.rule` — is a chain that every call site gets
 * subtly wrong at least once.
 */
export class ApiError extends Error {
  /** Stable machine-readable slug. **Branch on this**, never on `title` or `message`. */
  readonly code: ApiErrorCode;
  /** HTTP status, or `0` when the request never reached a server. */
  readonly status: number;
  /** Human-readable and invariant per error class. Safe as a heading. */
  readonly title: string;
  /** Occurrence-specific and safe to show a user. Never contains internals or secrets. */
  readonly detail: string;
  /** Path of the failing request, when the server named one. */
  readonly instance: string | null;
  /** Field-level failures. Populated only for `validation_error`. */
  readonly issues: ValidationIssue[];
  /** Seconds to wait. Present on `429` and on `report_not_ready`. */
  readonly retryAfterSeconds: number | null;
  /** The correlation id sent with the request, for pasting into a bug report. */
  readonly requestId: string | null;
  /** The raw document, for the rare caller that needs a per-error `details` key. */
  readonly problem: ProblemDocument | null;

  constructor(init: {
    code: ApiErrorCode;
    status: number;
    title: string;
    detail: string;
    instance?: string | null;
    issues?: ValidationIssue[];
    retryAfterSeconds?: number | null;
    requestId?: string | null;
    problem?: ProblemDocument | null;
    cause?: unknown;
  }) {
    super(init.detail, { cause: init.cause });
    this.name = 'ApiError';
    this.code = init.code;
    this.status = init.status;
    this.title = init.title;
    this.detail = init.detail;
    this.instance = init.instance ?? null;
    this.issues = init.issues ?? [];
    this.retryAfterSeconds = init.retryAfterSeconds ?? null;
    this.requestId = init.requestId ?? null;
    this.problem = init.problem ?? null;
  }

  /**
   * Whether retrying the identical request could plausibly succeed.
   *
   * Used by the query layer to decide whether to retry at all. Deliberately conservative:
   * a `409 report_not_ready` is retryable because the server said so with a
   * `retry_after`, but a `409 conflict` is not, and a `422` never is — retrying a request
   * the server has already told you is malformed is how a client turns one bad form
   * submission into a rate-limit ban.
   */
  get isRetryable(): boolean {
    if (this.code === 'report_not_ready' || this.code === 'rate_limited') return true;
    if (this.code === 'network_error' || this.code === 'timeout') return true;
    // 502/503/504 are downstream failures the server expects to recover from; 500 is an
    // unhandled bug, and repeating the request repeats the bug.
    return this.status === 502 || this.status === 503 || this.status === 504;
  }

  /** Whether the user has to re-authenticate before anything else will work. */
  get isAuthFailure(): boolean {
    return this.code === 'unauthenticated' || this.code === 'permission_denied';
  }
}

/** Type guard so a `catch (error: unknown)` block can narrow without a cast. */
export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

/**
 * A capability this UI needs and the HTTP contract does not define.
 *
 * `docs/frontend.md` §6 records three of these against `docs/api-reference.md` §2: there
 * is no collection endpoint for investigations or reports, no `GET` counterpart to
 * `POST /connectors/sync` for reading connector accounts, and no endpoint at all behind
 * the Trends and Competitors pages. Those are gaps in the *contract*, not unfinished work
 * in this app, and they cannot be closed from here.
 *
 * Modelled as an error rather than a silent empty result on purpose. An empty array would
 * render as "no connectors configured" — a confident, wrong statement about the system's
 * state — while this renders as "this endpoint is not specified yet" with the missing
 * contract named in `detail`, which is both true and actionable. Status `501` matches
 * `backend/schemas/common.py::problem_responses`, which already reserves 501 for "a
 * documented capability whose backing store does not exist yet".
 */
export class MissingEndpointError extends ApiError {
  /** The contract that has to be written, e.g. `GET /api/v1/connectors/accounts`. */
  readonly missingEndpoint: string;
  /** Where the gap is recorded, so the reader can find the open question. */
  readonly reference: string;

  constructor(missingEndpoint: string, reference: string, detail: string) {
    super({
      code: 'endpoint_not_specified',
      status: 501,
      title: 'Endpoint not specified',
      detail,
    });
    this.name = 'MissingEndpointError';
    this.missingEndpoint = missingEndpoint;
    this.reference = reference;
  }
}

export function isMissingEndpointError(error: unknown): error is MissingEndpointError {
  return error instanceof MissingEndpointError;
}

/**
 * Recover the §3.3 `code` from a problem document.
 *
 * `backend/core/exceptions.py::to_problem` encodes the code into `type` as
 * `https://omnisense.dev/errors/<code>` and does not emit a top-level `code` member, even
 * though `docs/api-reference.md` §3.3 documents one. `backend/schemas/common.py` records
 * that divergence and tells clients to branch on `type`. This function is the one place
 * that knows about it, so the rest of the app can use the field the documentation
 * promised — and so that the day the backend starts sending `code`, only this function
 * changes.
 */
function extractCode(problem: Partial<ProblemDocument> & { code?: unknown }, status: number) {
  if (typeof problem.code === 'string' && problem.code.length > 0) {
    return problem.code as ApiErrorCode;
  }
  if (typeof problem.type === 'string' && problem.type.startsWith(ERROR_TYPE_PREFIX)) {
    const slug = problem.type.slice(ERROR_TYPE_PREFIX.length).trim();
    if (slug.length > 0) return slug as ApiErrorCode;
  }
  // `title` is the code with underscores spaced out, so it is a lossless last resort.
  if (typeof problem.title === 'string' && problem.title.length > 0) {
    return problem.title.trim().replace(/\s+/g, '_') as ApiErrorCode;
  }
  return (status >= 500 ? 'internal_error' : 'http_error') satisfies ApiErrorCode;
}

/** Turn `title` back into something a human reads as a heading. */
function humanizeTitle(code: string): string {
  const spaced = code.replace(/_/g, ' ');
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

// --------------------------------------------------------------------------- //
// Configuration
// --------------------------------------------------------------------------- //

/**
 * Resolve the absolute or same-origin prefix every request is built on.
 *
 * Order matters and each branch has a distinct failure mode:
 *
 * 1. `NEXT_PUBLIC_API_BASE_URL` wins whenever it is set. It is read as a literal member
 *    expression, not through a computed key, because Next inlines `process.env.NEXT_PUBLIC_*`
 *    at build time by textual substitution — `process.env[name]` is `undefined` in the
 *    browser no matter what the variable is set to, and the failure is silent.
 * 2. In the browser with nothing configured, fall back to the same-origin rewrite that
 *    `next.config.mjs` already declares. This is the zero-config development path.
 * 3. On the server with nothing configured, throw. `fetch` cannot resolve a relative URL
 *    outside a browser, and the error it produces (`Failed to parse URL from /api/...`)
 *    names neither the missing variable nor the file that needed it.
 */
export function resolveApiBase(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (configured && configured.length > 0) {
    return `${configured.replace(/\/+$/, '')}${API_VERSION_PREFIX}`;
  }
  if (typeof window !== 'undefined') {
    return API_PROXY_PREFIX;
  }
  throw new Error(
    'NEXT_PUBLIC_API_BASE_URL is not set. Server-side fetches cannot use the ' +
      'same-origin /api/backend rewrite because fetch() outside a browser cannot ' +
      'resolve a relative URL. Copy frontend/.env.local.example to .env.local.',
  );
}

/** Supplies the bearer token for outgoing requests; see the module docstring. */
export type AuthTokenProvider = () => string | null | Promise<string | null>;

let authTokenProvider: AuthTokenProvider = () => null;

/**
 * Install the credential source.
 *
 * Left as a seam rather than implemented because `docs/frontend.md` §6 open question 5
 * leaves storage, refresh and mid-stream expiry undecided. Until it is answered, requests
 * go out unauthenticated and the API answers `401 unauthenticated` — which is a correct,
 * legible failure, unlike a token read from a storage mechanism nobody signed off on.
 */
export function setAuthTokenProvider(provider: AuthTokenProvider): void {
  authTokenProvider = provider;
}

/**
 * Mint a correlation id the backend will actually adopt.
 *
 * `backend/middleware/request_id.py::is_acceptable_request_id` accepts a UUID (dashed or
 * not) or a Crockford ULID and **silently discards anything else**, generating its own id
 * instead. A hand-rolled `req-${Date.now()}-${Math.random()}` therefore looks like it
 * works — the request succeeds, the header comes back populated — while the id in the
 * backend's logs is a different one, and the correlation this whole mechanism exists for
 * is quietly broken. So the fallback path below produces a syntactically valid UUIDv4 and
 * not merely something unique.
 */
export function newRequestId(): string {
  const cryptoObj = globalThis.crypto;
  if (cryptoObj && typeof cryptoObj.randomUUID === 'function') {
    return cryptoObj.randomUUID();
  }
  const bytes = new Uint8Array(16);
  if (cryptoObj && typeof cryptoObj.getRandomValues === 'function') {
    cryptoObj.getRandomValues(bytes);
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256);
    }
  }
  // Version 4, variant 10xx: the two bytes the pattern in the middleware checks.
  bytes[6] = ((bytes[6] ?? 0) & 0x0f) | 0x40;
  bytes[8] = ((bytes[8] ?? 0) & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20, 32),
  ].join('-');
}

// --------------------------------------------------------------------------- //
// Query strings
// --------------------------------------------------------------------------- //

/** A value that can appear in a query string, before serialisation. */
export type QueryValue = string | number | boolean | null | undefined | readonly string[];

/**
 * Serialise query parameters the way this API reads them.
 *
 * Three rules, each of which has a wrong answer that looks right:
 *
 * - **Arrays repeat the key.** `platform=reddit&platform=rss`, never `platform=reddit,rss`.
 *   The exception is the `include` family, whose values §4.2/§4.4/§4.7 define as a *csv
 *   enum*; those resource modules join before calling here.
 * - **`undefined` and `null` are dropped, `false` is kept.** `has_media=false` is a real
 *   filter with a different meaning from omitting it, and a truthiness check would erase
 *   it along with `limit=0`.
 * - **Empty arrays are dropped.** An empty `platforms` means "every enabled connector"
 *   (§4.1), so emitting `platform=` would send a filter matching one platform named "".
 */
export function buildQuery(params: Record<string, QueryValue> | undefined): string {
  if (!params) return '';
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue;
    if (Array.isArray(value)) {
      for (const item of value) {
        if (item !== undefined && item !== null && item !== '') search.append(key, String(item));
      }
      continue;
    }
    if (value === '') continue;
    search.append(key, String(value));
  }
  const encoded = search.toString();
  return encoded.length > 0 ? `?${encoded}` : '';
}

// --------------------------------------------------------------------------- //
// The fetch wrapper
// --------------------------------------------------------------------------- //

export interface ApiFetchOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  /** Serialised with `buildQuery`. */
  query?: Record<string, QueryValue>;
  /** JSON-encoded. Pass the request DTO from `@/types/`, not a hand-built object. */
  body?: unknown;
  /**
   * Idempotency key for the three `POST` endpoints (§3.5).
   *
   * Only set this where a duplicate would cost something real. Without it, two identical
   * `POST /investigations` calls create two investigations and bill twice — which is the
   * documented behaviour, not a bug, so the caller has to opt in.
   */
  idempotencyKey?: string;
  signal?: AbortSignal;
  timeoutMs?: number;
  /** Extra headers. Cannot override `X-Request-ID`; that one is minted per request. */
  headers?: Record<string, string>;
  /** Passed through to `fetch`; Next.js reads it for Server Component caching. */
  cache?: RequestCache;
  next?: { revalidate?: number | false; tags?: string[] };
}

/** A successful response plus the metadata callers occasionally need. */
export interface ApiResponse<T> {
  data: T;
  status: number;
  requestId: string | null;
  headers: Headers;
}

/**
 * Perform one API call and return the parsed body, or throw an `ApiError`.
 *
 * `apiFetchRaw` is the same call with the status and headers attached, for the two places
 * that genuinely need them: `POST /investigations` reads `Location`, and `POST
 * /connectors/sync` distinguishes a `200` dry-run validation from a `202` enqueue.
 */
export async function apiFetch<T>(path: string, options: ApiFetchOptions = {}): Promise<T> {
  const response = await apiFetchRaw<T>(path, options);
  return response.data;
}

export async function apiFetchRaw<T>(
  path: string,
  options: ApiFetchOptions = {},
): Promise<ApiResponse<T>> {
  const requestId = newRequestId();
  const url = `${resolveApiBase()}${path}${buildQuery(options.query)}`;

  const headers = new Headers(options.headers);
  headers.set('Accept', 'application/json');
  // Set last so a caller cannot shadow the correlation id it needs for support.
  headers.set('X-Request-ID', requestId);
  if (options.body !== undefined) {
    headers.set('Content-Type', 'application/json; charset=utf-8');
  }
  if (options.idempotencyKey) {
    headers.set('Idempotency-Key', options.idempotencyKey);
  }
  const token = await authTokenProvider();
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const timeout = AbortSignal.timeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  // Combined rather than replaced: dropping the caller's signal would leave a request
  // running after the component that started it unmounted, and dropping the timeout would
  // let a hung server hold a query in `pending` forever with no error state to render.
  const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;

  let response: Response;
  try {
    response = await fetch(url, {
      method: options.method ?? 'GET',
      headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal,
      cache: options.cache,
      ...(options.next ? { next: options.next } : {}),
    });
  } catch (cause) {
    throw networkFailure(cause, requestId, options.signal);
  }

  if (!response.ok) {
    throw await problemFromResponse(response, requestId);
  }

  return {
    data: await parseBody<T>(response, requestId),
    status: response.status,
    requestId: response.headers.get('X-Request-ID') ?? requestId,
    headers: response.headers,
  };
}

/**
 * Fetch a non-JSON representation, used by `GET /reports/{id}?format=markdown`.
 *
 * Separate from `apiFetch` rather than a mode of it because the two differ in what a
 * failure looks like: an error response is still `application/problem+json` even when the
 * success shape is `text/markdown`, so the error path is shared and the success path is
 * not.
 */
export async function apiFetchText(
  path: string,
  options: ApiFetchOptions & { accept: string },
): Promise<string> {
  const requestId = newRequestId();
  const url = `${resolveApiBase()}${path}${buildQuery(options.query)}`;
  const headers = new Headers(options.headers);
  headers.set('Accept', options.accept);
  headers.set('X-Request-ID', requestId);
  const token = await authTokenProvider();
  if (token) headers.set('Authorization', `Bearer ${token}`);

  const timeout = AbortSignal.timeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;

  let response: Response;
  try {
    response = await fetch(url, { method: options.method ?? 'GET', headers, signal });
  } catch (cause) {
    throw networkFailure(cause, requestId, options.signal);
  }
  if (!response.ok) throw await problemFromResponse(response, requestId);
  return response.text();
}

/**
 * Parse a success body, tolerating the two shapes that are not JSON objects.
 *
 * A `204` and a genuinely empty `200` both have no body, and `response.json()` throws on
 * both. Callers of those endpoints expect `void`, so an empty body resolves to
 * `undefined` rather than to a thrown `SyntaxError` that would surface as a mysterious
 * `malformed_response`.
 */
async function parseBody<T>(response: Response, requestId: string): Promise<T> {
  if (response.status === 204) return undefined as T;
  const text = await response.text();
  if (text.length === 0) return undefined as T;
  try {
    return JSON.parse(text) as T;
  } catch (cause) {
    throw new ApiError({
      code: 'malformed_response',
      status: response.status,
      title: 'Malformed response',
      detail:
        'The server returned a success status with a body that is not JSON. This is a ' +
        'server-side defect; the request itself may have succeeded.',
      requestId: response.headers.get('X-Request-ID') ?? requestId,
      cause,
    });
  }
}

/**
 * Normalise an error response into an `ApiError`.
 *
 * Every non-2xx from this API is `application/problem+json` (§3.3), but this must not
 * *assume* it: a 502 from a reverse proxy, a 413 from a body-size limit and a captive
 * portal's login page all arrive as non-2xx with an HTML body, and a client that calls
 * `response.json()` unconditionally turns each of them into a `SyntaxError` that says
 * nothing about the status that actually occurred.
 */
async function problemFromResponse(response: Response, requestId: string): Promise<ApiError> {
  const responseRequestId = response.headers.get('X-Request-ID') ?? requestId;
  const retryAfterHeader = response.headers.get('Retry-After');
  const retryAfterFromHeader = retryAfterHeader ? Number.parseInt(retryAfterHeader, 10) : NaN;

  const contentType = response.headers.get('Content-Type') ?? '';
  const looksLikeProblem =
    contentType.includes(PROBLEM_MEDIA_TYPE) || contentType.includes('application/json');

  if (looksLikeProblem) {
    try {
      const problem = (await response.json()) as ProblemDocument & { code?: string };
      const code = extractCode(problem, response.status);
      return new ApiError({
        code,
        status: typeof problem.status === 'number' ? problem.status : response.status,
        title:
          typeof problem.title === 'string' && problem.title.length > 0
            ? humanizeTitle(problem.title)
            : humanizeTitle(String(code)),
        detail:
          typeof problem.detail === 'string' && problem.detail.length > 0
            ? problem.detail
            : `The server responded with ${response.status}.`,
        instance: problem.instance ?? null,
        issues: problem.details?.errors ?? [],
        retryAfterSeconds: problem.retry_after ?? nullIfNaN(retryAfterFromHeader),
        requestId: responseRequestId,
        problem,
      });
    } catch {
      // Fall through: the body claimed to be JSON and was not. The status is still
      // meaningful and is the only thing worth reporting.
    }
  }

  return new ApiError({
    code: response.status >= 500 ? 'internal_error' : 'http_error',
    status: response.status,
    title: humanizeTitle(response.statusText || `HTTP ${response.status}`),
    detail: `The server responded with ${response.status} and no problem document.`,
    retryAfterSeconds: nullIfNaN(retryAfterFromHeader),
    requestId: responseRequestId,
  });
}

/**
 * Classify a `fetch` rejection, which is where the useful distinctions hide.
 *
 * `fetch` rejects identically for a DNS failure, a refused connection, a CORS rejection
 * and an abort, so the abort has to be separated out by inspecting the signals: an abort
 * the *caller* requested is a navigation or an unmount and must not be rendered as an
 * error at all, while an abort from the timeout is a real `504`-shaped failure the user
 * needs to see.
 */
function networkFailure(cause: unknown, requestId: string, callerSignal?: AbortSignal): ApiError {
  const aborted = cause instanceof DOMException && cause.name === 'AbortError';
  if (aborted && callerSignal?.aborted) {
    return new ApiError({
      code: 'network_error',
      status: 0,
      title: 'Request cancelled',
      detail: 'The request was cancelled before it completed.',
      requestId,
      cause,
    });
  }
  if (aborted || (cause instanceof DOMException && cause.name === 'TimeoutError')) {
    return new ApiError({
      code: 'timeout',
      status: 0,
      title: 'Timeout',
      detail: 'The API did not respond in time. It may be starting up or overloaded.',
      requestId,
      cause,
    });
  }
  return new ApiError({
    code: 'network_error',
    status: 0,
    title: 'Cannot reach the API',
    detail:
      'The request never reached the OmniSense API. Check that the backend is running ' +
      'and that NEXT_PUBLIC_API_BASE_URL points at it.',
    requestId,
    cause,
  });
}

function nullIfNaN(value: number): number | null {
  return Number.isFinite(value) ? value : null;
}
