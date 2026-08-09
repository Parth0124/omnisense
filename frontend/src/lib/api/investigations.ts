/**
 * `/api/v1/investigations` — the resource client.
 *
 * A thin, typed layer over `apiFetch`. Its value is that every call site gets the
 * same query-parameter spelling and the same return type: the endpoints take
 * repeated `state=` and `include=` parameters, and a component assembling those
 * by hand gets them subtly wrong in a way that returns a plausible-looking
 * wrong page.
 */
import { apiFetch } from '@/lib/api/client';
import type {
  CreateInvestigationRequest,
  InvestigationCreated,
  InvestigationDetail,
  InvestigationInclude,
  InvestigationState,
} from '@/types/investigation';

/**
 * Query parameters for the *collection* endpoint.
 *
 * Declared here rather than reusing `InvestigationQuery` from `@/types`, which
 * describes the *detail* endpoint's `include` and step paging. The two share a
 * resource and nothing else, and collapsing them would let `steps_limit` be
 * passed to a list call that silently ignores it.
 */
export interface ListInvestigationsQuery {
  limit?: number;
  cursor?: string;
  /** Repeated in the query string. OR within. */
  state?: readonly InvestigationState[];
}

/**
 * Start an investigation. Returns `202` with a stream link.
 *
 * `idempotencyKey` is passed by default and that is deliberate. Without one, a
 * user double-clicking Start — or a network blip that loses the response —
 * begins a second multi-minute run and bills for both. This is exactly the case
 * the header exists for, so the caller opts out rather than in.
 */
export async function createInvestigation(
  body: CreateInvestigationRequest,
  options: { idempotencyKey?: string | null } = {},
): Promise<InvestigationCreated> {
  const key =
    options.idempotencyKey === null
      ? undefined
      : (options.idempotencyKey ?? crypto.randomUUID());
  return apiFetch<InvestigationCreated>('/investigations', {
    method: 'POST',
    body,
    idempotencyKey: key,
  });
}

/** One investigation. `include` controls the optional sub-objects. */
export async function getInvestigation(
  id: string,
  include: readonly InvestigationInclude[] = [],
): Promise<InvestigationDetail> {
  return apiFetch<InvestigationDetail>(`/investigations/${encodeURIComponent(id)}`, {
    query: include.length ? { include: [...include] } : undefined,
    // Never cached. This is the resource whose whole point is that it changes
    // while you are looking at it.
    cache: 'no-store',
  });
}

/** Recent investigations for the signed-in tenant, newest first. */
export async function listInvestigations(
  query: ListInvestigationsQuery = {},
): Promise<InvestigationCreated[]> {
  return apiFetch<InvestigationCreated[]>('/investigations', {
    query: {
      limit: query.limit,
      cursor: query.cursor,
      state: query.state as readonly string[] | undefined,
    },
    cache: 'no-store',
  });
}

/**
 * Request cancellation. Cooperative — returns as soon as the state is written.
 *
 * The orchestrator notices at its next checkpoint, so the run does not stop the
 * instant this resolves. The UI reflects that by showing "cancelling" rather
 * than "cancelled" until the stream sends its terminal event.
 */
export async function cancelInvestigation(
  id: string,
  reason?: string,
): Promise<InvestigationDetail> {
  return apiFetch<InvestigationDetail>(
    `/investigations/${encodeURIComponent(id)}/cancel`,
    { method: 'POST', query: reason ? { reason } : undefined },
  );
}
