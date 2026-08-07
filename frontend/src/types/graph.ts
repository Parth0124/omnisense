/**
 * Knowledge-graph nodes, edges and the bounded neighbourhood the API returns.
 *
 * Mirrored from `docs/api-reference.md` §4.5 and the vocabulary in
 * `models/enums.py::EntityType` / `EdgeType`. `backend/schemas/graph.py` is still a
 * docstring stub; when it is written, this file is what it has to agree with.
 *
 * Two properties of this payload shape everything the graph page is allowed to do.
 *
 * **The neighbourhood is bounded by the API, not trimmed by the client.** `depth`,
 * `limit` and `max_neighbors` are server-side parameters, and `meta.truncated` reports
 * whether any node's neighbours were cut. An unbounded traversal is a Neo4j problem, not
 * a rendering problem — a client that asks for depth 3 on a 1,800-degree hub and then
 * filters the result in JavaScript has already paid for the query that will time the
 * server out (`docs/knowledge-graph.md`).
 *
 * **Edges are temporal.** Every edge carries a validity interval, and the server
 * evaluates it at `as_of`. An edge whose `valid_to` has passed is *historical fact*, not
 * stale data, and must be visually distinct from a current one: "Acme acquired Widget Co
 * in 2021" and "Acme is acquiring Widget Co" are different claims and the only thing
 * separating them on screen is that distinction.
 *
 * Node labels and edge types are **not re-spelled as TypeScript string literals anywhere
 * else in this app**. `GRAPH_NODE_TYPES` and `GRAPH_EDGE_TYPES` below are the single
 * source, and the legend is generated from them — a hand-maintained legend drifts from
 * the vocabulary the first time a label is added, and nothing fails when it does.
 */

import type { Open, PageInfo } from '@/types/api';

/**
 * Knowledge-graph node labels — `models/enums.py::EntityType`.
 *
 * Capitalised because the values are used verbatim as Neo4j labels (`MATCH (n:Company)`).
 * Lower-casing them here would force a translation table between this app and the graph.
 */
export const GRAPH_NODE_TYPES = [
  'Company',
  'Product',
  'Person',
  'Topic',
  'Technology',
  'Region',
  'Event',
] as const;

export type GraphNodeType = Open<(typeof GRAPH_NODE_TYPES)[number] | 'Unknown'>;

/**
 * Knowledge-graph relationship types — `models/enums.py::EdgeType`.
 *
 * The six of Design Doc §7 that §4.5 accepts as an `edge_types` filter. `SAME_AS` and
 * `DUPLICATE_OF` exist in the graph but are structural bookkeeping (an entity-resolution
 * merge, a dedup cluster link) rather than claims about the world, and §4.5 does not list
 * them as filterable — so they are typed but not offered in the legend's filter set.
 */
export const GRAPH_EDGE_TYPES = [
  'MENTIONS',
  'COMPETES_WITH',
  'ACQUIRED',
  'USES',
  'COMPLAINS_ABOUT',
  'LAUNCHED_BY',
] as const;

export type GraphEdgeType = Open<
  (typeof GRAPH_EDGE_TYPES)[number] | 'SAME_AS' | 'DUPLICATE_OF' | 'UNKNOWN'
>;

/**
 * One node of the returned neighbourhood.
 *
 * `labels` is an array because Neo4j nodes carry multiple labels; the first is the one to
 * colour by. `score` is the *search* score and is null for nodes reached by expansion
 * rather than matched by `q` — which is exactly why a node's visual weight must come from
 * `degree` or `distance` and not from a score that half the nodes do not have.
 */
export interface GraphNode {
  /** Prefixed and opaque: `prod_acme_cli`, `co_acme`. Never a bare UUID (§3.2). */
  id: string;
  labels: GraphNodeType[];
  canonical_name: string;
  aliases: string[];
  /** Full-text search score of a seed node, `0`–`1`. Null for expanded neighbours. */
  score: number | null;
  /** Total degree in the graph, not degree within this response. */
  degree: number;
  /** Hops from the nearest seed node. `0` for seeds. */
  distance: number;
}

/**
 * One edge, with the interval over which it is asserted to hold.
 *
 * `from` and `to` are the wire names. They are reserved words in Python, which is why
 * `backend/schemas/common.py::ResponseModel` sets `populate_by_name` — no such problem
 * here, so they are spelled as the contract spells them.
 *
 * A null `valid_to` means "still valid as far as the graph knows", not "unknown". The
 * distinction matters for `ACQUIRED`, where an open interval means the acquisition stands
 * and a closed one means it was unwound.
 */
export interface GraphEdge {
  id: string;
  type: GraphEdgeType;
  from: string;
  to: string;
  weight: number;
  valid_from: string | null;
  valid_to: string | null;
  /** Up to 3 supporting signal ids, present only with `include_signals=true`. */
  signal_ids: string[];
}

/**
 * What the server did to bound the traversal.
 *
 * `truncated` is `true` whenever any node's neighbours were cut by `max_neighbors`. §4.5
 * is explicit that **the UI must say so**: the graph on screen is then a sample, and a
 * viewer who reads it as complete will draw a conclusion about connectivity that the data
 * does not support.
 */
export interface GraphMeta {
  depth: number;
  /** The instant at which edge validity was evaluated. Defaults to now, server-side. */
  as_of: string;
  truncated: boolean;
  seed_count: number;
}

/** The `200` body of `GET /api/v1/graph/search` (§4.5). */
export interface GraphSearchResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  meta: GraphMeta;
  /** Pages over **seed** nodes only — expanded neighbours are not paginated. */
  page: PageInfo;
}

/**
 * Query parameters for `GET /api/v1/graph/search`.
 *
 * Exactly one of `q` or `node_id` is required; sending neither is a 422. They are typed
 * as two optional fields rather than a discriminated union because the search box and the
 * "expand this node" action write into the same state object, and a union would force a
 * cast every time the page switches between them.
 */
export interface GraphSearchQuery {
  /** Full-text entity search. Required unless `node_id` is given. */
  q?: string;
  /** Expand from a known entity instead of searching. Required unless `q` is given. */
  node_id?: string;
  node_types?: GraphNodeType[];
  edge_types?: GraphEdgeType[];
  /** `0`–`3`. `0` returns matched nodes only. Above 3 is a 422, not a clamp. */
  depth?: number;
  /** Caps **seed** nodes, not expanded neighbours. Max 200. */
  limit?: number;
  /** Per-node fan-out cap. This is what prevents a hub from exploding the response. */
  max_neighbors?: number;
  /** `0`–`1` search-score floor on seed nodes. */
  min_score?: number;
  /** Evaluate edge validity at this instant. Defaults to now. */
  as_of?: string;
  include_signals?: boolean;
  cursor?: string;
}

/**
 * Whether an edge's validity interval had already closed at the moment it was evaluated.
 *
 * Takes `as_of` from `meta` rather than reading the wall clock, because the server
 * evaluated the traversal at that instant and a client comparing against `Date.now()`
 * would disagree with the data it is rendering whenever a historical `as_of` was
 * requested — showing an edge as current on a graph explicitly asked for as of 2021.
 */
export function isEdgeExpired(edge: GraphEdge, asOf: string): boolean {
  if (edge.valid_to === null) return false;
  const closedAt = Date.parse(edge.valid_to);
  const evaluatedAt = Date.parse(asOf);
  if (Number.isNaN(closedAt) || Number.isNaN(evaluatedAt)) return false;
  return closedAt <= evaluatedAt;
}
