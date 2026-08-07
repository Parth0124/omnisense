/**
 * Reports: the evidence-backed document, its claims, and the citations that back them.
 *
 * Mirrored from `docs/api-reference.md` §4.4 rather than from `backend/schemas/report.py`,
 * which is still a docstring stub — when that module is written, this file is the thing
 * to diff it against.
 *
 * A report is the only artefact in OmniSense a human is expected to act on, which makes
 * the auditability rules here contractual rather than cosmetic:
 *
 * **A claim with an empty `citation_ids` is uncited and must look uncited.** The document
 * promises that every claim traces to a signal; a claim that silently renders as plain
 * prose is the highest-severity thing this page family can hide (`docs/frontend.md` §3.3).
 * That is why `citation_ids` is a required array and not an optional one — "absent" and
 * "empty" would otherwise be two spellings of the same failure, and only one of them
 * would be checked.
 *
 * **The quote is frozen.** `Citation.quote` was copied at report time and is displayed
 * verbatim. It is never re-fetched from `url`, because the source may have been edited or
 * deleted since, and a citation that silently updates itself is not a citation. `url` is
 * a convenience; the citation stays readable when that link is dead.
 *
 * **`retrieval_score` is a retrieval score.** It ranks a passage against the query that
 * found it. It is not confidence, it is not comparable across queries, and rendering it
 * in the confidence slot is the specific confusion `docs/glossary.md` exists to prevent.
 */

import type { Open } from '@/types/api';
import type { Platform } from '@/types/signal';

/**
 * The qualitative band that accompanies a confidence score.
 *
 * **The band comes from the API.** The client never derives it from the score. Two
 * components applying different cut-offs to the same number is the exact failure
 * `docs/frontend.md` §4.4 forbids, and it is invisible in review because both components
 * look correct in isolation.
 *
 * Open, and only `moderate` appears in the §4.4 example — the full vocabulary is not
 * published anywhere, so a component must render an unrecognised band as its own literal
 * text rather than bucketing it into the nearest known one.
 */
export type ConfidenceBand = Open<'low' | 'moderate' | 'high'>;

/**
 * Report-level confidence: a score, its band, and why.
 *
 * `rationale` is the reason this is not a bare number. A score on its own invites
 * over-trust; `docs/frontend.md` §4.4 requires the rationale to be reachable from the
 * badge, which is why it is required here and not optional.
 */
export interface ReportConfidence {
  /** `0.0`–`1.0`, computed by the Critic over the whole evidence set. */
  score: number;
  band: ConfidenceBand;
  rationale: string;
}

/**
 * One assertion inside a section, with the citations that support it.
 *
 * `confidence` here is per-claim and is a different quantity from the report-level score:
 * a well-supported claim can sit inside a report whose overall confidence is dragged down
 * by thin coverage elsewhere.
 */
export interface ReportClaim {
  id: string;
  text: string;
  /** `0.0`–`1.0`. Rendered through the same badge as the report score, never recomputed. */
  confidence: number;
  /** Ids into `Report.citations`. **Empty means uncited** — render it as such. */
  citation_ids: string[];
}

/**
 * One section of the document.
 *
 * `body` carries inline `[c1][c2]` markers referring to citation ids. They are part of the
 * generated prose, not a rendering instruction, so a renderer that fails to resolve one
 * must leave the marker visible rather than dropping it — a silently swallowed marker
 * turns a cited sentence into an uncited one.
 */
export interface ReportSection {
  id: string;
  heading: string;
  body: string;
  claims: ReportClaim[];
}

/**
 * A citation: the resolvable link from a claim to the signal that supports it.
 *
 * `signal_id` is `sig_`-prefixed and opaque (§3.2). `char_range` is a `[start, end)` pair
 * into the *stored* signal text, so it is only meaningful against the archived copy —
 * applying it to a re-fetched page will land on the wrong characters.
 */
export interface Citation {
  id: string;
  signal_id: string;
  platform: Platform;
  url: string | null;
  author: string | null;
  published_at: string;
  /** Frozen at report time. Displayed verbatim, never re-fetched. */
  quote: string;
  char_range: [number, number] | null;
  /** A retrieval score. Never rendered as confidence. */
  retrieval_score: number;
}

/** Volume counters for the document (§4.4). */
export interface ReportCounts {
  sections: number;
  claims: number;
  citations: number;
  /** Distinct signals cited; lower than `citations` when one signal backs several claims. */
  signals_cited: number;
}

/** The `200` body of `GET /api/v1/reports/{report_id}` (§4.4). */
export interface Report {
  id: string;
  investigation_id: string;
  /**
   * Reports are versioned and earlier versions stay fetchable, which is why the version
   * selector is part of the contract rather than a nicety: a report that was acted on is
   * evidence of what was known at the time, and it must remain retrievable after a
   * regeneration replaces it.
   */
  version: number;
  title: string;
  generated_at: string;
  confidence: ReportConfidence;
  sections: ReportSection[];
  /** Present with `include=citations`, which is the default. */
  citations: Citation[];
  counts: ReportCounts;
}

/** Sub-resources `GET /reports/{id}` will materialise on request (§4.4). */
export type ReportInclude = 'citations' | 'evidence' | 'lineage';

/** Query parameters for `GET /api/v1/reports/{report_id}`. */
export interface ReportQuery {
  include?: ReportInclude[];
  /** Omit for the latest. Values below 1 are rejected with 422. */
  version?: number;
}

/**
 * Index of citations by id, built once per report render.
 *
 * A helper type rather than a map built inside a component: `sections[].claims[]` refer to
 * citations by id, and resolving each one with `citations.find()` inside a render turns a
 * 40-citation report into a quadratic scan on every keystroke of an unrelated input.
 */
export type CitationIndex = ReadonlyMap<string, Citation>;

export function indexCitations(citations: readonly Citation[]): CitationIndex {
  return new Map(citations.map((citation) => [citation.id, citation]));
}
