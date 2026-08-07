/**
 * The Signal as it appears *on the wire*, which is not the Signal as it appears in the
 * pipeline — and the difference is the point of this file.
 *
 * `models/signal.py` is the pipeline's contract and grows a field whenever the pipeline
 * needs one: `embeddings[]`, `lineage.confidence_components`, `author.follower_count`.
 * `docs/api-reference.md` §4.7 publishes a deliberately chosen *subset* of that, and the
 * subset is the thing to mirror here. Embeddings are megabytes nobody asked for, and
 * `author.follower_count` is personal data that `docs/security-and-privacy.md` §6.1 keeps
 * out of responses entirely. Mirroring `models/signal.py` instead would teach this app to
 * expect fields the API is contractually forbidden from sending.
 *
 * The most visible consequence: `author` is a **string** here (`"u/example"`), not the
 * `Author` object of `models/signal.py`. A component that reaches for
 * `signal.author.display_name` is reading the pipeline model, not the API.
 *
 * Two smaller traps worth stating once:
 *
 * **`id` is not a UUID.** Signal ids are `sig_`-prefixed deterministic strings derived
 * from `(platform, native_id)` (`docs/signal-model.md` §4.1, api-reference §3.2). Every
 * other id in the API is a UUID string, and client-side UUID validation on this one field
 * rejects every real id in the system.
 *
 * **`relevance_score` is not confidence.** It is present only when `q` was supplied, it
 * ranks *this result against this query*, and it is meaningless across queries. Rendering
 * it in the same visual slot as `confidence` is the single most misleading thing this
 * page family can do (`docs/glossary.md`, commonly-confused pairs).
 */

import type { Open } from '@/types/api';

/**
 * Coarse source category — `models/enums.py::SourceCategory`, five buckets from Design
 * Doc §5 plus `unknown`.
 *
 * This, not `platform`, is what the dashboard splits volume by: there are two dozen
 * platforms and five categories, and a stacked chart with two dozen series communicates
 * nothing (`docs/frontend.md` §3.1).
 */
export type SourceCategory = Open<
  'social' | 'reviews' | 'enterprise' | 'research' | 'news' | 'unknown'
>;

/**
 * The concrete origin of a Signal — one member per connector module
 * (`models/enums.py::Platform`).
 *
 * Open by construction: `docs/signal-model.md` §7 makes adding a platform a
 * backward-compatible change *precisely because* every reader tolerates unknown members.
 * A connector shipped on Tuesday must not blank a dashboard rendered by a bundle built
 * on Monday.
 */
export type Platform = Open<
  // connectors/social/
  | 'reddit'
  | 'x'
  | 'youtube'
  | 'instagram'
  | 'tiktok'
  | 'linkedin'
  // connectors/reviews/
  | 'amazon'
  | 'play_store'
  | 'app_store'
  | 'trustpilot'
  | 'google_reviews'
  // connectors/enterprise/
  | 'slack'
  | 'jira'
  | 'confluence'
  | 'notion'
  | 'github'
  | 'salesforce'
  | 'hubspot'
  // connectors/research/
  | 'arxiv'
  | 'semantic_scholar'
  | 'papers_with_code'
  // connectors/news/
  | 'rss'
  | 'gdelt'
  | 'news_api'
  | 'unknown'
>;

/**
 * Discrete sentiment label accompanying the continuous polarity score.
 *
 * `mixed` is distinct from `neutral` and must not be folded into it in the UI: a review
 * that praises the hardware and condemns the software is strongly polarised in both
 * directions, and averaging it to neutral erases exactly the signal the Competitor and
 * Insight agents are looking for (`models/enums.py::SentimentLabel`).
 */
export type SentimentLabel = Open<'positive' | 'neutral' | 'negative' | 'mixed' | 'unknown'>;

/** Overall sentiment of a signal, as §4.7 serialises it. */
export interface SignalSentiment {
  /** `-1.0` maximally negative, `+1.0` maximally positive. */
  polarity: number;
  label: SentimentLabel;
}

/**
 * Platform counters plus the four cross-platform comparable axes.
 *
 * `raw` holds the platform's own counters verbatim and is **not comparable across
 * platforms**: a Reddit score of 400 and a YouTube view count of 400 are not the same
 * event. Only the normalised axes may be compared, because each is the empirical
 * percentile of the raw value within the same `(platform, content_type)` cohort
 * (`models/signal.py::Engagement`). Any chart that plots `raw` across platforms is wrong
 * in a way the chart itself will never reveal.
 *
 * Every axis is nullable because not every platform populates every axis — an RSS item
 * has no endorsement axis, and `null` there means "not measurable here", not "zero".
 */
export interface Engagement {
  raw: Record<string, number | null>;
  reach: number | null;
  endorsement: number | null;
  amplification: number | null;
  discussion: number | null;
  /** Weighted mean of the *available* axes. Null when no axis was populated at all. */
  score: number | null;
}

/** An entity mention attached to a signal, resolved to a knowledge-graph node. */
export interface SignalEntityRef {
  /** Prefixed and opaque: `ent_…`, `prod_…`, `co_…`. Never a bare UUID (§3.2). */
  id: string;
  /** The graph node label — `Company`, `Product`, … (`models/enums.py::EntityType`). */
  label: string;
  name: string;
}

/**
 * One item of `GET /api/v1/signals` (§4.7).
 *
 * Which fields are actually present depends on the `include` parameter, which defaults to
 * `summary`. Everything the summary projection omits is typed optional here rather than
 * required-and-nullable, so `include: 'content'` versus its absence is a compile-time
 * distinction rather than a runtime surprise.
 */
export interface SignalItem {
  /** `sig_<hex>`. Opaque. Not a UUID — see the module docstring. */
  id: string;
  source: SourceCategory;
  platform: Platform;
  /** BCP-47, or `und` when detection was inconclusive and the signal is unfiltered. */
  language: string;
  url: string | null;
  /** A display handle, already flattened by the API. Not the `Author` object. */
  author: string | null;
  /** Event time at the source, never ingestion time. Trend charts key off this. */
  timestamp: string;
  /** Present with `include=content`. Cleaned body text, possibly truncated. */
  content?: string;
  sentiment: SignalSentiment | null;
  engagement: Engagement;
  /**
   * How much an agent should trust a claim resting on this signal alone. Not sentiment
   * confidence, not retrieval relevance, and not the confidence shown on a report — that
   * one is computed by the Critic over a whole evidence set.
   */
  confidence: number;
  /** Present only when `q` was supplied. A retrieval score. Never render as confidence. */
  relevance_score?: number;
  topics: string[];
  /** Present with `include=entities`. */
  entities?: SignalEntityRef[];
}

/** Which projection to ask for. `summary` is the default (§4.7). */
export type SignalInclude = 'summary' | 'content' | 'entities' | 'lineage';

/** Sort key for `GET /signals`. `relevance` is only meaningful alongside `q`. */
export type SignalSort = 'relevance' | 'timestamp' | 'engagement' | 'confidence';

/**
 * Query parameters for `GET /api/v1/signals` (§4.7).
 *
 * Repeatable parameters are modelled as arrays and serialised as repeated keys, never as
 * a comma-joined string: `platform=reddit&platform=rss` ORs within a parameter while
 * ANDing across parameters, and a joined value would be read as one unknown platform
 * named `reddit,rss` and rejected with a 422.
 */
export interface SignalQuery {
  q?: string;
  platform?: Platform[];
  source_category?: SourceCategory[];
  entity_id?: string[];
  topic?: string[];
  language?: string[];
  sentiment?: 'positive' | 'neutral' | 'negative';
  min_confidence?: number;
  from?: string;
  to?: string;
  has_media?: boolean;
  sort?: SignalSort;
  order?: 'asc' | 'desc';
  include?: SignalInclude[];
  limit?: number;
  cursor?: string;
}
