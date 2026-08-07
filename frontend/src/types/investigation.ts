/**
 * Investigations and the execution-timeline events that narrate them.
 *
 * Mirrors `backend/schemas/investigation.py` for the REST shapes and
 * `backend/api/v1/stream.py` + `docs/api-reference.md` §5 for the SSE shapes. The two
 * describe the same run from different angles and must not be conflated: `GET
 * /investigations/{id}` is a *snapshot* the client may poll, the stream is the *live
 * view*, and `docs/frontend.md` §3.2 makes the stream primary with the snapshot as the
 * fallback on reconnect exhaustion and on late attach.
 *
 * Three divergences from the prose contract are encoded here on purpose, because a
 * client that trusts the prose will render nonsense against the built backend.
 *
 * **The state vocabulary is the persisted one.** §4.2 draws a state machine over
 * `queued → planning → collecting → retrieving → analyzing → reflecting → reporting →
 * completed` plus `timed_out`. `models/enums.py::InvestigationStatus` — the column that
 * actually holds this value — is coarser: there is nowhere to store `collecting` as
 * distinct from `retrieving`, and `timed_out` is recorded as `failed` with a reason.
 * `InvestigationState` below is the stored vocabulary. A progress UI keyed to the
 * documented one would sit on "collecting" forever.
 *
 * **Several documented counters are always null.** `counts.signals_considered`,
 * `counts.evidence`, `usage.tool_calls`, and a step's `evidence_count` and `tool_calls`
 * have no column and no source. They are `null`, not `0`, and rendering them as `0`
 * reports every investigation as having considered no evidence.
 *
 * **`depth` is null on a read.** It is accepted at creation and echoed in the `202`, but
 * `investigations` has no depth column, so `GET` cannot recover it. Defaulting the read
 * to `standard` in the UI would claim a budget the run may never have been given.
 */

import type { Open, Page, TimeWindow, Usage } from '@/types/api';
import type { Platform } from '@/types/signal';

/**
 * Terminal and non-terminal states of a run — `models/enums.py::InvestigationStatus`.
 *
 * `completed_with_findings` is not a failure. It is the terminal state when the Critic
 * loop hit `MAX_CRITIC_REVISIONS` without reaching an `accept` verdict
 * (`docs/agent-system.md` §13): the report ships, with its unresolved findings surfaced
 * rather than withheld or silently presented as clean. A UI that styles it like `failed`
 * hides a usable report; one that styles it like `completed` hides the caveat.
 */
export type InvestigationState = Open<
  | 'queued'
  | 'planning'
  | 'running'
  | 'reflecting'
  | 'completed'
  | 'completed_with_findings'
  | 'failed'
  | 'cancelled'
  | 'unknown'
>;

/** States after which nothing further will happen — and after which the stream closes. */
export const TERMINAL_INVESTIGATION_STATES = [
  'completed',
  'completed_with_findings',
  'failed',
  'cancelled',
] as const satisfies readonly InvestigationState[];

/**
 * Whether a run has reached a state it will never leave.
 *
 * Used to stop polling and, more importantly, to refuse to reconnect the stream. §5 is
 * explicit that the server closes after a terminal event and that clients must not
 * reconnect — an auto-reconnect on a terminal event is a client bug that presents as a
 * server loop, and it is the single most expensive mistake available to this page.
 */
export function isTerminalState(state: InvestigationState): boolean {
  return (TERMINAL_INVESTIGATION_STATES as readonly string[]).includes(state);
}

/** Lifecycle of one agent execution inside a run (`models/orm/investigation.py`). */
export type StepState = Open<
  'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'cancelled' | 'unknown'
>;

/** The ten agents of Design Doc §9 — `models/enums.py::AgentName`. */
export type AgentName = Open<
  | 'planner'
  | 'collector'
  | 'retriever'
  | 'trend'
  | 'competitor'
  | 'forecast'
  | 'insight'
  | 'strategy'
  | 'critic'
  | 'report'
  | 'unknown'
>;

/** Budget preset chosen at creation (§4.1). Not persisted — see the module docstring. */
export type InvestigationDepth = 'quick' | 'standard' | 'deep';

/** What the investigation is allowed to look at (§4.1). */
export interface InvestigationScope {
  /** Connector slugs. **Empty means every enabled connector**, not "none". */
  platforms?: Platform[];
  /** Entity ids or names to anchor retrieval on. Empty means unanchored. */
  entities?: string[];
  /** ISO 639-1 codes. Empty means all. */
  languages?: string[];
  time_window?: TimeWindow;
}

/** Hard ceilings on spend for one run. `null` means the deployment default (§4.1). */
export interface InvestigationBudget {
  max_tokens?: number | null;
  max_tool_calls?: number | null;
}

/** The `POST /api/v1/investigations` body (§4.1). */
export interface CreateInvestigationRequest {
  /** 1–2000 characters. A whitespace-only question is an empty question. */
  query: string;
  /** What the answer will be used for. Steers the Strategy agent. */
  objective?: string | null;
  depth?: InvestigationDepth;
  scope?: InvestigationScope;
  /** Run an incremental sync before retrieval. Increases latency measurably. */
  refresh_connectors?: boolean;
  /** 1–200. Hard stop for the orchestrator. */
  max_steps?: number | null;
  budget?: InvestigationBudget;
  /** HTTPS only — the backend rejects `http://` at the edge. */
  callback_url?: string | null;
  /** ≤16 keys, values ≤256 characters. Echoed back verbatim. */
  metadata?: Record<string, string>;
}

/**
 * Where to go next. Relative paths, never absolute URLs.
 *
 * The API is reached through a proxy, a tunnel and a browser origin the backend process
 * cannot see, so it declines to guess a scheme or a hostname. That makes these safe to
 * hand to `fetch` through the same client that produced them, and unsafe to render as an
 * external link.
 */
export interface InvestigationLinks {
  self: string;
  stream: string;
  report: string | null;
}

/** The `202 Accepted` body of §4.1. Never a `201` — every creating endpoint is async. */
export interface InvestigationCreated {
  id: string;
  state: InvestigationState;
  query: string;
  depth: InvestigationDepth;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  /**
   * Allocated eagerly so a client can subscribe before the report exists. Fetching it
   * before the run reaches reporting returns `409 report_not_ready` with a `retry_after`.
   */
  report_id: string | null;
  trace_id: string;
  links: InvestigationLinks;
}

/**
 * How far along the run is.
 *
 * `steps_total_estimate` is an estimate and not a denominator. The Critic loop can send
 * the graph round again (`docs/agent-system.md` §13), so the plan's step count is a lower
 * bound; a progress bar that divides by it will visibly run backwards mid-investigation.
 * It is `null` before the Planner has recorded a decomposition.
 */
export interface InvestigationProgress {
  steps_completed: number;
  steps_total_estimate: number | null;
}

/** Volume of what the run has produced. Nulls are honest, not zeroes — see the docstring. */
export interface InvestigationCounts {
  signals_considered: number | null;
  evidence: number | null;
  citations: number | null;
}

/**
 * Why a terminal run ended badly. Present only when `state` is `failed` or `cancelled`.
 *
 * A failed investigation is still a `200`: the request succeeded, the investigation did
 * not. Conflating the two makes a client retry an HTTP call that worked.
 *
 * `code` and `message` carry the same string today — `investigations.error` holds the
 * `code` of the raised `OmniSenseError` rather than a stack trace, precisely so that
 * handlers never pattern-match on message text.
 */
export interface InvestigationError {
  code: string;
  message: string;
  step_id: string | null;
}

/** One row of the step sub-collection (§4.2). */
export interface StepItem {
  id: string;
  /** Strictly increasing within an investigation. The stream's `seq` is a *different* counter. */
  seq: number;
  agent: AgentName;
  title: string;
  state: StepState;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  /** Always null today: `investigation_steps` has no such column. */
  tool_calls: number | null;
  /** Always null today: there is no evidence table to count from. */
  evidence_count: number | null;
}

/** The `200` body of `GET /api/v1/investigations/{id}` (§4.2). */
export interface InvestigationDetail {
  id: string;
  state: InvestigationState;
  query: string;
  /** Null on a read. See the module docstring — do not default it to `standard`. */
  depth: InvestigationDepth | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  progress: InvestigationProgress;
  counts: InvestigationCounts;
  report_id: string | null;
  trace_id: string;
  error: InvestigationError | null;
  links: InvestigationLinks;

  /**
   * The next three are **omitted**, not nulled, when `include` did not ask for them.
   * Omission is what keeps "you did not request this" distinguishable from "this is
   * empty", which matters most for `plan`: a null plan means *not planned yet*, and a
   * response that nulled an unrequested plan would throw that distinction away at the
   * last hop.
   */
  plan?: Record<string, unknown> | null;
  steps?: Page<StepItem>;
  usage?: Usage;
}

/** Sub-resources `GET /investigations/{id}` will materialise on request (§4.2). */
export type InvestigationInclude = 'plan' | 'steps' | 'evidence' | 'usage';

/** Query parameters for `GET /api/v1/investigations/{id}`. */
export interface InvestigationQuery {
  include?: InvestigationInclude[];
  /** Max 200. Values above the max are rejected with 422, never clamped. */
  steps_limit?: number;
  steps_cursor?: string;
}

// --------------------------------------------------------------------------- //
// The execution timeline (SSE, §5)
// --------------------------------------------------------------------------- //

/**
 * Event names carried by `GET /investigations/{id}/stream`.
 *
 * `stream.gap` is an OmniSense extension emitted by `backend/api/v1/stream.py` when a
 * range of events will never arrive — the resume point fell out of the replay buffer, or
 * backpressure dropped frames. It is deliberately sent **without an `id:` line** so that
 * a client's `Last-Event-ID` does not advance past events it never received; advancing it
 * would make the gap permanent across every future reconnect.
 *
 * Open, because §1 makes adding an event type additive and requires clients to ignore
 * names they do not know.
 */
export type TimelineEventType = Open<
  | 'step.started'
  | 'tool.called'
  | 'evidence.found'
  | 'step.completed'
  | 'error'
  | 'done'
  | 'stream.gap'
>;

/**
 * The envelope every `data` payload shares.
 *
 * `seq` is assigned by the orchestrator — the only party that sees every event of a run
 * in order — and is strictly increasing per investigation. A gap in `seq` means a
 * *dropped* event, never a reordered one, so the correct response is to surface it and
 * fall back to the snapshot, not to interpolate the missing rows.
 */
export interface TimelineEnvelope {
  investigation_id: string;
  seq: number;
  ts: string;
  /** Correlation id, present on every frame so an operator can grep one connection. */
  request_id?: string | null;
}

export interface StepStartedEvent extends TimelineEnvelope {
  type: 'step.started';
  step_id: string;
  agent: AgentName;
  title: string;
  parent_step_id?: string | null;
  plan_index?: number | null;
}

/**
 * An agent invoking a tool. Emitted twice for one call: once `started`, once terminal.
 *
 * `arguments` is redacted by the server for any tool holding credentials, so it may be
 * absent or partial and must never be assumed complete. A `failed` tool call does not
 * imply a failed step — the router can route around one.
 */
export interface ToolCalledEvent extends TimelineEnvelope {
  type: 'tool.called';
  step_id: string;
  tool: string;
  arguments?: Record<string, unknown> | null;
  duration_ms: number | null;
  status: Open<'started' | 'completed' | 'failed'>;
  error?: string | null;
}

/**
 * Evidence admitted by the Retriever or Collector.
 *
 * Payloads are deliberately small — ids and a snippet. Full evidence is hydrated on
 * demand through `GET /signals`, which is why `signal_id` is here and the signal body is
 * not. `signal_id` is `sig_`-prefixed and opaque; validating it as a UUID rejects every
 * real one.
 */
export interface EvidenceFoundEvent extends TimelineEnvelope {
  type: 'evidence.found';
  step_id: string;
  evidence_id: string;
  signal_id: string;
  platform: Platform;
  url: string | null;
  snippet: string;
  /** A retrieval score. Never rendered as confidence. */
  retrieval_score: number;
  retriever: Open<'vector' | 'keyword' | 'graph' | 'hybrid'>;
}

export interface StepCompletedEvent extends TimelineEnvelope {
  type: 'step.completed';
  step_id: string;
  agent: AgentName;
  state: Open<'completed' | 'skipped' | 'failed'>;
  duration_ms: number | null;
  evidence_count: number | null;
  tool_calls: number | null;
  usage?: { input_tokens: number; output_tokens: number } | null;
  summary?: string | null;
}

/** Terminal. The investigation itself failed; `code` is the §3.3 vocabulary. */
export interface TimelineErrorEvent extends TimelineEnvelope {
  type: 'error';
  code: string;
  message: string;
  step_id: string | null;
  retryable: boolean;
}

/** Terminal. Always the last event of a run that reached a terminal state. */
export interface DoneEvent extends TimelineEnvelope {
  type: 'done';
  state: InvestigationState;
  report_id: string | null;
  duration_ms: number | null;
  counts?: { steps: number; evidence: number; citations: number } | null;
  usage?: Usage | null;
}

/** OmniSense extension: events in `[from_seq, to_seq]` will never be delivered here. */
export interface StreamGapEvent extends TimelineEnvelope {
  type: 'stream.gap';
  from_seq: number;
  to_seq: number;
  reason: string;
}

/**
 * An event whose name this bundle does not recognise.
 *
 * Required by §1, not defensive programming: a backend deployed ahead of this bundle may
 * publish a name added after it was built, and the contract says to ignore it. Typing it
 * keeps "ignore" a compile-time-checked branch instead of a silent `default:`.
 */
export interface UnknownTimelineEvent extends TimelineEnvelope {
  type: Open<string>;
  [key: string]: unknown;
}

export type TimelineEvent =
  | StepStartedEvent
  | ToolCalledEvent
  | EvidenceFoundEvent
  | StepCompletedEvent
  | TimelineErrorEvent
  | DoneEvent
  | StreamGapEvent
  | UnknownTimelineEvent;

/** Event names after which the server closes and the client must not reconnect (§5). */
export const TERMINAL_EVENT_TYPES = ['error', 'done'] as const;

export function isTerminalEvent(event: TimelineEvent): boolean {
  return event.type === 'error' || event.type === 'done';
}

/**
 * The timeline as the UI holds it: steps in `seq` order, each with its tool calls and
 * evidence nested underneath.
 *
 * Derived from the event log rather than fetched, and stored in the TanStack Query cache
 * beside the investigation itself. It is *not* mirrored into Zustand: a second copy of
 * the timeline is the defect `docs/frontend.md` §3.2 names explicitly, because the two
 * copies diverge the first time a reconnect replays an event the store already applied.
 */
export interface TimelineStep {
  step_id: string;
  seq: number;
  agent: AgentName;
  title: string;
  state: StepState;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  summary: string | null;
  tool_calls: ToolCallRecord[];
  evidence: EvidenceFoundEvent[];
}

/** One tool invocation, folded from its `started` and terminal `tool.called` events. */
export interface ToolCallRecord {
  tool: string;
  status: Open<'started' | 'completed' | 'failed'>;
  arguments: Record<string, unknown> | null;
  duration_ms: number | null;
  error: string | null;
  /** `seq` of the frame that opened the call; the sort key that keeps calls in order. */
  seq: number;
}
