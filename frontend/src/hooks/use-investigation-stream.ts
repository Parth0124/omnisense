'use client';

/**
 * Turns the SSE event stream into the state the timeline renders.
 *
 * All the interpretation lives here rather than in the component, for one
 * reason: the reducer below is the thing most likely to be wrong, and a reducer
 * is testable in a way a component full of `useEffect` is not.
 *
 * **The step map is derived, never accumulated blindly.** A `step.started` for a
 * node that is already `done` is a re-delivery after reconnect, not a restart,
 * and treating it as a restart would make a finished step spin again in front of
 * the user. Every transition is guarded.
 *
 * **Skipping is inferred.** The router branches, so a run with no fresh-data
 * steps never enters the Collector — and nothing announces that. When a later
 * step starts, everything before it that never ran is marked skipped rather than
 * left pending, because a permanently spinning step reads as a hang.
 */

import * as React from 'react';
import { PIPELINE, type TimelineStep } from '@/components/investigation/execution-timeline';
import type { FeedEntry } from '@/components/investigation/execution-timeline';
import { streamInvestigation, type StreamStatus } from '@/lib/stream';
import type { TimelineEvent } from '@/types/investigation';

const ORDER: readonly string[] = PIPELINE.map((step) => step.id);

interface State {
  steps: Map<string, TimelineStep>;
  feed: FeedEntry[];
  counters: { evidence: number; insights: number; steps: number };
  gaps: Array<[number, number]>;
  status: StreamStatus;
  error: string | null;
  finished: boolean;
  reportId: string | null;
}

const INITIAL: State = {
  steps: new Map(),
  feed: [],
  counters: { evidence: 0, insights: 0, steps: 0 },
  gaps: [],
  status: 'connecting',
  error: null,
  finished: false,
  reportId: null,
};

/** Cap the feed. A long run emits hundreds of events and the DOM is not free. */
const MAX_FEED = 200;

type Action =
  | { kind: 'event'; event: TimelineEvent }
  | { kind: 'status'; status: StreamStatus }
  | { kind: 'gap'; from: number; to: number }
  | { kind: 'error'; message: string };

function setStep(
  steps: Map<string, TimelineStep>,
  id: string,
  patch: Partial<TimelineStep>,
): Map<string, TimelineStep> {
  const next = new Map(steps);
  next.set(id, { id, status: 'pending', ...next.get(id), ...patch });
  return next;
}

/**
 * Mark everything before `id` that never ran as skipped.
 *
 * Without this a branch the router declined to take sits at `pending` for the
 * rest of the run and reads as a step that is stuck. Only `pending` steps are
 * touched — a `done` or `failed` step keeps its outcome.
 */
function skipEarlier(
  steps: Map<string, TimelineStep>,
  id: string,
): Map<string, TimelineStep> {
  const index = ORDER.indexOf(id);
  if (index <= 0) return steps;
  let next = steps;
  for (const earlier of ORDER.slice(0, index)) {
    if ((next.get(earlier)?.status ?? 'pending') === 'pending') {
      next = setStep(next, earlier, { status: 'skipped' });
    }
  }
  return next;
}

function reduce(state: State, action: Action): State {
  switch (action.kind) {
    case 'status':
      return { ...state, status: action.status };

    case 'gap':
      return { ...state, gaps: [...state.gaps, [action.from, action.to]] };

    case 'error':
      return { ...state, error: action.message, status: 'failed' };

    case 'event': {
      const { event } = action;
      // The server's payload sits alongside the envelope fields rather than
      // under a `data` key. Read through an index type because the union's
      // known members do not declare the progress fields this UI uses -- and
      // an unknown event type is a contract-permitted case, not an error.
      const data = event as unknown as Record<string, unknown>;
      const type = String((event as { type?: unknown }).type ?? '');
      const node = typeof data.node === 'string' ? data.node : undefined;
      const message =
        typeof data.message === 'string' ? data.message : describe(type);
      const at = event.ts ? Date.parse(event.ts) : Date.now();

      let steps = state.steps;
      let finished = state.finished;
      let reportId = state.reportId;
      let error = state.error;

      if (node) {
        if (type === 'step.started') {
          const current = steps.get(node)?.status;
          // Guarded: a replayed `step.started` for a finished node must not
          // restart it. This happens on every reconnect.
          if (current !== 'done' && current !== 'failed') {
            steps = skipEarlier(steps, node);
            steps = setStep(steps, node, { status: 'running', startedAt: at, message });
          }
        } else if (type === 'step.completed') {
          const started = steps.get(node)?.startedAt;
          steps = setStep(steps, node, {
            status: 'done',
            message,
            durationMs: started ? at - started : undefined,
          });
        }
      }

      if (type === 'error') {
        error = message;
        finished = true;
        // Whatever was running when it failed owns the failure. Leaving it
        // spinning would suggest the run is still going.
        for (const [id, step] of steps) {
          if (step.status === 'running') steps = setStep(steps, id, { status: 'failed' });
        }
      }

      if (type === 'done') {
        finished = true;
        reportId = typeof data.report_id === 'string' ? data.report_id : null;
        for (const [id, step] of steps) {
          if (step.status === 'running') steps = setStep(steps, id, { status: 'done' });
        }
        // A run that finished cannot have pending steps -- the router simply
        // did not take those branches.
        for (const id of ORDER) {
          if ((steps.get(id)?.status ?? 'pending') === 'pending') {
            steps = setStep(steps, id, { status: 'skipped' });
          }
        }
      }

      const counters = {
        evidence: numberOr(data.evidence_count, state.counters.evidence),
        insights: numberOr(data.insight_count, state.counters.insights),
        steps: numberOr(data.step_count, state.counters.steps),
      };

      const entry: FeedEntry = { seq: event.seq, type: type, node, message, at };

      return {
        ...state,
        steps,
        counters,
        finished,
        reportId,
        error,
        feed: [entry, ...state.feed].slice(0, MAX_FEED),
      };
    }
  }
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function describe(type: string): string {
  switch (type) {
    case 'done':
      return 'Investigation complete';
    case 'error':
      return 'The investigation failed';
    case 'evidence.found':
      return 'Evidence found';
    case 'tool.called':
      return 'Tool called';
    default:
      return type;
  }
}

export interface UseInvestigationStream extends State {
  elapsedMs: number;
}

export function useInvestigationStream(
  investigationId: string | null,
  options: { token?: string | null; enabled?: boolean } = {},
): UseInvestigationStream {
  const [state, dispatch] = React.useReducer(reduce, INITIAL);
  const [elapsedMs, setElapsed] = React.useState(0);
  const startedAt = React.useRef<number>(Date.now());

  const enabled = options.enabled ?? true;
  const token = options.token ?? null;

  React.useEffect(() => {
    if (!investigationId || !enabled) return;
    startedAt.current = Date.now();

    const handle = streamInvestigation(
      investigationId,
      {
        onEvent: (event) => dispatch({ kind: 'event', event }),
        onGap: (from, to) => dispatch({ kind: 'gap', from, to }),
        onError: (err) => dispatch({ kind: 'error', message: err.message }),
        onStatusChange: (status) => dispatch({ kind: 'status', status }),
      },
      { token },
    );

    // Without this the connection outlives the component, and on a page that
    // re-renders that is one leaked stream per render.
    return () => handle.close();
  }, [investigationId, enabled, token]);

  // A one-second tick, stopped as soon as the run ends. Left running it would
  // re-render a finished page forever for no reason.
  React.useEffect(() => {
    if (state.finished) return;
    const id = window.setInterval(() => setElapsed(Date.now() - startedAt.current), 1000);
    return () => window.clearInterval(id);
  }, [state.finished]);

  return { ...state, elapsedMs };
}
