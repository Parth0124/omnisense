'use client';

/**
 * The live execution timeline. The screen a user watches while a run happens.
 *
 * An investigation takes minutes. The difference between that being acceptable
 * and being unbearable is entirely whether the person can see what is going on,
 * so this component's job is to answer three questions continuously: what is
 * happening right now, what has already been established, and is it still alive.
 *
 * **Steps are shown before they run.** The whole pipeline is rendered from the
 * start, with future steps dimmed. A timeline that appends rows as they arrive
 * gives no sense of how much is left, so a run four steps from finishing looks
 * identical to one that has just begun. Showing the shape up front means
 * progress is legible at a glance.
 *
 * **The step order is a plan, not a promise.** The router branches — a run with
 * no fresh-data steps skips the Collector, and the Critic loop can send the
 * graph round again. So a step that never arrives is marked skipped rather than
 * left spinning forever, and the count is described as an estimate.
 *
 * **Liveness is shown explicitly.** A stream that dropped and is retrying looks
 * exactly like a step that is taking a long time, and those need different
 * reactions from the user. The connection state has its own indicator.
 *
 * **Gaps are admitted.** `docs/api-reference.md` §5 says a gap in `seq` means
 * events were lost. Rendering the remaining events as a complete timeline would
 * quietly under-report what the run did.
 */

import * as React from 'react';
import {
  AlertTriangle,
  Check,
  ChevronRight,
  CircleDashed,
  Loader2,
  Minus,
  X,
} from 'lucide-react';
import { Badge, Card } from '@/components/ui/primitives';
import { cn, formatDuration } from '@/lib/utils';
import type { StreamStatus } from '@/lib/stream';

/**
 * The pipeline, in the order the router usually walks it.
 *
 * Duplicated from `workers/investigation_worker.py`'s `_ORDER` rather than
 * fetched, and that is a real trade: the copy can drift. It is made here because
 * the alternative is an empty timeline until the first event arrives, and the
 * first event is the thing the user is waiting to stop waiting for. The
 * descriptions say what each stage *does* in plain terms — someone watching this
 * should not need to know the system's internal vocabulary.
 */
export const PIPELINE = [
  { id: 'planner', label: 'Plan', description: 'Breaking the question into parts' },
  { id: 'collector', label: 'Collect', description: 'Fetching fresh data from sources' },
  { id: 'retriever', label: 'Retrieve', description: 'Searching the corpus for evidence' },
  { id: 'graph_expansion', label: 'Expand', description: 'Following the knowledge graph' },
  { id: 'trend', label: 'Trends', description: 'Measuring change over time' },
  { id: 'competitor', label: 'Competitors', description: 'Building the competitive picture' },
  { id: 'forecast', label: 'Forecast', description: 'Projecting measured series' },
  { id: 'insight', label: 'Insights', description: 'Synthesising what the evidence means' },
  { id: 'strategy', label: 'Strategy', description: 'Forming recommendations' },
  { id: 'critic', label: 'Verify', description: 'Checking every citation resolves' },
  { id: 'report', label: 'Report', description: 'Writing the document' },
] as const;

export type StepStatus = 'pending' | 'running' | 'done' | 'skipped' | 'failed';

export interface TimelineStep {
  id: string;
  status: StepStatus;
  /** What the step accomplished, from the server. Never composed here. */
  message?: string;
  startedAt?: number;
  durationMs?: number;
}

export interface TimelineCounters {
  evidence: number;
  insights: number;
  steps: number;
}

/* --------------------------------------------------------------- Step row */

const STATUS_ICON: Record<StepStatus, React.ReactNode> = {
  pending: <CircleDashed className="size-4 text-muted-foreground/50" strokeWidth={1.75} />,
  running: <Loader2 className="size-4 animate-spin text-primary" strokeWidth={2} />,
  done: <Check className="size-4 text-[hsl(var(--positive))]" strokeWidth={2.5} />,
  skipped: <Minus className="size-4 text-muted-foreground/50" strokeWidth={2} />,
  failed: <X className="size-4 text-[hsl(var(--negative))]" strokeWidth={2.5} />,
};

function StepRow({
  step,
  meta,
  isLast,
}: {
  step: TimelineStep;
  meta: (typeof PIPELINE)[number];
  isLast: boolean;
}) {
  const running = step.status === 'running';
  const inactive = step.status === 'pending' || step.status === 'skipped';

  return (
    <li className="relative flex gap-3.5 pb-1">
      {/* The connector line. Drawn behind the icon rather than as a border on
          the row, so it does not break where a row grows to two lines. */}
      {!isLast ? (
        <span
          aria-hidden
          className="absolute left-[11px] top-7 h-[calc(100%-1rem)] w-px bg-border"
        />
      ) : null}

      <span
        className={cn(
          'relative z-10 mt-1 grid size-[22px] shrink-0 place-items-center rounded-full border bg-card',
          running ? 'border-primary/50' : 'border-border',
        )}
      >
        {STATUS_ICON[step.status]}
      </span>

      <div className={cn('min-w-0 flex-1 pb-3', inactive && 'opacity-45')}>
        <div className="flex items-baseline justify-between gap-3">
          <span
            className={cn(
              'text-sm',
              running ? 'font-medium text-foreground' : 'text-foreground/90',
            )}
          >
            {meta.label}
          </span>
          {step.durationMs != null ? (
            <span className="tabular shrink-0 text-[11px] text-muted-foreground">
              {formatDuration(step.durationMs)}
            </span>
          ) : null}
        </div>

        <p
          className={cn(
            'mt-0.5 text-xs leading-relaxed',
            running ? 'text-primary/90' : 'text-muted-foreground',
            running && 'animate-breathe',
          )}
        >
          {/* The server's message when there is one -- it carries the real
              numbers. The static description is only a placeholder for a step
              that has not reported yet. */}
          {step.message ?? meta.description}
        </p>
      </div>
    </li>
  );
}

/* ------------------------------------------------------------- Connection */

function ConnectionPill({ status }: { status: StreamStatus }) {
  const config: Record<StreamStatus, { tone: Parameters<typeof Badge>[0]['tone']; label: string }> =
    {
      connecting: { tone: 'neutral', label: 'Connecting' },
      live: { tone: 'positive', label: 'Live' },
      reconnecting: { tone: 'caution', label: 'Reconnecting' },
      closed: { tone: 'neutral', label: 'Finished' },
      failed: { tone: 'negative', label: 'Disconnected' },
    };
  const { tone, label } = config[status];
  return (
    <Badge tone={tone}>
      <span
        aria-hidden
        className={cn(
          'size-1.5 rounded-full bg-current',
          status === 'live' && 'animate-breathe',
        )}
      />
      {label}
    </Badge>
  );
}

/* ---------------------------------------------------------------- Counters */

function Counter({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="tabular text-lg font-semibold leading-none">{value}</div>
      <div className="mt-1 text-[11px] uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- Timeline */

export interface ExecutionTimelineProps {
  steps: Map<string, TimelineStep>;
  counters: TimelineCounters;
  status: StreamStatus;
  /** Missing `seq` ranges, if any events were lost. */
  gaps: Array<[number, number]>;
  error?: string | null;
  elapsedMs?: number;
}

export function ExecutionTimeline({
  steps,
  counters,
  status,
  gaps,
  error,
  elapsedMs,
}: ExecutionTimelineProps) {
  const completed = PIPELINE.filter((s) => steps.get(s.id)?.status === 'done').length;

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between gap-4 border-b border-border/70 px-5 py-3.5">
        <div className="flex items-center gap-2.5">
          <h3 className="text-sm font-semibold tracking-tight">Progress</h3>
          <span className="tabular text-xs text-muted-foreground">
            {/* "of ~11" -- the tilde is doing real work. The Critic loop can
                add steps, so this is a lower bound and a bare "of 11" would
                make a progress bar run backwards. */}
            {completed} of ~{PIPELINE.length}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {elapsedMs != null ? (
            <span className="tabular text-xs text-muted-foreground">
              {formatDuration(elapsedMs)}
            </span>
          ) : null}
          <ConnectionPill status={status} />
        </div>
      </div>

      {/* A thin determinate bar. Deliberately not a percentage label: the
          denominator is an estimate, and a number invites arithmetic the
          estimate cannot support. */}
      <div className="h-0.5 w-full bg-border/60">
        <div
          className="h-full bg-primary transition-[width] duration-500 ease-out"
          style={{ width: `${(completed / PIPELINE.length) * 100}%` }}
          role="progressbar"
          aria-valuenow={completed}
          aria-valuemin={0}
          aria-valuemax={PIPELINE.length}
          aria-label="Investigation progress"
        />
      </div>

      <div className="grid grid-cols-3 gap-4 border-b border-border/70 px-5 py-4">
        <Counter label="Evidence" value={counters.evidence} />
        <Counter label="Insights" value={counters.insights} />
        <Counter label="Steps run" value={counters.steps} />
      </div>

      <ol className="scroll-slim max-h-[26rem] overflow-y-auto px-5 py-4">
        {PIPELINE.map((meta, index) => (
          <StepRow
            key={meta.id}
            meta={meta}
            step={steps.get(meta.id) ?? { id: meta.id, status: 'pending' }}
            isLast={index === PIPELINE.length - 1}
          />
        ))}
      </ol>

      {gaps.length > 0 ? (
        <div className="flex items-start gap-2 border-t border-border/70 bg-[hsl(var(--caution))]/8 px-5 py-3">
          <AlertTriangle
            className="mt-0.5 size-3.5 shrink-0 text-[hsl(var(--caution))]"
            strokeWidth={2}
          />
          <p className="text-xs leading-relaxed text-muted-foreground">
            Some progress updates were lost in transit, so this timeline may be
            missing steps. The investigation itself is unaffected — reload when it
            finishes to see the complete record.
          </p>
        </div>
      ) : null}

      {error ? (
        <div className="flex items-start gap-2 border-t border-border/70 bg-destructive/8 px-5 py-3">
          <X className="mt-0.5 size-3.5 shrink-0 text-destructive" strokeWidth={2.5} />
          <p className="text-xs leading-relaxed text-destructive/90">{error}</p>
        </div>
      ) : null}
    </Card>
  );
}

/* ------------------------------------------------------------ Live feed */

export interface FeedEntry {
  seq: number;
  type: string;
  node?: string;
  message: string;
  at: number;
}

/**
 * The raw event feed, newest first.
 *
 * Alongside the step list rather than instead of it, because they answer
 * different questions. The steps say where the run is; the feed says what just
 * happened — which is what someone stares at when a step is taking longer than
 * they expected and they want evidence that anything is still moving.
 */
export function LiveFeed({ entries }: { entries: FeedEntry[] }) {
  return (
    <Card className="overflow-hidden">
      <div className="border-b border-border/70 px-5 py-3.5">
        <h3 className="text-sm font-semibold tracking-tight">Activity</h3>
      </div>
      <div className="scroll-slim max-h-[26rem] overflow-y-auto">
        {entries.length === 0 ? (
          <p className="px-5 py-8 text-center text-xs text-muted-foreground">
            Waiting for the first update…
          </p>
        ) : (
          <ul className="divide-y divide-border/50">
            {entries.map((entry) => (
              <li
                key={entry.seq}
                className="animate-slide-in flex items-start gap-2.5 px-5 py-2.5"
              >
                <ChevronRight
                  className="mt-0.5 size-3 shrink-0 text-muted-foreground/60"
                  strokeWidth={2}
                />
                <div className="min-w-0 flex-1">
                  <p className="text-xs leading-relaxed text-foreground/90">
                    {entry.message}
                  </p>
                  {entry.node ? (
                    <span className="mt-0.5 inline-block text-[10px] uppercase tracking-wider text-muted-foreground/70">
                      {entry.node.replace(/_/g, ' ')}
                    </span>
                  ) : null}
                </div>
                <time className="tabular shrink-0 text-[10px] text-muted-foreground/70">
                  {new Date(entry.at).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                  })}
                </time>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}
