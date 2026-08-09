'use client';

/**
 * The live investigation view. Where a user spends the minutes a run takes.
 *
 * Two panes side by side, answering two different questions. The timeline says
 * *where the run is* — which stages are done, which is active, how far there is
 * to go. The activity feed says *what just happened* — which is what someone
 * looks at when a stage is taking longer than they expected and they want
 * evidence that anything is still moving. Either alone leaves one of those
 * questions unanswered.
 *
 * **The stream is the source of truth while running; the API is afterwards.**
 * SSE carries progress, and when it ends the detail endpoint is fetched once for
 * the authoritative record. Polling the API alongside the stream would show two
 * slightly different states — the eventual-consistency gap between a checkpoint
 * write and a status update — and the user would see a step flicker between
 * done and running.
 *
 * **A finished run reached by a fresh page load must still render.** Someone
 * opening a link an hour later gets no live events, so the fallback is not an
 * edge case: it is how most views of a completed investigation happen.
 */

import * as React from 'react';
import Link from 'next/link';
import type { Route } from 'next';
import { useParams } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import { FileText, XCircle } from 'lucide-react';
import { PageHeader } from '@/components/layout/app-shell';
import {
  ExecutionTimeline,
  LiveFeed,
  PIPELINE,
  type TimelineStep,
} from '@/components/investigation/execution-timeline';
import { Button, Card, Skeleton } from '@/components/ui/primitives';
import { useInvestigationStream } from '@/hooks/use-investigation-stream';
import { cancelInvestigation, getInvestigation } from '@/lib/api/investigations';
import { isTerminalState } from '@/types/investigation';

export default function InvestigationPage() {
  const params = useParams<{ id: string }>();
  const investigationId = params?.id ?? '';

  // Fetched once for the question text and the state on arrival. Not polled:
  // the stream is what keeps this page current, and two sources of truth
  // updating at different rates is what makes a step flicker.
  const detail = useQuery({
    queryKey: ['investigation', investigationId],
    queryFn: () => getInvestigation(investigationId, ['plan']),
    enabled: Boolean(investigationId),
  });

  const alreadyFinished = detail.data ? isTerminalState(detail.data.state) : false;

  const stream = useInvestigationStream(investigationId, {
    // No point opening a stream for a run that ended before the page loaded --
    // the server would replay the whole history and immediately close.
    enabled: Boolean(investigationId) && detail.isSuccess && !alreadyFinished,
  });

  // When the stream ends, refetch once for the authoritative record: the final
  // status, the report id, and anything the last event did not carry.
  React.useEffect(() => {
    if (stream.finished) void detail.refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream.finished]);

  const [cancelling, setCancelling] = React.useState(false);
  async function cancel() {
    setCancelling(true);
    try {
      await cancelInvestigation(investigationId, 'cancelled from the UI');
      await detail.refetch();
    } finally {
      setCancelling(false);
    }
  }

  /**
   * For a run that finished before this page opened there are no events, so the
   * timeline is reconstructed from the recorded step count: everything up to it
   * is done, the rest was skipped. Coarser than the live view — there are no
   * per-step durations — and honest about what it is.
   */
  const steps: Map<string, TimelineStep> = React.useMemo(() => {
    if (!alreadyFinished || stream.steps.size > 0) return stream.steps;
    const completed = detail.data?.progress?.steps_completed ?? 0;
    const rebuilt = new Map<string, TimelineStep>();
    PIPELINE.forEach((meta, index) => {
      rebuilt.set(meta.id, {
        id: meta.id,
        status: index < completed ? 'done' : 'skipped',
      });
    });
    return rebuilt;
  }, [alreadyFinished, stream.steps, detail.data?.progress?.steps_completed]);

  const reportId = stream.reportId ?? detail.data?.report_id ?? null;
  const state = detail.data?.state;
  const running = state ? !isTerminalState(state) : false;

  if (detail.isLoading) {
    return (
      <div className="px-8 py-8">
        <Skeleton className="mb-6 h-8 w-2/3" />
        <div className="grid gap-5 lg:grid-cols-2">
          <Skeleton className="h-96" />
          <Skeleton className="h-96" />
        </div>
      </div>
    );
  }

  if (detail.isError || !detail.data) {
    return (
      <div className="px-8 py-8">
        <Card className="px-6 py-10 text-center">
          <h2 className="text-sm font-semibold">Investigation not found</h2>
          <p className="mx-auto mt-2 max-w-sm text-sm text-muted-foreground">
            It may have been removed, or it belongs to another account.
          </p>
          <Link href="/" className="mt-4 inline-block">
            <Button variant="secondary" size="sm">
              Start a new investigation
            </Button>
          </Link>
        </Card>
      </div>
    );
  }

  return (
    <>
      <PageHeader
        breadcrumb={{ label: 'Investigations', href: '/investigations' }}
        title={detail.data.query}
        description={
          detail.data.error
            ? undefined
            : running
              ? 'Running now — this page updates as each stage completes.'
              : 'This investigation has finished.'
        }
        actions={
          <>
            {reportId ? (
              <Link href={`/reports/${reportId}` as Route}>
                <Button size="sm">
                  <FileText className="size-3.5" strokeWidth={2} />
                  View report
                </Button>
              </Link>
            ) : null}
            {running ? (
              <Button variant="secondary" size="sm" onClick={cancel} loading={cancelling}>
                <XCircle className="size-3.5" strokeWidth={2} />
                Cancel
              </Button>
            ) : null}
          </>
        }
      />

      <div className="px-8 py-6">
        {detail.data.error ? (
          <Card className="mb-5 border-destructive/30 bg-destructive/8 px-5 py-4">
            <div className="flex items-start gap-2.5">
              <XCircle className="mt-0.5 size-4 shrink-0 text-destructive" strokeWidth={2} />
              <div>
                <p className="text-sm font-medium text-destructive">
                  This investigation did not complete
                </p>
                <p className="mt-1 text-xs leading-relaxed text-destructive/85">
                  {detail.data.error.message}
                </p>
              </div>
            </div>
          </Card>
        ) : null}

        {/* Two panes on a wide screen, stacked on a narrow one with the timeline
            first -- it is the one that answers "how much longer". */}
        <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <ExecutionTimeline
            steps={steps}
            counters={stream.counters}
            status={alreadyFinished ? 'closed' : stream.status}
            gaps={stream.gaps}
            error={stream.error}
            elapsedMs={running ? stream.elapsedMs : undefined}
          />
          <LiveFeed entries={stream.feed} />
        </div>

        {reportId && !running ? (
          <Card className="mt-5 flex items-center justify-between gap-4 px-5 py-4">
            <div>
              <p className="text-sm font-medium">The report is ready</p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Every claim links to the source that supports it.
              </p>
            </div>
            <Link href={`/reports/${reportId}` as Route}>
              <Button size="sm">Read it</Button>
            </Link>
          </Card>
        ) : null}
      </div>
    </>
  );
}
