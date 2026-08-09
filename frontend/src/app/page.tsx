'use client';

/**
 * The home screen: ask a question.
 *
 * A dashboard of charts was the obvious alternative and would be the wrong
 * first screen. The product does one thing, and putting six summary tiles in
 * front of it makes a user hunt for the button that does it. Recent runs sit
 * below the composer, where they are reachable without competing with the
 * primary action.
 */
import * as React from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import type { Route } from 'next';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Clock } from 'lucide-react';
import { QueryComposer } from '@/components/investigation/query-composer';
import { Badge, Card, Skeleton } from '@/components/ui/primitives';
import { createInvestigation, listInvestigations } from '@/lib/api/investigations';
import { isApiError } from '@/lib/api/client';
import { formatRelative } from '@/lib/utils';
import type { CreateInvestigationRequest, InvestigationState } from '@/types/investigation';

const STATE_TONE: Record<string, Parameters<typeof Badge>[0]['tone']> = {
  queued: 'neutral',
  planning: 'primary',
  running: 'primary',
  reflecting: 'primary',
  completed: 'positive',
  completed_with_findings: 'caution',
  failed: 'negative',
  cancelled: 'neutral',
};

function stateLabel(state: InvestigationState): string {
  // `completed_with_findings` is a success with caveats, and "Completed with
  // findings" is the only phrasing that does not read as either a plain success
  // or a failure. It is long, and the length is the point.
  return String(state).replace(/_/g, ' ');
}

export default function HomePage() {
  const router = useRouter();

  const recent = useQuery({
    queryKey: ['investigations', 'recent'],
    queryFn: () => listInvestigations({ limit: 8 }),
  });

  const start = useMutation({
    mutationFn: (request: CreateInvestigationRequest) => createInvestigation(request),
    onSuccess: (created) => {
      // Straight to the live view. The whole design of the 202 response is that
      // the user watches the run rather than waiting on a request, and bouncing
      // them to a list would hide the thing they just started.
      router.push(`/investigations/${created.id}`);
    },
  });

  const errorMessage = start.error
    ? isApiError(start.error)
      ? start.error.detail || start.error.message
      : 'Could not start the investigation. Check your connection and try again.'
    : null;

  return (
    <div className="mx-auto max-w-3xl px-8 py-14">
      <div className="mb-9">
        <h1 className="text-2xl font-semibold tracking-tight">
          What do you want to find out?
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          OmniSense plans an investigation, gathers evidence from your connected
          sources, and writes a report where every claim links back to a document
          you can open.
        </p>
      </div>

      <Card className="p-6">
        <QueryComposer
          onSubmit={async (request) => {
            await start.mutateAsync(request);
          }}
          submitting={start.isPending}
          error={errorMessage}
        />
      </Card>

      <section className="mt-12">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold tracking-tight">
          <Clock className="size-3.5 text-muted-foreground" strokeWidth={2} />
          Recent investigations
        </h2>

        {recent.isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-14 w-full" />
            <Skeleton className="h-14 w-full" />
          </div>
        ) : recent.isError ? (
          <Card className="px-5 py-6 text-center">
            <p className="text-sm text-muted-foreground">
              Could not load recent investigations.
            </p>
          </Card>
        ) : !recent.data?.length ? (
          <Card className="px-5 py-8 text-center">
            <p className="text-sm text-muted-foreground">
              Nothing yet. Your investigations will appear here.
            </p>
          </Card>
        ) : (
          <ul className="space-y-2">
            {recent.data.map((item) => (
              <li key={item.id}>
                <Link
                  href={`/investigations/${item.id}` as Route}
                  className="flex items-center justify-between gap-4 rounded-lg border border-border/70 bg-card/50 px-4 py-3 transition-colors hover:border-border hover:bg-card"
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm">{item.query}</span>
                    <span className="mt-0.5 block text-xs text-muted-foreground">
                      {formatRelative(item.created_at)}
                    </span>
                  </span>
                  <Badge tone={STATE_TONE[String(item.state)] ?? 'neutral'}>
                    {stateLabel(item.state)}
                  </Badge>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
