'use client';

/**
 * Every investigation, newest first.
 *
 * The state filter is a row of toggles rather than a dropdown because the
 * question people arrive with is usually "what is running right now" — and a
 * dropdown makes that two clicks and hides the answer behind a closed menu.
 */
import * as React from 'react';
import Link from 'next/link';
import type { Route } from 'next';
import { useQuery } from '@tanstack/react-query';
import { PageHeader } from '@/components/layout/app-shell';
import { Badge, Button, Card, Skeleton } from '@/components/ui/primitives';
import { listInvestigations } from '@/lib/api/investigations';
import { formatRelative, cn } from '@/lib/utils';
import type { InvestigationState } from '@/types/investigation';

const FILTERS: Array<{ label: string; states: InvestigationState[] }> = [
  { label: 'All', states: [] },
  { label: 'Running', states: ['queued', 'planning', 'running', 'reflecting'] },
  { label: 'Completed', states: ['completed', 'completed_with_findings'] },
  { label: 'Failed', states: ['failed', 'cancelled'] },
];

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

export default function InvestigationsPage() {
  const [filter, setFilter] = React.useState(0);
  const states = FILTERS[filter]?.states ?? [];

  const investigations = useQuery({
    queryKey: ['investigations', states],
    queryFn: () => listInvestigations({ limit: 50, state: states.length ? states : undefined }),
    // Something on this page is usually in flight, and a list that does not
    // move while a run finishes is a list people reload by hand.
    refetchInterval: 10_000,
  });

  return (
    <>
      <PageHeader
        title="Investigations"
        description="Every run, newest first."
        actions={
          <Link href={'/' as Route}>
            <Button size="sm">New investigation</Button>
          </Link>
        }
      />

      <div className="px-8 py-6">
        <div className="mb-4 flex gap-1.5">
          {FILTERS.map((option, index) => (
            <button
              key={option.label}
              type="button"
              onClick={() => setFilter(index)}
              className={cn(
                'rounded-md px-3 py-1.5 text-xs transition-colors',
                index === filter
                  ? 'bg-primary/12 font-medium text-primary'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground',
              )}
            >
              {option.label}
            </button>
          ))}
        </div>

        {investigations.isLoading ? (
          <div className="space-y-2">
            {[0, 1, 2].map((index) => (
              <Skeleton key={index} className="h-16 w-full" />
            ))}
          </div>
        ) : investigations.isError ? (
          <Card className="px-5 py-8 text-center">
            <p className="text-sm text-muted-foreground">Could not load investigations.</p>
          </Card>
        ) : !investigations.data?.length ? (
          <Card className="px-5 py-10 text-center">
            <p className="text-sm text-muted-foreground">
              Nothing here yet.{' '}
              <Link href={'/' as Route} className="text-primary hover:underline">
                Start an investigation
              </Link>
              .
            </p>
          </Card>
        ) : (
          <ul className="space-y-2">
            {investigations.data.map((item) => (
              <li key={item.id}>
                <Link
                  href={`/investigations/${item.id}` as Route}
                  className="flex items-center justify-between gap-4 rounded-lg border border-border/70 bg-card/50 px-4 py-3.5 transition-colors hover:border-border hover:bg-card"
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm">{item.query}</span>
                    <span className="mt-0.5 block text-xs text-muted-foreground">
                      {formatRelative(item.created_at)}
                    </span>
                  </span>
                  <Badge tone={STATE_TONE[String(item.state)] ?? 'neutral'}>
                    {String(item.state).replace(/_/g, ' ')}
                  </Badge>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}
