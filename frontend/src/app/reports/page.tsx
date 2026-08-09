'use client';

/**
 * Reports, reached through the investigations that produced them.
 *
 * There is no "list every report" endpoint, and adding one would be the wrong
 * shape: a report only means something in the context of the question it answers
 * and the run that produced it. So this page lists *completed investigations*
 * and links to their reports, which is the same set of documents arranged the
 * way people actually look for them — by what they asked, not by a document id.
 */
import Link from 'next/link';
import type { Route } from 'next';
import { useQuery } from '@tanstack/react-query';
import { FileText } from 'lucide-react';
import { PageHeader } from '@/components/layout/app-shell';
import { Badge, Card, Skeleton } from '@/components/ui/primitives';
import { listInvestigations } from '@/lib/api/investigations';
import { formatRelative } from '@/lib/utils';

export default function ReportsPage() {
  const finished = useQuery({
    queryKey: ['investigations', 'completed'],
    queryFn: () =>
      listInvestigations({
        limit: 50,
        state: ['completed', 'completed_with_findings'],
      }),
  });

  const withReports = (finished.data ?? []).filter((item) => item.report_id);

  return (
    <>
      <PageHeader
        title="Reports"
        description="Finished investigations and the documents they produced."
      />

      <div className="px-8 py-6">
        {finished.isLoading ? (
          <div className="space-y-2">
            {[0, 1, 2].map((index) => (
              <Skeleton key={index} className="h-16 w-full" />
            ))}
          </div>
        ) : finished.isError ? (
          <Card className="px-5 py-8 text-center">
            <p className="text-sm text-muted-foreground">Could not load reports.</p>
          </Card>
        ) : !withReports.length ? (
          <Card className="px-5 py-10 text-center">
            <p className="text-sm text-muted-foreground">
              No reports yet. They appear once an investigation reaches its reporting
              stage.
            </p>
          </Card>
        ) : (
          <ul className="space-y-2">
            {withReports.map((item) => (
              <li key={item.id}>
                <Link
                  href={`/reports/${item.report_id}` as Route}
                  className="flex items-center gap-3 rounded-lg border border-border/70 bg-card/50 px-4 py-3.5 transition-colors hover:border-border hover:bg-card"
                >
                  <FileText
                    className="size-4 shrink-0 text-muted-foreground"
                    strokeWidth={1.75}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm">{item.query}</span>
                    <span className="mt-0.5 block text-xs text-muted-foreground">
                      {formatRelative(item.completed_at ?? item.created_at)}
                    </span>
                  </span>
                  {String(item.state) === 'completed_with_findings' ? (
                    // Surfaced in the list, not only inside the document. A
                    // report with stated gaps should be identifiable before it
                    // is opened.
                    <Badge tone="caution">has gaps</Badge>
                  ) : null}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}
