'use client';

/**
 * The report. The thing the whole system exists to produce.
 *
 * Three decisions shape this page, and all three are about not overstating what
 * the report knows.
 *
 * **The gaps section is rendered first, not last.** Every instinct says to put
 * limitations at the bottom, and that is exactly why most readers never see
 * them. `docs/architecture.md` §7.3 permits a smaller, honestly-labelled answer
 * instead of a failure — a promise that only holds if the label is somewhere a
 * reader encounters before forming a view.
 *
 * **`409 report_not_ready` is a state, not an error.** A report row exists from
 * the moment its investigation starts, so arriving early is normal. The page
 * polls and says what it is waiting for, rather than showing a failure for
 * something that is thirty seconds away.
 *
 * **Uncited sections are marked.** A report where two of six sections carry no
 * citation should not render identically to one where all six do. The reader
 * cannot tell otherwise, and the prose reads the same either way.
 */

import * as React from 'react';
import Link from 'next/link';
import type { Route } from 'next';
import { useParams } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import { AlertTriangle, ArrowLeft, Download, FileWarning } from 'lucide-react';
import { PageHeader } from '@/components/layout/app-shell';
import { CitationChip } from '@/components/reports/citation';
import { ConfidenceBadge } from '@/components/reports/confidence-badge';
import { Badge, Button, Card, Skeleton, Spinner } from '@/components/ui/primitives';
import { getReport, isReportNotReady, type ReportSectionItem } from '@/lib/api/reports';
import { formatRelative } from '@/lib/utils';

function Section({ section, offset }: { section: ReportSectionItem; offset: number }) {
  const uncited = section.citations.length === 0;

  return (
    <section className="border-t border-border/60 py-6 first:border-t-0 first:pt-0">
      <div className="mb-2.5 flex items-center gap-2.5">
        <h2 className="text-base font-semibold tracking-tight">{section.heading}</h2>
        {uncited ? (
          <Badge tone="caution">
            <FileWarning className="size-3" strokeWidth={2} />
            uncited
          </Badge>
        ) : (
          <span className="text-[11px] text-muted-foreground">
            {section.citations.length} source
            {section.citations.length === 1 ? '' : 's'}
          </span>
        )}
      </div>

      {/* `whitespace-pre-wrap` because the body is generated prose with real
          paragraph breaks. Collapsing them would run the whole section into one
          block, and a markdown renderer would be a second place for untrusted
          text to be interpreted. */}
      <div className="whitespace-pre-wrap text-sm leading-[1.75] text-foreground/90">
        {section.body}
      </div>

      {section.citations.length > 0 ? (
        <div className="mt-3.5 flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-[11px] text-muted-foreground">Sources:</span>
          {section.citations.map((citation, index) => (
            <CitationChip key={citation.id} citation={citation} index={offset + index + 1} />
          ))}
        </div>
      ) : null}
    </section>
  );
}

export default function ReportPage() {
  const params = useParams<{ id: string }>();
  const reportId = params?.id ?? '';

  const report = useQuery({
    queryKey: ['report', reportId],
    queryFn: () => getReport(reportId),
    enabled: Boolean(reportId),
    // Poll only while the report is being written. `isReportNotReady` is the
    // signal; anything else stops the interval so a finished page is not
    // refetching forever.
    refetchInterval: (query) =>
      isReportNotReady(query.state.error) ? 3_000 : false,
    retry: (_count, error) => isReportNotReady(error),
  });

  if (report.isLoading) {
    return (
      <div className="mx-auto max-w-3xl px-8 py-10">
        <Skeleton className="mb-4 h-8 w-2/3" />
        <Skeleton className="mb-2 h-4 w-full" />
        <Skeleton className="mb-2 h-4 w-full" />
        <Skeleton className="h-4 w-1/2" />
      </div>
    );
  }

  if (isReportNotReady(report.error)) {
    return (
      <div className="mx-auto max-w-lg px-8 py-20 text-center">
        <Spinner className="mx-auto size-6 text-primary" />
        <h2 className="mt-4 text-sm font-semibold">The report is still being written</h2>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          The investigation has not reached its reporting stage yet. This page will
          update on its own.
        </p>
      </div>
    );
  }

  if (report.isError || !report.data) {
    return (
      <div className="mx-auto max-w-lg px-8 py-20 text-center">
        <h2 className="text-sm font-semibold">Report not found</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          It may have been removed, or it belongs to another account.
        </p>
        <Link href="/" className="mt-4 inline-block">
          <Button variant="secondary" size="sm">
            <ArrowLeft className="size-3.5" strokeWidth={2} />
            Back
          </Button>
        </Link>
      </div>
    );
  }

  const data = report.data;

  // A running offset so citation numbers are unique across the document. Per
  // section they would restart at 1, and "[1]" would mean five different
  // sources in a five-section report.
  let offset = 0;

  return (
    <>
      <PageHeader
        breadcrumb={{
          label: 'Investigation',
          href: `/investigations/${data.investigation_id}` as Route,
        }}
        title={data.title}
        description={`Generated ${formatRelative(data.created_at)} · ${data.citation_count} citation${
          data.citation_count === 1 ? '' : 's'
        }`}
        actions={
          data.download_url ? (
            <a href={data.download_url}>
              <Button variant="secondary" size="sm">
                <Download className="size-3.5" strokeWidth={2} />
                Download
              </Button>
            </a>
          ) : undefined
        }
      />

      <article className="mx-auto max-w-3xl px-8 py-8">
        <div className="mb-6 flex flex-wrap items-center gap-2">
          <ConfidenceBadge score={data.confidence} band={data.confidence_band} />
          {!data.is_current ? (
            <Badge tone="neutral">superseded by v{data.version + 1}</Badge>
          ) : (
            <Badge tone="neutral">version {data.version}</Badge>
          )}
        </div>

        {/* Gaps first. See the module docstring -- limitations at the bottom are
            limitations nobody reads. */}
        {data.gaps.length > 0 ? (
          <Card className="mb-8 border-[hsl(var(--caution))]/30 bg-[hsl(var(--caution))]/6 px-5 py-4">
            <div className="flex items-start gap-2.5">
              <AlertTriangle
                className="mt-0.5 size-4 shrink-0 text-[hsl(var(--caution))]"
                strokeWidth={2}
              />
              <div className="min-w-0">
                <h2 className="text-sm font-medium">
                  What this investigation could not establish
                </h2>
                <ul className="mt-2 space-y-1.5">
                  {data.gaps.map((gap) => (
                    <li
                      key={gap}
                      className="text-xs leading-relaxed text-muted-foreground"
                    >
                      · {gap}
                    </li>
                  ))}
                </ul>
                <p className="mt-2.5 text-[11px] leading-relaxed text-muted-foreground/80">
                  Read the findings below in light of these.
                </p>
              </div>
            </div>
          </Card>
        ) : null}

        {data.summary ? (
          <div className="mb-8 rounded-lg border-l-2 border-primary/50 bg-card/40 px-5 py-4">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Summary
            </h2>
            <p className="whitespace-pre-wrap text-sm leading-[1.75] text-foreground/95">
              {data.summary}
            </p>
          </div>
        ) : null}

        {data.uncited_sections.length > 0 ? (
          <p className="mb-6 text-xs leading-relaxed text-muted-foreground">
            {data.uncited_sections.length} section
            {data.uncited_sections.length === 1 ? ' carries' : 's carry'} no citation
            and {data.uncited_sections.length === 1 ? 'is' : 'are'} marked below.
          </p>
        ) : null}

        {data.sections.length === 0 ? (
          <Card className="px-5 py-10 text-center">
            <p className="text-sm text-muted-foreground">
              This report has no sections. Every drafted claim cited evidence that
              could not be resolved, so nothing could be published.
            </p>
          </Card>
        ) : (
          <div>
            {data.sections
              .slice()
              .sort((a, b) => a.ordinal - b.ordinal)
              .map((section) => {
                const element = (
                  <Section key={section.id} section={section} offset={offset} />
                );
                offset += section.citations.length;
                return element;
              })}
          </div>
        )}
      </article>
    </>
  );
}
