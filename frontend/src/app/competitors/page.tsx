'use client';

/**
 * Competitive positioning for one company or product.
 *
 * The column that matters is **basis**, and it is given equal visual weight to
 * the strength score rather than tucked away. `stated` means a document says
 * these two compete; `inferred` means the system concluded it from
 * co-occurrence — which in a corpus of mentions is equally true of a company and
 * its own supplier, its largest customer, and the analyst who covers both.
 *
 * A table that showed only names and a bar would present those two as the same
 * claim. Competitive positioning is what a reader acts on, so that conflation is
 * the most consequential one this UI could make.
 *
 * `as_of` is echoed and displayed because it defaults to *now* on the server: a
 * client that omitted it cannot otherwise reproduce the result it is looking at.
 */
import * as React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Search } from 'lucide-react';
import { PageHeader } from '@/components/layout/app-shell';
import { Badge, Button, Card, Input, Skeleton } from '@/components/ui/primitives';
import { apiFetch } from '@/lib/api/client';
import { toPercent } from '@/lib/utils';

interface CompetitorItem {
  id: string;
  name: string;
  type: string;
  strength: number | null;
  basis: 'stated' | 'inferred' | 'derived' | 'unknown';
  market: string | null;
  confidence: number | null;
  evidence_count: number;
  citations: string[];
}

interface CompetitorsResponse {
  subject: string;
  as_of: string;
  results: CompetitorItem[];
  total: number;
}

const BASIS_TONE: Record<string, Parameters<typeof Badge>[0]['tone']> = {
  stated: 'positive',
  derived: 'primary',
  inferred: 'caution',
  unknown: 'neutral',
};

const BASIS_MEANING: Record<string, string> = {
  stated: 'A document says these two compete. Sourced.',
  derived: 'Recorded in the knowledge graph from an earlier extraction.',
  inferred: 'Concluded from co-occurrence — equally true of a supplier or a customer.',
  unknown: 'The basis was not recorded.',
};

export default function CompetitorsPage() {
  const [term, setTerm] = React.useState('');
  const [subject, setSubject] = React.useState('');

  const competitors = useQuery({
    queryKey: ['competitors', subject],
    queryFn: () =>
      apiFetch<CompetitorsResponse>(
        `/graph/entities/${encodeURIComponent(subject)}/competitors`,
        { query: { limit: 25 } },
      ),
    enabled: subject.length > 0,
  });

  return (
    <>
      <PageHeader
        title="Competitors"
        description="Who competes with whom, and on what evidence."
      />

      <div className="px-8 py-6">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            setSubject(term.trim());
          }}
          className="mb-6 flex gap-2"
        >
          <Input
            value={term}
            onChange={(event) => setTerm(event.target.value)}
            placeholder="Company or product name…"
            className="max-w-md"
          />
          <Button type="submit" disabled={!term.trim()}>
            <Search className="size-3.5" strokeWidth={2} />
            Find rivals
          </Button>
        </form>

        {!subject ? (
          <Card className="px-5 py-10 text-center">
            <p className="text-sm text-muted-foreground">
              Enter a company or product. Aliases work — “Big Blue” finds IBM.
            </p>
          </Card>
        ) : competitors.isLoading ? (
          <Skeleton className="h-64 w-full" />
        ) : competitors.isError ? (
          <Card className="px-5 py-8 text-center">
            <p className="text-sm text-muted-foreground">
              Could not read the graph. Competitive data needs Neo4j.
            </p>
          </Card>
        ) : !competitors.data?.results.length ? (
          <Card className="px-5 py-10 text-center">
            <p className="text-sm text-muted-foreground">
              No recorded competitors for “{subject}”. That may mean none were
              extracted yet, rather than that none exist.
            </p>
          </Card>
        ) : (
          <>
            <p className="mb-3 text-xs text-muted-foreground">
              As the graph was believed at{' '}
              {new Date(competitors.data.as_of).toLocaleString()}.
            </p>
            <Card className="overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/70 text-left text-xs text-muted-foreground">
                    <th className="px-5 py-2.5 font-medium">Competitor</th>
                    <th className="px-3 py-2.5 font-medium">Basis</th>
                    <th className="px-3 py-2.5 font-medium">Strength</th>
                    <th className="px-3 py-2.5 text-right font-medium">Evidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {competitors.data.results.map((rival) => (
                    <tr key={rival.id}>
                      <td className="px-5 py-3">
                        <span className="block">{rival.name}</span>
                        {rival.market ? (
                          <span className="text-xs text-muted-foreground">
                            {rival.market}
                          </span>
                        ) : null}
                      </td>
                      <td className="px-3 py-3">
                        <span title={BASIS_MEANING[rival.basis]}>
                          <Badge tone={BASIS_TONE[rival.basis] ?? 'neutral'}>
                            {rival.basis}
                          </Badge>
                        </span>
                      </td>
                      <td className="px-3 py-3">
                        {rival.strength == null ? (
                          // Null and 0.0 are different claims: "nobody scored
                          // this" versus "assessed and negligible".
                          <span className="text-xs text-muted-foreground">not scored</span>
                        ) : (
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-20 overflow-hidden rounded-full bg-border/50">
                              <div
                                className="h-full rounded-full bg-primary/60"
                                style={{ width: `${toPercent(rival.strength)}%` }}
                              />
                            </div>
                            <span className="tabular text-xs text-muted-foreground">
                              {toPercent(rival.strength)}
                            </span>
                          </div>
                        )}
                      </td>
                      <td className="tabular px-3 py-3 text-right text-xs text-muted-foreground">
                        {rival.evidence_count}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </>
        )}
      </div>
    </>
  );
}
