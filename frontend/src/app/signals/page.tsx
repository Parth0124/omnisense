'use client';

/**
 * The corpus browser.
 *
 * Deliberately a *browser* rather than a search interface. Full-text search over
 * signals lives behind `/signals/search`, and conflating the two would put a
 * search box in front of a list whose default order is chronological — which
 * trains people to search when what they wanted was to see what arrived today.
 *
 * Duplicates are marked rather than hidden. A press release syndicated to six
 * platforms is one thing that happened, and a reader counting six rows is
 * overcounting the evidence.
 */
import { useQuery } from '@tanstack/react-query';
import { Copy } from 'lucide-react';
import { PageHeader } from '@/components/layout/app-shell';
import { Badge, Card, Skeleton } from '@/components/ui/primitives';
import { apiFetch } from '@/lib/api/client';
import { formatRelative } from '@/lib/utils';

interface SignalItem {
  id: string;
  platform: string;
  source: string;
  url: string | null;
  timestamp: string;
  title: string | null;
  text: string;
  text_truncated: boolean;
  sentiment: { score: number; band: string } | null;
  is_canonical: boolean;
  duplicate_of: string | null;
}

interface SignalPage {
  items: SignalItem[];
  limit: number;
  next_cursor: string | null;
  has_more: boolean;
}

const SENTIMENT_TONE: Record<string, Parameters<typeof Badge>[0]['tone']> = {
  positive: 'positive',
  neutral: 'neutral',
  negative: 'negative',
  mixed: 'caution',
};

export default function SignalsPage() {
  const signals = useQuery({
    queryKey: ['signals'],
    queryFn: () => apiFetch<SignalPage>('/signals', { query: { limit: 30 } }),
  });

  return (
    <>
      <PageHeader
        title="Signals"
        description="Everything collected, newest first. Each one is a single thing somebody said."
      />
      <div className="px-8 py-6">
        {signals.isLoading ? (
          <div className="space-y-2">
            {[0, 1, 2].map((index) => (
              <Skeleton key={index} className="h-20 w-full" />
            ))}
          </div>
        ) : signals.isError ? (
          <Card className="px-5 py-8 text-center">
            <p className="text-sm text-muted-foreground">
              Could not load signals. The corpus may not be reachable.
            </p>
          </Card>
        ) : !signals.data?.items.length ? (
          <Card className="px-5 py-10 text-center">
            <p className="text-sm text-muted-foreground">
              No signals yet. Configure a source and run a sync to start collecting.
            </p>
          </Card>
        ) : (
          <ul className="space-y-2">
            {signals.data.items.map((signal) => (
              <li key={signal.id}>
                <Card className="px-4 py-3.5">
                  <div className="mb-1.5 flex flex-wrap items-center gap-2">
                    <Badge tone="neutral">{signal.platform}</Badge>
                    {signal.sentiment ? (
                      <Badge tone={SENTIMENT_TONE[signal.sentiment.band] ?? 'neutral'}>
                        {signal.sentiment.band}
                      </Badge>
                    ) : null}
                    {!signal.is_canonical ? (
                      <Badge tone="caution">
                        <Copy className="size-3" strokeWidth={2} />
                        duplicate
                      </Badge>
                    ) : null}
                    <span className="ml-auto text-xs text-muted-foreground">
                      {formatRelative(signal.timestamp)}
                    </span>
                  </div>
                  {signal.title ? (
                    <div className="text-sm font-medium">{signal.title}</div>
                  ) : null}
                  <p className="mt-1 line-clamp-3 text-xs leading-relaxed text-muted-foreground">
                    {signal.text}
                    {signal.text_truncated ? '…' : ''}
                  </p>
                </Card>
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}
