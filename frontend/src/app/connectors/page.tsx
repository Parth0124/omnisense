'use client';

/**
 * The source catalogue.
 *
 * The distinction this page exists to draw is between *enabled* and
 * *configured*. A connector can be switched on with no credentials, in which
 * case it returns nothing — and that is indistinguishable from a working
 * connector on a quiet day unless the UI says which is which. It is the most
 * common "why is this source empty" question, and answering it here saves the
 * round trip to a log.
 *
 * `tos_blocked` is shown as its own state rather than as an error. Several
 * platforms have no lawful third-party API for the data this product wants, so
 * the connector refuses by design. Rendering that as a failure would send
 * someone to debug a decision.
 */
import { useQuery } from '@tanstack/react-query';
import { CheckCircle2, Circle, KeyRound, ShieldAlert } from 'lucide-react';
import { PageHeader } from '@/components/layout/app-shell';
import { Badge, Card, Skeleton } from '@/components/ui/primitives';
import { apiFetch } from '@/lib/api/client';

interface ConnectorItem {
  slug: string;
  platform: string;
  category: string;
  enabled: boolean;
  configured: boolean;
  requires_tos_review: boolean;
  supports_incremental: boolean;
  supports_backfill: boolean;
  auth_type: string;
  version: string;
}

function StateBadge({ item }: { item: ConnectorItem }) {
  if (item.requires_tos_review) {
    return (
      <Badge tone="caution">
        <ShieldAlert className="size-3" strokeWidth={2} />
        review required
      </Badge>
    );
  }
  if (!item.enabled) {
    return (
      <Badge tone="neutral">
        <Circle className="size-3" strokeWidth={2} />
        off
      </Badge>
    );
  }
  if (!item.configured) {
    return (
      <Badge tone="caution">
        <KeyRound className="size-3" strokeWidth={2} />
        needs credentials
      </Badge>
    );
  }
  return (
    <Badge tone="positive">
      <CheckCircle2 className="size-3" strokeWidth={2} />
      ready
    </Badge>
  );
}

export default function ConnectorsPage() {
  const connectors = useQuery({
    queryKey: ['connectors'],
    queryFn: () => apiFetch<ConnectorItem[]>('/connectors'),
  });

  const byCategory = (connectors.data ?? []).reduce<Record<string, ConnectorItem[]>>(
    (groups, item) => {
      (groups[item.category] ??= []).push(item);
      return groups;
    },
    {},
  );

  return (
    <>
      <PageHeader
        title="Sources"
        description="Where OmniSense collects evidence from. A source that is on but unconfigured returns nothing."
      />
      <div className="px-8 py-6">
        {connectors.isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        ) : connectors.isError ? (
          <Card className="px-5 py-8 text-center">
            <p className="text-sm text-muted-foreground">Could not load the catalogue.</p>
          </Card>
        ) : (
          <div className="space-y-7">
            {Object.entries(byCategory)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([category, items]) => (
                <section key={category}>
                  <h2 className="mb-2.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    {category}
                  </h2>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {items.map((item) => (
                      <Card key={item.slug} className="px-4 py-3.5">
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <div className="truncate text-sm font-medium">{item.slug}</div>
                            <div className="mt-0.5 text-xs text-muted-foreground">
                              {item.auth_type === 'none'
                                ? 'no credentials needed'
                                : `${item.auth_type} auth`}
                              {item.supports_incremental ? ' · incremental' : ''}
                            </div>
                          </div>
                          <StateBadge item={item} />
                        </div>
                      </Card>
                    ))}
                  </div>
                </section>
              ))}
          </div>
        )}
      </div>
    </>
  );
}
