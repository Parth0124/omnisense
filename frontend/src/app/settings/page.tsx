'use client';

/**
 * Settings, and an honest account of what this build can configure.
 *
 * Most of the system's behaviour is set by environment variables read at process
 * start — model tiers, retrieval weights, rate limits, connector credentials.
 * Surfacing those as editable fields here would be a lie: a change would either
 * do nothing or need a restart nobody warned about.
 *
 * So this page shows what is actually true — the live agent configuration, read
 * from `/agents` — and is explicit that changing it means changing the
 * deployment. A settings screen that admits it is read-only is more useful than
 * one whose controls silently have no effect.
 */
import { useQuery } from '@tanstack/react-query';
import { Info } from 'lucide-react';
import { PageHeader } from '@/components/layout/app-shell';
import { Badge, Card, Skeleton } from '@/components/ui/primitives';
import { apiFetch } from '@/lib/api/client';

interface AgentDescriptor {
  name: string;
  tier: string;
  blocking: boolean;
  tools: string[];
  output_schema: string;
  prompt: { version: string; sha256: string; fragments: string[] } | null;
}

export default function SettingsPage() {
  const agents = useQuery({
    queryKey: ['agents'],
    queryFn: () => apiFetch<AgentDescriptor[]>('/agents'),
  });

  return (
    <>
      <PageHeader
        title="Settings"
        description="What this deployment is running."
      />

      <div className="px-8 py-6">
        <Card className="mb-6 flex items-start gap-2.5 px-5 py-4">
          <Info className="mt-0.5 size-4 shrink-0 text-primary" strokeWidth={2} />
          <p className="text-xs leading-relaxed text-muted-foreground">
            Configuration comes from the environment and is read at process start.
            This page reports what is live rather than offering controls that would
            need a restart to take effect — see <code className="font-mono">.env.example</code>{' '}
            for every setting and its options.
          </p>
        </Card>

        <h2 className="mb-3 text-sm font-semibold tracking-tight">Agents</h2>

        {agents.isLoading ? (
          <div className="space-y-2">
            {[0, 1, 2].map((index) => (
              <Skeleton key={index} className="h-16 w-full" />
            ))}
          </div>
        ) : agents.isError ? (
          <Card className="px-5 py-8 text-center">
            <p className="text-sm text-muted-foreground">Could not read the agent roster.</p>
          </Card>
        ) : (
          <Card className="divide-y divide-border/50">
            {(agents.data ?? []).map((agent) => (
              <div key={agent.name} className="px-5 py-3.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium capitalize">{agent.name}</span>
                  <Badge tone="neutral">{agent.tier}</Badge>
                  {agent.blocking ? <Badge tone="caution">blocking</Badge> : null}
                  {agent.prompt ? (
                    <span
                      className="ml-auto font-mono text-[10px] text-muted-foreground"
                      // The hash covers the shared fragments too, so a change to
                      // the safety prompt moves every agent's -- which is
                      // correct, because it changed what every agent was told.
                      title={`prompt ${agent.prompt.version} · fragments: ${agent.prompt.fragments.join(', ')}`}
                    >
                      {agent.prompt.version} · {agent.prompt.sha256.slice(0, 10)}
                    </span>
                  ) : (
                    <Badge tone="negative">prompt failed to load</Badge>
                  )}
                </div>
                <p className="mt-1.5 text-xs text-muted-foreground">
                  {agent.tools.length > 0 ? (
                    <>
                      May call:{' '}
                      <span className="font-mono">{agent.tools.join(', ')}</span>
                    </>
                  ) : (
                    'No tools granted.'
                  )}
                </p>
              </div>
            ))}
          </Card>
        )}
      </div>
    </>
  );
}
