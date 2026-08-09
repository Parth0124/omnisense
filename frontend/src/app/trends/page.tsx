'use client';

/**
 * Topic volume over a window.
 *
 * A bar per topic rather than a line chart over time, and the reason is what the
 * data actually supports: `topic_activity` returns one aggregate count per topic
 * for the window, not a series. Drawing a line through a single point per topic
 * would invent a trajectory the query never measured — which is precisely the
 * kind of fabrication the rest of this system is built to avoid.
 *
 * Sentiment is shown as a tint on the bar and `null` is rendered distinctly from
 * neutral. "Nobody assessed the sentiment of these mentions" and "the mentions
 * were neutral" are different facts, and a UI that draws them the same asserts
 * an assessment that was never made.
 */
import * as React from 'react';
import { useQuery } from '@tanstack/react-query';
import { PageHeader } from '@/components/layout/app-shell';
import { Card, Select, Skeleton } from '@/components/ui/primitives';
import { apiFetch } from '@/lib/api/client';
import { cn, formatRelative } from '@/lib/utils';

interface TopicActivity {
  topic_id: string;
  topic: string;
  mentions: number;
  avg_sentiment: number | null;
  last_mentioned_at: string | null;
}

function sentimentClass(score: number | null): string {
  if (score == null) return 'bg-muted-foreground/30';
  if (score >= 0.2) return 'bg-[hsl(var(--positive))]/60';
  if (score <= -0.2) return 'bg-[hsl(var(--negative))]/60';
  return 'bg-primary/50';
}

export default function TrendsPage() {
  const [windowDays, setWindowDays] = React.useState(30);

  const topics = useQuery({
    queryKey: ['topic-activity', windowDays],
    queryFn: () =>
      apiFetch<TopicActivity[]>('/graph/topics/activity', {
        query: { window_days: windowDays, limit: 25 },
      }),
  });

  const peak = Math.max(1, ...(topics.data ?? []).map((topic) => topic.mentions));

  return (
    <>
      <PageHeader
        title="Trends"
        description="What is being talked about, and how much, over a window."
        actions={
          <Select
            value={String(windowDays)}
            onChange={(event) => setWindowDays(Number(event.target.value))}
            className="w-40"
            aria-label="Window"
          >
            <option value="7">Last 7 days</option>
            <option value="30">Last 30 days</option>
            <option value="90">Last 90 days</option>
          </Select>
        }
      />

      <div className="px-8 py-6">
        {topics.isLoading ? (
          <div className="space-y-2">
            {[0, 1, 2, 3].map((index) => (
              <Skeleton key={index} className="h-11 w-full" />
            ))}
          </div>
        ) : topics.isError ? (
          <Card className="px-5 py-8 text-center">
            <p className="text-sm text-muted-foreground">Could not load topic activity.</p>
          </Card>
        ) : !topics.data?.length ? (
          <Card className="px-5 py-10 text-center">
            <p className="text-sm text-muted-foreground">
              No topic activity in this window. Topics appear once signals have been
              enriched and written to the knowledge graph.
            </p>
          </Card>
        ) : (
          <Card className="divide-y divide-border/50">
            {topics.data.map((topic) => (
              <div key={topic.topic_id} className="px-5 py-3">
                <div className="mb-1.5 flex items-baseline justify-between gap-4">
                  <span className="truncate text-sm">{topic.topic}</span>
                  <span className="tabular shrink-0 text-xs text-muted-foreground">
                    {topic.mentions.toLocaleString()}
                    {topic.last_mentioned_at
                      ? ` · ${formatRelative(topic.last_mentioned_at)}`
                      : ''}
                  </span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-border/50">
                  <div
                    className={cn('h-full rounded-full', sentimentClass(topic.avg_sentiment))}
                    style={{ width: `${(topic.mentions / peak) * 100}%` }}
                  />
                </div>
                {topic.avg_sentiment == null ? (
                  <p className="mt-1 text-[10px] text-muted-foreground/70">
                    sentiment not assessed
                  </p>
                ) : null}
              </div>
            ))}
          </Card>
        )}
      </div>
    </>
  );
}
