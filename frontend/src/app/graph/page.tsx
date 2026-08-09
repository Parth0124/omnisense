'use client';

/**
 * The knowledge graph explorer: search an entity, then see what it connects to.
 *
 * **A radial layout, not a force simulation.** A force layout looks impressive
 * and settles somewhere different on every render, so the same neighbourhood
 * looks like a different graph each visit and nothing is where the user left it.
 * A deterministic radial arrangement puts the seed in the middle and its
 * neighbours on a ring in a stable order — the same query always draws the same
 * picture, which is what makes it navigable rather than decorative.
 *
 * **Rendered with plain SVG.** Cytoscape is in the dependency list and would be
 * the right answer for a thousand-node canvas with panning and clustering. For
 * the fifty-node neighbourhood this endpoint returns, it is a large runtime for
 * an ellipse and a line.
 *
 * **Edge labels are shown.** An unlabelled graph tells you two companies are
 * connected; the interesting part is whether that edge says *competes with* or
 * *acquired*, and hiding it behind a hover makes the picture pleasant and
 * useless.
 */
import * as React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Search } from 'lucide-react';
import { PageHeader } from '@/components/layout/app-shell';
import { Badge, Button, Card, Input, Skeleton } from '@/components/ui/primitives';
import { apiFetch } from '@/lib/api/client';

interface EntityHit {
  id: string;
  name: string;
  type: string;
  description: string | null;
  source_count: number;
}

interface GraphNode {
  id: string;
  name: string;
  type: string;
}

interface GraphEdge {
  source: string;
  target: string;
  predicate: string;
  confidence: number | null;
}

interface SubgraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
}

const WIDTH = 720;
const HEIGHT = 460;
const RADIUS = 168;

function Canvas({ data, seedId }: { data: SubgraphResponse; seedId: string }) {
  // Seed at the centre, everything else on a ring in the order the server
  // returned it -- which is stable for a given query, so the picture does not
  // rearrange itself between visits.
  const others = data.nodes.filter((node) => node.id !== seedId);
  const positions = new Map<string, { x: number; y: number }>();
  positions.set(seedId, { x: WIDTH / 2, y: HEIGHT / 2 });
  others.forEach((node, index) => {
    const angle = (index / Math.max(1, others.length)) * Math.PI * 2 - Math.PI / 2;
    positions.set(node.id, {
      x: WIDTH / 2 + Math.cos(angle) * RADIUS,
      y: HEIGHT / 2 + Math.sin(angle) * RADIUS,
    });
  });

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="h-auto w-full"
      role="img"
      aria-label={`Neighbourhood of ${seedId}: ${data.nodes.length} entities, ${data.edges.length} relationships`}
    >
      {data.edges.map((edge, index) => {
        const from = positions.get(edge.source);
        const to = positions.get(edge.target);
        if (!from || !to) return null;
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;
        return (
          <g key={`${edge.source}-${edge.predicate}-${edge.target}-${index}`}>
            <line
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke="hsl(var(--border))"
              // Confidence as opacity. A weak edge should look weak; drawing
              // every relationship identically presents an inference and a
              // sourced fact as the same claim.
              strokeOpacity={0.4 + (edge.confidence ?? 0.3) * 0.6}
              strokeWidth={1.2}
            />
            <text
              x={midX}
              y={midY - 3}
              textAnchor="middle"
              className="fill-muted-foreground"
              style={{ fontSize: 8 }}
            >
              {edge.predicate.toLowerCase().replace(/_/g, ' ')}
            </text>
          </g>
        );
      })}

      {data.nodes.map((node) => {
        const position = positions.get(node.id);
        if (!position) return null;
        const isSeed = node.id === seedId;
        return (
          <g key={node.id}>
            <circle
              cx={position.x}
              cy={position.y}
              r={isSeed ? 22 : 15}
              fill={isSeed ? 'hsl(var(--primary) / 0.22)' : 'hsl(var(--card))'}
              stroke={isSeed ? 'hsl(var(--primary))' : 'hsl(var(--border))'}
              strokeWidth={1.5}
            />
            <text
              x={position.x}
              y={position.y + (isSeed ? 38 : 29)}
              textAnchor="middle"
              className="fill-foreground"
              style={{ fontSize: 10 }}
            >
              {node.name.length > 22 ? `${node.name.slice(0, 21)}…` : node.name}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function GraphPage() {
  const [term, setTerm] = React.useState('');
  const [query, setQuery] = React.useState('');
  const [seed, setSeed] = React.useState<EntityHit | null>(null);

  const results = useQuery({
    queryKey: ['graph-search', query],
    queryFn: () =>
      apiFetch<{ results: EntityHit[] }>('/graph/search', { query: { q: query, limit: 10 } }),
    enabled: query.length > 0,
  });

  const subgraph = useQuery({
    queryKey: ['subgraph', seed?.id],
    queryFn: () =>
      apiFetch<SubgraphResponse>('/graph/subgraph', {
        method: 'POST',
        body: { entity_ids: [seed!.id], depth: 1, limit: 40 },
      }),
    enabled: Boolean(seed),
  });

  return (
    <>
      <PageHeader
        title="Knowledge graph"
        description="Search for an entity, then explore what it connects to."
      />

      <div className="px-8 py-6">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            setQuery(term.trim());
            setSeed(null);
          }}
          className="mb-5 flex gap-2"
        >
          <Input
            value={term}
            onChange={(event) => setTerm(event.target.value)}
            placeholder="Search companies, products, topics…"
            className="max-w-md"
          />
          <Button type="submit" disabled={!term.trim()}>
            <Search className="size-3.5" strokeWidth={2} />
            Search
          </Button>
        </form>

        {results.isFetching ? (
          <Skeleton className="h-24 w-full max-w-md" />
        ) : results.data?.results.length ? (
          <div className="mb-6 flex flex-wrap gap-2">
            {results.data.results.map((hit) => (
              <button
                key={hit.id}
                type="button"
                onClick={() => setSeed(hit)}
                className={`rounded-lg border px-3 py-2 text-left text-xs transition-colors ${
                  seed?.id === hit.id
                    ? 'border-primary/50 bg-primary/10'
                    : 'border-border/70 bg-card/50 hover:border-border hover:bg-card'
                }`}
              >
                <span className="block font-medium">{hit.name}</span>
                <span className="mt-0.5 block text-muted-foreground">
                  {hit.type} · {hit.source_count} source{hit.source_count === 1 ? '' : 's'}
                </span>
              </button>
            ))}
          </div>
        ) : query && !results.isFetching ? (
          <Card className="max-w-md px-5 py-6 text-center">
            <p className="text-sm text-muted-foreground">
              Nothing matched “{query}”.
            </p>
          </Card>
        ) : null}

        {seed ? (
          subgraph.isLoading ? (
            <Skeleton className="h-96 w-full" />
          ) : subgraph.data && subgraph.data.nodes.length > 0 ? (
            <Card className="overflow-hidden">
              <div className="flex items-center justify-between border-b border-border/70 px-5 py-3">
                <h3 className="text-sm font-semibold">{seed.name}</h3>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">
                    {subgraph.data.nodes.length} entities · {subgraph.data.edges.length}{' '}
                    relationships
                  </span>
                  {subgraph.data.truncated ? (
                    <Badge tone="caution">partial</Badge>
                  ) : null}
                </div>
              </div>
              <div className="p-4">
                <Canvas data={subgraph.data} seedId={seed.id} />
              </div>
              {subgraph.data.truncated ? (
                <p className="border-t border-border/70 px-5 py-2.5 text-xs text-muted-foreground">
                  The edge limit was reached — this is a subset of the neighbourhood,
                  not the whole of it.
                </p>
              ) : null}
            </Card>
          ) : (
            <Card className="px-5 py-10 text-center">
              <p className="text-sm text-muted-foreground">
                {seed.name} has no recorded relationships yet.
              </p>
            </Card>
          )
        ) : null}
      </div>
    </>
  );
}
