// Superseded by `execution-timeline.tsx`, which renders the whole pipeline
// rather than one step at a time. Kept as a placeholder for a future
// per-step detail view (inputs, tool calls, raw output) reachable by clicking
// a row in the timeline.

export function AgentStep() {
  return (
    <div data-testid="AgentStep" className="text-xs text-muted-foreground">
      Per-step detail is not built yet. The timeline shows each stage and what it
      produced.
    </div>
  );
}
