'use client';

/**
 * The main input of the product: where a research question is written.
 *
 * Everything about this component is shaped by one fact — pressing Start commits
 * the user to several minutes and a real amount of money. So the form's job is
 * not only to collect a question but to make the *consequences* legible before
 * the click, and to make a badly-shaped question obvious while there is still
 * time to fix it.
 *
 * **Depth is presented as a time and cost trade, not as a preference.** "Quick /
 * Standard / Deep" tells a user nothing; "~1 min, fewer sources" tells them what
 * they are choosing. The estimates are approximate and labelled as such.
 *
 * **Refresh sources is off by default and explains itself.** Turning it on
 * dispatches connector syncs, which is minutes of extra wall clock and a slice
 * of a third-party rate limit shared across the deployment. That cost is
 * invisible from the UI unless it is written down next to the switch.
 *
 * **The examples are clickable and deliberately specific.** A user facing an
 * empty box types something vague, gets a vague report, and concludes the
 * product is vague. Concrete examples teach the shape of a good question faster
 * than placeholder text can.
 */

import * as React from 'react';
import { ArrowRight, Info } from 'lucide-react';
import { Button, Label, Select, Textarea } from '@/components/ui/primitives';
import { cn } from '@/lib/utils';
import type { CreateInvestigationRequest, InvestigationDepth } from '@/types/investigation';

const MAX_QUERY_CHARS = 2000;

/** Where the character counter starts being shown at all. */
const COUNTER_VISIBLE_FROM = 1600;

const DEPTHS: Array<{
  value: InvestigationDepth;
  label: string;
  detail: string;
}> = [
  { value: 'quick', label: 'Quick', detail: '~1 min · fewer sources, no forecasting' },
  { value: 'standard', label: 'Standard', detail: '~3 min · full pipeline' },
  { value: 'deep', label: 'Deep', detail: '~8 min · wider retrieval, more revision' },
];

const EXAMPLES = [
  'How is Acme positioning against Globex in the mid-market, and what are customers saying about the difference?',
  'What are the recurring complaints about battery life across review sites in the last quarter?',
  'Which companies have entered the lithium supply chain in the past six months?',
] as const;

export interface QueryComposerProps {
  onSubmit: (request: CreateInvestigationRequest) => void | Promise<void>;
  submitting?: boolean;
  error?: string | null;
}

export function QueryComposer({ onSubmit, submitting, error }: QueryComposerProps) {
  const [query, setQuery] = React.useState('');
  const [objective, setObjective] = React.useState('');
  const [depth, setDepth] = React.useState<InvestigationDepth>('standard');
  const [refresh, setRefresh] = React.useState(false);
  const [touched, setTouched] = React.useState(false);

  const trimmed = query.trim();
  const tooLong = query.length > MAX_QUERY_CHARS;
  // Not a hard block, just a nudge: some perfectly good questions are short.
  const looksThin = trimmed.length > 0 && trimmed.length < 15;
  const invalid = trimmed.length === 0 || tooLong;

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setTouched(true);
    if (invalid || submitting) return;
    await onSubmit({
      query: trimmed,
      objective: objective.trim() || null,
      depth,
      refresh_connectors: refresh,
    });
  }

  return (
    <form onSubmit={submit} className="space-y-5">
      <div>
        <Label htmlFor="query" hint="What do you want to find out?">
          Research question
        </Label>
        <Textarea
          id="query"
          name="query"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onBlur={() => setTouched(true)}
          rows={3}
          autoGrowTo={280}
          invalid={touched && invalid}
          placeholder="e.g. How is Acme's battery strategy performing against competitors, and what are customers complaining about?"
          className="mt-2"
          // Cmd/Ctrl+Enter submits. The primary action on a form whose main
          // field is a multi-line textarea is otherwise a mouse trip, and this
          // is the shortcut people already expect from every chat interface.
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
              void submit(event as unknown as React.FormEvent);
            }
          }}
        />
        <div className="mt-1.5 flex min-h-[1.25rem] items-start justify-between gap-4">
          <p
            className={cn(
              'text-xs',
              touched && invalid ? 'text-destructive' : 'text-muted-foreground',
            )}
          >
            {touched && trimmed.length === 0
              ? 'An investigation needs a question.'
              : tooLong
                ? `${query.length.toLocaleString()} characters — the limit is ${MAX_QUERY_CHARS.toLocaleString()}.`
                : looksThin
                  ? 'Specific questions produce better-evidenced answers.'
                  : 'Press ⌘↵ to start.'}
          </p>
          {query.length >= COUNTER_VISIBLE_FROM ? (
            <span
              className={cn(
                'tabular shrink-0 text-xs',
                tooLong ? 'text-destructive' : 'text-muted-foreground',
              )}
            >
              {query.length.toLocaleString()}/{MAX_QUERY_CHARS.toLocaleString()}
            </span>
          ) : null}
        </div>
      </div>

      {query.length === 0 ? (
        <div>
          <p className="mb-2 text-xs text-muted-foreground">Or start from an example:</p>
          <div className="flex flex-col gap-1.5">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => setQuery(example)}
                className="rounded-md border border-border/70 bg-card/50 px-3 py-2 text-left text-xs leading-relaxed text-muted-foreground transition-colors hover:border-primary/40 hover:bg-accent hover:text-foreground"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <Label htmlFor="depth">Depth</Label>
          <Select
            id="depth"
            value={depth}
            onChange={(event) => setDepth(event.target.value as InvestigationDepth)}
            className="mt-2"
          >
            {DEPTHS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
          <p className="mt-1.5 text-xs text-muted-foreground">
            {DEPTHS.find((option) => option.value === depth)?.detail}
          </p>
        </div>

        <div>
          <Label htmlFor="objective" hint="optional">
            What will you use this for?
          </Label>
          <Textarea
            id="objective"
            value={objective}
            onChange={(event) => setObjective(event.target.value)}
            rows={1}
            autoGrowTo={120}
            placeholder="e.g. a board update on competitive risk"
            className="mt-2"
          />
          <p className="mt-1.5 text-xs text-muted-foreground">
            Steers what the recommendations optimise for.
          </p>
        </div>
      </div>

      <label className="flex cursor-pointer items-start gap-3 rounded-lg border border-border/70 bg-card/40 p-3.5 transition-colors hover:border-border">
        <input
          type="checkbox"
          checked={refresh}
          onChange={(event) => setRefresh(event.target.checked)}
          className="mt-0.5 size-4 shrink-0 cursor-pointer rounded border-input accent-[hsl(var(--primary))]"
        />
        <span className="min-w-0">
          <span className="block text-sm">Fetch fresh data first</span>
          <span className="mt-0.5 block text-xs leading-relaxed text-muted-foreground">
            Runs a connector sync before retrieval. Adds several minutes and uses
            shared source quota — leave off unless the question is about something
            that happened very recently.
          </span>
        </span>
      </label>

      {error ? (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/8 px-3.5 py-3">
          <Info className="mt-0.5 size-3.5 shrink-0 text-destructive" strokeWidth={2} />
          <p className="text-xs leading-relaxed text-destructive/90">{error}</p>
        </div>
      ) : null}

      <div className="flex items-center justify-between gap-4 pt-1">
        <p className="text-xs text-muted-foreground">
          Every claim in the report will link to a source you can open.
        </p>
        <Button type="submit" size="lg" loading={submitting} disabled={invalid}>
          {submitting ? 'Starting…' : 'Start investigation'}
          {!submitting ? <ArrowRight className="size-4" strokeWidth={2} /> : null}
        </Button>
      </div>
    </form>
  );
}
