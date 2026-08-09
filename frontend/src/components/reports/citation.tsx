'use client';

/**
 * A citation marker and its popover.
 *
 * The quote is displayed verbatim and never re-fetched. It was verified against
 * the stored signal at report time; re-fetching the page now could return
 * something different — the article was edited, the post was deleted — and
 * silently swapping in the new text would break the one guarantee the report
 * makes.
 *
 * `char_range` indexes into the *archived* copy for the same reason. Applying it
 * to live content lands on the wrong characters, so it is never used to
 * highlight anything outside the stored quote.
 */
import * as React from 'react';
import { ExternalLink, Quote } from 'lucide-react';
import { Card } from '@/components/ui/primitives';
import type { ReportCitation } from '@/lib/api/reports';

export function CitationChip({
  citation,
  index,
}: {
  citation: ReportCitation;
  index: number;
}) {
  const [open, setOpen] = React.useState(false);

  return (
    <span className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        onBlur={() => setOpen(false)}
        aria-expanded={open}
        className="mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded bg-primary/15 px-1 align-super font-mono text-[10px] font-medium text-primary transition-colors hover:bg-primary/25"
      >
        {index}
      </button>

      {open ? (
        // `onMouseDown` prevented so clicking inside the popover does not blur
        // the trigger and close it before the click lands -- the classic
        // popover bug where links inside are unclickable.
        <Card
          raised
          onMouseDown={(event) => event.preventDefault()}
          className="absolute left-0 top-6 z-30 w-80 p-3.5 shadow-xl"
        >
          <div className="flex items-start gap-2">
            <Quote className="mt-0.5 size-3 shrink-0 text-muted-foreground" strokeWidth={2} />
            <blockquote className="text-xs leading-relaxed text-foreground/90">
              {citation.quote}
            </blockquote>
          </div>
          <div className="mt-3 flex items-center justify-between gap-3 border-t border-border/70 pt-2.5">
            <span className="truncate font-mono text-[10px] text-muted-foreground">
              {citation.signal_id}
            </span>
            <a
              href={`/signals/${encodeURIComponent(citation.signal_id)}`}
              className="inline-flex shrink-0 items-center gap-1 text-[11px] text-primary hover:underline"
            >
              Open source
              <ExternalLink className="size-3" strokeWidth={2} />
            </a>
          </div>
        </Card>
      ) : null}
    </span>
  );
}
