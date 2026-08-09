'use client';

/**
 * The confidence badge.
 *
 * A band and a number, never a number alone. The score is an aggregate of
 * judgements rather than a measurement, so a bare "0.63" invites a reader to
 * treat the second digit as meaningful — and to compare two reports on a
 * precision neither of them has.
 *
 * The rationale is reachable from the badge rather than buried, because a
 * confidence figure with no stated reason is exactly the kind of number people
 * either over-trust or ignore entirely.
 */
import { Badge } from '@/components/ui/primitives';
import { toPercent } from '@/lib/utils';

export type ConfidenceBand = 'low' | 'moderate' | 'high';

const TONE: Record<ConfidenceBand, Parameters<typeof Badge>[0]['tone']> = {
  high: 'positive',
  moderate: 'caution',
  low: 'negative',
};

const MEANING: Record<ConfidenceBand, string> = {
  high: 'Multiple independent sources state this directly.',
  moderate: 'Supported, but resting on limited or partly-conflicting evidence.',
  low: 'Thin evidence. Treat as a lead rather than a finding.',
};

export function ConfidenceBadge({
  score,
  band,
  rationale,
  showScore = true,
}: {
  score: number;
  band: ConfidenceBand;
  rationale?: string | null;
  showScore?: boolean;
}) {
  return (
    <span
      className="inline-flex items-center gap-1.5"
      title={rationale || MEANING[band]}
    >
      <Badge tone={TONE[band]}>
        {band} confidence
        {showScore ? (
          <span className="tabular opacity-70">{toPercent(score)}%</span>
        ) : null}
      </Badge>
    </span>
  );
}
