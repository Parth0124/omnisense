/**
 * Small shared helpers. Kept deliberately tiny -- a `utils` module is where
 * unrelated things accumulate, so anything here has to be used in more than one
 * place and belong to no particular feature.
 */
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Compose Tailwind classes, resolving conflicts in favour of the last one.
 *
 * `twMerge` on top of `clsx` because plain concatenation loses: `"p-2 p-4"`
 * applies whichever CSS rule the stylesheet happens to emit last, not the one
 * the caller wrote last. That makes a component's `className` override
 * unreliable in exactly the case it exists for -- a caller adjusting spacing on
 * a shared component.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * A duration in milliseconds as a short human string.
 *
 * Sub-second values render as milliseconds rather than "0.4s", because the
 * timeline shows per-step durations and a run where six steps all read "0.0s"
 * tells the reader nothing about which was slow.
 */
export function formatDuration(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return '--';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

/**
 * Relative time, coarse on purpose.
 *
 * "3 minutes ago" rather than "3 minutes and 14 seconds ago": the precision is
 * not information anyone acts on, and it forces a re-render every second to stay
 * truthful.
 */
export function formatRelative(iso: string | null | undefined): string {
  if (!iso) return '--';
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '--';
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 45) return 'just now';
  if (seconds < 90) return 'a minute ago';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} minutes ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`;
  const days = Math.round(hours / 24);
  return `${days} day${days === 1 ? '' : 's'} ago`;
}

/** Clamp a 0-1 score into a percentage integer, tolerating nulls. */
export function toPercent(score: number | null | undefined): number {
  if (score == null || Number.isNaN(score)) return 0;
  return Math.round(Math.min(1, Math.max(0, score)) * 100);
}
