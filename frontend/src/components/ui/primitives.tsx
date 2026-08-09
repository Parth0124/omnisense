/**
 * The small set of primitives this app actually uses.
 *
 * One file rather than one per component. The shadcn convention is a file each,
 * which earns its keep in a design system consumed by many teams; here it would
 * be eleven files averaging fifteen lines, and the cost of that is that nobody
 * reads any of them. Grouped, the whole visual vocabulary is one scroll.
 *
 * Everything is styled through the CSS variables in `globals.css`. No component
 * here names a colour, which is what keeps the light theme working without a
 * second implementation.
 */
'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';

/* ------------------------------------------------------------------ Button */

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
type ButtonSize = 'sm' | 'md' | 'lg';

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    'bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/95 shadow-sm',
  secondary:
    'bg-secondary text-secondary-foreground hover:bg-secondary/80 border border-border',
  ghost: 'text-foreground/80 hover:bg-accent hover:text-foreground',
  danger: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-xs gap-1.5',
  md: 'h-9 px-4 text-sm gap-2',
  lg: 'h-11 px-6 text-sm gap-2',
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant = 'primary', size = 'md', loading, disabled, children, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      // `disabled` while loading, so a slow request cannot be submitted twice by
      // an impatient click. The visual state alone would not prevent it.
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cn(
        'inline-flex items-center justify-center rounded-md font-medium',
        'transition-colors duration-150',
        'disabled:pointer-events-none disabled:opacity-50',
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className,
      )}
      {...props}
    >
      {loading ? <Spinner className="size-3.5" /> : null}
      {children}
    </button>
  );
});

/* ----------------------------------------------------------------- Spinner */

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={cn('animate-spin', className)}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path
        d="M12 2a10 10 0 0 1 10 10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* -------------------------------------------------------------------- Card */

export function Card({
  className,
  raised,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & { raised?: boolean }) {
  return (
    <div
      className={cn(
        'rounded-xl border border-border',
        raised ? 'bg-[hsl(var(--surface-raised))]' : 'bg-card',
        className,
      )}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('px-5 pt-4 pb-3', className)} {...props} />;
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3 className={cn('text-sm font-semibold tracking-tight', className)} {...props} />
  );
}

export function CardBody({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('px-5 pb-5', className)} {...props} />;
}

/* ------------------------------------------------------------------- Input */

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  invalid?: boolean;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(function Input(
  { className, invalid, ...props },
  ref,
) {
  return (
    <input
      ref={ref}
      aria-invalid={invalid || undefined}
      className={cn(
        'w-full rounded-md border bg-background/60 px-3 py-2 text-sm',
        'placeholder:text-muted-foreground/70',
        'transition-colors focus:border-primary/60 focus:bg-background',
        'disabled:cursor-not-allowed disabled:opacity-50',
        invalid ? 'border-destructive/70' : 'border-input',
        className,
      )}
      {...props}
    />
  );
});

/* ---------------------------------------------------------------- Textarea */

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  invalid?: boolean;
  /** Grow with content up to this many pixels, then scroll. */
  autoGrowTo?: number;
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  function Textarea({ className, invalid, autoGrowTo = 320, onInput, ...props }, ref) {
    const inner = React.useRef<HTMLTextAreaElement | null>(null);

    // Auto-grow rather than a fixed box. The main input of this product is a
    // research question, which is routinely two or three sentences; a
    // three-row textarea makes the user scroll inside their own question while
    // composing it, which is the point at which they most need to see it whole.
    const grow = React.useCallback(
      (element: HTMLTextAreaElement | null) => {
        if (!element) return;
        element.style.height = 'auto';
        element.style.height = `${Math.min(element.scrollHeight, autoGrowTo)}px`;
      },
      [autoGrowTo],
    );

    React.useEffect(() => grow(inner.current), [grow, props.value]);

    return (
      <textarea
        ref={(node) => {
          inner.current = node;
          if (typeof ref === 'function') ref(node);
          else if (ref) ref.current = node;
          grow(node);
        }}
        aria-invalid={invalid || undefined}
        onInput={(event) => {
          grow(event.currentTarget);
          onInput?.(event);
        }}
        className={cn(
          'w-full resize-none rounded-md border bg-background/60 px-3.5 py-3 text-sm leading-relaxed',
          'placeholder:text-muted-foreground/70',
          'transition-colors focus:border-primary/60 focus:bg-background',
          'scroll-slim disabled:cursor-not-allowed disabled:opacity-50',
          invalid ? 'border-destructive/70' : 'border-input',
          className,
        )}
        {...props}
      />
    );
  },
);

/* ------------------------------------------------------------------ Select */

export const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(function Select({ className, children, ...props }, ref) {
  return (
    <select
      ref={ref}
      className={cn(
        'w-full appearance-none rounded-md border border-input bg-background/60 px-3 py-2 text-sm',
        'transition-colors focus:border-primary/60 focus:bg-background',
        // The native arrow is drawn light-on-light by some engines in a dark
        // theme, so it is replaced with one that follows the text colour.
        "bg-[url(\"data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2'%3e%3cpolyline points='6 9 12 15 18 9'/%3e%3c/svg%3e\")]",
        'bg-[length:1rem] bg-[right_0.6rem_center] bg-no-repeat pr-9',
        className,
      )}
      {...props}
    >
      {children}
    </select>
  );
});

/* ------------------------------------------------------------------- Label */

export function Label({
  className,
  hint,
  children,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement> & { hint?: string }) {
  return (
    <label className={cn('block', className)} {...props}>
      <span className="text-xs font-medium text-foreground/85">{children}</span>
      {hint ? (
        <span className="ml-2 text-xs font-normal text-muted-foreground">{hint}</span>
      ) : null}
    </label>
  );
}

/* ------------------------------------------------------------------- Badge */

type BadgeTone = 'neutral' | 'primary' | 'positive' | 'caution' | 'negative';

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: 'bg-secondary text-secondary-foreground border-border',
  primary: 'bg-primary/12 text-primary border-primary/25',
  positive: 'bg-[hsl(var(--positive))]/12 text-[hsl(var(--positive))] border-[hsl(var(--positive))]/25',
  caution: 'bg-[hsl(var(--caution))]/12 text-[hsl(var(--caution))] border-[hsl(var(--caution))]/25',
  negative: 'bg-[hsl(var(--negative))]/12 text-[hsl(var(--negative))] border-[hsl(var(--negative))]/25',
};

export function Badge({
  tone = 'neutral',
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-medium',
        BADGE_TONES[tone],
        className,
      )}
      {...props}
    />
  );
}

/* ---------------------------------------------------------------- Skeleton */

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-md bg-muted', className)} />;
}
