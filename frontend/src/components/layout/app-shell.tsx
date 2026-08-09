/**
 * The application chrome: a fixed sidebar and a scrolling content column.
 *
 * The layout choice that matters is that the *content* column scrolls, not the
 * page. An investigation timeline can run to a hundred rows, and with a
 * page-level scroll the navigation disappears the moment the interesting part
 * arrives — so the user watching a run loses the way back to their other work
 * precisely while they are waiting.
 */
import Link from 'next/link';
import type { Route } from 'next';
import type { ReactNode } from 'react';
import { Sidebar } from '@/components/layout/sidebar';

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-dvh overflow-hidden bg-background">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <main className="scroll-slim min-h-0 flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}

/**
 * A page header. Consistent spacing across every screen, in one place.
 *
 * Extracted because the alternative is each page inventing its own top margin,
 * and the resulting half-rem differences between screens are individually
 * invisible and collectively make an app feel unfinished.
 */
export function PageHeader({
  title,
  description,
  actions,
  breadcrumb,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
  // `Route` rather than `string`: `typedRoutes` is on, and taking a bare
  // string here would push the cast to every caller -- which is where a typo
  // in a path stops being a compile error.
  breadcrumb?: { label: string; href: Route };
}) {
  return (
    <header className="border-b border-border/70 px-8 pb-5 pt-7">
      {breadcrumb ? (
        <Link
          href={breadcrumb.href}
          className="mb-2 inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          <span aria-hidden>&larr;</span> {breadcrumb.label}
        </Link>
      ) : null}
      <div className="flex items-start justify-between gap-6">
        <div className="min-w-0">
          <h1 className="truncate text-xl font-semibold tracking-tight">{title}</h1>
          {description ? (
            <p className="mt-1 text-sm text-muted-foreground">{description}</p>
          ) : null}
        </div>
        {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
      </div>
    </header>
  );
}
