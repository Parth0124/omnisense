'use client';

/**
 * Primary navigation.
 *
 * Flat and short on purpose. Six destinations is few enough that grouping them
 * into collapsible sections would add a click to reach anything while hiding
 * half the product from a new user.
 */
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Activity,
  GitBranch,
  LayoutDashboard,
  Plug,
  Search,
  TrendingUp,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const NAV = [
  { href: '/', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/investigations', label: 'Investigations', icon: Activity },
  { href: '/signals', label: 'Signals', icon: Search },
  { href: '/trends', label: 'Trends', icon: TrendingUp },
  { href: '/graph', label: 'Knowledge graph', icon: GitBranch },
  { href: '/connectors', label: 'Sources', icon: Plug },
] as const;

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-border/70 bg-card/40">
      <div className="flex h-16 items-center gap-2.5 px-5">
        <div
          className="grid size-7 place-items-center rounded-md bg-primary/15 text-primary"
          aria-hidden
        >
          <span className="text-sm font-bold">O</span>
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold tracking-tight">OmniSense</div>
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Market intelligence
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 px-3 py-2">
        {NAV.map(({ href, label, icon: Icon }) => {
          // Exact match for the dashboard, prefix match for everything else, so
          // `/investigations/abc` keeps Investigations highlighted. Prefix
          // matching on `/` would light up every route.
          const active = href === '/' ? pathname === '/' : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? 'page' : undefined}
              className={cn(
                'flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors',
                active
                  ? 'bg-primary/10 font-medium text-primary'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground',
              )}
            >
              <Icon className="size-4 shrink-0" strokeWidth={1.75} />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-border/70 px-5 py-3">
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Every claim traces to a source you can open.
        </p>
      </div>
    </aside>
  );
}
