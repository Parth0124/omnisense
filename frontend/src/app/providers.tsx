'use client';

/**
 * Client-side providers.
 *
 * The query client is created inside `useState` rather than at module scope.
 * At module scope it would be shared across every request in a server render,
 * which leaks one user's cached data into another's page — the single most
 * consequential mistake available when wiring TanStack Query into the App
 * Router.
 */
import * as React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = React.useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Investigation and signal data changes underneath the user, so a
            // long stale time would show yesterday's run as current. Thirty
            // seconds is short enough to feel live and long enough that
            // navigating back and forth does not refetch on every click.
            staleTime: 30_000,
            retry: (failureCount, error) => {
              // 4xx will not become a 2xx. Retrying a 404 three times just
              // delays telling the user it does not exist.
              const status = (error as { status?: number })?.status;
              if (status && status >= 400 && status < 500) return false;
              return failureCount < 2;
            },
            refetchOnWindowFocus: false,
          },
        },
      }),
  );

  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
