# frontend/

*Next.js App Router client (Design Doc §11).*

TypeScript · TailwindCSS · shadcn/ui · Zustand · TanStack Query.

## Layout

| Path | Purpose |
| --- | --- |
| `src/app/` | App Router pages: Dashboard, Investigations, Reports, Knowledge Graph, Trends, Competitors, Settings. |
| `src/components/` | Feature components, grouped by page; `ui/` holds shadcn primitives. |
| `src/lib/api/` | Typed client, one module per backend resource. |
| `src/lib/stream.ts` | Consumes the SSE execution timeline. |
| `src/hooks/` | TanStack Query hooks. |
| `src/stores/` | Zustand stores for client-only state. |
| `src/providers/` | Query and theme providers. |
| `src/types/` | Types mirroring the backend contracts. |

## The four components that matter

Per Design Doc §11 the distinguishing UI is: the **streaming execution
timeline**, **graph visualization**, **citations**, and **confidence
indicators**. Everything else is supporting chrome.

## Rules

- **TanStack Query owns server state. Zustand owns client state.** They must not
  overlap — duplicated server state in Zustand is the most likely source of bugs
  here.
- Only `NEXT_PUBLIC_*` variables reach the browser. Never put a secret behind
  that prefix.
- Server Components by default; add `'use client'` only where interactivity
  requires it.

## Run

```bash
npm install     # or: make install-frontend
npm run dev     # or: make frontend
```

Copy `.env.local.example` to `.env.local` first.

## See also

[`docs/api-reference.md`](../docs/api-reference.md) ·
[`docs/coding-standards.md`](../docs/coding-standards.md)
