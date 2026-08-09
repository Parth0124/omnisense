/**
 * The SSE client for investigation timelines.
 *
 * `docs/api-reference.md` §5 defines the stream: named events, a monotonic `seq`
 * on each, `Last-Event-ID` for resume, and two terminal types after which the
 * server closes and the client must not reconnect.
 *
 * **Why not the browser's `EventSource`.** It cannot send headers, and this API
 * authenticates with a bearer token. The usual workaround is a token in the
 * query string, which puts a credential in the server's access log, in the
 * browser's history, and in the `Referer` of anything the page subsequently
 * loads. `fetch` with a `ReadableStream` costs us a hand-written frame parser
 * and buys a credential that stays in a header.
 *
 * That trade has a second consequence worth stating: `EventSource` reconnects on
 * its own and we now have to. The reconnection policy below is therefore
 * explicit rather than inherited, which is an improvement — the built-in one
 * retries forever, including against a terminal stream.
 *
 * **Gaps are surfaced, not hidden.** Every event carries `seq`. When one arrives
 * out of order the consumer is told, because §5 promises that a gap means loss
 * and the UI has to be able to say "some steps are not shown" rather than
 * silently rendering an incomplete timeline as if it were complete.
 */

import { API_VERSION_PREFIX, resolveApiBase } from '@/lib/api/client';
import type { TimelineEvent } from '@/types/investigation';

/** Terminal event types. After one of these the server closes and we stop. */
export const TERMINAL_EVENTS: readonly string[] = ['done', 'error'];

/**
 * Reconnection backoff, in milliseconds.
 *
 * Capped at 15s and jittered. Uncapped exponential backoff on a page a user is
 * watching means the stream appears dead long before it gives up; no jitter
 * means every client reconnects in lockstep after a deploy, which is how a
 * recovering API is immediately re-saturated by its own clients.
 */
const BACKOFF_MS = [500, 1_000, 2_000, 5_000, 10_000, 15_000] as const;
const MAX_RECONNECT_ATTEMPTS = 8;

function backoffFor(attempt: number): number {
  // Non-null: `attempt` is clamped to the array's own bounds, so the index is
  // always valid. TypeScript cannot see that through `Math.min`.
  const base = BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)]!;
  return base + Math.random() * base * 0.3;
}

export interface StreamHandlers {
  /** One timeline event, in order. */
  onEvent: (event: TimelineEvent) => void;
  /** The stream reached a terminal event. No reconnect will follow. */
  onDone?: (event: TimelineEvent) => void;
  /**
   * A gap was detected or reported. `from`/`to` are the missing `seq` range.
   *
   * Surfaced so the UI can say so. A timeline missing three steps looks exactly
   * like a run that had three fewer steps.
   */
  onGap?: (from: number, to: number) => void;
  /** Transport-level failure, after retries are exhausted. */
  onError?: (error: Error) => void;
  /** Connection state, for the "live / reconnecting / closed" indicator. */
  onStatusChange?: (status: StreamStatus) => void;
}

export type StreamStatus = 'connecting' | 'live' | 'reconnecting' | 'closed' | 'failed';

export interface StreamHandle {
  /** Stop the stream and release the connection. Idempotent. */
  close: () => void;
}

interface ParsedFrame {
  event: string;
  data: string;
  id: string | null;
}

/**
 * Parse one SSE frame from its raw text.
 *
 * Written by hand because we are not using `EventSource`. The format is simple
 * but has two details that bite: a line may repeat (`data:` twice means two
 * lines joined by `\n`), and a leading space after the colon is part of the
 * syntax rather than the value.
 */
function parseFrame(raw: string): ParsedFrame | null {
  const lines = raw.split('\n');
  let event = 'message';
  let id: string | null = null;
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(':')) continue; // comment / heartbeat
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    // One optional leading space is syntax, not content.
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);

    if (field === 'event') event = value;
    else if (field === 'data') dataLines.push(value);
    else if (field === 'id') id = value;
  }

  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join('\n'), id };
}

/**
 * Open an investigation's timeline stream.
 *
 * Returns immediately with a handle; events arrive through the callbacks. The
 * caller is responsible for calling `close()` — in React that means a cleanup
 * function, and forgetting it leaks a connection per mount, which on a page
 * that re-renders is a connection per render.
 */
export function streamInvestigation(
  investigationId: string,
  handlers: StreamHandlers,
  options: { token?: string | null; signal?: AbortSignal } = {},
): StreamHandle {
  const controller = new AbortController();
  let closed = false;
  let lastSeq = -1;
  let attempt = 0;

  const setStatus = (status: StreamStatus) => handlers.onStatusChange?.(status);

  if (options.signal) {
    options.signal.addEventListener('abort', () => close(), { once: true });
  }

  function close(): void {
    if (closed) return;
    closed = true;
    controller.abort();
    setStatus('closed');
  }

  async function connect(): Promise<void> {
    if (closed) return;
    setStatus(attempt === 0 ? 'connecting' : 'reconnecting');

    const url = `${resolveApiBase()}${API_VERSION_PREFIX}/investigations/${encodeURIComponent(
      investigationId,
    )}/stream`;

    const headers: Record<string, string> = { Accept: 'text/event-stream' };
    if (options.token) headers.Authorization = `Bearer ${options.token}`;
    // The resume point. On a reconnect this is what stops the server replaying
    // the whole run from the beginning — which would re-render every step the
    // user has already watched.
    if (lastSeq >= 0) headers['Last-Event-ID'] = String(lastSeq);

    let response: Response;
    try {
      response = await fetch(url, {
        headers,
        signal: controller.signal,
        cache: 'no-store',
      });
    } catch (error) {
      if (closed) return;
      return scheduleReconnect(error);
    }

    if (!response.ok || !response.body) {
      // 4xx is not retryable: a 401 or a 404 will be a 401 or a 404 again, and
      // retrying eight times against it just delays telling the user.
      if (response.status >= 400 && response.status < 500) {
        setStatus('failed');
        handlers.onError?.(
          new Error(
            response.status === 404
              ? 'This investigation does not exist, or belongs to another account.'
              : `Cannot open the stream (HTTP ${response.status}).`,
          ),
        );
        closed = true;
        return;
      }
      return scheduleReconnect(new Error(`HTTP ${response.status}`));
    }

    attempt = 0;
    setStatus('live');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (!closed) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Frames are separated by a blank line. Anything after the last
        // separator is a partial frame and stays in the buffer — dispatching it
        // would deliver half a JSON document.
        let separator = buffer.indexOf('\n\n');
        while (separator !== -1) {
          const raw = buffer.slice(0, separator);
          buffer = buffer.slice(separator + 2);
          dispatch(raw);
          separator = buffer.indexOf('\n\n');
        }
      }
    } catch (error) {
      if (!closed) return scheduleReconnect(error);
    } finally {
      reader.releaseLock();
    }

    // The server closed without a terminal event: the run is still going and
    // the connection dropped. Reconnect from `lastSeq`.
    if (!closed) return scheduleReconnect(new Error('stream ended without a terminal event'));
  }

  function dispatch(raw: string): void {
    const frame = parseFrame(raw);
    if (!frame) return;

    let payload: TimelineEvent;
    try {
      payload = JSON.parse(frame.data) as TimelineEvent;
    } catch {
      // One unparseable frame must not kill the stream. It is a lost progress
      // update, which the gap accounting below will surface if it mattered.
      return;
    }

    const type = String(frame.event || payload.type);
    const event = { ...payload, type } as TimelineEvent;

    // `stream.gap` is the server telling us it dropped events for this
    // connection. Forwarded rather than swallowed: §5 promises gaps mean loss.
    if (type === 'stream.gap') {
      const gap = payload as { from_seq?: number; to_seq?: number };
      handlers.onGap?.(
        Number(gap.from_seq ?? lastSeq + 1),
        Number(gap.to_seq ?? lastSeq + 1),
      );
      return;
    }

    const seq = typeof payload.seq === 'number' ? payload.seq : lastSeq + 1;

    // A replay after reconnect can legitimately resend what we already have.
    // Dropping it here keeps the timeline idempotent without the consumer
    // needing to dedupe.
    if (seq <= lastSeq) return;

    if (lastSeq >= 0 && seq > lastSeq + 1) {
      handlers.onGap?.(lastSeq + 1, seq - 1);
    }
    lastSeq = seq;

    handlers.onEvent(event);

    if (TERMINAL_EVENTS.includes(type)) {
      handlers.onDone?.(event);
      close();
    }
  }

  function scheduleReconnect(error: unknown): void {
    if (closed) return;
    if (attempt >= MAX_RECONNECT_ATTEMPTS) {
      setStatus('failed');
      handlers.onError?.(
        error instanceof Error ? error : new Error('The connection was lost.'),
      );
      closed = true;
      return;
    }
    const delay = backoffFor(attempt);
    attempt += 1;
    setStatus('reconnecting');
    window.setTimeout(() => void connect(), delay);
  }

  void connect();

  return { close };
}
