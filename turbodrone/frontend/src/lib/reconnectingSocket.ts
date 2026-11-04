/**
 * A WebSocket that reconnects forever (new in this fork; upstream opened /ws once and never
 * reconnected). Works in the page and in Web Workers.
 *
 * - Backoff: 250 ms, doubling, capped at 2 s; reset after a successful open.
 * - A connect attempt that hangs for 3 s is abandoned and retried.
 * - Optional idle timeout: for streams that should talk constantly (telemetry at 10 Hz), a
 *   silent socket is closed and reopened.
 * - send() returns false when the socket is not open, so callers can fall back (LAND and
 *   E-STOP go over HTTP) or tell the operator.
 */

export type LinkStatus = "connecting" | "open" | "closed";

export interface ReconnectingSocketOptions {
  url: string;
  /** Called with each parsed JSON text message. Binary and non-JSON messages are ignored. */
  onMessage?: (msg: unknown) => void;
  /** Called on every status change; attempts counts failed connects since the last open. */
  onStatus?: (status: LinkStatus, attempts: number) => void;
  /** Close and reopen if no message arrives for this long while open (0 = never). */
  idleTimeoutMs?: number;
  minDelayMs?: number;
  maxDelayMs?: number;
  connectTimeoutMs?: number;
}

type Timer = ReturnType<typeof setTimeout>;

export class ReconnectingSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: Timer | null = null;
  private connectTimer: Timer | null = null;
  private idleTimer: Timer | null = null;
  private attempts = 0;
  private running = false;
  private readonly opts: Required<ReconnectingSocketOptions>;

  constructor(opts: ReconnectingSocketOptions) {
    this.opts = {
      onMessage: () => {},
      onStatus: () => {},
      idleTimeoutMs: 0,
      minDelayMs: 250,
      maxDelayMs: 2000,
      connectTimeoutMs: 3000,
      ...opts,
    };
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.connect();
  }

  /** Stop for good (component unmount). No reconnect after this. */
  stop(): void {
    this.running = false;
    this.clearTimers();
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      ws.onopen = ws.onclose = ws.onerror = ws.onmessage = null;
      try {
        ws.close();
      } catch {
        /* already closed */
      }
    }
    this.opts.onStatus("closed", this.attempts);
  }

  isOpen(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /** Send one JSON message. Returns false (and sends nothing) when the socket is not open. */
  send(msg: object): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    try {
      this.ws.send(JSON.stringify(msg));
      return true;
    } catch {
      return false;
    }
  }

  /** Force a reconnect now (used when the caller decides the link is dead). */
  reconnectNow(): void {
    if (!this.running) return;
    this.ws?.close();
  }

  private connect(): void {
    if (!this.running) return;
    this.clearTimers();
    this.opts.onStatus("connecting", this.attempts);

    let ws: WebSocket;
    try {
      ws = new WebSocket(this.opts.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    // A connect that hangs (no refusal, no answer) is abandoned after connectTimeoutMs.
    this.connectTimer = setTimeout(() => {
      if (ws.readyState === WebSocket.CONNECTING) ws.close();
    }, this.opts.connectTimeoutMs);

    ws.onopen = () => {
      if (this.ws !== ws) return;
      if (this.connectTimer) clearTimeout(this.connectTimer);
      this.attempts = 0;
      this.armIdleTimer();
      this.opts.onStatus("open", 0);
    };

    ws.onmessage = (ev: MessageEvent) => {
      if (this.ws !== ws) return;
      this.armIdleTimer();
      if (typeof ev.data !== "string") return;
      let msg: unknown;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      this.opts.onMessage(msg);
    };

    // onerror is always followed by onclose; reconnect from onclose only.
    ws.onerror = () => {};

    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.opts.onStatus("closed", this.attempts);
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (!this.running) return;
    this.clearTimers();
    const delay = Math.min(this.opts.maxDelayMs, this.opts.minDelayMs * 2 ** this.attempts);
    this.attempts += 1;
    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }

  private armIdleTimer(): void {
    if (!this.opts.idleTimeoutMs) return;
    if (this.idleTimer) clearTimeout(this.idleTimer);
    const ws = this.ws;
    this.idleTimer = setTimeout(() => {
      if (ws && this.ws === ws) ws.close(); // onclose schedules the reconnect
    }, this.opts.idleTimeoutMs);
  }

  private clearTimers(): void {
    for (const t of [this.reconnectTimer, this.connectTimer, this.idleTimer]) {
      if (t) clearTimeout(t);
    }
    this.reconnectTimer = this.connectTimer = this.idleTimer = null;
  }
}
