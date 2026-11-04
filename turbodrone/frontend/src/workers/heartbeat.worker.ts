/// <reference lib="webworker" />
/**
 * Heartbeat worker (new in this fork).
 *
 * Owns its own WebSocket to the control endpoint and sends {"type":"hb","client":"ui"} every
 * 200 ms (5 Hz). It runs off the main thread, so React renders, a busy page, or background-tab
 * timer throttling cannot stop it. The governor goes neutral after 1.5 s without a heartbeat
 * and lands after 5 s, so the heartbeat must never depend on the page's own timers.
 *
 * But a heartbeat must also mean "the operator's page works": a hung page cannot E-STOP. So the
 * worker pings the page every 500 ms with postMessage (message events are not timer-throttled
 * in background tabs) and stops heartbeating when the page has not answered for 3 s.
 *
 * Messages from the page:  {type:"start", url}  |  {type:"stop"}  |  {type:"pong"}
 * Messages to the page:    {type:"status", open, sent, rateHz, pageAlive}  about once a second
 *                          {type:"ping"}  every 500 ms
 */
import { ReconnectingSocket } from "../lib/reconnectingSocket";

const HB_PERIOD_MS = 200;
const PING_PERIOD_MS = 500;
const PAGE_LIVENESS_MS = 3000;
const HEARTBEAT = { type: "hb", client: "ui" } as const;

let lastPongAt = performance.now();
let pingTimer: ReturnType<typeof setInterval> | null = null;

let sock: ReconnectingSocket | null = null;
let hbTimer: ReturnType<typeof setInterval> | null = null;
let statusTimer: ReturnType<typeof setInterval> | null = null;
let sent = 0;
let sentAtLastMeasure = 0;
let lastMeasureAt = 0;
let rateHz: number | null = null; // null until the first full one-second window

function pageAlive(): boolean {
  return performance.now() - lastPongAt <= PAGE_LIVENESS_MS;
}

function postStatus(): void {
  self.postMessage({ type: "status", open: sock?.isOpen() ?? false, sent, rateHz, pageAlive: pageAlive() });
}

/** Once a second: measure the send rate over the last window, then report. */
function measure(): void {
  const now = performance.now();
  const dt = (now - lastMeasureAt) / 1000;
  rateHz = dt > 0 ? (sent - sentAtLastMeasure) / dt : 0;
  sentAtLastMeasure = sent;
  lastMeasureAt = now;
  postStatus();
}

function start(url: string): void {
  if (sock) return;
  sock = new ReconnectingSocket({
    url,
    onStatus: () => postStatus(),
  });
  sock.start();
  lastPongAt = performance.now(); // start-up grace
  pingTimer = setInterval(() => self.postMessage({ type: "ping" }), PING_PERIOD_MS);
  hbTimer = setInterval(() => {
    if (!pageAlive()) return; // the page is hung: let the governor time out (neutral, then land)
    if (sock?.send(HEARTBEAT)) sent += 1;
  }, HB_PERIOD_MS);
  lastMeasureAt = performance.now();
  statusTimer = setInterval(measure, 1000);
}

function stop(): void {
  if (hbTimer) clearInterval(hbTimer);
  if (statusTimer) clearInterval(statusTimer);
  if (pingTimer) clearInterval(pingTimer);
  hbTimer = statusTimer = pingTimer = null;
  sock?.stop();
  sock = null;
}

self.onmessage = (ev: MessageEvent) => {
  const msg = ev.data as { type?: string; url?: string };
  if (msg?.type === "pong") lastPongAt = performance.now();
  else if (msg?.type === "start" && typeof msg.url === "string") start(msg.url);
  else if (msg?.type === "stop") stop();
};
