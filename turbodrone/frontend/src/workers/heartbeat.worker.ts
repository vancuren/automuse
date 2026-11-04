/// <reference lib="webworker" />
/**
 * Heartbeat worker (new in this fork).
 *
 * Owns its own WebSocket to the control endpoint and sends {"type":"hb","client":"ui"} every
 * 200 ms (5 Hz). It runs off the main thread, so React renders, a busy page, or background-tab
 * timer throttling cannot stop it. The governor goes neutral after 1.5 s without a heartbeat
 * and lands after 5 s, so the heartbeat must never depend on the page's own timers.
 *
 * Messages from the page:  {type:"start", url}  |  {type:"stop"}
 * Messages to the page:    {type:"status", open, sent, rateHz}  about once a second
 */
import { ReconnectingSocket } from "../lib/reconnectingSocket";

const HB_PERIOD_MS = 200;
const HEARTBEAT = { type: "hb", client: "ui" } as const;

let sock: ReconnectingSocket | null = null;
let hbTimer: ReturnType<typeof setInterval> | null = null;
let statusTimer: ReturnType<typeof setInterval> | null = null;
let sent = 0;
let sentAtLastStatus = 0;
let lastStatusAt = 0;

function postStatus(): void {
  const now = performance.now();
  const dt = (now - lastStatusAt) / 1000;
  const rateHz = dt > 0 ? (sent - sentAtLastStatus) / dt : 0;
  sentAtLastStatus = sent;
  lastStatusAt = now;
  self.postMessage({ type: "status", open: sock?.isOpen() ?? false, sent, rateHz });
}

function start(url: string): void {
  if (sock) return;
  sock = new ReconnectingSocket({
    url,
    onStatus: () => postStatus(),
  });
  sock.start();
  hbTimer = setInterval(() => {
    if (sock?.send(HEARTBEAT)) sent += 1;
  }, HB_PERIOD_MS);
  lastStatusAt = performance.now();
  statusTimer = setInterval(postStatus, 1000);
}

function stop(): void {
  if (hbTimer) clearInterval(hbTimer);
  if (statusTimer) clearInterval(statusTimer);
  hbTimer = statusTimer = null;
  sock?.stop();
  sock = null;
}

self.onmessage = (ev: MessageEvent) => {
  const msg = ev.data as { type?: string; url?: string };
  if (msg?.type === "start" && typeof msg.url === "string") start(msg.url);
  else if (msg?.type === "stop") stop();
};
