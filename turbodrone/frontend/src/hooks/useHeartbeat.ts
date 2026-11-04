import { useEffect, useState } from "react";
import { URLS } from "../config";

export interface HeartbeatStatus {
  open: boolean;
  sent: number;
  rateHz: number | null; // null until the worker has measured a full second
  error: string | null;  // the worker failed to load or crashed
}

/**
 * Starts the heartbeat Web Worker (5 Hz {"type":"hb","client":"ui"} on its own socket),
 * answers its liveness pings, and reports its status about once a second. New in this fork.
 */
export function useHeartbeat(): HeartbeatStatus {
  const [status, setStatus] = useState<HeartbeatStatus>({ open: false, sent: 0, rateHz: null, error: null });

  useEffect(() => {
    // If the worker cannot even be created, report it; never let it take the dashboard (and
    // its E-STOP button) down with it.
    let worker: Worker;
    try {
      worker = new Worker(new URL("../workers/heartbeat.worker.ts", import.meta.url), {
        type: "module",
      });
    } catch (e) {
      setStatus((s) => ({ ...s, open: false, rateHz: 0, error: e instanceof Error ? e.message : String(e) }));
      return;
    }
    worker.onmessage = (ev: MessageEvent) => {
      const d = ev.data as Partial<HeartbeatStatus> & { type?: string };
      if (d?.type === "ping") {
        worker.postMessage({ type: "pong" }); // "this page is alive": the worker heartbeats only while we answer
        return;
      }
      if (d?.type === "status") {
        setStatus({
          open: Boolean(d.open),
          sent: Number(d.sent ?? 0),
          rateHz: typeof d.rateHz === "number" ? d.rateHz : null,
          error: null,
        });
      }
    };
    // A worker that fails to load or throws sends no heartbeat: the governor will hold, then land.
    worker.onerror = (ev: ErrorEvent) => {
      setStatus((s) => ({ ...s, open: false, rateHz: 0, error: ev.message || "heartbeat worker failed" }));
    };
    worker.postMessage({ type: "start", url: URLS.control });
    return () => {
      // React StrictMode mounts twice in dev: terminate so only one heartbeat socket exists.
      worker.postMessage({ type: "stop" });
      worker.terminate();
    };
  }, []);

  return status;
}
