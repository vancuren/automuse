import { useEffect, useState } from "react";
import { URLS } from "../config";

export interface HeartbeatStatus {
  open: boolean;
  sent: number;
  rateHz: number | null; // null until the worker has measured a full second
}

/**
 * Starts the heartbeat Web Worker (5 Hz {"type":"hb","client":"ui"} on its own socket) and
 * reports its status about once a second. New in this fork.
 */
export function useHeartbeat(): HeartbeatStatus {
  const [status, setStatus] = useState<HeartbeatStatus>({ open: false, sent: 0, rateHz: null });

  useEffect(() => {
    const worker = new Worker(new URL("../workers/heartbeat.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (ev: MessageEvent) => {
      const d = ev.data as Partial<HeartbeatStatus> & { type?: string };
      if (d?.type === "status") {
        setStatus({
          open: Boolean(d.open),
          sent: Number(d.sent ?? 0),
          rateHz: typeof d.rateHz === "number" ? d.rateHz : null,
        });
      }
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
