import { useEffect, useRef, useState } from "react";
import { URLS } from "../config";
import { isTelemetry, type Telemetry } from "../lib/contracts";
import { ReconnectingSocket, type LinkStatus } from "../lib/reconnectingSocket";

/** Telemetry older than this is "stale": flight state unknown, takeoff disabled. */
export const TELEMETRY_STALE_MS = 1000;

export interface TelemetryLink {
  telemetry: Telemetry | null; // last message received (may be stale)
  status: LinkStatus;
  stale: boolean;
  rateHz: number;
}

/** Subscribes to /ws/telemetry (Telemetry at 10 Hz). New in this fork. */
export function useTelemetry(): TelemetryLink {
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [status, setStatus] = useState<LinkStatus>("connecting");
  const [stale, setStale] = useState(true);
  const [rateHz, setRateHz] = useState(0);
  const lastAt = useRef(0);
  const count = useRef(0);

  useEffect(() => {
    const sock = new ReconnectingSocket({
      url: URLS.telemetry,
      idleTimeoutMs: 3000, // 10 Hz stream: 3 s of silence means the socket is dead
      onStatus: setStatus,
      onMessage: (msg) => {
        if (!isTelemetry(msg)) return;
        lastAt.current = performance.now();
        count.current += 1;
        setTelemetry(msg);
        setStale(false);
      },
    });
    sock.start();

    const staleTimer = setInterval(() => {
      setStale(performance.now() - lastAt.current > TELEMETRY_STALE_MS);
    }, 200);
    let lastCount = 0;
    const rateTimer = setInterval(() => {
      setRateHz(count.current - lastCount);
      lastCount = count.current;
    }, 1000);

    return () => {
      sock.stop();
      clearInterval(staleTimer);
      clearInterval(rateTimer);
    };
  }, []);

  return { telemetry, status, stale, rateHz };
}
