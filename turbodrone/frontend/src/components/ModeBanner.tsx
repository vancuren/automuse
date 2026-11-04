import type { Telemetry } from "../lib/contracts";
import type { LinkStatus } from "../lib/reconnectingSocket";

/**
 * Full-width banners at the top of the screen (new in this fork):
 *  1. REAL / TWIN, from telemetry.flight_mode. Unknown until telemetry arrives.
 *  2. DISCONNECTED while the control socket is down (auto-reconnect is running).
 *  3. NO TELEMETRY while telemetry is stale.
 */
export function ModeBanner({ telemetry, stale }: { telemetry: Telemetry | null; stale: boolean }) {
  const mode = telemetry?.flight_mode ?? null;
  if (mode === "real") {
    return (
      <div
        data-testid="mode-banner"
        data-mode="real"
        className="banner-real flex h-10 shrink-0 items-center justify-center gap-4 text-lg font-black uppercase tracking-[0.3em] text-white"
      >
        <span>REAL DRONE</span>
        <span className="text-sm font-semibold tracking-[0.2em] text-red-100">
          Loiley X69 · props live · {telemetry?.armed ? "ARMED" : "takeoff locked until ARM"}
        </span>
        {stale && <span className="text-sm text-yellow-200">(last known)</span>}
      </div>
    );
  }
  if (mode === "twin") {
    return (
      <div
        data-testid="mode-banner"
        data-mode="twin"
        className="flex h-10 shrink-0 items-center justify-center gap-4 bg-sky-800 text-lg font-black uppercase tracking-[0.3em] text-sky-50"
      >
        <span>TWIN</span>
        <span className="text-sm font-semibold tracking-[0.2em] text-sky-200">
          digital twin on 127.0.0.1 · wire-level X69 emulator · no aircraft
        </span>
      </div>
    );
  }
  return (
    <div
      data-testid="mode-banner"
      data-mode="unknown"
      className="flex h-10 shrink-0 items-center justify-center bg-slate-700 text-lg font-black uppercase tracking-[0.3em] text-slate-100"
    >
      MODE UNKNOWN · waiting for telemetry
    </div>
  );
}

function HeartbeatBanner({ problem }: { problem: string }) {
  return (
    <div
      data-testid="heartbeat-banner"
      className="flex h-11 w-full shrink-0 animate-pulse items-center justify-center gap-4 bg-red-700 text-xl font-black uppercase tracking-[0.25em] text-white"
    >
      <span>NO HEARTBEAT</span>
      <span className="text-sm font-semibold normal-case tracking-normal text-red-100">
        {problem} · the governor holds after 1.5 s and lands after 5 s · reload this page
      </span>
    </div>
  );
}

export function LinkBanner({
  controlStatus,
  attempts,
  telemetryStale,
  backendUrl,
  heartbeatProblem,
}: {
  controlStatus: LinkStatus;
  attempts: number;
  telemetryStale: boolean;
  backendUrl: string;
  heartbeatProblem: string | null; // why the governor is (or will be) missing our heartbeat
}) {
  if (heartbeatProblem && heartbeatProblem.startsWith("heartbeat worker failed")) {
    return <HeartbeatBanner problem={heartbeatProblem} />;
  }
  if (controlStatus !== "open") {
    return (
      <div
        data-testid="disconnected-banner"
        className="flex h-11 w-full shrink-0 animate-pulse items-center justify-center gap-4 bg-red-700 text-xl font-black uppercase tracking-[0.25em] text-white"
      >
        <span>DISCONNECTED</span>
        <span className="text-sm font-semibold normal-case tracking-normal text-red-100">
          control link to {backendUrl} is down · reconnecting{attempts > 0 ? ` (attempt ${attempts})` : ""}
          {" "}· LAND and E-STOP use the HTTP fallback
        </span>
      </div>
    );
  }
  if (heartbeatProblem) return <HeartbeatBanner problem={heartbeatProblem} />;
  if (telemetryStale) {
    return (
      <div
        data-testid="telemetry-banner"
        className="flex h-9 w-full shrink-0 items-center justify-center gap-4 bg-amber-600 text-base font-black uppercase tracking-[0.25em] text-amber-50"
      >
        <span>NO TELEMETRY</span>
        <span className="text-sm font-semibold normal-case tracking-normal">
          flight state unknown · takeoff disabled · LAND and E-STOP still work
        </span>
      </div>
    );
  }
  return null;
}
