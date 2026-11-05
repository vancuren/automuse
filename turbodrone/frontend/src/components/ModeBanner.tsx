import type { Telemetry } from "../lib/contracts";
import { isCameraOnly } from "../lib/gates";
import type { LinkStatus } from "../lib/reconnectingSocket";

const PILL = "flex h-7 items-center gap-2 whitespace-nowrap rounded-full px-3 text-xs font-medium";

/**
 * What this session is flying, as one pill in the header (new in this fork):
 *  - Real drone, from telemetry.flight_mode. Red, the only pill with a filled dot.
 *  - Camera only: twin mode with the camera_only flag (real camera, no RC packets). Amber.
 *  - Simulator: the digital twin.
 *  - Connecting, until telemetry arrives.
 */
export function ModeBanner({ telemetry, stale }: { telemetry: Telemetry | null; stale: boolean }) {
  const mode = telemetry?.flight_mode ?? null;
  if (mode === "real") {
    return (
      <span
        data-testid="mode-banner"
        data-mode="real"
        title="Loiley X69. Props are live."
        className={`${PILL} bg-red-500/10 text-red-300 ring-1 ring-inset ring-red-500/30`}
      >
        <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
        Real drone
        {stale && <span className="text-red-300/60">last known</span>}
      </span>
    );
  }
  if (mode === "twin" && isCameraOnly(telemetry)) {
    return (
      <span
        data-testid="mode-banner"
        data-mode="camera"
        title="Live drone camera. Flight is simulated by the governor; no RC packets are sent."
        className={`${PILL} bg-amber-500/10 text-amber-300 ring-1 ring-inset ring-amber-500/30`}
      >
        Camera only · no flight
      </span>
    );
  }
  if (mode === "twin") {
    return (
      <span
        data-testid="mode-banner"
        data-mode="twin"
        title="Digital twin on this Mac: a wire-level X69 emulator, no aircraft."
        className={`${PILL} bg-white/[0.06] text-neutral-300`}
      >
        Simulator
      </span>
    );
  }
  return (
    <span data-testid="mode-banner" data-mode="unknown" className={`${PILL} bg-white/[0.06] text-neutral-500`}>
      Connecting
    </span>
  );
}

const NOTICE = "flex h-9 w-full shrink-0 items-center justify-center gap-2 px-4 text-[13px]";

function HeartbeatBanner({ problem }: { problem: string }) {
  return (
    <div data-testid="heartbeat-banner" className={`${NOTICE} bg-red-500/15 text-red-200`}>
      <span className="font-semibold text-red-100">No heartbeat</span>
      <span className="truncate">
        {problem}. The governor holds after 1.5 s and lands after 5 s. Reload this page.
      </span>
    </div>
  );
}

/** One slim notice under the header while a link is down. Nothing when all is well. */
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
      <div data-testid="disconnected-banner" className={`${NOTICE} bg-red-500/15 text-red-200`}>
        <span className="font-semibold text-red-100">Disconnected</span>
        <span className="truncate">
          Reconnecting to {backendUrl}
          {attempts > 0 ? ` (attempt ${attempts})` : ""}. Land and Emergency stop still work.
        </span>
      </div>
    );
  }
  if (heartbeatProblem) return <HeartbeatBanner problem={heartbeatProblem} />;
  if (telemetryStale) {
    return (
      <div data-testid="telemetry-banner" className={`${NOTICE} bg-amber-500/15 text-amber-200`}>
        <span className="font-semibold text-amber-100">No telemetry</span>
        <span className="truncate">Flight state unknown, takeoff is off. Land and Emergency stop still work.</span>
      </div>
    );
  }
  return null;
}
