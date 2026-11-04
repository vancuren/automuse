import type { CommandCapabilities } from "../hooks/useControls";
import type { Telemetry } from "./contracts";
import type { LinkStatus } from "./reconnectingSocket";

/**
 * Governor safety flags under which the governor refuses takeoff (names from the gov
 * workstream's governor). Disabling on them only ever makes the button stricter than the
 * governor, never looser; unknown flag names simply do not block.
 */
const TAKEOFF_BLOCKING_FLAGS: Record<string, string> = {
  video_stale: "no video",
  heartbeat_missing: "no UI heartbeat",
  governor_fault: "governor fault",
  rc_tick_stale: "RC loop stalled",
};

/**
 * Camera-only runs (scripts/run_stack.sh --mode camera): twin mode with the drone's real camera and
 * no RC packets at all. The backend says so with this safety flag (src/mhd/backend/modes.py
 * CAMERA_ONLY_FLAG); it is not in TAKEOFF_BLOCKING_FLAGS, so it never blocks the (simulated) takeoff.
 */
export const CAMERA_ONLY_FLAG = "camera_only";

export function isCameraOnly(telemetry: Telemetry | null): boolean {
  return telemetry?.flight_mode === "twin" && (telemetry.safety_flags ?? []).includes(CAMERA_ONLY_FLAG);
}

/**
 * Why TAKEOFF is disabled, or null when it is allowed (new in this fork).
 * Takeoff needs a live control link, fresh telemetry, the drone on the ground, and in real
 * mode the operator must have typed ARM this session (telemetry.armed). The backend enforces
 * the same rules; this only keeps the button honest.
 */
export function takeoffBlocker(
  telemetry: Telemetry | null,
  stale: boolean,
  control: LinkStatus,
  caps: CommandCapabilities,
): string | null {
  if (!caps.takeoff) return "takeoff not supported";
  if (control !== "open") return "control link down";
  if (!telemetry || stale) return "no telemetry";
  if (telemetry.flight_state !== "ON_GROUND") return "not on the ground";
  if (telemetry.flight_mode === "real" && !telemetry.armed) return "type ARM first";
  for (const flag of telemetry.safety_flags ?? []) {
    if (Object.prototype.hasOwnProperty.call(TAKEOFF_BLOCKING_FLAGS, flag)) return TAKEOFF_BLOCKING_FLAGS[flag];
  }
  return null;
}
