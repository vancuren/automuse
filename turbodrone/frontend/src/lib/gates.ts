import type { CommandCapabilities } from "../hooks/useControls";
import type { Telemetry } from "./contracts";
import type { LinkStatus } from "./reconnectingSocket";

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
  return null;
}
