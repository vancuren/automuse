/** Colors and labels shared by the panels (new in this fork). */
import type { AutopilotState, FlightState } from "./contracts";

/**
 * Subject outline color by autopilot state. Colour is reserved for meaning: white while the
 * autopilot is still looking, green once it follows, orange when unsure, red when lost.
 */
export const AUTOPILOT_COLORS: Record<AutopilotState, string> = {
  IDLE: "#a3a3a3",
  FIND: "#fafafa",
  CONFIRM: "#fafafa",
  FOLLOW: "#22c55e",  // locked and following
  DOUBT: "#fb923c",   // tracker unsure, hovering
  LOST: "#ef4444",
  SEARCH: "#fafafa",
  GIVE_UP: "#ef4444",
};

/** One or two plain words for the autopilot state. */
export const AUTOPILOT_LABEL: Record<AutopilotState, string> = {
  IDLE: "Idle",
  FIND: "Looking",
  CONFIRM: "Confirming",
  FOLLOW: "Following",
  DOUBT: "Unsure",
  LOST: "Lost",
  SEARCH: "Searching",
  GIVE_UP: "Gave up",
};

export const AUTOPILOT_HELP: Record<AutopilotState, string> = {
  IDLE: "no mission",
  FIND: "looking for the subject",
  CONFIRM: "confirming the subject",
  FOLLOW: "following",
  DOUBT: "unsure, holding position",
  LOST: "subject lost, holding",
  SEARCH: "searching",
  GIVE_UP: "gave up, hovering",
};

export const FLIGHT_STATE_STYLE: Record<FlightState, { dot: string; label: string }> = {
  ON_GROUND: { dot: "#737373", label: "On the ground" },
  TAKING_OFF: { dot: "#fafafa", label: "Taking off" },
  AIRBORNE: { dot: "#22c55e", label: "Airborne" },
  LANDING: { dot: "#f59e0b", label: "Landing" },
  ESTOPPED: { dot: "#ef4444", label: "Stopped" },
};

export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "–";
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  return `${Math.round(ms)} ms`;
}

export function fmtClock(totalS: number): string {
  const s = Math.max(0, Math.floor(totalS));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
