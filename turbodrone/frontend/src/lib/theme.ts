/** Colors and labels shared by the panels (new in this fork). */
import type { AutopilotState, FlightState } from "./contracts";

/** Mask/box color by autopilot state: green means following, warm colors mean trouble. */
export const AUTOPILOT_COLORS: Record<AutopilotState, string> = {
  IDLE: "#94a3b8",    // slate
  FIND: "#38bdf8",    // sky: looking for the subject
  CONFIRM: "#facc15", // yellow: candidate, confirming
  FOLLOW: "#22c55e",  // green: locked and following
  DOUBT: "#fb923c",   // orange: tracker unsure, hovering
  LOST: "#ef4444",    // red
  SEARCH: "#a78bfa",  // violet: searching after loss
  GIVE_UP: "#f43f5e", // rose
};

export const AUTOPILOT_HELP: Record<AutopilotState, string> = {
  IDLE: "no mission",
  FIND: "looking for the subject",
  CONFIRM: "confirming the subject",
  FOLLOW: "following",
  DOUBT: "unsure: holding position",
  LOST: "subject lost: holding",
  SEARCH: "searching (yaw scan)",
  GIVE_UP: "gave up: hovering",
};

export const FLIGHT_STATE_STYLE: Record<FlightState, { bg: string; fg: string; label: string }> = {
  ON_GROUND: { bg: "#334155", fg: "#e2e8f0", label: "ON GROUND" },
  TAKING_OFF: { bg: "#0369a1", fg: "#e0f2fe", label: "TAKING OFF" },
  AIRBORNE: { bg: "#15803d", fg: "#dcfce7", label: "AIRBORNE" },
  LANDING: { bg: "#c2410c", fg: "#ffedd5", label: "LANDING" },
  ESTOPPED: { bg: "#b91c1c", fg: "#fef2f2", label: "E-STOPPED" },
};

/** Model badge colors: the tuned Glimmer is the star of the show. */
export function modelBadgeStyle(model: string): { bg: string; fg: string } {
  const m = model.toLowerCase();
  if (m.includes("glimmer") && m.includes("lora")) return { bg: "#1d4ed8", fg: "#dbeafe" };
  if (m.includes("glimmer")) return { bg: "#1e40af", fg: "#dbeafe" };
  if (m.includes("spark")) return { bg: "#7e22ce", fg: "#f3e8ff" };
  if (m.includes("keyword")) return { bg: "#475569", fg: "#f1f5f9" };
  return { bg: "#334155", fg: "#e2e8f0" };
}

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
