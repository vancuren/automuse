/**
 * TypeScript mirror of src/mhd/contracts/messages.py (contract v0.1, frozen by the integrator).
 *
 * Field names and literal values must match the pydantic models exactly: the backend validates
 * with extra="forbid", so a stray field makes a message invalid. tests/ui/test_contract_mirror.py
 * compares this file with the pydantic models and fails on drift.
 *
 * Timestamps: the UI never sends ts_ms. Browser clocks are not the shared time.monotonic clock,
 * so the backend stamps arrival time itself.
 */

export const CONTRACT_VERSION = "0.1";
export const FRAME_W = 1280;
export const FRAME_H = 720;

/** Axis-aligned box in 1280x720 pixels. */
export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export type SubjectState = "searching" | "candidate" | "confirmed" | "lost";
export type ObsSource = "sam_api" | "sam_mlx" | "tracker" | "oracle";

export interface SubjectObs {
  type: "subject_obs";
  mission_id?: string | null;
  frame_id: number;
  capture_ms: number;
  state: SubjectState;
  track_id?: number | null;
  box?: Box | null;
  confidence: number;
  tracker_score: number;
  confirm_age_ms: number;
  mask_poly?: [number, number][] | null;
  source: ObsSource;
  detector_latency_ms?: number | null;
  render_id?: number | null;
}

export interface Candidate {
  index: number;
  box: Box;
  confidence: number;
  color_score: number;
}

export interface Candidates {
  type: "candidates";
  mission_id: string;
  frame_id: number;
  items: Candidate[];
}

export interface Select {
  type: "select";
  mission_id: string;
  index: number;
}

/** UI -> backend /ws. Sent only while at least one axis is nonzero. */
export interface OperatorInput {
  type: "axes";
  throttle: number;
  yaw: number;
  pitch: number;
  roll: number;
  mode: "abs" | "inc";
  ts_ms?: number | null;
}

/** UI -> backend /ws at 5 Hz, from the heartbeat Web Worker. */
export interface Heartbeat {
  type: "hb";
  client: string;
  ts_ms?: number | null;
}

export const COMMAND_NAMES = [
  "takeoff", "land", "estop", "estop_reset", "pause", "resume", "confirm_touchdown",
  "arm", "disarm", "tilt_up", "tilt_down", "tilt_stop", "pack_swap",
] as const;
export type CommandName = (typeof COMMAND_NAMES)[number];

/** UI -> backend, over /ws or (LAND and E-STOP only) HTTP POST /command. */
export interface OperatorCommand {
  type: "cmd";
  command: CommandName;
  arm_phrase?: string | null;
  pack_id?: string | null;
}

export interface Intent {
  action: "follow" | "hover" | "land" | "clarify" | "refuse";
  subject_phrase: string;
  category: "person" | "car" | "animal" | "object" | "none";
  color: string;
  distance: "close" | "medium" | "far";
  reason_code: "none" | "ambiguous" | "identity_request" | "unsafe" | "unsupported";
}

export interface Mission {
  type: "mission";
  mission_id: string;
  intent: Intent;
  transcript: string;
  model: string;          // "glimmer-30b+lora", "glimmer-30b", "muse-spark-1.3", "keyword"
  latency_ms: number;
  reply?: string | null;
}

export const AGENT_EVENT_KINDS = [
  "ptt_down", "ptt_up", "partial", "final", "intent", "reply", "error", "safety_word", "status",
] as const;
export type AgentEventKind = (typeof AGENT_EVENT_KINDS)[number];

export interface AgentEvent {
  type: "agent_event";
  kind: AgentEventKind;
  text?: string | null;
  data?: Record<string, unknown> | null;
  ts_ms?: number | null;
}

export const FLIGHT_STATES = ["ON_GROUND", "TAKING_OFF", "AIRBORNE", "LANDING", "ESTOPPED"] as const;
export type FlightState = (typeof FLIGHT_STATES)[number];

export const AUTOPILOT_STATES = [
  "IDLE", "FIND", "CONFIRM", "FOLLOW", "DOUBT", "LOST", "SEARCH", "GIVE_UP",
] as const;
export type AutopilotState = (typeof AUTOPILOT_STATES)[number];

/** backend -> UI at 10 Hz over /ws/telemetry. */
export interface Telemetry {
  type: "telemetry";
  ts_ms: number;
  flight_mode: "twin" | "real";
  flight_state: FlightState;
  autopilot_state: AutopilotState;
  armed: boolean;
  sticks_wire: Record<string, number>;
  sticks_by_source: Record<string, Record<string, number>>;
  active_source: string;
  frame_age_ms?: number | null;
  obs_age_ms?: number | null;
  latency_ms: Record<string, number>;
  flight_time_s: number;
  pack_id?: string | null;
  safety_flags: string[];
  mission?: Mission | null;
  subject?: SubjectObs | null;
}

/* ── Small runtime guards for messages arriving off the wire ─────────────────────────────── */

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function isTelemetry(v: unknown): v is Telemetry {
  return (
    isObject(v) &&
    v.type === "telemetry" &&
    (v.flight_mode === "twin" || v.flight_mode === "real") &&
    typeof v.flight_state === "string" &&
    typeof v.autopilot_state === "string"
  );
}

export function isAgentEvent(v: unknown): v is AgentEvent {
  return isObject(v) && v.type === "agent_event" && typeof v.kind === "string";
}

export function isCandidates(v: unknown): v is Candidates {
  return isObject(v) && v.type === "candidates" && Array.isArray(v.items);
}

export function isMission(v: unknown): v is Mission {
  return isObject(v) && v.type === "mission" && isObject(v.intent) && typeof v.model === "string";
}

/** Best-effort: an "intent" AgentEvent may carry a whole Mission, or just an Intent, in data. */
export function missionFromIntentEvent(ev: AgentEvent): Mission | null {
  const d = ev.data;
  if (!isObject(d)) return null;
  if (isObject(d.intent) && typeof d.model === "string") {
    return {
      type: "mission",
      mission_id: String(d.mission_id ?? "pending"),
      intent: d.intent as unknown as Intent,
      transcript: String(d.transcript ?? ""),
      model: d.model,
      latency_ms: Number(d.latency_ms ?? 0),
      reply: (d.reply as string | undefined) ?? null,
    };
  }
  if (typeof d.action === "string") {
    return {
      type: "mission",
      mission_id: "pending",
      intent: d as unknown as Intent,
      transcript: "",
      model: String(ev.text ?? "unknown"),
      latency_ms: 0,
      reply: null,
    };
  }
  return null;
}
