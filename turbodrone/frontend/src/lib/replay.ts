/**
 * Flight replay (new): saved flights from the backend's recorder (src/mhd/backend/recorder.py).
 * Read-only: GET /flights, /flights/{id}/events, /flights/{id}/frame/{n}. Nothing here can
 * send a command.
 */
import { URLS } from "../config";
import type { Telemetry } from "./contracts";
import { AUTOPILOT_LABEL, FLIGHT_STATE_STYLE } from "./theme";

export interface FlightCommand {
  t: number;
  kind: "cmd" | "mission";
  command?: string;
  transcript?: string | null;
  subject?: string | null;
}

export interface FlightSummary {
  id: string;
  started_unix?: number;
  flight_mode?: string;
  duration_ms?: number;
  frames?: number;
  complete: boolean;
  commands: FlightCommand[];
}

export type MarkKind = "command" | "voice" | "mission" | "reply" | "state" | "autopilot" | "safety";

/** One line of the flight's story: something the operator, the voice agent or the governor did. */
export interface ReplayMark {
  t: number; // ms from the start of the recording
  kind: MarkKind;
  label: string;
  detail?: string;
  failed?: boolean;
}

export interface Replay {
  id: string;
  durationMs: number;
  telemetry: { t: number; d: Telemetry }[];
  frames: { t: number; i: number }[];
  marks: ReplayMark[];
}

export const COMMAND_LABEL: Record<string, string> = {
  takeoff: "Take off",
  land: "Land",
  estop: "Emergency stop",
  estop_reset: "Emergency stop reset",
  arm: "Armed",
  disarm: "Disarmed",
  pause: "Pause",
  resume: "Resume",
  confirm_touchdown: "Touchdown confirmed",
  pack_swap: "Battery pack changed",
};

// "http" is POST /command: the voice agent's stop/land words, or the UI's fallback when its socket is down.
const SOURCE_LABEL: Record<string, string> = { ui: "button", http: "http request", agent: "voice" };

type Obj = Record<string, unknown>;
const str = (v: unknown): string | undefined => (typeof v === "string" && v ? v : undefined);

function markFor(k: string, d: Obj): Omit<ReplayMark, "t"> | null {
  if (k === "cmd") {
    const name = str(d.command) ?? "";
    const label = COMMAND_LABEL[name];
    if (!label) return null; // camera tilt and the like are not part of the story
    const source = SOURCE_LABEL[str(d.source) ?? ""] ?? str(d.source);
    if (d.ok === false) return { kind: "command", label, detail: `refused${str(d.reason) ? `: ${str(d.reason)}` : ""}`, failed: true };
    return { kind: "command", label, detail: source };
  }
  if (k === "mission") {
    const intent = (d.intent ?? {}) as Obj;
    const what = [str(intent.action), str(intent.subject_phrase)].filter(Boolean).join(" · ");
    const ms = typeof d.latency_ms === "number" ? `${(d.latency_ms / 1000).toFixed(1)} s` : undefined;
    return {
      kind: "mission",
      label: `Mission: ${what || "unknown"}`,
      detail: [str(d.model), ms, d.ok === false ? `refused${str(d.reason) ? `: ${str(d.reason)}` : ""}` : undefined]
        .filter(Boolean)
        .join(" · "),
      failed: d.ok === false,
    };
  }
  if (k !== "agent" || d.type !== "agent_event") return null;
  const text = str(d.text);
  const data = (d.data ?? {}) as Obj;
  switch (d.kind) {
    case "final":
      return text ? { kind: "voice", label: `“${text}”`, detail: str(data.model) } : null;
    case "reply":
      return text ? { kind: "reply", label: text } : null;
    case "error":
      return text ? { kind: "safety", label: text, failed: true } : null;
    case "safety_word":
      return text ? { kind: "voice", label: `“${text}”`, detail: `safety word${str(data.command) ? ` → ${str(data.command)}` : ""}` } : null;
    case "status": {
      if (data.kind === "flight_state") {
        const to = str(data.to) ?? "";
        const label = (FLIGHT_STATE_STYLE as Record<string, { label: string }>)[to]?.label ?? to;
        return { kind: "state", label, detail: str(data.cause) };
      }
      if (data.kind === "state") {
        const to = str(data.to) ?? "";
        return { kind: "autopilot", label: (AUTOPILOT_LABEL as Record<string, string>)[to] ?? to, detail: str(data.reason) };
      }
      if (data.kind === "safety" || data.kind === "announce") return text ? { kind: "safety", label: text } : null;
      return null;
    }
    default:
      return null;
  }
}

export async function fetchFlights(): Promise<FlightSummary[]> {
  const res = await fetch(URLS.flights, { cache: "no-store" });
  if (!res.ok) throw new Error(`flights: HTTP ${res.status}`);
  const body = (await res.json()) as { flights?: FlightSummary[] };
  return body.flights ?? [];
}

export async function fetchReplay(id: string): Promise<Replay> {
  const res = await fetch(`${URLS.flights}/${encodeURIComponent(id)}/events`, { cache: "no-store" });
  if (!res.ok) throw new Error(`flight ${id}: HTTP ${res.status}`);
  const replay: Replay = { id, durationMs: 0, telemetry: [], frames: [], marks: [] };
  for (const line of (await res.text()).split("\n")) {
    if (!line) continue;
    let ev: { t: number; k: string; d?: Obj; i?: number };
    try {
      ev = JSON.parse(line);
    } catch {
      continue; // a line cut short by a crash
    }
    if (typeof ev.t !== "number") continue;
    replay.durationMs = Math.max(replay.durationMs, ev.t);
    if (ev.k === "tlm" && ev.d) replay.telemetry.push({ t: ev.t, d: ev.d as unknown as Telemetry });
    else if (ev.k === "frame" && typeof ev.i === "number") replay.frames.push({ t: ev.t, i: ev.i });
    else if (ev.d) {
      const m = markFor(ev.k, ev.d);
      if (m) replay.marks.push({ t: ev.t, ...m });
    }
  }
  replay.marks.sort((a, b) => a.t - b.t);
  return replay;
}

export function frameUrl(id: string, index: number): string {
  return `${URLS.flights}/${encodeURIComponent(id)}/frame/${index}`;
}

/** Index of the last item with t <= at, or -1 (items are sorted by t). */
export function indexAt(items: { t: number }[], at: number): number {
  let lo = 0;
  let hi = items.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].t <= at) {
      found = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return found;
}
