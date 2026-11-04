import { useCallback, useEffect, useRef, useState } from "react";
import { URLS } from "../config";
import {
  isAgentEvent,
  isCandidates,
  isMission,
  missionFromIntentEvent,
  type AgentEvent,
  type Candidates,
  type Mission,
  type Select,
} from "../lib/contracts";
import { ReconnectingSocket, type LinkStatus } from "../lib/reconnectingSocket";

/** A line in the voice log shown under the push-to-talk button. */
export interface VoiceLine {
  id: number;
  kind: AgentEvent["kind"];
  text: string;
  meta?: string; // e.g. which ASR model and how long it took
  at: number; // Date.now(), display only
}

/** A governor or autopilot transition relayed by the backend as a status AgentEvent. */
export interface SystemEvent {
  id: number;
  text: string;
  at: number;
}

/** The agent's own health, from its periodic "agent alive" status event. */
export interface AgentHealth {
  text: string;
  safetyListener: boolean | null;
  asr: string | null;
  intent: string | null;
  micSilent: boolean;
  at: number; // Date.now() of the last health message
}

/** Status events whose data.kind marks them as governor/autopilot events (backend hub). */
const SYSTEM_EVENT_KINDS = new Set(["state", "flight_state", "announce", "safety"]);
const MAX_EVENTS = 20;

const MAX_LINES = 30;
/** Candidate boxes are a snapshot of one frame: drop them before they drift far from the scene. */
const CANDIDATES_TTL_MS = 10000;

/**
 * The /ws/agent?role=ui link (new in this fork): push-to-talk events out, transcript, intent,
 * replies and look-alike candidates in, Select out.
 */
export function useAgentLink() {
  const sockRef = useRef<ReconnectingSocket | null>(null);
  const pttRef = useRef(false);
  const nextId = useRef(1);

  const [status, setStatus] = useState<LinkStatus>("connecting");
  const [ptt, setPtt] = useState(false);
  const [pttError, setPttError] = useState<string | null>(null);
  const [partial, setPartial] = useState("");
  const [lines, setLines] = useState<VoiceLine[]>([]);
  const [agentMission, setAgentMission] = useState<Mission | null>(null);
  const [agentMissionAt, setAgentMissionAt] = useState(0);
  const [candidates, setCandidates] = useState<Candidates | null>(null);
  const [candidatesAt, setCandidatesAt] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);

  const [events, setEvents] = useState<SystemEvent[]>([]);
  const [health, setHealth] = useState<AgentHealth | null>(null);

  const addLine = useCallback((kind: AgentEvent["kind"], text: string, meta?: string) => {
    const line: VoiceLine = { id: nextId.current++, kind, text, meta, at: Date.now() };
    setLines((prev) => [...prev.slice(-(MAX_LINES - 1)), line]);
  }, []);

  const addEvent = useCallback((text: string) => {
    const ev: SystemEvent = { id: nextId.current++, text, at: Date.now() };
    setEvents((prev) => [...prev.slice(-(MAX_EVENTS - 1)), ev]);
  }, []);

  useEffect(() => {
    const sock = new ReconnectingSocket({
      url: URLS.agent,
      onStatus: setStatus,
      onMessage: (msg) => {
        if (isCandidates(msg)) {
          setCandidates(msg);
          setCandidatesAt(Date.now());
          setSelected(null);
          return;
        }
        if (isMission(msg)) {
          setAgentMission(msg);
          setAgentMissionAt(Date.now());
          return;
        }
        if (!isAgentEvent(msg)) return;
        const text = msg.text ?? "";
        const data = (msg.data ?? {}) as Record<string, unknown>;
        switch (msg.kind) {
          case "partial":
            setPartial(text);
            break;
          case "final": {
            setPartial("");
            const asrMs = typeof data.asr_ms === "number" ? ` · ${(data.asr_ms / 1000).toFixed(2)} s` : "";
            const meta = typeof data.model === "string" ? `${data.model}${asrMs}` : undefined;
            if (text) addLine("final", text, meta);
            break;
          }
          case "status":
            if ("safety_listener" in data) {
              // The agent's periodic health report: shown in the panel caption, not logged.
              setHealth({
                text: text || "agent alive",
                safetyListener: typeof data.safety_listener === "boolean" ? data.safety_listener : null,
                asr: typeof data.asr === "string" ? data.asr : null,
                intent: typeof data.intent === "string" ? data.intent : null,
                micSilent: data.mic_silent === true,
                at: Date.now(),
              });
            } else if (typeof data.kind === "string" && SYSTEM_EVENT_KINDS.has(data.kind)) {
              if (text) addEvent(text); // governor/autopilot transition: the Safety panel feed
            } else if (text) {
              addLine("status", text);
            }
            break;
          case "safety_word": {
            const cmd = typeof data.command === "string" ? ` → ${data.command}` : "";
            const ms = typeof data.decode_ms === "number" ? `local listener · ${Math.round(data.decode_ms)} ms` : "local listener";
            addLine("safety_word", `heard “${text}”${cmd}`, ms);
            break;
          }
          case "intent": {
            const m = missionFromIntentEvent(msg);
            if (m) {
              setAgentMission(m);
              setAgentMissionAt(Date.now());
            }
            break;
          }
          case "reply":
          case "error":
            if (text) addLine(msg.kind, text);
            break;
          default:
            break; // ptt_down / ptt_up echoes need no display
        }
      },
    });
    sockRef.current = sock;
    sock.start();
    return () => {
      sock.stop();
      sockRef.current = null;
    };
  }, [addLine, addEvent]);

  // Candidates expire so a stale picker never lingers over the video.
  useEffect(() => {
    if (!candidates) return;
    const t = setTimeout(() => setCandidates(null), CANDIDATES_TTL_MS - (Date.now() - candidatesAt));
    return () => clearTimeout(t);
  }, [candidates, candidatesAt]);

  /** The tie was resolved elsewhere (autopilot is following, or a new mission started). */
  const clearCandidates = useCallback(() => setCandidates(null), []);

  const sendEvent = useCallback((ev: AgentEvent): boolean => {
    return sockRef.current?.send(ev) ?? false;
  }, []);

  /** Push-to-talk pressed (Space or the on-screen button). Repeats are ignored. */
  const pttDown = useCallback(() => {
    if (pttRef.current) return;
    pttRef.current = true;
    setPtt(true);
    setPartial("");
    const ok = sendEvent({ type: "agent_event", kind: "ptt_down" });
    setPttError(ok ? null : "Voice link down: not listening");
  }, [sendEvent]);

  /** Push-to-talk released (also called on blur so the agent never records forever). */
  const pttUp = useCallback(() => {
    if (!pttRef.current) return;
    pttRef.current = false;
    setPtt(false);
    const ok = sendEvent({ type: "agent_event", kind: "ptt_up" });
    if (!ok) setPttError("Voice link down: release not delivered");
  }, [sendEvent]);

  /** Operator picked a look-alike: Select goes to perception through the backend. */
  const selectCandidate = useCallback(
    (index: number): boolean => {
      if (!candidates) return false;
      const msg: Select = { type: "select", mission_id: candidates.mission_id, index };
      const ok = sockRef.current?.send(msg) ?? false;
      if (ok) {
        setSelected(index);
        addLine("status", `Picked candidate ${index}`);
        setTimeout(() => setCandidates((c) => (c === candidates ? null : c)), 1500);
      }
      return ok;
    },
    [candidates, addLine],
  );

  return {
    status,
    ptt,
    pttError,
    partial,
    lines,
    events,
    health,
    agentMission,
    agentMissionAt,
    candidates,
    selected,
    pttDown,
    pttUp,
    selectCandidate,
    clearCandidates,
  };
}
