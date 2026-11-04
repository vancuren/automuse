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
  at: number; // Date.now(), display only
}

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

  const addLine = useCallback((kind: AgentEvent["kind"], text: string) => {
    const line: VoiceLine = { id: nextId.current++, kind, text, at: Date.now() };
    setLines((prev) => [...prev.slice(-(MAX_LINES - 1)), line]);
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
        switch (msg.kind) {
          case "partial":
            setPartial(text);
            break;
          case "final":
            setPartial("");
            if (text) addLine("final", text);
            break;
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
          case "safety_word":
          case "status":
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
  }, [addLine]);

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
