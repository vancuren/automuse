import { useEffect, useState } from "react";
import type { AgentHealth, VoiceLine } from "../hooks/useAgentLink";
import type { LinkStatus } from "../lib/reconnectingSocket";
import Panel from "./Panel";

const KIND_STYLE: Record<string, string> = {
  final: "text-slate-100",
  reply: "text-emerald-300",
  error: "text-red-300",
  safety_word: "font-bold text-orange-300",
  status: "text-slate-400",
};

const KIND_LABEL: Record<string, string> = {
  final: "you",
  reply: "drone",
  error: "error",
  safety_word: "safety",
  status: "agent",
};

/** The agent re-sends its health every 5 s; silence for this long means it is gone. */
const AGENT_SILENT_MS = 12000;

/** Caption: is the agent alive, and is the always-on local stop/land listener running? */
function HealthCaption({ agentStatus, health }: { agentStatus: LinkStatus; health: AgentHealth | null }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  if (agentStatus !== "open") return <span className="font-semibold text-red-300">voice link {agentStatus}</span>;
  if (!health) return <span>waiting for the agent…</span>;
  if (now - health.at > AGENT_SILENT_MS) return <span className="font-semibold text-amber-300">agent silent</span>;
  if (health.micSilent) return <span className="font-semibold text-red-300">microphone silent: check permission</span>;
  if (health.safetyListener === false) return <span className="font-semibold text-red-300">LOCAL STOP/LAND LISTENER OFF</span>;
  return (
    <span data-testid="agent-health">
      agent alive · local stop/land listener {health.safetyListener ? "on" : "?"}
      {health.asr ? ` · ASR ${health.asr}` : ""}
      {health.intent ? ` · intent ${health.intent}` : ""}
    </span>
  );
}

/** Push-to-talk state and the live transcript (new). */
export default function VoicePanel({
  ptt,
  partial,
  lines,
  agentStatus,
  health,
  pttError,
}: {
  ptt: boolean;
  partial: string;
  lines: VoiceLine[];
  agentStatus: LinkStatus;
  health: AgentHealth | null;
  pttError: string | null;
}) {
  const recent = lines.slice(-6);
  return (
    <Panel title="Voice" testId="voice-panel" caption={<HealthCaption agentStatus={agentStatus} health={health} />}>
      <div className="flex flex-col gap-1.5">
        <div
          data-testid="ptt-indicator"
          data-active={ptt ? "1" : "0"}
          className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-semibold ${
            ptt ? "bg-red-600/90 text-white" : "bg-slate-800 text-slate-400"
          }`}
        >
          <span className={`inline-block h-3 w-3 rounded-full ${ptt ? "animate-pulse bg-white" : "bg-slate-600"}`} />
          {ptt ? "LISTENING… release Space to send" : "Hold Space (or the talk button) to give a mission"}
        </div>
        {pttError && <div className="text-xs font-semibold text-red-300">{pttError}</div>}
        <div data-testid="transcript" className="min-h-[4.5rem] space-y-0.5 text-sm">
          {recent.map((l) => (
            <div key={l.id} className={`truncate ${KIND_STYLE[l.kind] ?? "text-slate-300"}`}>
              <span className="mr-1.5 inline-block w-12 text-right text-[10px] uppercase tracking-wider text-slate-500">
                {KIND_LABEL[l.kind] ?? l.kind}
              </span>
              {l.text}
              {l.meta && <span className="ml-2 font-mono text-[10px] text-slate-500">{l.meta}</span>}
            </div>
          ))}
          {partial && (
            <div data-testid="partial-transcript" className="truncate italic text-sky-200">
              <span className="mr-1.5 inline-block w-12 text-right text-[10px] uppercase tracking-wider text-slate-500">
                hearing
              </span>
              {partial}…
            </div>
          )}
        </div>
      </div>
    </Panel>
  );
}
