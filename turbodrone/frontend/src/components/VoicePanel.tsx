import { useEffect, useState } from "react";
import type { AgentHealth, VoiceLine } from "../hooks/useAgentLink";
import type { LinkStatus } from "../lib/reconnectingSocket";
import Panel from "./Panel";

const KIND_STYLE: Record<string, string> = {
  final: "text-neutral-100",
  reply: "text-neutral-400",
  error: "text-red-400",
  safety_word: "text-amber-300",
  status: "text-neutral-500",
};

const KIND_LABEL: Record<string, string> = {
  final: "You",
  reply: "Drone",
  error: "Error",
  safety_word: "Safety",
  status: "",
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
  if (agentStatus !== "open") return <span className="text-red-400">Voice link {agentStatus}</span>;
  if (!health) return <span>Starting</span>;
  if (now - health.at > AGENT_SILENT_MS) return <span className="text-amber-300">Voice agent is silent</span>;
  if (health.micSilent) return <span className="text-red-400">Microphone silent, check permission</span>;
  if (health.safetyListener === false) return <span className="text-red-400">Stop/land listener is off</span>;
  return (
    <span
      data-testid="agent-health"
      title={[health.asr && `speech: ${health.asr}`, health.intent && `intent: ${health.intent}`].filter(Boolean).join(" · ")}
    >
      Stop/land listener {health.safetyListener ? "on" : "?"}
    </span>
  );
}

/** Push-to-talk state and the conversation with the drone (new). */
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
      <div className="flex flex-col gap-2">
        <div
          data-testid="ptt-indicator"
          data-active={ptt ? "1" : "0"}
          className={`flex items-center gap-2 text-[13px] ${ptt ? "text-white" : "text-neutral-500"}`}
        >
          <span className={`h-2 w-2 rounded-full ${ptt ? "soft-pulse bg-red-500" : "bg-neutral-700"}`} />
          {ptt ? "Listening. Release Space to send." : "Hold Space and say what to follow."}
        </div>
        {pttError && <div className="text-xs text-red-400">{pttError}</div>}
        <div data-testid="transcript" className="min-h-[3.5rem] space-y-1.5 text-[13px]">
          {recent.map((l) => (
            <div key={l.id} className={`flex gap-2.5 ${KIND_STYLE[l.kind] ?? "text-neutral-300"}`} title={l.meta}>
              <span className="w-10 shrink-0 text-neutral-600">{KIND_LABEL[l.kind] ?? l.kind}</span>
              <span className="min-w-0 flex-1">{l.text}</span>
            </div>
          ))}
          {partial && (
            <div data-testid="partial-transcript" className="flex gap-2.5 text-neutral-300">
              <span className="w-10 shrink-0 text-neutral-600">You</span>
              <span className="min-w-0 flex-1">{partial}…</span>
            </div>
          )}
        </div>
      </div>
    </Panel>
  );
}
