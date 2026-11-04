import type { VoiceLine } from "../hooks/useAgentLink";
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

/** Push-to-talk state and the live transcript (new). */
export default function VoicePanel({
  ptt,
  partial,
  lines,
  agentStatus,
  pttError,
}: {
  ptt: boolean;
  partial: string;
  lines: VoiceLine[];
  agentStatus: LinkStatus;
  pttError: string | null;
}) {
  const recent = lines.slice(-6);
  return (
    <Panel
      title="Voice"
      testId="voice-panel"
      caption={agentStatus === "open" ? "Muse Voice Transcribe · local stop/land listener" : `voice link ${agentStatus}`}
    >
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
