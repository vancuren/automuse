import type { Mission } from "../lib/contracts";
import { fmtMs, modelBadgeStyle } from "../lib/theme";
import Panel from "./Panel";

/** Intent JSON with keys and values colored, one field per line. */
function IntentJson({ intent }: { intent: Record<string, unknown> }) {
  const entries = Object.entries(intent);
  return (
    <pre data-testid="intent-json" className="overflow-x-auto rounded-lg bg-slate-950 px-3 py-2 font-mono text-[13px] leading-5">
      <span className="text-slate-500">{"{"}</span>
      {entries.map(([k, v], i) => (
        <div key={k} className="pl-4">
          <span className="text-sky-300">"{k}"</span>
          <span className="text-slate-500">: </span>
          <span className={k === "action" ? "font-bold text-emerald-300" : "text-amber-200"}>
            {JSON.stringify(v)}
          </span>
          {i < entries.length - 1 && <span className="text-slate-500">,</span>}
        </div>
      ))}
      <span className="text-slate-500">{"}"}</span>
    </pre>
  );
}

/**
 * Mission card (new): what the language model decided, which model answered, how long it took.
 * Language models only choose missions; the governor and the classical autopilot fly.
 */
export default function MissionCard({ mission, provisional }: { mission: Mission | null; provisional: boolean }) {
  if (!mission) {
    return (
      <Panel title="Mission" testId="mission-card">
        <div className="py-3 text-sm text-slate-500">
          No mission yet. Hold <kbd className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-300">Space</kbd> and say
          something like “follow the person in the blue shirt”.
        </div>
      </Panel>
    );
  }
  const badge = modelBadgeStyle(mission.model);
  return (
    <Panel
      title="Mission"
      testId="mission-card"
      caption={<span className="font-mono">{mission.mission_id}{provisional ? " · from agent" : ""}</span>}
    >
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span
            data-testid="model-badge"
            className="rounded-md px-2.5 py-1 font-mono text-sm font-bold"
            style={{ background: badge.bg, color: badge.fg }}
          >
            {mission.model}
          </span>
          <span data-testid="mission-latency" className="rounded-md bg-slate-800 px-2 py-1 font-mono text-sm text-slate-200">
            {fmtMs(mission.latency_ms)}
          </span>
          <span className="text-xs text-slate-500">model that answered · speech end to mission</span>
        </div>
        {mission.transcript && (
          <div className="text-[15px] italic text-slate-200">“{mission.transcript}”</div>
        )}
        <IntentJson intent={mission.intent as unknown as Record<string, unknown>} />
        {mission.reply && <div className="text-sm text-emerald-300">🔊 {mission.reply}</div>}
      </div>
    </Panel>
  );
}
