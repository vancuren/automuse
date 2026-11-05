import type { Mission } from "../lib/contracts";
import { fmtMs } from "../lib/theme";
import Panel from "./Panel";

function show(v: unknown): string {
  if (v === null || v === undefined || v === "") return "–";
  return typeof v === "string" ? v : JSON.stringify(v);
}

/** The intent the language model returned, one field per row. */
function IntentFields({ intent }: { intent: Record<string, unknown> }) {
  const entries = Object.entries(intent).filter(([, v]) => v !== null && v !== undefined && v !== "" && v !== "none");
  return (
    <dl data-testid="intent-json" className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[13px]">
      {entries.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-neutral-500">{k.replace(/_/g, " ")}</dt>
          <dd className={`min-w-0 truncate ${k === "action" ? "font-medium text-neutral-100" : "text-neutral-300"}`}>{show(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Mission (new): what the language model decided, which model answered, how long it took.
 * Language models only choose missions; the governor and the classical autopilot fly.
 */
export default function MissionCard({ mission, provisional }: { mission: Mission | null; provisional: boolean }) {
  if (!mission) {
    return (
      <Panel title="Mission" testId="mission-card">
        <div className="text-[13px] text-neutral-500">
          None yet. Try “follow the person in the blue shirt”.
        </div>
      </Panel>
    );
  }
  return (
    <Panel
      title="Mission"
      testId="mission-card"
      caption={<span title={mission.mission_id}>{provisional ? "from the voice agent" : ""}</span>}
    >
      <div className="flex flex-col gap-3">
        {mission.transcript && <div className="text-[15px] leading-snug text-neutral-100">“{mission.transcript}”</div>}
        <IntentFields intent={mission.intent as unknown as Record<string, unknown>} />
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-neutral-500">
          <span data-testid="model-badge" className="rounded-md bg-white/[0.06] px-1.5 py-0.5 font-mono text-[11px] text-neutral-300">
            {mission.model}
          </span>
          <span>answered in</span>
          <span data-testid="mission-latency" className="num text-neutral-300">
            {fmtMs(mission.latency_ms)}
          </span>
        </div>
        {mission.reply && <div className="text-[13px] text-neutral-400">{mission.reply}</div>}
      </div>
    </Panel>
  );
}
