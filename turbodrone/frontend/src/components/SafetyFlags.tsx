import type { SystemEvent } from "../hooks/useAgentLink";
import type { Telemetry } from "../lib/contracts";
import Panel from "./Panel";

const RED_WORDS = ["estop", "land", "stale", "missing", "lost", "error", "exception", "wrong", "link", "fail"];

/** Safety flags raised by the governor (new). Red for anything that neutralizes or lands. */
export default function SafetyFlags({
  telemetry,
  stale,
  events,
}: {
  telemetry: Telemetry | null;
  stale: boolean;
  events: SystemEvent[];
}) {
  const flags = telemetry?.safety_flags ?? [];
  return (
    <Panel title="Safety" testId="safety-flags" caption="governor flags and recent events">
      {stale ? (
        <div className="text-sm text-amber-300">unknown: telemetry stale</div>
      ) : flags.length === 0 ? (
        <div data-testid="safety-clear" className="text-sm font-semibold text-emerald-400">
          ✓ all clear
        </div>
      ) : (
        <div className="flex flex-wrap gap-1">
          {flags.map((f) => {
            const red = RED_WORDS.some((w) => f.toLowerCase().includes(w));
            return (
              <span
                key={f}
                data-testid="safety-flag"
                className={`rounded-md px-2 py-0.5 font-mono text-xs font-semibold ${
                  red ? "bg-red-700 text-red-50" : "bg-amber-600 text-amber-50"
                }`}
              >
                {f}
              </span>
            );
          })}
        </div>
      )}
      <div data-testid="system-events" className="mt-2 space-y-0.5 border-t border-slate-800 pt-1.5">
        {events.slice(-4).map((e) => (
          <div key={e.id} className="truncate font-mono text-[11px] text-slate-400">
            <span className="mr-2 text-slate-600">{new Date(e.at).toLocaleTimeString([], { hour12: false })}</span>
            {e.text}
          </div>
        ))}
      </div>
    </Panel>
  );
}
