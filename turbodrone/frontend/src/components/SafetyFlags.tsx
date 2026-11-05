import type { SystemEvent } from "../hooks/useAgentLink";
import type { Telemetry } from "../lib/contracts";
import Panel from "./Panel";

const RED_WORDS = ["estop", "land", "stale", "missing", "lost", "error", "exception", "wrong", "link", "fail", "fault"];

/** Safety flags raised by the governor, and what it did most recently (new). */
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
    <Panel
      title="Safety"
      testId="safety-flags"
      caption={
        stale ? (
          <span className="text-amber-300">unknown, telemetry is stale</span>
        ) : flags.length === 0 ? (
          <span data-testid="safety-clear">All clear</span>
        ) : undefined
      }
    >
      {!stale && flags.length > 0 && (
        <div className="mb-2.5 flex flex-wrap gap-1.5">
          {flags.map((f) => {
            const red = RED_WORDS.some((w) => f.toLowerCase().includes(w));
            return (
              <span
                key={f}
                data-testid="safety-flag"
                className={`rounded-md px-1.5 py-0.5 text-xs ${
                  red ? "bg-red-500/15 text-red-300" : "bg-amber-500/15 text-amber-300"
                }`}
              >
                {f.replace(/_/g, " ")}
              </span>
            );
          })}
        </div>
      )}
      <div data-testid="system-events" className="space-y-1">
        {events.slice(-4).map((e) => (
          <div key={e.id} className="flex gap-2.5 text-xs text-neutral-500">
            <span className="num shrink-0 text-neutral-600">
              {new Date(e.at).toLocaleTimeString([], { hour12: false })}
            </span>
            <span className="min-w-0 flex-1 truncate" title={e.text}>
              {e.text}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}
