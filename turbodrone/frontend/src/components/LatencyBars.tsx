import { useRef } from "react";
import type { Telemetry } from "../lib/contracts";
import { fmtMs } from "../lib/theme";
import Panel from "./Panel";

interface BarSpec {
  key: string;
  label: string;
  hint: string;
  warn: number; // governor/budget threshold: amber from here
  hard: number; // red from here; also the full-scale value
  get: (t: Telemetry) => number | null | undefined;
  /** Keep showing the last value for this long when the current telemetry has none (ms). */
  holdMs?: number;
}

/** Thresholds follow PLAN.md 4.5 (staleness) and 4.9 (G0 latency gates). */
const BARS: BarSpec[] = [
  { key: "frame", label: "Frame age", hint: "camera capture → backend; neutral 0.7 s, land 2 s", warn: 700, hard: 2000, get: (t) => t.frame_age_ms },
  { key: "obs", label: "Observation age", hint: "capture → subject observation; neutral above 0.4 s", warn: 400, hard: 1000, get: (t) => t.obs_age_ms },
  {
    key: "detector",
    label: "Detector",
    hint: "SAM 3.1 detection; gate p95 1.5 s",
    warn: 1500,
    hard: 3000,
    get: (t) => t.latency_ms?.detector ?? t.subject?.detector_latency_ms,
    // Perception reports detector_latency_ms only on the frames a detection finished (1-2 Hz), so most
    // telemetry has none: show the last detection's latency until it is 3 s old.
    holdMs: 3000,
  },
  {
    key: "intent",
    label: "Intent",
    hint: "speech end → mission; Glimmer deadline 2.5 s",
    warn: 2500,
    hard: 6000,
    get: (t) => t.latency_ms?.intent ?? t.mission?.latency_ms,
  },
];

const PRIMARY_KEYS = new Set(["detector", "intent"]);

/** Latency bars (new): the four numbers that decide whether the autopilot may fly. */
export default function LatencyBars({ telemetry, stale }: { telemetry: Telemetry | null; stale: boolean }) {
  const extras = Object.entries(telemetry?.latency_ms ?? {}).filter(([k]) => !PRIMARY_KEYS.has(k));
  const last = useRef<Record<string, { v: number; at: number }>>({});
  const now = performance.now();
  return (
    <Panel title="Latency" testId="latency-bars" caption={stale ? "stale" : "live"}>
      <div className="flex flex-col gap-1.5">
        {BARS.map((b) => {
          const raw = telemetry ? b.get(telemetry) : null;
          let v = typeof raw === "number" && Number.isFinite(raw) ? raw : null;
          if (b.holdMs) {
            const held = last.current[b.key];
            if (v !== null) last.current[b.key] = { v, at: now };
            else if (held && now - held.at <= b.holdMs) v = held.v;
          }
          const pct = v === null ? 0 : Math.min(100, (v / b.hard) * 100);
          const color = v === null ? "bg-slate-700" : v >= b.hard ? "bg-red-500" : v >= b.warn ? "bg-amber-400" : "bg-emerald-500";
          return (
            <div key={b.key} data-testid={`latency-${b.key}`} title={b.hint}>
              <div className="flex items-baseline justify-between text-xs">
                <span className="text-slate-300">{b.label}</span>
                <span className="font-mono text-slate-100">{fmtMs(v)}</span>
              </div>
              <div className="relative h-2 w-full overflow-hidden rounded bg-slate-800">
                <div className={`h-full ${color} ${stale ? "opacity-40" : ""}`} style={{ width: `${pct}%` }} />
                <div className="absolute top-0 h-full w-0.5 bg-white/60" style={{ left: `${(b.warn / b.hard) * 100}%` }} />
              </div>
            </div>
          );
        })}
        {extras.length > 0 && (
          <div className="truncate font-mono text-[10px] text-slate-500">
            {extras.map(([k, v]) => `${k} ${fmtMs(v)}`).join(" · ")}
          </div>
        )}
      </div>
    </Panel>
  );
}
