import type { Axes, ControlMode } from "../hooks/useControls";
import type { Telemetry } from "../lib/contracts";
import Panel from "./Panel";
import UiButton from "./UiButton";

const AXES: { key: keyof Axes; label: string; hint: string }[] = [
  { key: "throttle", label: "THR", hint: "+ climbs" },
  { key: "yaw", label: "YAW", hint: "+ turns clockwise" },
  { key: "pitch", label: "PIT", hint: "+ forward" },
  { key: "roll", label: "ROL", hint: "+ right" },
];

/** One wire-count bar: 0..255, center 128, filled from the center toward the value. */
function WireBar({ value }: { value: number | undefined }) {
  const v = typeof value === "number" ? Math.max(0, Math.min(255, value)) : 128;
  const centerPct = (128 / 255) * 100;
  const valuePct = (v / 255) * 100;
  const left = Math.min(centerPct, valuePct);
  const width = Math.abs(valuePct - centerPct);
  return (
    <div className="relative h-3 flex-1 overflow-hidden rounded bg-slate-800">
      <div className="absolute top-0 h-full bg-sky-400" style={{ left: `${left}%`, width: `${Math.max(width, 0.4)}%` }} />
      <div className="absolute top-0 h-full w-px bg-white/70" style={{ left: `${centerPct}%` }} />
    </div>
  );
}

/**
 * Stick bars (new): the final wire counts the governor sent this tick, which source is in
 * control, and what this UI is sending (axes go out only while nonzero).
 */
export default function StickBars({
  telemetry,
  stale,
  localAxes,
  sendingAxes,
  mode,
  setMode,
  gamepadConnected,
}: {
  telemetry: Telemetry | null;
  stale: boolean;
  localAxes: Axes;
  sendingAxes: boolean;
  mode: ControlMode;
  setMode: (m: ControlMode) => void;
  gamepadConnected: boolean;
}) {
  const wire = telemetry?.sticks_wire ?? {};
  const sources = Object.keys(telemetry?.sticks_by_source ?? {});
  const active = telemetry?.active_source ?? "–";
  return (
    <Panel
      title="Sticks"
      testId="stick-bars"
      caption={
        <span>
          active source{" "}
          <span data-testid="active-source" className="rounded bg-sky-900 px-1.5 py-0.5 font-mono font-bold text-sky-100">
            {stale ? "?" : active}
          </span>
        </span>
      }
    >
      <div className="flex flex-col gap-1">
        {AXES.map((a) => (
          <div key={a.key} className="flex items-center gap-2" title={a.hint}>
            <span className="w-8 font-mono text-[11px] text-slate-400">{a.label}</span>
            <WireBar value={stale ? undefined : wire[a.key]} />
            <span data-testid={`wire-${a.key}`} className="w-9 text-right font-mono text-xs text-slate-100">
              {stale || wire[a.key] === undefined ? "–" : wire[a.key]}
            </span>
          </div>
        ))}
        <div className="flex flex-wrap items-center gap-1 pt-0.5">
          {sources.map((s) => (
            <span
              key={s}
              className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${
                s === active ? "bg-sky-700 text-white" : "bg-slate-800 text-slate-400"
              }`}
            >
              {s}
            </span>
          ))}
        </div>
        <div className="flex items-center justify-between gap-2 pt-0.5 text-[11px]">
          <span data-testid="local-input" data-sending={sendingAxes ? "1" : "0"} className={sendingAxes ? "text-sky-300" : "text-slate-500"}>
            {sendingAxes
              ? `you: T${localAxes.throttle} Y${localAxes.yaw} P${localAxes.pitch} R${localAxes.roll}`
              : "you: idle (no axes sent)"}
          </span>
          <span className="flex gap-1">
            <UiButton
              onClick={() => setMode("inc")}
              className={`rounded px-1.5 py-0.5 ${mode === "inc" ? "bg-sky-700 text-white" : "bg-slate-800 text-slate-400"}`}
              title="W/A/S/D pitch and roll, arrows throttle and yaw"
            >
              keys
            </UiButton>
            <UiButton
              onClick={() => gamepadConnected && setMode("abs")}
              disabled={!gamepadConnected}
              className={`rounded px-1.5 py-0.5 ${
                mode === "abs" ? "bg-emerald-700 text-white" : gamepadConnected ? "bg-slate-800 text-slate-300" : "bg-slate-800/50 text-slate-600"
              }`}
              title={gamepadConnected ? "Use the gamepad" : "No gamepad detected"}
            >
              pad
            </UiButton>
          </span>
        </div>
      </div>
    </Panel>
  );
}
