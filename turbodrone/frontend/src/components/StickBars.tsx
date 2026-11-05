import type { Axes, ControlMode } from "../hooks/useControls";
import type { Telemetry } from "../lib/contracts";
import Panel from "./Panel";
import UiButton from "./UiButton";

const AXES: { key: keyof Axes; label: string; hint: string }[] = [
  { key: "throttle", label: "Throttle", hint: "+ climbs" },
  { key: "yaw", label: "Yaw", hint: "+ turns clockwise" },
  { key: "pitch", label: "Pitch", hint: "+ forward" },
  { key: "roll", label: "Roll", hint: "+ right" },
];

/** One wire-count bar: 0..255, center 128, filled from the center toward the value. */
export function WireBar({ value }: { value: number | undefined }) {
  const v = typeof value === "number" ? Math.max(0, Math.min(255, value)) : 128;
  const centerPct = (128 / 255) * 100;
  const valuePct = (v / 255) * 100;
  const left = Math.min(centerPct, valuePct);
  const width = Math.abs(valuePct - centerPct);
  return (
    <div className="relative h-1 flex-1 rounded-full bg-white/[0.08]">
      <div className="absolute top-0 h-full rounded-full bg-neutral-200" style={{ left: `${left}%`, width: `${width}%` }} />
      <div className="absolute -top-0.5 h-2 w-px bg-white/40" style={{ left: `${centerPct}%` }} />
    </div>
  );
}

/**
 * Sticks (new): the final wire counts the governor sent this tick, which source is in
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
  const active = telemetry?.active_source ?? "–";
  const seg = "rounded-md px-2 py-0.5 text-xs";
  return (
    <Panel
      title="Sticks"
      testId="stick-bars"
      caption={
        <span title={`sources: ${Object.keys(telemetry?.sticks_by_source ?? {}).join(", ") || "–"}`}>
          flown by <span data-testid="active-source" className="text-neutral-300">{stale ? "?" : active}</span>
        </span>
      }
    >
      <div className="flex flex-col gap-2">
        {AXES.map((a) => (
          <div key={a.key} className="flex items-center gap-3 text-xs" title={a.hint}>
            <span className="w-24 shrink-0 text-neutral-500">{a.label}</span>
            <WireBar value={stale ? undefined : wire[a.key]} />
            <span data-testid={`wire-${a.key}`} className="num w-14 shrink-0 text-right text-neutral-300">
              {stale || wire[a.key] === undefined ? "–" : wire[a.key]}
            </span>
          </div>
        ))}
        <div className="flex items-center justify-between gap-2 pt-1 text-xs">
          <span data-testid="local-input" data-sending={sendingAxes ? "1" : "0"} className={sendingAxes ? "num text-neutral-200" : "text-neutral-600"}>
            {sendingAxes
              ? `You: T${localAxes.throttle} Y${localAxes.yaw} P${localAxes.pitch} R${localAxes.roll}`
              : "W A S D and arrows fly by hand"}
          </span>
          <span className="flex rounded-lg bg-white/[0.06] p-0.5">
            <UiButton
              onClick={() => setMode("inc")}
              className={`${seg} ${mode === "inc" ? "bg-white/10 text-neutral-100" : "text-neutral-500"}`}
              title="W/A/S/D pitch and roll, arrows throttle and yaw"
            >
              Keys
            </UiButton>
            <UiButton
              onClick={() => gamepadConnected && setMode("abs")}
              disabled={!gamepadConnected}
              className={`${seg} ${
                mode === "abs" ? "bg-white/10 text-neutral-100" : gamepadConnected ? "text-neutral-400" : "text-neutral-700"
              }`}
              title={gamepadConnected ? "Use the gamepad" : "No gamepad detected"}
            >
              Gamepad
            </UiButton>
          </span>
        </div>
      </div>
    </Panel>
  );
}
