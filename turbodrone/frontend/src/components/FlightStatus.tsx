import { PACK_IDS, PACK_LAND_S, PACK_WARN_S } from "../config";
import type { Telemetry } from "../lib/contracts";
import type { LinkStatus } from "../lib/reconnectingSocket";
import { AUTOPILOT_COLORS, FLIGHT_STATE_STYLE, fmtClock } from "../lib/theme";
import UiButton from "./UiButton";

/** Flight-state badge and autopilot badge (new). Gray "UNKNOWN" when telemetry is stale. */
export function FlightStateBadges({ telemetry, stale }: { telemetry: Telemetry | null; stale: boolean }) {
  if (!telemetry || stale) {
    return (
      <div className="flex items-center gap-2">
        <span
          data-testid="flight-state"
          data-state="UNKNOWN"
          className="rounded-lg bg-slate-700 px-4 py-1.5 text-xl font-black tracking-[0.18em] text-slate-200"
        >
          UNKNOWN
        </span>
      </div>
    );
  }
  // A state this UI does not know (version skew) must never read as ON GROUND.
  const fs = FLIGHT_STATE_STYLE[telemetry.flight_state] ?? {
    bg: "#7c2d12",
    fg: "#fff7ed",
    label: `? ${String(telemetry.flight_state)}`,
  };
  const apColor = AUTOPILOT_COLORS[telemetry.autopilot_state] ?? AUTOPILOT_COLORS.IDLE;
  return (
    <div className="flex items-center gap-2">
      <span
        data-testid="flight-state"
        data-state={telemetry.flight_state}
        className={`whitespace-nowrap rounded-lg px-4 py-1.5 text-xl font-black tracking-[0.18em] ${
          telemetry.flight_state === "ESTOPPED" ? "animate-pulse" : ""
        }`}
        style={{ background: fs.bg, color: fs.fg }}
      >
        {fs.label}
      </span>
      <span
        data-testid="autopilot-state"
        data-state={telemetry.autopilot_state}
        className="whitespace-nowrap rounded-lg border-2 px-3 py-1 text-base font-bold tracking-[0.14em]"
        style={{ borderColor: apColor, color: apColor }}
      >
        AUTOPILOT {telemetry.autopilot_state}
      </span>
      {telemetry.flight_mode === "real" && (
        <span
          data-testid="armed-chip"
          className={`rounded-md px-2 py-1 text-xs font-bold tracking-wider ${
            telemetry.armed ? "bg-red-600 text-white" : "bg-slate-700 text-slate-300"
          }`}
        >
          {telemetry.armed ? "ARMED" : "DISARMED"}
        </span>
      )}
    </div>
  );
}

/**
 * Per-pack flight timer (new). Amber from VITE_PACK_WARN_S (5:00), red from 30 s before
 * the governor's auto-land at VITE_PACK_LAND_S (6:30). Pack swap only while on the ground.
 */
export function PackTimer({
  telemetry,
  stale,
  onPackSwap,
}: {
  telemetry: Telemetry | null;
  stale: boolean;
  onPackSwap: (packId: string) => void;
}) {
  const t = telemetry?.flight_time_s ?? 0;
  const pack = telemetry?.pack_id ?? "?";
  const level = t >= PACK_LAND_S - 30 ? "red" : t >= PACK_WARN_S ? "amber" : "ok";
  const colors = {
    ok: "text-emerald-300 border-slate-700",
    amber: "text-amber-300 border-amber-500",
    red: "text-red-300 border-red-500 animate-pulse",
  }[level];
  const pct = Math.min(100, (t / PACK_LAND_S) * 100);
  const onGround = !stale && telemetry?.flight_state === "ON_GROUND";
  return (
    <div className="flex items-center gap-2">
      <div
        data-testid="pack-timer"
        data-level={level}
        className={`flex min-w-[190px] flex-col rounded-lg border bg-slate-900 px-3 py-1 ${colors}`}
        title={`warns at ${fmtClock(PACK_WARN_S)}, auto-land at ${fmtClock(PACK_LAND_S)}`}
      >
        <div className="flex items-baseline justify-between gap-3">
          <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-400">Pack {pack}</span>
          <span className="font-mono text-xl font-bold">{fmtClock(t)}</span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded bg-slate-800">
          <div
            className={`h-full ${level === "ok" ? "bg-emerald-500" : level === "amber" ? "bg-amber-500" : "bg-red-500"}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="text-[10px] text-slate-500">
          {level === "red" ? "AUTO-LAND SOON" : level === "amber" ? "LAND SOON" : `auto-land at ${fmtClock(PACK_LAND_S)}`}
        </div>
      </div>
      <div className="flex flex-col gap-1">
        {PACK_IDS.map((id) => (
          <UiButton
            key={id}
            data-testid={`pack-${id}`}
            disabled={!onGround || id === telemetry?.pack_id}
            onClick={() => onPackSwap(id)}
            className={`whitespace-nowrap rounded px-2 py-0.5 text-[11px] font-semibold ${
              id === telemetry?.pack_id
                ? "bg-slate-600 text-white"
                : onGround
                  ? "bg-slate-800 text-slate-300 hover:bg-slate-700"
                  : "cursor-not-allowed bg-slate-800/60 text-slate-600"
            }`}
            title={onGround ? `Swap to pack ${id}` : "Pack swap only on the ground"}
          >
            pack {id}
          </UiButton>
        ))}
      </div>
    </div>
  );
}

function Dot({ ok, warn }: { ok: boolean; warn?: boolean }) {
  return (
    <span
      className={`inline-block h-2.5 w-2.5 shrink-0 rounded-full ${ok ? "bg-emerald-400" : warn ? "bg-amber-400" : "bg-red-500"}`}
    />
  );
}

/** Link status chips: control socket, heartbeat worker, telemetry, voice. */
export function LinkChips({
  control,
  hbOpen,
  hbRate,
  telemetryStale,
  telemetryRate,
  agent,
}: {
  control: LinkStatus;
  hbOpen: boolean;
  hbRate: number | null;
  telemetryStale: boolean;
  telemetryRate: number | null;
  agent: LinkStatus;
}) {
  const chip =
    "flex items-center gap-1.5 whitespace-nowrap rounded-md bg-slate-900 px-2 py-1 font-mono text-[11px] text-slate-300";
  const hbOk = hbOpen && (hbRate === null || hbRate > 3.5);
  return (
    <div data-testid="link-chips" className="grid grid-cols-2 gap-1">
      <span className={chip} data-testid="link-control" data-status={control}>
        <Dot ok={control === "open"} /> CTRL {control}
      </span>
      <span className={chip} data-testid="link-hb">
        <Dot ok={hbOk} warn={hbOpen} /> HB {hbRate === null ? "–" : hbRate.toFixed(1)} Hz
      </span>
      <span className={chip} data-testid="link-telemetry">
        <Dot ok={!telemetryStale} /> TLM {telemetryRate === null ? "–" : telemetryRate} Hz
      </span>
      <span className={chip} data-testid="link-agent" data-status={agent}>
        <Dot ok={agent === "open"} warn={agent === "connecting"} /> VOICE {agent}
      </span>
    </div>
  );
}
