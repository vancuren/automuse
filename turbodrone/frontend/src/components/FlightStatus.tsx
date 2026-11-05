import { PACK_IDS, PACK_LAND_S, PACK_WARN_S } from "../config";
import type { Telemetry } from "../lib/contracts";
import type { LinkStatus } from "../lib/reconnectingSocket";
import { AUTOPILOT_COLORS, AUTOPILOT_LABEL, FLIGHT_STATE_STYLE, fmtClock } from "../lib/theme";
import UiButton from "./UiButton";

/** Flight state and autopilot state, in words. "Waiting" while telemetry is stale. */
export function FlightStateBadges({ telemetry, stale }: { telemetry: Telemetry | null; stale: boolean }) {
  if (!telemetry || stale) {
    return (
      <div className="flex items-center gap-2.5 text-[15px] font-medium text-neutral-500">
        <span className="h-2 w-2 rounded-full bg-neutral-600" />
        <span data-testid="flight-state" data-state="UNKNOWN">
          Waiting for the drone
        </span>
      </div>
    );
  }
  // A state this UI does not know (version skew) must never read as "On the ground".
  const fs = FLIGHT_STATE_STYLE[telemetry.flight_state] ?? {
    dot: "#f59e0b",
    label: `Unknown state ${String(telemetry.flight_state)}`,
  };
  const ap = telemetry.autopilot_state;
  const apColor = AUTOPILOT_COLORS[ap] ?? AUTOPILOT_COLORS.IDLE;
  const flying = telemetry.flight_state === "AIRBORNE";
  return (
    <div className="flex items-center gap-2.5 whitespace-nowrap text-[15px] font-medium">
      <span
        className={`h-2 w-2 rounded-full ${telemetry.flight_state === "ESTOPPED" ? "soft-pulse" : ""}`}
        style={{ background: fs.dot }}
      />
      <span data-testid="flight-state" data-state={telemetry.flight_state} className="text-neutral-100">
        {fs.label}
      </span>
      <span
        data-testid="autopilot-state"
        data-state={ap}
        className={flying && ap !== "IDLE" ? "" : "hidden"}
        style={{ color: apColor }}
      >
        <span className="mr-2.5 text-neutral-600">/</span>
        {AUTOPILOT_LABEL[ap] ?? ap}
      </span>
      {telemetry.flight_mode === "real" && (
        <span
          data-testid="armed-chip"
          className={`ml-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${
            telemetry.armed ? "bg-red-500/15 text-red-300" : "bg-white/[0.06] text-neutral-400"
          }`}
        >
          {telemetry.armed ? "Armed" : "Locked"}
        </span>
      )}
    </div>
  );
}

/** Lewei board: a battery reading older than this (no reply from the drone) is not shown as live. */
const FC_STALE_MS = 3000;

function fmtAge(ms: number): string {
  const s = Math.floor(ms / 1000);
  return s < 120 ? `${s} s` : `${Math.floor(s / 60)} min`;
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
  const level = t >= PACK_LAND_S - 30 ? "red" : t >= PACK_WARN_S ? "amber" : "ok";
  const tone = { ok: "text-neutral-200", amber: "text-amber-300", red: "text-red-400 soft-pulse" }[level];
  const bar = { ok: "bg-neutral-400", amber: "bg-amber-400", red: "bg-red-500" }[level];
  const pct = Math.min(100, (t / PACK_LAND_S) * 100);
  const onGround = !stale && telemetry?.flight_state === "ON_GROUND";
  // Lewei board only: the flight controller's own battery reading (informational, 0.1 V steps). Once the
  // drone's last reply is over 3 s old (it powered itself off, Wi-Fi gone) the reading is not live: a
  // grey "—" and its age, never the last volts as if current. No age at all (older backends): as before.
  const volts = telemetry?.latency_ms?.fc_battery_v;
  const fcAge = telemetry?.latency_ms?.fc_telemetry_age;
  const fcOldFor = typeof fcAge === "number" && Number.isFinite(fcAge) && fcAge > FC_STALE_MS ? fmtAge(fcAge) : null;
  return (
    <div className="flex items-center gap-3">
      <div
        data-testid="pack-timer"
        data-level={level}
        className="flex items-center gap-2.5"
        title={`Flight time on this battery pack. Warns at ${fmtClock(PACK_WARN_S)}, lands by itself at ${fmtClock(PACK_LAND_S)}.`}
      >
        {typeof volts === "number" &&
          Number.isFinite(volts) &&
          (fcOldFor !== null ? (
            <span
              data-testid="fc-battery"
              data-old="1"
              className="num whitespace-nowrap text-[13px] text-slate-500"
              title={`no reply from the drone for ${fcOldFor}: last reading ${volts.toFixed(1)} V (Lewei board)`}
            >
              — V{" "}
              <span data-testid="fc-battery-age" className="text-[11px]">
                {fcOldFor} ago
              </span>
            </span>
          ) : (
            <span
              data-testid="fc-battery"
              data-old="0"
              className={`num text-[13px] ${stale ? "text-slate-500" : volts < 3.5 ? "text-amber-300" : "text-neutral-400"}`}
              title="battery voltage reported by the flight controller (Lewei board)"
            >
              {volts.toFixed(1)} V
            </span>
          ))}
        <div className="h-1 w-14 overflow-hidden rounded-full bg-white/10">
          <div className={`h-full rounded-full ${bar}`} style={{ width: `${pct}%` }} />
        </div>
        <span className={`num text-[15px] font-medium ${tone}`}>{fmtClock(t)}</span>
        {level !== "ok" && (
          <span className={`text-xs ${level === "red" ? "text-red-400" : "text-amber-300"}`}>
            {level === "red" ? "landing soon" : "land soon"}
          </span>
        )}
      </div>
      <div className="flex rounded-lg bg-white/[0.06] p-0.5" title="Battery pack in the drone">
        {PACK_IDS.map((id) => {
          const current = id === telemetry?.pack_id;
          return (
            <UiButton
              key={id}
              data-testid={`pack-${id}`}
              disabled={!onGround || current}
              onClick={() => onPackSwap(id)}
              className={`whitespace-nowrap rounded-md px-2 py-1 text-xs font-medium ${
                current
                  ? "bg-white/10 text-neutral-100"
                  : onGround
                    ? "text-neutral-400 hover:text-neutral-100"
                    : "cursor-not-allowed text-neutral-600"
              }`}
              title={onGround ? `Switch to pack ${id}` : "Change packs on the ground"}
            >
              Pack {id}
            </UiButton>
          );
        })}
      </div>
    </div>
  );
}

function Dot({ ok, warn }: { ok: boolean; warn?: boolean }) {
  return <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${ok ? "bg-emerald-400" : warn ? "bg-amber-400" : "bg-red-500"}`} />;
}

/** Link status: control socket, heartbeat worker, telemetry, voice. A dot each; rates on hover. */
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
  const chip = "flex items-center gap-1.5 whitespace-nowrap text-xs text-neutral-500";
  const hbOk = hbOpen && (hbRate === null || hbRate > 3.5);
  return (
    <div data-testid="link-chips" className="flex items-center gap-3.5">
      <span className={chip} data-testid="link-control" data-status={control} title={`Control link: ${control}`}>
        <Dot ok={control === "open"} /> Control
      </span>
      <span className={chip} data-testid="link-hb" title={`Heartbeat: ${hbRate === null ? "–" : hbRate.toFixed(1)} Hz`}>
        <Dot ok={hbOk} warn={hbOpen} /> Heartbeat
      </span>
      <span
        className={chip}
        data-testid="link-telemetry"
        title={`Telemetry: ${telemetryRate === null ? "–" : telemetryRate} Hz`}
      >
        <Dot ok={!telemetryStale} /> Telemetry
      </span>
      <span className={chip} data-testid="link-agent" data-status={agent} title={`Voice link: ${agent}`}>
        <Dot ok={agent === "open"} warn={agent === "connecting"} /> Voice
      </span>
    </div>
  );
}
