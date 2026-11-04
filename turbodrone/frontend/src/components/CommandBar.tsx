import { useEffect, useRef, useState, type FormEvent, type PointerEvent, type ReactNode } from "react";
import type { CameraTiltDirection, CommandCapabilities } from "../hooks/useControls";
import type { CommandReport } from "../lib/commands";
import type { CommandName, Telemetry } from "../lib/contracts";
import { takeoffBlocker } from "../lib/gates";
import type { LinkStatus } from "../lib/reconnectingSocket";
import UiButton from "./UiButton";

type Send = (command: CommandName, extra?: { arm_phrase?: string; pack_id?: string }) => boolean;

/** Real mode only: the operator types ARM once per session before takeoff is accepted. */
function ArmGate({ telemetry, send, onGround }: { telemetry: Telemetry; send: Send; onGround: boolean }) {
  const [phrase, setPhrase] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  if (telemetry.armed) {
    return (
      <div className="flex flex-col items-start gap-1">
        <span data-testid="armed-ok" className="rounded-md bg-red-600 px-2 py-1 text-xs font-black tracking-widest text-white">
          ARMED
        </span>
        <UiButton
          data-testid="disarm-btn"
          disabled={!onGround}
          onClick={() => send("disarm")}
          className="rounded bg-slate-800 px-2 py-0.5 text-[11px] text-slate-300 disabled:opacity-40"
        >
          disarm
        </UiButton>
      </div>
    );
  }
  const ready = phrase === "ARM";
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!ready) return;
    if (send("arm", { arm_phrase: "ARM" })) {
      setPhrase("");
      inputRef.current?.blur(); // hand the keyboard back to the hotkeys
    }
  };
  return (
    <form onSubmit={submit} className="flex flex-col gap-1" data-testid="arm-gate">
      <label className="text-[10px] font-semibold uppercase tracking-[0.16em] text-red-300">type ARM to unlock takeoff</label>
      <div className="flex gap-1">
        <input
          ref={inputRef}
          data-testid="arm-input"
          value={phrase}
          onChange={() => setPhrase((inputRef.current?.value ?? "").toUpperCase().slice(0, 8))}
          placeholder="ARM"
          autoComplete="off"
          spellCheck={false}
          className="w-20 rounded-md border border-red-500/70 bg-slate-950 px-2 py-1 font-mono text-lg font-bold tracking-widest text-red-100 outline-none placeholder:text-slate-700 focus:border-red-400"
        />
        <UiButton
          data-testid="arm-btn"
          disabled={!ready}
          onClick={() => {
            if (ready && send("arm", { arm_phrase: "ARM" })) {
              setPhrase("");
              inputRef.current?.blur();
            }
          }}
          className={`rounded-md px-3 py-1 text-sm font-black tracking-widest ${
            ready ? "bg-red-600 text-white hover:bg-red-500" : "cursor-not-allowed bg-slate-800 text-slate-600"
          }`}
        >
          ARM
        </UiButton>
      </div>
    </form>
  );
}

/** Hold-to-act button: pointer down starts, release / cancel / capture loss stops. */
function HoldButton({
  onStart,
  onStop,
  className,
  children,
  testId,
  title,
}: {
  onStart: () => void;
  onStop: () => void;
  className: string;
  children: ReactNode;
  testId?: string;
  title?: string;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  return (
    <UiButton
      ref={ref}
      data-testid={testId}
      title={title}
      onPointerDown={(e: PointerEvent<HTMLButtonElement>) => {
        ref.current?.setPointerCapture(e.pointerId);
        onStart();
      }}
      onPointerUp={onStop}
      onPointerCancel={onStop}
      onLostPointerCapture={onStop}
      className={className}
    >
      {children}
    </UiButton>
  );
}

/** Reset E-STOP needs two clicks within 3 s. */
function EstopReset({ send }: { send: Send }) {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 3000);
    return () => clearTimeout(t);
  }, [armed]);
  return (
    <UiButton
      data-testid="estop-reset-btn"
      onClick={() => {
        if (armed) {
          send("estop_reset");
          setArmed(false);
        } else setArmed(true);
      }}
      className={`rounded-lg px-3 py-2 text-sm font-bold ${armed ? "bg-yellow-400 text-black" : "bg-slate-700 text-slate-100 hover:bg-slate-600"}`}
    >
      {armed ? "click again to reset E-STOP" : "Reset E-STOP…"}
    </UiButton>
  );
}

export default function CommandBar({
  telemetry,
  stale,
  control,
  caps,
  send,
  onTakeoff,
  onLand,
  onEstop,
  ptt,
  onPttDown,
  onPttUp,
  tilt,
  setTilt,
  lastCommand,
}: {
  telemetry: Telemetry | null;
  stale: boolean;
  control: LinkStatus;
  caps: CommandCapabilities;
  send: Send;
  onTakeoff: () => void;
  onLand: () => void;
  onEstop: () => void;
  ptt: boolean;
  onPttDown: () => void;
  onPttUp: () => void;
  tilt: CameraTiltDirection;
  setTilt: (d: CameraTiltDirection) => void;
  lastCommand: CommandReport | null;
}) {
  const blocker = takeoffBlocker(telemetry, stale, control, caps);
  const fs = !stale ? telemetry?.flight_state : undefined;
  const real = telemetry?.flight_mode === "real";

  return (
    <div className="flex h-[112px] shrink-0 items-stretch justify-between gap-4 border-t border-slate-800 bg-slate-950 px-4 py-2.5">
      {/* LEFT: takeoff (+ ARM gate in real mode) and LAND */}
      <div className="flex items-stretch gap-3">
        <div className="flex flex-col justify-center gap-1">
          <UiButton
            data-testid="takeoff-btn"
            disabled={blocker !== null}
            onClick={onTakeoff}
            title={blocker ?? "Take off (button only, never by voice)"}
            className={`rounded-xl px-6 py-3 text-xl font-black tracking-[0.12em] ${
              blocker === null
                ? "bg-emerald-600 text-white shadow-lg hover:bg-emerald-500 active:scale-95"
                : "cursor-not-allowed bg-slate-800 text-slate-500"
            }`}
          >
            TAKEOFF
          </UiButton>
          <span data-testid="takeoff-blocker" className="text-center text-[11px] text-slate-500">
            {blocker ?? "ready"}
          </span>
        </div>
        {real && telemetry && <ArmGate telemetry={telemetry} send={send} onGround={fs === "ON_GROUND"} />}
        <UiButton
          data-testid="land-btn"
          disabled={!caps.land}
          onPointerDown={onLand}
          title="Land (L). Falls back to HTTP if the socket is down."
          className="flex min-w-[200px] flex-col items-center justify-center rounded-xl border-2 border-orange-300/70 bg-orange-600 px-8 text-white shadow-lg hover:bg-orange-500 active:scale-95"
        >
          <span className="text-3xl font-black tracking-[0.2em]">LAND</span>
          <span className="text-xs font-semibold text-orange-100">key L</span>
        </UiButton>
      </div>

      {/* CENTER: context buttons, push-to-talk, and the last command result */}
      <div className="flex min-w-0 flex-1 items-center justify-center gap-3">
        <div className="flex flex-col items-end gap-1">
          <div className="flex gap-2">
            {fs === "ESTOPPED" && <EstopReset send={send} />}
            {fs === "LANDING" && (
              <UiButton
                data-testid="touchdown-btn"
                onClick={() => send("confirm_touchdown")}
                className="rounded-lg bg-orange-800 px-3 py-2 text-sm font-bold text-orange-50 hover:bg-orange-700"
              >
                Confirm touchdown
              </UiButton>
            )}
            {fs === "AIRBORNE" && (
              <>
                <UiButton
                  data-testid="pause-btn"
                  onClick={() => send("pause")}
                  className="rounded-lg bg-slate-700 px-3 py-2 text-sm font-bold text-slate-100 hover:bg-slate-600"
                >
                  Pause autopilot
                </UiButton>
                <UiButton
                  data-testid="resume-btn"
                  onClick={() => send("resume")}
                  className="rounded-lg bg-slate-700 px-3 py-2 text-sm font-bold text-slate-100 hover:bg-slate-600"
                >
                  Resume
                </UiButton>
              </>
            )}
          </div>
          {caps.camera_tilt && (
            <div className="flex items-center gap-1 text-[11px] text-slate-400">
              camera
              <HoldButton
                testId="tilt-up"
                title="Hold to tilt the camera up (PageUp)"
                onStart={() => setTilt(1)}
                onStop={() => setTilt(0)}
                className={`rounded px-2 py-1 font-semibold ${tilt === 1 ? "bg-violet-600 text-white" : "bg-slate-800 text-slate-300"}`}
              >
                ▲ up
              </HoldButton>
              <HoldButton
                testId="tilt-down"
                title="Hold to tilt the camera down (PageDown)"
                onStart={() => setTilt(-1)}
                onStop={() => setTilt(0)}
                className={`rounded px-2 py-1 font-semibold ${tilt === -1 ? "bg-violet-600 text-white" : "bg-slate-800 text-slate-300"}`}
              >
                ▼ down
              </HoldButton>
            </div>
          )}
          {lastCommand && (
            <div
              data-testid="last-command"
              data-channel={lastCommand.channel}
              className={`max-w-[360px] truncate font-mono text-[11px] ${lastCommand.ok ? "text-slate-400" : "font-bold text-red-400"}`}
            >
              {lastCommand.command.toUpperCase()} · {lastCommand.detail} · {new Date(lastCommand.at).toLocaleTimeString()}
            </div>
          )}
        </div>
        <HoldButton
          testId="ptt-btn"
          title="Hold to talk (Space)"
          onStart={onPttDown}
          onStop={onPttUp}
          className={`flex min-w-[260px] flex-col items-center justify-center self-stretch rounded-2xl border-2 px-6 ${
            ptt ? "border-white bg-red-600 text-white" : "border-sky-400/60 bg-sky-800 text-sky-50 hover:bg-sky-700"
          }`}
        >
          <span className="text-2xl font-black tracking-[0.14em]">{ptt ? "● LISTENING" : "🎙 HOLD TO TALK"}</span>
          <span className="text-xs font-semibold opacity-80">Space · missions only, never takeoff</span>
        </HoldButton>
      </div>

      {/* RIGHT: E-STOP, alone and far from LAND */}
      <UiButton
        data-testid="estop-btn"
        onPointerDown={onEstop}
        title="Emergency stop (Esc): cuts the motors. Falls back to HTTP if the socket is down."
        className="flex min-w-[260px] flex-col items-center justify-center rounded-2xl border-4 border-yellow-300 bg-red-700 text-yellow-100 shadow-xl hover:bg-red-600 active:scale-95"
      >
        <span className="text-4xl font-black tracking-[0.18em]">E-STOP</span>
        <span className="text-xs font-bold text-yellow-200">key Esc · cuts motors</span>
      </UiButton>
    </div>
  );
}
