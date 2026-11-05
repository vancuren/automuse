import { useEffect, useRef, useState, type FormEvent, type PointerEvent, type ReactNode } from "react";
import type { CameraTiltDirection, CommandCapabilities } from "../hooks/useControls";
import type { CommandReport } from "../lib/commands";
import type { CommandName, Telemetry } from "../lib/contracts";
import { takeoffBlocker } from "../lib/gates";
import type { LinkStatus } from "../lib/reconnectingSocket";
import UiButton from "./UiButton";

type Send = (command: CommandName, extra?: { arm_phrase?: string; pack_id?: string }) => boolean;

const QUIET = "h-9 rounded-lg bg-white/[0.06] px-3 text-[13px] font-medium text-neutral-200 hover:bg-white/10";

/** Real mode only: the operator types ARM once per session before takeoff is accepted. */
function ArmGate({ telemetry, send, onGround }: { telemetry: Telemetry; send: Send; onGround: boolean }) {
  const [phrase, setPhrase] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  if (telemetry.armed) {
    return (
      <div className="flex items-center gap-2">
        <span data-testid="armed-ok" className="flex h-7 items-center gap-1.5 rounded-full bg-red-500/15 px-2.5 text-xs font-medium text-red-300">
          <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
          Armed
        </span>
        <UiButton
          data-testid="disarm-btn"
          disabled={!onGround}
          onClick={() => send("disarm")}
          className="text-xs text-neutral-500 hover:text-neutral-200 disabled:opacity-40"
        >
          Disarm
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
    <form onSubmit={submit} className="flex items-center gap-2" data-testid="arm-gate">
      <input
        ref={inputRef}
        data-testid="arm-input"
        value={phrase}
        onChange={() => setPhrase((inputRef.current?.value ?? "").toUpperCase().slice(0, 8))}
        placeholder="Type ARM"
        aria-label="Type ARM to unlock takeoff"
        autoComplete="off"
        spellCheck={false}
        className="h-12 w-[104px] rounded-xl bg-transparent px-3 text-center text-[15px] font-medium text-white outline-none ring-1 ring-inset ring-white/15 placeholder:text-neutral-600 focus:ring-white/50"
      />
      <UiButton
        data-testid="arm-btn"
        type="submit"
        disabled={!ready}
        className={`h-12 rounded-xl px-4 text-[15px] font-semibold ${
          ready ? "bg-white text-black hover:bg-neutral-200" : "cursor-not-allowed bg-white/[0.04] text-neutral-600"
        }`}
      >
        Arm
      </UiButton>
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
      className={armed ? "h-9 rounded-lg bg-amber-400 px-3 text-[13px] font-semibold text-black" : QUIET}
    >
      {armed ? "Click again to reset" : "Reset emergency stop"}
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
  // Real mode, in the order the operator works: type ARM, then Take off. Once armed it steps aside.
  const armGate = real && telemetry ? <ArmGate telemetry={telemetry} send={send} onGround={fs === "ON_GROUND"} /> : null;
  const tiltBtn = (on: boolean) =>
    `h-9 w-9 rounded-lg text-[13px] ${on ? "bg-white text-black" : "bg-white/[0.06] text-neutral-300 hover:bg-white/10"}`;

  return (
    <div className="flex h-[92px] shrink-0 items-center justify-between gap-6 border-t border-white/[0.07] px-5">
      {/* LEFT: takeoff (+ ARM gate in real mode) and LAND */}
      <div className="flex items-center gap-3">
        {armGate && !telemetry?.armed && armGate}
        <div className="relative">
          <UiButton
            data-testid="takeoff-btn"
            disabled={blocker !== null}
            onClick={onTakeoff}
            title={blocker ?? "Take off (button only, never by voice)"}
            className={`h-12 min-w-[150px] rounded-xl px-6 text-[15px] font-semibold ${
              blocker === null
                ? "bg-white text-black hover:bg-neutral-200 active:scale-[0.98]"
                : "cursor-not-allowed bg-white/[0.04] text-neutral-600"
            }`}
          >
            Take off
          </UiButton>
          <span
            data-testid="takeoff-blocker"
            className={`absolute left-0 top-full mt-1 whitespace-nowrap text-xs ${blocker === null ? "text-neutral-600" : "text-neutral-400"}`}
          >
            {blocker ?? "ready"}
          </span>
        </div>
        {armGate && telemetry?.armed && armGate}
        <UiButton
          data-testid="land-btn"
          onPointerDown={onLand}
          title="Land (L). Falls back to HTTP if the socket is down."
          className="flex h-12 min-w-[180px] items-center justify-center gap-2.5 rounded-xl bg-white/10 px-6 text-[15px] font-semibold text-white hover:bg-white/[0.16] active:scale-[0.98]"
        >
          Land
          <kbd>L</kbd>
        </UiButton>
      </div>

      {/* CENTER: push-to-talk, with the buttons that only apply right now beside it */}
      <div className="flex min-w-0 flex-1 items-center justify-center gap-3">
        {fs === "ESTOPPED" && <EstopReset send={send} />}
        {fs === "LANDING" && (
          <UiButton data-testid="touchdown-btn" onClick={() => send("confirm_touchdown")} className={QUIET}>
            Confirm touchdown
          </UiButton>
        )}
        {fs === "AIRBORNE" && (
          <>
            <UiButton data-testid="pause-btn" onClick={() => send("pause")} className={QUIET}>
              Pause
            </UiButton>
            <UiButton data-testid="resume-btn" onClick={() => send("resume")} className={QUIET}>
              Resume
            </UiButton>
          </>
        )}
        <div className="relative">
          <HoldButton
            testId="ptt-btn"
            title="Hold to talk (Space). Missions only, never takeoff."
            onStart={onPttDown}
            onStop={onPttUp}
            className={`flex h-12 min-w-[280px] items-center justify-center gap-2.5 rounded-full px-6 text-[15px] font-medium ${
              ptt ? "bg-white text-black" : "bg-white/[0.06] text-neutral-200 ring-1 ring-inset ring-white/10 hover:bg-white/10"
            }`}
          >
            {ptt ? (
              <>
                <span className="soft-pulse h-2 w-2 rounded-full bg-red-500" />
                Listening
              </>
            ) : (
              <>
                Hold <kbd>Space</kbd> to talk
              </>
            )}
          </HoldButton>
          {lastCommand && (
            <div
              data-testid="last-command"
              data-channel={lastCommand.channel}
              className={`num absolute left-1/2 top-full mt-1 max-w-[420px] -translate-x-1/2 truncate whitespace-nowrap text-xs ${
                lastCommand.ok ? "text-neutral-600" : "text-red-400"
              }`}
            >
              {lastCommand.command.toUpperCase()} · {lastCommand.detail} · {new Date(lastCommand.at).toLocaleTimeString()}
            </div>
          )}
        </div>
        {caps.camera_tilt && (
          <div className="flex items-center gap-1" title="Camera tilt (hold). PageUp / PageDown.">
            <HoldButton
              testId="tilt-up"
              title="Hold to tilt the camera up (PageUp)"
              onStart={() => setTilt(1)}
              onStop={() => setTilt(0)}
              className={tiltBtn(tilt === 1)}
            >
              ↑
            </HoldButton>
            <HoldButton
              testId="tilt-down"
              title="Hold to tilt the camera down (PageDown)"
              onStart={() => setTilt(-1)}
              onStop={() => setTilt(0)}
              className={tiltBtn(tilt === -1)}
            >
              ↓
            </HoldButton>
          </div>
        )}
      </div>

      {/* RIGHT: emergency stop, alone and far from Land */}
      <UiButton
        data-testid="estop-btn"
        onPointerDown={onEstop}
        title="Emergency stop (Esc): cuts the motors. Falls back to HTTP if the socket is down."
        className="flex h-12 min-w-[240px] items-center justify-center gap-2.5 rounded-xl bg-red-600 px-6 text-[15px] font-semibold text-white hover:bg-red-500 active:scale-[0.98]"
      >
        Emergency stop
        <kbd>Esc</kbd>
      </UiButton>
    </div>
  );
}
