import { useCallback, useEffect, useRef, useState } from "react";
import { URLS } from "../config";
import { ReconnectingSocket, type LinkStatus } from "../lib/reconnectingSocket";
import {
  buildCommand,
  postCommandWithRetry,
  SAFETY_COMMANDS,
  type CommandReport,
} from "../lib/commands";
import type { CommandName, FlightState, OperatorInput } from "../lib/contracts";

/*
 * Operator stick input and commands. Forked from turbodrone's useControls and patched for
 * safety (see docs/ws/ui.md):
 *  - the control socket auto-reconnects; its status drives the DISCONNECTED banner;
 *  - {"type":"axes"} is sent only while at least one axis is nonzero, and never while the
 *    window is blurred or hidden; axes are zeroed on blur, visibilitychange and the Meta key;
 *  - every command is {"type":"cmd","command":...}; LAND and E-STOP fall back to HTTP;
 *  - the speed tier, profile and plugin controls are gone (the governor pins them);
 *  - the TrackPoint (pointer-lock) mode is gone: pointer lock swallows Esc, which is E-STOP.
 */

/* ── Shared types ───────────────────────────────────────────────────────────────────────── */
export type ControlMode = "inc" | "abs"; // keyboard (incremental) | gamepad (absolute)

export interface Axes {
  throttle: number; // -1 … +1  (down / up)
  yaw: number;      // -1 … +1  (left / right, +yaw turns clockwise seen from above)
  pitch: number;    // -1 … +1  (back / forward)
  roll: number;     // -1 … +1  (left / right)
}

export interface CommandCapabilities {
  takeoff: boolean;
  land: boolean;
  estop: boolean;
  camera_tilt: boolean;
  camera_switch: boolean;
  speed_control: boolean; // reported by upstream /capabilities; ignored (tier is pinned)
}

export type CameraTiltDirection = -1 | 0 | 1;

const ZERO_AXES: Axes = { throttle: 0, yaw: 0, pitch: 0, roll: 0 };
const TX_HZ = 30;
const GAMEPAD_DEADZONE = 0.15;
/** How long telemetry gets to confirm LAND / E-STOP before we also send it over HTTP. */
const SAFETY_CONFIRM_MS = 600;
/** A takeoff the governor accepted shows TAKING_OFF well within this. */
const TAKEOFF_CONFIRM_MS = 1200;

/** Keyboard stick map, by physical key (e.code) so Shift/Caps Lock cannot strand a key. */
const KEY_AXES: Record<string, { axis: keyof Axes; dir: 1 | -1 }> = {
  KeyW: { axis: "pitch", dir: +1 },
  KeyS: { axis: "pitch", dir: -1 },
  KeyA: { axis: "roll", dir: -1 },
  KeyD: { axis: "roll", dir: +1 },
  ArrowUp: { axis: "throttle", dir: +1 },
  ArrowDown: { axis: "throttle", dir: -1 },
  ArrowLeft: { axis: "yaw", dir: -1 },
  ArrowRight: { axis: "yaw", dir: +1 },
};

function isZero(a: Axes): boolean {
  return a.throttle === 0 && a.yaw === 0 && a.pitch === 0 && a.roll === 0;
}

function round3(v: number): number {
  return Math.round(Math.max(-1, Math.min(1, v)) * 1000) / 1000;
}

/** True while the operator is typing in a text field (the ARM box): stick keys stay off. */
export function isTypingInField(): boolean {
  const el = document.activeElement;
  return el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement;
}

/** Sticks may only be sent while this page is visible and focused. */
function inputAllowed(): boolean {
  return document.visibilityState === "visible" && document.hasFocus();
}

/**
 * Does telemetry show that a safety command took effect? ON_GROUND counts for LAND only when
 * no takeoff was sent recently: right after TAKEOFF, telemetry may still say ON_GROUND while
 * the governor is already taking off.
 */
function safetyConfirmed(command: CommandName, state: FlightState | null, takeoffRecent = false): boolean {
  if (state === null) return false;
  if (command === "estop") return state === "ESTOPPED";
  if (command === "land") return state === "LANDING" || state === "ESTOPPED" || (state === "ON_GROUND" && !takeoffRecent);
  return true;
}

/** How long after a TAKEOFF an ON_GROUND telemetry state is not trusted as "landed". */
const TAKEOFF_RECENT_MS = 5000;

export function useControls(opts: { flightState: FlightState | null }) {
  /* ------- mutable refs (read by timers and listeners) ------- */
  const axesRef = useRef<Axes>({ ...ZERO_AXES });
  const modeRef = useRef<ControlMode>("inc");
  const pressedRef = useRef<Set<string>>(new Set());
  const tiltRef = useRef<CameraTiltDirection>(0);
  const sockRef = useRef<ReconnectingSocket | null>(null);
  const flightStateRef = useRef<FlightState | null>(opts.flightState);
  flightStateRef.current = opts.flightState;
  // When telemetry last showed each safety command in effect (performance.now()). A command
  // confirmed at any moment after it was sent is never re-sent, even if the state moved on
  // (for example an E-STOP followed quickly by a reset).
  const confirmedAtRef = useRef<Record<string, number>>({});
  const takeoffSentAtRef = useRef(-Infinity);
  const sendSeqRef = useRef(0); // bumps on every command sent: late reports must not hide newer ones
  const takeoffRecent = () => performance.now() - takeoffSentAtRef.current < TAKEOFF_RECENT_MS;
  useEffect(() => {
    for (const command of SAFETY_COMMANDS) {
      if (safetyConfirmed(command, opts.flightState, performance.now() - takeoffSentAtRef.current < TAKEOFF_RECENT_MS)) {
        confirmedAtRef.current[command] = performance.now();
      }
    }
  }, [opts.flightState]);

  /* ------- state that triggers re-renders ------- */
  const [axes, setAxes] = useState<Axes>({ ...ZERO_AXES });
  const [mode, setModeSt] = useState<ControlMode>("inc");
  const [linkStatus, setLinkStatus] = useState<LinkStatus>("connecting");
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [sendingAxes, setSendingAxes] = useState(false);
  const [lastCommand, setLastCommand] = useState<CommandReport | null>(null);
  const [gamepadConnected, setGamepadConnected] = useState<boolean>(false);
  const [droneType, setDroneType] = useState<string>("unknown");
  const [commandCapabilities, setCommandCapabilities] = useState<CommandCapabilities>({
    takeoff: true,
    land: true,
    estop: true,
    camera_tilt: false,
    camera_switch: false,
    speed_control: false,
  });
  const [cameraTiltDirection, setCameraTiltDirectionSt] = useState<CameraTiltDirection>(0);

  const setMode = useCallback((m: ControlMode) => {
    modeRef.current = m;
    setModeSt(m);
    pressedRef.current.clear();
    axesRef.current = { ...ZERO_AXES };
    setAxes({ ...ZERO_AXES });
  }, []);

  /** Release every stick at once (blur, hidden tab, Meta key, mode change). */
  const zeroAxes = useCallback(() => {
    pressedRef.current.clear();
    if (!isZero(axesRef.current)) {
      axesRef.current = { ...ZERO_AXES };
      setAxes({ ...ZERO_AXES });
    }
  }, []);

  /* --------------- control socket (/ws), auto-reconnecting --------------- */
  useEffect(() => {
    const sock = new ReconnectingSocket({
      url: URLS.control,
      onStatus: (status, attempts) => {
        setLinkStatus(status);
        setReconnectAttempts(attempts);
      },
    });
    sockRef.current = sock;
    sock.start();
    return () => {
      sock.stop();
      sockRef.current = null;
    };
  }, []);

  /* --------------- commands ---------------------------------------------- */
  const report = useCallback((r: Omit<CommandReport, "at">) => {
    setLastCommand({ ...r, at: Date.now() });
  }, []);

  const sendCommand = useCallback(
    (command: CommandName, extra: { arm_phrase?: string; pack_id?: string } = {}): boolean => {
      const msg = buildCommand(command, extra);
      const safety = SAFETY_COMMANDS.has(command);
      const seq = ++sendSeqRef.current;
      if (command === "takeoff") takeoffSentAtRef.current = performance.now();

      if (sockRef.current?.send(msg)) {
        report({ command, channel: "ws", ok: true, detail: "sent on control socket" });
        if (command === "takeoff") {
          // Commands over the socket get no reply: watch telemetry to tell the operator
          // when the governor refused (no video, no heartbeat, not armed, ...).
          setTimeout(() => {
            if (sendSeqRef.current !== seq) return; // a newer command's report is showing
            if (flightStateRef.current === "ON_GROUND") {
              report({ command, channel: "ws", ok: false, detail: "NOT ACCEPTED: still on the ground (see Safety flags)" });
            }
          }, TAKEOFF_CONFIRM_MS);
        }
        if (safety) {
          // Belt and braces: if telemetry does not show the command took effect, the socket
          // may be open but unread. Send it again over HTTP (both are idempotent).
          const sentAt = performance.now();
          setTimeout(() => {
            if (safetyConfirmed(command, flightStateRef.current, takeoffRecent())) return;
            if ((confirmedAtRef.current[command] ?? -1) >= sentAt) return;
            void postCommandWithRetry(msg, 4).then((ok) =>
              report({
                command,
                channel: "http",
                ok,
                detail: ok ? "not confirmed by telemetry: re-sent over HTTP" : "HTTP re-send FAILED",
              }),
            );
          }, SAFETY_CONFIRM_MS);
        }
        return true;
      }

      if (safety) {
        report({ command, channel: "http", ok: true, detail: "socket down: sending over HTTP…" });
        void postCommandWithRetry(msg).then((ok) =>
          report({
            command,
            channel: "http",
            ok,
            detail: ok ? "sent over HTTP (socket down)" : "FAILED: backend unreachable",
          }),
        );
        return true;
      }

      report({ command, channel: "none", ok: false, detail: "NOT SENT: control link down" });
      return false;
    },
    [report],
  );

  const takeOff = useCallback(() => sendCommand("takeoff"), [sendCommand]);
  const land = useCallback(() => sendCommand("land"), [sendCommand]);
  const emergencyStop = useCallback(() => sendCommand("estop"), [sendCommand]);

  /* --------------- capabilities (retried until the backend answers) ------ */
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const fetchCapabilities = async () => {
      try {
        const response = await fetch(URLS.capabilities, { cache: "no-store" });
        if (!response.ok) throw new Error(String(response.status));
        const data = await response.json();
        if (cancelled) return;
        setDroneType(data?.drone_type ?? "unknown");
        setCommandCapabilities({
          takeoff: Boolean(data?.commands?.takeoff),
          land: Boolean(data?.commands?.land),
          estop: Boolean(data?.commands?.estop),
          camera_tilt: Boolean(data?.commands?.camera_tilt),
          camera_switch: Boolean(data?.commands?.camera_switch),
          speed_control: Boolean(data?.commands?.speed_control),
        });
      } catch {
        // Keep optimistic defaults (LAND and E-STOP always shown) and try again shortly.
        if (!cancelled) timer = setTimeout(fetchCapabilities, 3000);
      }
    };

    void fetchCapabilities();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  /* --------------- zero everything on blur / hidden tab ------------------ */
  useEffect(() => {
    const release = () => {
      zeroAxes();
      if (tiltRef.current !== 0) {
        tiltRef.current = 0;
        setCameraTiltDirectionSt(0);
        sendCommand("tilt_stop");
      }
    };
    const onVisibility = () => {
      if (document.visibilityState !== "visible") release();
    };
    window.addEventListener("blur", release);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("blur", release);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [zeroAxes, sendCommand]);

  /* --------------- gamepad detection ------------------------------------ */
  useEffect(() => {
    const checkGamepad = () => {
      const pads = navigator.getGamepads ? Array.from(navigator.getGamepads()) : [];
      const hasGamepad = pads.some((gp) => gp !== null && gp.connected);
      setGamepadConnected(hasGamepad);
      if (!hasGamepad && modeRef.current === "abs") setMode("inc"); // unplugged: back to keys
    };
    checkGamepad();
    window.addEventListener("gamepadconnected", checkGamepad);
    window.addEventListener("gamepaddisconnected", checkGamepad);
    const poll = setInterval(checkGamepad, 1000);
    return () => {
      window.removeEventListener("gamepadconnected", checkGamepad);
      window.removeEventListener("gamepaddisconnected", checkGamepad);
      clearInterval(poll);
    };
  }, [setMode]);

  /* --------------- keyboard sticks (W/A/S/D + arrows) ------------------- */
  useEffect(() => {
    if (mode !== "inc") return;

    const recompute = () => {
      const next: Axes = { ...ZERO_AXES };
      for (const code of pressedRef.current) {
        const m = KEY_AXES[code];
        if (m) next[m.axis] += m.dir; // opposite keys cancel
      }
      axesRef.current = next;
      setAxes(next);
    };

    const down = (e: KeyboardEvent) => {
      // macOS drops keyup events while Cmd is held, so any Meta chord releases the sticks.
      if (e.metaKey || e.key === "Meta") {
        zeroAxes();
        return;
      }
      if (!(e.code in KEY_AXES) || isTypingInField()) return;
      e.preventDefault(); // arrows must not scroll the page
      if (!pressedRef.current.has(e.code)) {
        pressedRef.current.add(e.code);
        recompute();
      }
    };

    const up = (e: KeyboardEvent) => {
      if (e.key === "Meta") {
        zeroAxes();
        return;
      }
      if (!(e.code in KEY_AXES)) return;
      e.preventDefault();
      if (pressedRef.current.delete(e.code)) recompute();
    };

    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      zeroAxes();
    };
  }, [mode, zeroAxes]);

  /* --------------- camera tilt keys (hold PageUp / PageDown) ------------ */
  const setCameraTiltDirection = useCallback(
    (direction: CameraTiltDirection) => {
      if (tiltRef.current === direction) return;
      tiltRef.current = direction;
      setCameraTiltDirectionSt(direction);
      sendCommand(direction > 0 ? "tilt_up" : direction < 0 ? "tilt_down" : "tilt_stop");
    },
    [sendCommand],
  );

  // The governor stops a tilt by itself 3 s after the last tilt_up/tilt_down (the UI may die
  // mid-press), so re-send the held direction every second, quietly (no status-line report).
  useEffect(() => {
    if (cameraTiltDirection === 0) return;
    const command = cameraTiltDirection > 0 ? "tilt_up" : "tilt_down";
    const keepAlive = setInterval(() => sockRef.current?.send(buildCommand(command)), 1000);
    return () => clearInterval(keepAlive);
  }, [cameraTiltDirection]);

  useEffect(() => {
    if (!commandCapabilities.camera_tilt) return;
    const map: Record<string, CameraTiltDirection> = { PageUp: 1, PageDown: -1 };
    const down = (e: KeyboardEvent) => {
      const direction = map[e.code];
      if (!direction) return;
      e.preventDefault();
      setCameraTiltDirection(direction);
    };
    const up = (e: KeyboardEvent) => {
      const direction = map[e.code];
      if (!direction) return;
      e.preventDefault();
      if (tiltRef.current === direction) setCameraTiltDirection(0);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [commandCapabilities.camera_tilt, setCameraTiltDirection]);

  /* --------------- Xbox-style gamepad (absolute) ------------------------ */
  useEffect(() => {
    if (mode !== "abs") return;
    const dz = (v: number) => (Math.abs(v) < GAMEPAD_DEADZONE ? 0 : v);
    let raf = 0;
    const poll = () => {
      // The first connected pad, whatever its index (a re-plugged pad may come back as 1).
      const gp = Array.from(navigator.getGamepads()).find((g) => g !== null && g.connected) ?? null;
      if (gp && inputAllowed()) {
        // Left stick: axes[0] X, axes[1] Y. Right stick: axes[2] X, axes[3] Y. +Y is down.
        const next: Axes = {
          roll: dz(gp.axes[0] ?? 0),
          pitch: dz(-(gp.axes[1] ?? 0)),
          yaw: dz(gp.axes[2] ?? 0),
          throttle: dz(-(gp.axes[3] ?? 0)),
        };
        const prev = axesRef.current;
        axesRef.current = next;
        if (
          next.roll !== prev.roll || next.pitch !== prev.pitch ||
          next.yaw !== prev.yaw || next.throttle !== prev.throttle
        ) {
          setAxes(next);
        }
      } else {
        zeroAxes();
      }
      raf = requestAnimationFrame(poll);
    };
    raf = requestAnimationFrame(poll);
    return () => {
      cancelAnimationFrame(raf);
      zeroAxes();
    };
  }, [mode, zeroAxes]);

  /* --------------- network TX at 30 Hz, only while nonzero -------------- */
  useEffect(() => {
    let wasSending = false;
    const interval = setInterval(() => {
      const a = axesRef.current;
      const rounded: Axes = {
        throttle: round3(a.throttle),
        yaw: round3(a.yaw),
        pitch: round3(a.pitch),
        roll: round3(a.roll),
      };
      let sending = false;
      // All-zero messages are not input (contract): never send one.
      if (!isZero(rounded) && inputAllowed()) {
        const msg: OperatorInput = { type: "axes", mode: modeRef.current, ...rounded };
        sending = sockRef.current?.send(msg) ?? false;
      }
      if (sending !== wasSending) {
        wasSending = sending;
        setSendingAxes(sending);
      }
    }, 1000 / TX_HZ);
    return () => clearInterval(interval);
  }, []);

  /* ------------- hook return ------------------------------------------- */
  return {
    axes,
    mode,
    setMode,
    gamepadConnected,
    droneType,
    commandCapabilities,
    linkStatus,
    reconnectAttempts,
    sendingAxes,
    lastCommand,
    sendCommand,
    takeOff,
    land,
    emergencyStop,
    cameraTiltDirection,
    setCameraTiltDirection,
  };
}
