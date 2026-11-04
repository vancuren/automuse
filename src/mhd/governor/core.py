"""The safety governor: owns every RC packet (PLAN.md 4.5).

`Governor.tick(now_ms)` runs once per RC tick (25 Hz) on the RC thread, called by GovernorStrategy
just before turbodrone builds the packet. Inputs arrive from other threads through the submit_*,
command and set_mission methods, which only update state under a lock.

Order inside one tick:
  1. E-stop latch: 0x02 in every packet, neutral sticks, until estop_reset.
  2. Flight-state timers (TAKING_OFF -> AIRBORNE after the neutral hold) and the flight timer.
  3. Safety landings: heartbeat missing 5 s, video lag 2 s, flight timer 6:30.
  4. Sticks by priority: E-stop > land > human > autopilot; staleness checks neutralize the
     autopilot (and a missing heartbeat neutralizes everything). Safety neutral bypasses slew.
  5. Wire counts: sign, deadband compensation (autopilot only), caps, symmetric rounding.
  6. Flags: takeoff (N packets), land re-asserted 5x per second, camera tilt (human only),
     flip and calibrate always masked.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from mhd.autopilot import AutopilotRun, autopilot_start, autopilot_step, autopilot_stop, bearing_rad
from mhd.contracts import AutopilotState, FlightState, Mission, SubjectObs, now_ms

from .config import AXES, AutopilotConfig, GovernorConfig, logs_dir
from .flight_timer import FlightTimer
from .stats import PeriodEstimator, SlidingMin
from .tick_log import TickLog
from .wire import CENTER, neutral_wire, to_wire

logger = logging.getLogger(__name__)

ZERO = {"throttle": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0}
INF = math.inf
AGENT_COMMANDS = frozenset({"pause", "land"})     # voice/LLM side: never takeoff, never E-stop
ALL_COMMANDS = frozenset({
    "takeoff", "land", "estop", "estop_reset", "pause", "resume", "confirm_touchdown",
    "arm", "disarm", "tilt_up", "tilt_down", "tilt_stop", "pack_swap",
})


class GovernorError(RuntimeError):
    pass


@dataclass
class TickOutput:
    """What one tick decided. Written into the X69 model by GovernorStrategy."""
    wire: dict
    takeoff: bool = False
    land: bool = False
    stop: bool = False
    tilt_state: int = 0           # turbodrone convention: 0 neutral, 1 down, 2 up
    flight_state: FlightState = FlightState.on_ground
    source: str = "none"          # none | human | autopilot | landing | estop | takeoff | shutdown | fallback
    safety_flags: tuple = ()


@dataclass
class CommandResult:
    ok: bool
    flight_state: str
    reason: Optional[str] = None

    def as_dict(self) -> dict:
        d = {"ok": self.ok, "flight_state": self.flight_state}
        if self.reason:
            d["reason"] = self.reason
        return d


@dataclass
class _Snapshot:
    """Read-only copy of the last tick for telemetry (swapped atomically, never mutated)."""
    ts_ms: float = 0.0
    flight_state: FlightState = FlightState.on_ground
    autopilot_state: AutopilotState = AutopilotState.idle
    armed: bool = False
    wire: dict = field(default_factory=neutral_wire)
    by_source: dict = field(default_factory=dict)
    source: str = "none"
    frame_age_ms: Optional[float] = None
    obs_age_ms: Optional[float] = None
    latency_ms: dict = field(default_factory=dict)
    flight_time_s: float = 0.0
    pack_id: Optional[str] = None
    safety_flags: tuple = ()
    tick_us: float = 0.0


def _finite(x: float) -> Optional[float]:
    return None if x is None or x == INF or x != x else round(float(x), 1)


class Governor:
    def __init__(self, cfg: GovernorConfig, ap_params: AutopilotConfig, flight_mode: str, *,
                 clock: Callable[[], float] = now_ms, log_dir: Optional[Path] = None,
                 persist_timer: bool = True, tick_log: Optional[bool] = None):
        if flight_mode not in ("twin", "real"):
            raise GovernorError(f"flight_mode must be 'twin' or 'real', got {flight_mode!r}")
        if flight_mode == "real" and cfg.twin_headless_bypass:
            raise GovernorError("twin_headless_bypass is set: refusing to run in real mode")
        self.cfg = cfg
        self.ap_params = ap_params
        self.flight_mode = flight_mode
        self._clock = clock
        self._lock = threading.RLock()
        log_dir = Path(log_dir) if log_dir else logs_dir()
        self.flight_timer = FlightTimer(log_dir / "flight_timer.json" if persist_timer else None)
        use_log = cfg.tick_log if tick_log is None else tick_log
        self.tick_log = TickLog(log_dir, enabled=use_log)

        # Flight state machine
        self._state = FlightState.on_ground
        self._state_since = clock()
        self._armed = flight_mode == "twin"     # twin: auto-armed; real: type ARM first
        self._estop_latched = False             # read by the exception fallback without the lock
        self._takeoff_left = 0
        self._land_cause: Optional[str] = None
        self._land_is_safety = False
        self._land_last_ms = -INF
        self._timer_land_fired = False
        self._timer_warned = False
        self._shutdown = False

        # Human input
        self._human_target = dict(ZERO)
        self._human_mode = "abs"
        self._human_last_ms: Optional[float] = None
        self._human_ramp = dict(ZERO)
        self._hb_last_ms: Optional[float] = None
        self._tilt_dir = 0
        self._tilt_until_ms = -INF

        # Video frames
        self._frame_capture_ms: Optional[float] = None
        self._frame_floor = SlidingMin(cfg.video.floor_window_s * 1000.0)

        # Observations
        self._obs: Optional[SubjectObs] = None
        self._obs_floor = SlidingMin(cfg.observation.floor_window_s * 1000.0)
        self._obs_period = PeriodEstimator()
        self._obs_hist: deque = deque(maxlen=200)       # (capture_ms, bearing) for the wrong-way check
        self._obs_hist_frame = -1

        # Mission and autopilot
        self._mission: Optional[Mission] = None
        self._ap = AutopilotRun()
        self._ap_paused = False
        self._ap_fault_until = -INF
        self._ap_last_sticks = dict(ZERO)
        self._ap_step_ms = 0.0
        self._wrong_way_hold = False
        self._slew = dict(ZERO)
        self._applied = dict(ZERO)                       # normalized, effective, last tick
        self._yaw_hist: deque = deque(maxlen=150)        # (t_ms, yaw rate rad/s) actually commanded (6 s)
        self.events: deque = deque(maxlen=200)           # autopilot/safety events for the hub to relay

        # Tick bookkeeping
        self._last_tick_ms: Optional[float] = None
        self.last_tick_ts: Optional[float] = None        # published for watchdogs
        self.tick_count = 0
        self.fault_count = 0
        self._fault_since: Optional[float] = None
        self._tick_us = deque(maxlen=2000)
        self._snapshot = _Snapshot(armed=self._armed, pack_id=self.flight_timer.current_pack)

        self._hk_stop = threading.Event()
        self._hk_thread: Optional[threading.Thread] = None

    # ==========================================================================================
    # Inputs (any thread)

    def heartbeat(self, now: Optional[float] = None) -> None:
        with self._lock:
            self._hb_last_ms = self._clock() if now is None else now

    def submit_human(self, axes: dict, mode: str = "abs", now: Optional[float] = None) -> bool:
        """Operator sticks. Returns True if this counted as input (nonzero after the deadzone).

        An all-zero message is not input: it releases the sticks immediately but never claims
        control, so an idle UI streaming zeros cannot block the autopilot.
        """
        now = self._clock() if now is None else now
        dz = self.cfg.human.deadzone
        vals = {}
        for a in AXES:
            try:
                v = float(axes.get(a, 0.0) or 0.0)
            except (TypeError, ValueError):
                v = 0.0
            if v != v:
                v = 0.0
            v = max(-1.0, min(1.0, v))
            vals[a] = 0.0 if abs(v) < dz else v
        nonzero = any(vals[a] != 0.0 for a in AXES)
        with self._lock:
            if not nonzero:
                self._release_human()
                return False
            if mode in ("inc", "incremental"):
                mode = "inc"
            else:
                mode = "abs"            # "abs", "mouse", anything else: absolute
            if mode != self._human_mode:
                self._human_ramp = dict(ZERO)
            self._human_target = vals
            self._human_mode = mode
            self._human_last_ms = now
            return True

    def ui_disconnected(self) -> None:
        """The /ws socket closed: human input and tilt go to zero now."""
        with self._lock:
            self._release_human()
            self._tilt_dir = 0
            self._tilt_until_ms = -INF

    def set_tilt(self, direction: int, *, hold_s: Optional[float] = None, now: Optional[float] = None) -> None:
        """Camera tilt: +1 up, -1 down, 0 stop. Human only; stops by itself after hold_s."""
        now = self._clock() if now is None else now
        d = 1 if direction > 0 else (-1 if direction < 0 else 0)
        with self._lock:
            self._tilt_dir = d
            hold = self.cfg.tilt.max_hold_s if hold_s is None else hold_s
            self._tilt_until_ms = now + hold * 1000.0 if d else -INF

    def note_frame(self, capture_ms: float, receive_ms: Optional[float] = None) -> None:
        """A decoded video frame reached the backend (for video-age checks)."""
        receive_ms = self._clock() if receive_ms is None else receive_ms
        with self._lock:
            if self._frame_capture_ms is None or capture_ms >= self._frame_capture_ms:
                self._frame_capture_ms = capture_ms
            self._frame_floor.add(receive_ms, max(0.0, receive_ms - capture_ms))

    def submit_obs(self, obs: SubjectObs, receive_ms: Optional[float] = None) -> bool:
        """Newest SubjectObs from perception. Out-of-order, foreign-mission or future obs are dropped."""
        receive_ms = self._clock() if receive_ms is None else receive_ms
        with self._lock:
            mid = self._mission.mission_id if self._mission else None
            if obs.mission_id is not None and mid is not None and obs.mission_id != mid:
                return False
            if obs.capture_ms > receive_ms + 50.0:
                return False                     # clock mismatch: never trust a future frame
            if self._obs is not None and obs.capture_ms < self._obs.capture_ms:
                return False                     # out of order
            self._obs = obs
            self._obs_floor.add(receive_ms, max(0.0, receive_ms - obs.capture_ms))
            self._obs_period.add(obs.capture_ms)
            if obs.box is not None and obs.frame_id != self._obs_hist_frame:
                self._obs_hist_frame = obs.frame_id
                self._obs_hist.append((obs.capture_ms, bearing_rad(obs.box.cx, self.ap_params)))
            return True

    def set_mission(self, mission: Mission, now: Optional[float] = None) -> dict:
        """Apply an agent Mission. follow starts the autopilot (FIND); hover pauses; land lands."""
        now = self._clock() if now is None else now
        action = mission.intent.action
        with self._lock:
            if action == "follow":
                self._mission = mission
                self._ap = autopilot_start(self._ap, mission.mission_id, now,
                                           distance=mission.intent.distance,
                                           category=mission.intent.category)
                self._ap_paused = False
                self._wrong_way_hold = False
                self._obs = None
                self._obs_hist.clear()
                return {"ok": True, "action": "follow", "find": True}
            if action == "hover":
                self._ap_paused = True
                return {"ok": True, "action": "hover"}
        if action == "land":
            r = self.command("land", source="agent", now=now)
            return {"ok": r.ok, "action": "land", "flight_state": r.flight_state}
        return {"ok": True, "action": action, "note": "no flight change"}

    def stop_mission(self, now: Optional[float] = None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            self._ap = autopilot_stop(self._ap, now)
            self._mission = None

    def command(self, cmd: str, *, arm_phrase: Optional[str] = None, pack_id: Optional[str] = None,
                source: str = "ui", now: Optional[float] = None) -> CommandResult:
        """Operator commands. E-stop and land are accepted from every human source in every state."""
        now = self._clock() if now is None else now
        with self._lock:
            res = self._command_locked(cmd, arm_phrase, pack_id, source, now)
        level = logging.WARNING if cmd in ("estop", "land") or not res.ok else logging.INFO
        logger.log(level, "[governor] cmd=%s source=%s -> ok=%s state=%s %s", cmd, source, res.ok,
                   res.flight_state, res.reason or "")
        self.events.append({"kind": "command", "command": cmd, "source": source, "ok": res.ok,
                            "reason": res.reason, "flight_state": res.flight_state, "ts_ms": now})
        return res

    def begin_shutdown(self) -> None:
        """Backend is stopping: every following packet is neutral sticks plus land."""
        with self._lock:
            self._shutdown = True

    # ==========================================================================================
    # Commands

    def _command_locked(self, cmd, arm_phrase, pack_id, source, now) -> CommandResult:
        st = self._state
        ok = lambda: CommandResult(True, self._state.value)          # noqa: E731
        no = lambda why: CommandResult(False, self._state.value, why)  # noqa: E731
        if cmd not in ALL_COMMANDS:
            return no("unknown_command")
        if source == "agent" and cmd not in AGENT_COMMANDS:
            return no("not_allowed_from_agent")

        if cmd == "estop":
            self._estop_latched = True
            self._set_state(FlightState.estopped, now)
            self._ap = autopilot_stop(self._ap, now)
            self._ap_paused = True
            return ok()
        if cmd == "estop_reset":
            if st != FlightState.estopped:
                return no("not_estopped")
            self._estop_latched = False
            self._set_state(FlightState.on_ground, now)
            if self.flight_mode == "real" and self.cfg.estop.rearm_after_reset:
                self._armed = False
            return ok()
        if st == FlightState.estopped and cmd in ("takeoff", "land", "resume", "confirm_touchdown"):
            return no("estopped")

        if cmd == "land":
            self._begin_landing(f"{source}_land", safety=False, now=now)
            return ok()
        if cmd == "takeoff":
            if self._shutdown:
                return no("shutting_down")
            if st != FlightState.on_ground:
                return no("not_on_ground")
            if not self._armed:
                return no("not_armed")
            if self.cfg.takeoff.require_heartbeat and not self._hb_bypass() and self._hb_age(now) >= \
                    self.cfg.heartbeat.neutral_after_s * 1000.0:
                return no("no_heartbeat")
            if self._frame_lag(now) >= self.cfg.video.neutral_lag_s * 1000.0:
                return no("no_video")
            self._set_state(FlightState.taking_off, now)
            self._takeoff_left = self.cfg.takeoff.packets
            self._timer_land_fired = False
            self._timer_warned = False
            return ok()
        if cmd == "confirm_touchdown":
            if st in (FlightState.landing, FlightState.taking_off, FlightState.airborne):
                self._set_state(FlightState.on_ground, now)
                return ok()
            return no("not_flying")
        if cmd == "pause":
            self._ap_paused = True
            return ok()
        if cmd == "resume":
            if st == FlightState.landing:
                # Explicit cancel of a (safety or human) landing. The autopilot stays paused; a second
                # resume restarts it. A landing cause that still holds re-triggers on the next tick.
                self._set_state(FlightState.airborne, now)
                self._ap_paused = True
                return ok()
            self._ap_paused = False
            self._wrong_way_hold = False
            if self._ap.mode == AutopilotState.give_up and self._mission is not None:
                self._ap = autopilot_start(self._ap, self._mission.mission_id, now,
                                           distance=self._mission.intent.distance,
                                           category=self._mission.intent.category)
            return ok()
        if cmd == "arm":
            if self.flight_mode == "real" and (arm_phrase or "").strip() != self.cfg.arming.phrase:
                return no("wrong_arm_phrase")
            self._armed = True
            return ok()
        if cmd == "disarm":
            self._armed = False
            return ok()
        if cmd in ("tilt_up", "tilt_down", "tilt_stop"):
            self._tilt_dir = {"tilt_up": 1, "tilt_down": -1, "tilt_stop": 0}[cmd]
            self._tilt_until_ms = now + self.cfg.tilt.max_hold_s * 1000.0 if self._tilt_dir else -INF
            return ok()
        if cmd == "pack_swap":
            if st != FlightState.on_ground:
                return no("not_on_ground")
            self.flight_timer.swap_pack(pack_id)
            return ok()
        return no("unhandled")

    def _set_state(self, new: FlightState, now: float) -> None:
        if new != self._state:
            self.events.append({"kind": "flight_state", "from": self._state.value, "to": new.value,
                                "cause": self._land_cause if new == FlightState.landing else None,
                                "ts_ms": now})
        self._state = new
        self._state_since = now
        if new != FlightState.taking_off:
            self._takeoff_left = 0
        if new != FlightState.landing:
            self._land_cause = None
            self._land_is_safety = False

    def _begin_landing(self, cause: str, *, safety: bool, now: float) -> None:
        if self._state == FlightState.estopped:
            return
        self._land_cause = cause
        self._land_is_safety = safety
        if self._state != FlightState.landing:
            self._land_last_ms = -INF               # first land flag goes out on the next tick
            self._set_state(FlightState.landing, now)
            self._land_cause = cause
            self._land_is_safety = safety
        self._ap_paused = True
        if safety:
            logger.warning("[governor] SAFETY LANDING: %s", cause)

    def _release_human(self) -> None:
        self._human_target = dict(ZERO)
        self._human_last_ms = None
        self._human_ramp = dict(ZERO)

    # ==========================================================================================
    # Ages (lock held)

    def _hb_bypass(self) -> bool:
        return self.flight_mode == "twin" and self.cfg.twin_headless_bypass

    def _hb_age(self, now: float) -> float:
        return INF if self._hb_last_ms is None else now - self._hb_last_ms

    def _frame_age(self, now: float) -> float:
        return INF if self._frame_capture_ms is None else now - self._frame_capture_ms

    def _frame_lag(self, now: float) -> float:
        """Frame age beyond the best-case pipeline latency of the last 10 s (capped)."""
        age = self._frame_age(now)
        floor = self._frame_floor.value(now) or 0.0
        return age - min(floor, self.cfg.video.floor_cap_s * 1000.0)

    def _obs_age(self, now: float) -> float:
        return INF if self._obs is None else now - self._obs.capture_ms

    def _obs_lag(self, now: float) -> float:
        age = self._obs_age(now)
        floor = self._obs_floor.value(now) or 0.0
        return age - min(floor, self.cfg.observation.floor_cap_s * 1000.0)

    def _obs_stale_limit(self) -> float:
        oc = self.cfg.observation
        period = self._obs_period.period_ms or 0.0
        return max(oc.stale_min_s * 1000.0, oc.stale_periods * period)

    # ==========================================================================================
    # The tick (RC thread)

    def tick(self, now: Optional[float] = None) -> TickOutput:
        t0 = time.perf_counter()
        now = self._clock() if now is None else now
        with self._lock:
            out, rec = self._tick_locked(now)
        us = (time.perf_counter() - t0) * 1e6
        self._tick_us.append(us)
        self.tick_count += 1
        self._fault_since = None
        self.last_tick_ts = now
        rec["tick_us"] = round(us, 1)
        self.tick_log.record(rec)
        return out

    def _tick_locked(self, now: float) -> tuple[TickOutput, dict]:
        cfg = self.cfg
        dt = 0.0 if self._last_tick_ms is None else max(0.0, min(0.5, (now - self._last_tick_ms) / 1000.0))
        self._last_tick_ms = now
        flags: list[str] = []

        hb_age = self._hb_age(now)
        hb_bypass = self._hb_bypass()
        hb_neutral = not hb_bypass and hb_age >= cfg.heartbeat.neutral_after_s * 1000.0
        hb_land = not hb_bypass and hb_age >= cfg.heartbeat.land_after_s * 1000.0
        frame_lag = self._frame_lag(now)
        video_neutral = frame_lag >= cfg.video.neutral_lag_s * 1000.0
        video_land = frame_lag >= cfg.video.land_lag_s * 1000.0
        obs_lag = self._obs_lag(now)
        obs_stale = obs_lag > self._obs_stale_limit()
        human_active = (self._human_last_ms is not None
                        and now - self._human_last_ms <= cfg.human.stale_s * 1000.0)
        if not human_active and self._human_last_ms is not None:
            self._release_human()                  # stale: counts as zero from now on
        if hb_neutral:
            flags.append("heartbeat_missing")
        if video_neutral:
            flags.append("video_stale")
        if self._ap.mode != AutopilotState.idle and obs_stale:
            flags.append("obs_stale")

        # 1-2. Flight state timers.
        st = self._state
        if st == FlightState.taking_off and now - self._state_since >= cfg.takeoff.hold_s * 1000.0:
            self._set_state(FlightState.airborne, now)
            st = self._state
        flying = st in (FlightState.taking_off, FlightState.airborne, FlightState.landing)
        if flying:
            self.flight_timer.add_flight_time(dt)
        flight_s = self.flight_timer.flight_s()
        if flight_s >= cfg.flight_timer.warn_s:
            flags.append("flight_timer_warn")
            if flying and not self._timer_warned:
                self._timer_warned = True
                self.events.append({"kind": "safety", "text": "battery timer warning", "ts_ms": now})

        # 3. Safety landings (only while taking off or airborne).
        if st in (FlightState.taking_off, FlightState.airborne) and not self._shutdown:
            cause = None
            if hb_land:
                cause = "heartbeat_lost"
            elif video_land:
                cause = "video_frozen"
            elif flight_s >= cfg.flight_timer.land_s and not self._timer_land_fired:
                self._timer_land_fired = True
                cause = "flight_timer"
            if cause:
                self._begin_landing(cause, safety=True, now=now)
                self.events.append({"kind": "safety", "text": f"safety landing: {cause}", "ts_ms": now})
                st = self._state

        # 4. Autopilot step (every tick while a mission is active, so its timers keep running).
        ap_allowed = (st == FlightState.airborne and not self._shutdown and not hb_neutral
                      and not human_active and not self._ap_paused and not self._wrong_way_hold
                      and not video_neutral and not obs_stale and now >= self._ap_fault_until)
        ap_sticks = dict(ZERO)
        if self._ap.mode != AutopilotState.idle:
            ap_sticks = self._step_autopilot(now, ap_allowed, flags)
            if self._ap.mode == AutopilotState.follow and cfg.wrong_way.enabled and not self._wrong_way_hold:
                if self._wrong_way_check():
                    self._wrong_way_hold = True
                    self.events.append({"kind": "safety", "text": "wrong-way: hovering", "ts_ms": now})
                    logger.error("[governor] WRONG-WAY detector fired: hovering")
        if self._wrong_way_hold:
            flags.append("wrong_way")
        if self._ap_paused and self._ap.mode != AutopilotState.idle:
            flags.append("autopilot_paused")
        if now < self._ap_fault_until:
            flags.append("autopilot_fault")

        # 5. Sticks by priority.
        human_sticks = self._human_sticks(dt) if human_active else dict(ZERO)
        sticks, source, caps, compensate = dict(ZERO), "none", cfg.caps.autopilot, False
        if self._shutdown:
            source = "shutdown"
        elif st == FlightState.estopped:
            source = "estop"
        elif st == FlightState.landing:
            source = "landing"
        elif st == FlightState.taking_off:
            source = "takeoff"
        elif st == FlightState.airborne and not hb_neutral:
            if human_active:
                sticks, source, caps = human_sticks, "human", cfg.caps.human
            elif ap_allowed and self._ap.mode != AutopilotState.idle:
                sticks, source, compensate = self._slew_limit(ap_sticks, dt), "autopilot", True
        if source != "autopilot":
            self._slew = dict(ZERO)            # safety neutral and handover bypass the slew limiter
        wire = to_wire(sticks, cfg.wire, caps, compensate) if source in ("human", "autopilot") else neutral_wire()

        # 6. Flags.
        takeoff = land = stop = False
        if self._shutdown:
            land = True
            stop = self._estop_latched
        elif st == FlightState.estopped:
            stop = True
        elif st == FlightState.taking_off and self._takeoff_left > 0:
            takeoff = True
            self._takeoff_left -= 1
        elif st == FlightState.landing:
            if now - self._land_last_ms >= 1000.0 / cfg.landing.reassert_per_s - 1.0:
                land = True
                self._land_last_ms = now
        tilt = 0
        if st != FlightState.estopped and not self._shutdown and now < self._tilt_until_ms:
            tilt = 2 if self._tilt_dir > 0 else (1 if self._tilt_dir < 0 else 0)

        # Bookkeeping for dead reckoning and the wrong-way check: what was actually sent.
        self._applied = self._effective_normalized(wire)
        self._yaw_hist.append((now, self._applied["yaw"] * self.ap_params.plant.yaw_rate_per_count * 127.0))

        out = TickOutput(wire=wire, takeoff=takeoff, land=land, stop=stop, tilt_state=tilt,
                         flight_state=st, source=source, safety_flags=tuple(flags))
        self._snapshot = _Snapshot(
            ts_ms=now, flight_state=st, autopilot_state=self._ap.mode, armed=self._armed, wire=wire,
            by_source={"human": human_sticks, "autopilot": ap_sticks,
                       "output": {a: round(self._applied[a], 4) for a in AXES}},
            source=source, frame_age_ms=_finite(self._frame_age(now)), obs_age_ms=_finite(self._obs_age(now)),
            latency_ms={k: v for k, v in (
                ("frame_lag", _finite(frame_lag)), ("obs_lag", _finite(obs_lag)),
                ("frame_floor", _finite(self._frame_floor.value())), ("obs_floor", _finite(self._obs_floor.value())),
                ("obs_period", _finite(self._obs_period.period_ms)), ("hb_age", _finite(hb_age)),
                ("autopilot_step", round(self._ap_step_ms, 3)),
            ) if v is not None},
            flight_time_s=round(flight_s, 1), pack_id=self.flight_timer.current_pack,
            safety_flags=tuple(flags) + ((f"landing:{self._land_cause}",) if st == FlightState.landing else ()),
        )
        rec = {
            "t": round(now, 1), "fs": st.value, "ap": self._ap.mode.value, "src": source,
            "w": [wire["throttle"], wire["yaw"], wire["pitch"], wire["roll"]],
            "to": int(takeoff), "ld": int(land), "es": int(stop), "tl": tilt,
            "fl": flags, "obs_lag": _finite(obs_lag), "frame_lag": _finite(frame_lag),
            "hb": _finite(hb_age), "brg": None if self._ap.dbg_bearing is None else round(self._ap.dbg_bearing, 4),
            "es_size": None if self._ap.dbg_e_s is None else round(self._ap.dbg_e_s, 4), "gate": self._ap.dbg_gate,
        }
        return out, rec

    # ------------------------------------------------------------------------------------------
    def _step_autopilot(self, now: float, can_move: bool, flags: list) -> dict:
        t0 = time.perf_counter()
        try:
            sticks, new_state, events = autopilot_step(
                self._obs, self._ap, self.ap_params, now, can_move=can_move,
                applied_yaw=self._applied["yaw"], applied_pitch=self._applied["pitch"])
        except Exception:
            logger.exception("[governor] autopilot step failed: autopilot output neutral")
            self._ap_fault_until = now + self.cfg.autopilot.fault_neutral_s * 1000.0
            return dict(ZERO)
        self._ap_step_ms = (time.perf_counter() - t0) * 1000.0
        if self._ap_step_ms > self.cfg.autopilot.max_step_ms:
            flags.append("autopilot_slow")
        try:
            clean = {a: float(sticks.get(a, 0.0)) for a in AXES}
            valid = all(math.isfinite(v) for v in clean.values())
        except (TypeError, ValueError):
            valid = False
        if not valid:
            logger.error("[governor] autopilot returned invalid sticks %s: neutral", sticks)
            self._ap_fault_until = now + self.cfg.autopilot.fault_neutral_s * 1000.0
            return dict(ZERO)
        self._ap = new_state
        for ev in events:
            self.events.append(ev)
        self._ap_last_sticks = clean
        return clean

    def _human_sticks(self, dt: float) -> dict:
        if self._human_mode == "abs":
            return dict(self._human_target)
        out = {}
        for a in AXES:
            d = self._human_target[a]
            limit = self.cfg.caps.human.get(a) / float(self.cfg.wire.full_scale_counts)
            r = self._human_ramp[a]
            if d == 0.0:
                r = 0.0
            else:
                if r * d < 0:
                    r = 0.0                       # reversal: through zero, no "boost"
                r += math.copysign(self.cfg.human.inc_ramp_per_s * dt, d)
                r = max(-limit * abs(d), min(limit * abs(d), r))
            self._human_ramp[a] = r
            out[a] = r
        return out

    def _slew_limit(self, target: dict, dt: float) -> dict:
        """Magnitude increases are limited to slew.autopilot_per_s; moving toward zero is immediate."""
        step = self.cfg.slew.autopilot_per_s * dt
        out = {}
        for a in AXES:
            cur, tgt = self._slew[a], target.get(a, 0.0)
            if cur * tgt < 0:
                cur = 0.0                           # sign change: drop to zero first
            if abs(tgt) <= abs(cur):
                cur = tgt                           # slowing down is never limited
            else:
                cur = cur + max(-step, min(step, tgt - cur))
            self._slew[a] = cur
            out[a] = cur
        return out

    def _effective_normalized(self, wire: dict) -> dict:
        """Wire counts back to normalized, minus the firmware deadband (what actually moves the drone)."""
        w = self.cfg.wire
        out = {}
        for a in AXES:
            off = (wire[a] - CENTER) * w.sign.get(a)
            d = w.deadband_counts.get(a)
            eff = math.copysign(max(0.0, abs(off) - d), off) if off else 0.0
            out[a] = eff / float(w.full_scale_counts)
        return out

    def _wrong_way_check(self) -> bool:
        """PLAN 4.5 step 5: observed bearing change over 1.5 theta vs. the change predicted from
        the yaw we actually commanded (turning right moves the subject left: d(beta) = -yaw_rate dt).
        Fires when both exceed min_change_rad with opposite signs."""
        if len(self._obs_hist) < 2:
            return False
        ww = self.cfg.wrong_way
        theta_ms = self.ap_params.delay.theta_s * 1000.0
        t2, b2 = self._obs_hist[-1]
        t1_target = t2 - ww.window_thetas * theta_ms
        first = None
        for t, b in self._obs_hist:
            if t <= t1_target:
                first = (t, b)
            else:
                break
        if first is None:
            return False
        t1, b1 = first
        observed = b2 - b1
        lag = self.ap_params.plant.yaw_lag_s * 1000.0
        a, b = t1 - lag, t2 - lag
        predicted = 0.0
        prev_t = None
        for t, rate in self._yaw_hist:
            if prev_t is not None and t > a and prev_t < b:
                seg = (min(t, b) - max(prev_t, a)) / 1000.0
                if seg > 0:
                    predicted -= rate * seg
            prev_t = t
        return (abs(observed) > ww.min_change_rad and abs(predicted) > ww.min_change_rad
                and observed * predicted < 0)

    # ==========================================================================================
    # Read-only views (any thread)

    @property
    def state(self) -> FlightState:
        return self._state

    @property
    def estop_latched(self) -> bool:
        return self._estop_latched

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def mission(self) -> Optional[Mission]:
        return self._mission

    @property
    def autopilot(self) -> AutopilotRun:
        return self._ap

    @property
    def latest_obs(self) -> Optional[SubjectObs]:
        return self._obs

    def snapshot(self) -> _Snapshot:
        return self._snapshot

    def tick_stats(self) -> dict:
        vals = sorted(self._tick_us)
        if not vals:
            return {"n": 0}
        pick = lambda q: vals[min(len(vals) - 1, int(q * len(vals)))]  # noqa: E731
        return {"n": len(vals), "p50_us": round(pick(0.5), 1), "p99_us": round(pick(0.99), 1),
                "max_us": round(vals[-1], 1), "faults": self.fault_count}

    def drain_events(self) -> list:
        out = []
        while self.events:
            try:
                out.append(self.events.popleft())
            except IndexError:
                break
        return out

    # ==========================================================================================
    # Fault handling and housekeeping

    def note_fault(self, now: Optional[float] = None) -> bool:
        """Called by the strategy fallback after a tick exception. Returns True if the fallback
        should also assert land (faults have persisted for faults.land_after_s)."""
        now = self._clock() if now is None else now
        self.fault_count += 1
        if self._fault_since is None:
            self._fault_since = now
        return (now - self._fault_since) >= self.cfg.faults.land_after_s * 1000.0

    def start_background(self) -> None:
        """Housekeeping thread: persists the flight timer (file I/O never happens on the RC thread)."""
        if self._hk_thread and self._hk_thread.is_alive():
            return
        self._hk_stop.clear()
        self._hk_thread = threading.Thread(target=self._housekeeping, name="GovernorHousekeeping", daemon=True)
        self._hk_thread.start()

    def _housekeeping(self) -> None:
        while not self._hk_stop.wait(1.0):
            self.flight_timer.save_if_dirty()
            last = self.last_tick_ts
            if last is not None and self._clock() - last > 500.0:
                logger.error("[governor] RC tick is %.0f ms old: RC thread stalled?", self._clock() - last)

    def close(self) -> None:
        self._hk_stop.set()
        if self._hk_thread:
            self._hk_thread.join(timeout=2.0)
        self.flight_timer.save_if_dirty()
        self.tick_log.close()
