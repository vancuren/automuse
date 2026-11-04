"""100 Hz kinematic model of the X69, driven by decoded RC packets (PLAN.md section 4.2).

World frame (ENU, meters): x east, y north, z up. Yaw psi is the compass heading in radians,
clockwise from north seen from above, so +yaw stick (nose right) increases psi. The drone's
forward unit vector is (sin psi, cos psi) and its right unit vector is (cos psi, -sin psi).

Model, per axis:
- sticks pass a per-axis firmware deadband, then scale per count beyond it;
- yaw rate, forward/right speed, and climb rate follow first-order lags toward the command,
  with quadratic drag on horizontal speed; with throttle centered the altitude holds;
- the airframe tilts by atan(a_thrust / g) (with a short lag), so the camera pitches down when
  accelerating forward and rolls when accelerating sideways;
- headless mode interprets pitch/roll relative to the heading at takeoff;
- flags: takeoff spools up and climbs to the hover height, land descends and disarms,
  E-stop cuts the motors (free fall), flip dips altitude briefly;
- optical-flow hold drifts: an Ornstein-Uhlenbeck random walk on velocity, stronger over low texture;
- camera tilt servo moves per press (rising edge of a tilt bit) or while held;
- link loss after loss_timeout_s without RC control packets (keepalives never count);
- battery timer: lands when the pack is empty and refuses takeoff afterwards.

RcFaultInjector (packet loss, burst loss, order-preserving jitter) and RcLinkStats
(inter-arrival p50/p99 with a 60 ms alarm) live here too; the server applies them on the UDP path.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from . import protocol as P

G = 9.81
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "twin" / "x69.yaml"


def load_params(path: Optional[str | Path] = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _lag(value: float, target: float, tau: float, dt: float) -> float:
    """Exact discrete first-order lag step."""
    if tau <= 0:
        return target
    return target + (value - target) * math.exp(-dt / tau)


def _counts_beyond_deadband(count: int, deadband: int, invert: bool) -> float:
    """Signed counts beyond the deadband (0 inside it)."""
    off = int(count) - P.CENTER
    if invert:
        off = -off
    mag = abs(off) - deadband
    if mag <= 0:
        return 0.0
    return math.copysign(mag, off)


# Flight phases of the simulated drone (not the governor's flight state).
LANDED = "landed"
SPOOLING = "spooling"      # takeoff accepted, motors spinning up
TAKING_OFF = "taking_off"
FLYING = "flying"
LANDING = "landing"
FALLING = "falling"        # motors cut in the air (E-stop or link 'drop')
PHASES = (LANDED, SPOOLING, TAKING_OFF, FLYING, LANDING, FALLING)


@dataclass
class DroneState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0              # rad, clockwise from north
    vx: float = 0.0               # world m/s
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0         # rad/s
    tilt_fwd: float = 0.0         # rad, + = nose down
    tilt_right: float = 0.0       # rad, + = right side down
    cam_tilt: float = 0.0         # servo angle, rad, + = up
    phase: str = LANDED
    motors_on: bool = False
    headless: bool = False
    headless_ref_yaw: float = 0.0
    flight_time_s: float = 0.0    # motors-on time on the current pack
    battery_empty: bool = False
    link_lost: bool = False
    landing_reason: Optional[str] = None
    phase_t: float = 0.0          # time spent in the current phase
    flip_t: Optional[float] = None
    collisions: int = 0


@dataclass
class EventLog:
    """Counts of RC events, for /status and tests."""
    takeoff: int = 0
    land: int = 0
    estop: int = 0
    calibrate: int = 0
    flip: int = 0
    tilt_presses: int = 0
    ignored_takeoff: int = 0
    unknown_flag_packets: int = 0
    link_losses: int = 0
    last_events: deque = field(default_factory=lambda: deque(maxlen=50))

    def add(self, t: float, name: str, detail: str = "") -> None:
        self.last_events.append({"t": round(t, 3), "event": name, "detail": detail})


class X69Dynamics:
    """Kinematic X69. Feed packets with on_packet(), advance with step(dt) or advance_to(t)."""

    def __init__(self, params: Optional[dict] = None, seed: int = 0) -> None:
        self.p = params or load_params()
        self.rng = random.Random(seed)
        self.dt = 1.0 / float(self.p.get("rate_hz", 100))
        self.t = 0.0
        self.s = DroneState()
        self.events = EventLog()
        self.sticks = {"roll": P.CENTER, "pitch": P.CENTER, "throttle": P.CENTER, "yaw": P.CENTER}
        self.last_rc_t: Optional[float] = None
        self.last_flags2 = 0
        self.tilt_dir = 0
        self.drift_v = [0.0, 0.0]
        self.drift_yaw = 0.0
        self.drift_scale = 1.0          # TwinControl.drift multiplier (0 = no drift)
        self.texture = 1.0              # ground texture quality 0..1 (from the scenario)
        self.bounds: Optional[tuple[float, float, float, float]] = None   # xmin, ymin, xmax, ymax
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, x: float = 0.0, y: float = 0.0, yaw_deg: float = 0.0) -> None:
        cam = self.p["camera"]
        self.s = DroneState(x=x, y=y, yaw=math.radians(yaw_deg),
                            cam_tilt=math.radians(cam.get("tilt_init_deg", 0.0)),
                            headless=bool(self.p.get("headless", {}).get("enabled", False)))
        self.sticks = {k: P.CENTER for k in self.sticks}
        self.last_flags2 = 0
        self.tilt_dir = 0
        self.drift_v = [0.0, 0.0]
        self.drift_yaw = 0.0
        self.last_rc_t = None
        self.events = EventLog()

    def set_phase(self, phase: str, reason: Optional[str] = None) -> None:
        if phase != self.s.phase:
            self.events.add(self.t, "phase", f"{self.s.phase}->{phase}" + (f" ({reason})" if reason else ""))
        self.s.phase = phase
        self.s.phase_t = 0.0
        if phase == LANDING:
            self.s.landing_reason = reason

    # ------------------------------------------------------------------ input
    def on_packet(self, pkt: P.RcPacket, t: Optional[float] = None) -> None:
        """Apply one decoded control packet received at sim time t (default: now)."""
        t = self.t if t is None else t
        self.last_rc_t = t
        if self.s.link_lost:
            self.s.link_lost = False
            self.events.add(self.t, "link_restored")
            if self.s.phase == LANDING and self.s.landing_reason == "link_loss" and \
                    self.p["link"].get("resume_on_relink", False):
                self.set_phase(FLYING, "relink")
        self.sticks = pkt.sticks()
        if pkt.unknown_flags1 or pkt.unknown_flags2:
            self.events.unknown_flag_packets += 1
        hb = self.p.get("headless", {}).get("flag1_bit")
        if hb and (pkt.flags1 & int(hb)):
            self.s.headless = not self.s.headless
            self.s.headless_ref_yaw = self.s.yaw
        # Order matters: E-stop wins over everything in the same packet.
        if pkt.estop:
            self._estop()
        elif pkt.land:
            self._land("flag")
        elif pkt.takeoff:
            self._takeoff()
        if pkt.calibrate:
            self.events.calibrate += 1
            self.events.add(self.t, "calibrate", "ignored" if self.s.motors_on else "ok")
        if pkt.flip and self.s.phase == FLYING and self.s.flip_t is None:
            self.events.flip += 1
            self.s.flip_t = 0.0
            self.events.add(self.t, "flip")
        self._tilt_input(pkt)
        self.last_flags2 = pkt.flags2

    def _tilt_input(self, pkt: P.RcPacket) -> None:
        cam = self.p["camera"]
        up, down = pkt.tilt_up, pkt.tilt_down
        if cam.get("tilt_mode", "per_press") == "per_press":
            rising_up = up and not (self.last_flags2 & P.FLAG2_TILT_UP)
            rising_down = down and not (self.last_flags2 & P.FLAG2_TILT_DOWN)
            step = math.radians(cam.get("tilt_step_deg", 6.0))
            if rising_up:
                self._move_tilt(+step)
            if rising_down:
                self._move_tilt(-step)
            self.tilt_dir = 0
        else:
            self.tilt_dir = (1 if up else 0) - (1 if down else 0)
            if (up and not (self.last_flags2 & P.FLAG2_TILT_UP)) or \
                    (down and not (self.last_flags2 & P.FLAG2_TILT_DOWN)):
                self.events.tilt_presses += 1

    def _move_tilt(self, delta: float, count: bool = True) -> None:
        cam = self.p["camera"]
        lo, hi = math.radians(cam["tilt_min_deg"]), math.radians(cam["tilt_max_deg"])
        self.s.cam_tilt = max(lo, min(hi, self.s.cam_tilt + delta))
        if count:
            self.events.tilt_presses += 1

    def _takeoff(self) -> None:
        if self.s.phase != LANDED:
            return     # repeated takeoff flags while airborne are no-ops
        if self.s.battery_empty:
            self.events.ignored_takeoff += 1
            self.events.add(self.t, "takeoff_refused", "battery empty")
            return
        self.events.takeoff += 1
        self.s.motors_on = True
        self.s.headless_ref_yaw = self.s.yaw
        self.set_phase(SPOOLING)

    def _land(self, reason: str) -> None:
        if self.s.phase in (SPOOLING,):
            self.events.land += 1
            self.s.motors_on = False
            self.set_phase(LANDED, reason)
            return
        if self.s.phase in (TAKING_OFF, FLYING):
            self.events.land += 1
            self.set_phase(LANDING, reason)
        # land while landed, landing, or falling: no-op

    def _estop(self) -> None:
        self.events.estop += 1
        if not self.s.motors_on and self.s.phase in (LANDED, FALLING):
            return
        self.s.motors_on = False
        falls = self.p.get("estop", {}).get("falls", True)
        if self.s.z > 0.01 and falls:
            self.set_phase(FALLING, "estop")
        else:
            self._touchdown("estop")
        self.events.add(self.t, "estop")

    def _touchdown(self, reason: str) -> None:
        s = self.s
        s.z = 0.0
        s.vx = s.vy = s.vz = 0.0
        s.yaw_rate = 0.0
        s.tilt_fwd = s.tilt_right = 0.0
        s.motors_on = False
        s.flip_t = None
        self.set_phase(LANDED, reason)

    # ------------------------------------------------------------------ simulation
    def advance_to(self, t: float) -> None:
        """Step at the fixed rate until sim time reaches t (wall-clock friendly)."""
        steps = 0
        while self.t + self.dt <= t + 1e-9:
            self.step(self.dt)
            steps += 1
            if steps > 1000:          # never spin more than 10 s of catch-up in one call
                self.t = t
                break

    def step(self, dt: float) -> None:
        p, s = self.p, self.s
        self.t += dt
        s.phase_t += dt

        # Link loss: only RC control packets count; the keepalive thread can outlive the RC thread.
        timeout = float(p["link"]["loss_timeout_s"])
        if s.motors_on and not s.link_lost and self.last_rc_t is not None and \
                self.t - self.last_rc_t > timeout:
            s.link_lost = True
            self.events.link_losses += 1
            self.events.add(self.t, "link_lost")
            action = p["link"].get("loss_action", "land")
            if action == "land":
                self._land("link_loss")
            elif action == "drop":
                self._estop()

        # Battery timer.
        bat = p.get("battery", {})
        if s.motors_on:
            s.flight_time_s += dt
            if bat.get("enabled", True) and not s.battery_empty and \
                    s.flight_time_s >= float(bat.get("flight_time_s", 480)):
                s.battery_empty = True
                self.events.add(self.t, "battery_empty")
                self._land("battery")

        # Camera tilt while held.
        if self.tilt_dir:
            rate = math.radians(p["camera"].get("tilt_rate_dps", 30.0))
            self._move_tilt(self.tilt_dir * rate * dt, count=False)

        # Effective stick commands (counts beyond deadband), only while flying with a live link.
        db, inv = p["sticks"]["deadband_counts"], p["sticks"]["invert"]
        # With a lost link the sticks read as centered (hover); 'land' and 'drop' changed the phase above.
        live = s.phase == FLYING and not s.link_lost
        c = {ax: (_counts_beyond_deadband(self.sticks[ax], int(db[ax]), bool(inv[ax])) if live else 0.0)
             for ax in ("roll", "pitch", "throttle", "yaw")}

        # Yaw.
        yaw_cmd = math.radians(p["yaw"]["rate_per_count_dps"]) * c["yaw"]
        if s.motors_on:
            self.drift_yaw = self._ou(self.drift_yaw,
                                      math.radians(p["drift"]["yaw_sigma_dps"]) * self.drift_scale,
                                      p["drift"]["tau_s"], dt)
            yaw_cmd += self.drift_yaw if s.phase != LANDED else 0.0
        if s.phase in (FLYING, TAKING_OFF, LANDING):
            s.yaw_rate = _lag(s.yaw_rate, yaw_cmd, p["yaw"]["tau_s"], dt)
        else:
            s.yaw_rate = _lag(s.yaw_rate, 0.0, p["yaw"]["tau_s"], dt)
        s.yaw = (s.yaw + s.yaw_rate * dt) % (2 * math.pi)

        # Horizontal velocity in the stick frame (body, or takeoff heading in headless mode).
        ref_yaw = s.headless_ref_yaw if s.headless else s.yaw
        fwd = (math.sin(ref_yaw), math.cos(ref_yaw))
        right = (math.cos(ref_yaw), -math.sin(ref_yaw))
        v_fwd = s.vx * fwd[0] + s.vy * fwd[1]
        v_right = s.vx * right[0] + s.vy * right[1]
        cmd_fwd = p["pitch"]["speed_per_count_mps"] * c["pitch"]
        cmd_right = p["roll"]["speed_per_count_mps"] * c["roll"]
        airborne = s.phase in (TAKING_OFF, FLYING, LANDING) and s.z > 0.02
        if airborne:
            # Optical-flow drift (a velocity random walk), stronger over poor texture.
            gain = 1.0 + (float(p["drift"]["low_texture_gain"]) - 1.0) * (1.0 - max(0.0, min(1.0, self.texture)))
            sigma = float(p["drift"]["sigma_mps"]) * gain * self.drift_scale
            self.drift_v[0] = self._ou(self.drift_v[0], sigma, p["drift"]["tau_s"], dt)
            self.drift_v[1] = self._ou(self.drift_v[1], sigma, p["drift"]["tau_s"], dt)
            d_fwd = self.drift_v[0] * fwd[0] + self.drift_v[1] * fwd[1]
            d_right = self.drift_v[0] * right[0] + self.drift_v[1] * right[1]
            new_fwd = _lag(v_fwd, cmd_fwd + d_fwd, p["pitch"]["tau_s"], dt)
            new_right = _lag(v_right, cmd_right + d_right, p["roll"]["tau_s"], dt)
            k = float(p["drag"]["quadratic_per_m"])
            new_fwd -= k * new_fwd * abs(new_fwd) * dt
            new_right -= k * new_right * abs(new_right) * dt
            # Thrust acceleration = net acceleration + drag compensation; the airframe tilts by atan(a/g).
            a_fwd = (new_fwd - v_fwd) / dt + k * new_fwd * abs(new_fwd)
            a_right = (new_right - v_right) / dt + k * new_right * abs(new_right)
            max_tilt = math.radians(p["attitude"]["max_tilt_deg"])
            tf = max(-max_tilt, min(max_tilt, math.atan2(a_fwd, G)))
            tr = max(-max_tilt, min(max_tilt, math.atan2(a_right, G)))
            s.tilt_fwd = _lag(s.tilt_fwd, tf, p["attitude"]["tau_s"], dt)
            s.tilt_right = _lag(s.tilt_right, tr, p["attitude"]["tau_s"], dt)
            v_fwd, v_right = new_fwd, new_right
        elif s.phase == FALLING:
            # Coasts horizontally while falling (no thrust, no tilt control).
            pass
        else:
            v_fwd = v_right = 0.0
            s.tilt_fwd = _lag(s.tilt_fwd, 0.0, p["attitude"]["tau_s"], dt)
            s.tilt_right = _lag(s.tilt_right, 0.0, p["attitude"]["tau_s"], dt)
        s.vx = v_fwd * fwd[0] + v_right * right[0]
        s.vy = v_fwd * fwd[1] + v_right * right[1]

        # Vertical.
        alt = p["altitude"]
        if s.phase == LANDED:
            s.vz = 0.0
            s.z = 0.0
        elif s.phase == SPOOLING:
            s.vz = 0.0
            if s.phase_t >= float(alt.get("takeoff_spool_s", 0.5)):
                self.set_phase(TAKING_OFF)
        elif s.phase == TAKING_OFF:
            s.vz = _lag(s.vz, float(alt["takeoff_climb_mps"]), 0.15, dt)
            if s.z >= float(alt["hover_height_m"]):
                s.z = float(alt["hover_height_m"])
                s.vz = 0.0
                self.set_phase(FLYING)
        elif s.phase == FLYING:
            vz_cmd = p["throttle"]["climb_per_count_mps"] * c["throttle"]
            s.vz = _lag(s.vz, vz_cmd, p["throttle"]["tau_s"], dt)
            if s.z + s.vz * dt < float(alt["floor_m"]) and s.vz < 0:
                s.vz = 0.0
            if s.z + s.vz * dt > float(alt["ceiling_m"]) and s.vz > 0:
                s.vz = 0.0
            # Flip: a short altitude dip, then recovery to where it was.
            if s.flip_t is not None:
                s.flip_t += dt
                dur = float(p["flip"]["duration_s"])
                if s.flip_t < dur:
                    s.vz += -float(p["flip"]["altitude_loss_m"]) * math.pi / dur * math.sin(
                        2 * math.pi * s.flip_t / dur)
                else:
                    s.flip_t = None
        elif s.phase == LANDING:
            s.vz = _lag(s.vz, -float(alt["land_descent_mps"]), 0.2, dt)
        elif s.phase == FALLING:
            s.vz -= G * dt

        s.x += s.vx * dt
        s.y += s.vy * dt
        s.z += s.vz * dt
        if s.z <= 0.0 and s.phase in (LANDING, FALLING) and s.vz <= 0:
            self._touchdown("landed" if s.phase == LANDING else "impact")
        s.z = max(0.0, s.z)

        # Arena walls: stop at the bounds.
        if self.bounds is not None:
            xmin, ymin, xmax, ymax = self.bounds
            hit = False
            if s.x < xmin or s.x > xmax:
                s.x = max(xmin, min(xmax, s.x)); s.vx = 0.0; hit = True
            if s.y < ymin or s.y > ymax:
                s.y = max(ymin, min(ymax, s.y)); s.vy = 0.0; hit = True
            if hit and s.phase != LANDED:
                s.collisions += 1

    def _ou(self, value: float, sigma: float, tau: float, dt: float) -> float:
        """Ornstein-Uhlenbeck step with stationary std sigma and correlation time tau."""
        if sigma <= 0:
            return value * math.exp(-dt / max(tau, 1e-6))
        a = math.exp(-dt / tau)
        return value * a + sigma * math.sqrt(1 - a * a) * self.rng.gauss(0.0, 1.0)

    # ------------------------------------------------------------------ outputs
    def camera_pose(self) -> dict:
        """Camera position and orientation for the renderer and the truth projection."""
        s, cam = self.s, self.p["camera"]
        fwd = (math.sin(s.yaw), math.cos(s.yaw))
        off_f, off_u = float(cam.get("offset_forward_m", 0.0)), float(cam.get("offset_up_m", 0.0))
        return {
            "x": s.x + fwd[0] * off_f,
            "y": s.y + fwd[1] * off_f,
            "z": max(0.03, s.z + 0.05 + off_u),     # the camera sits a few cm above the skids
            "yaw": s.yaw,
            "pitch": s.cam_tilt - s.tilt_fwd,        # rad, + = looking up
            "roll": s.tilt_right,                    # rad, + = right side down
            "hfov_deg": float(cam["hfov_deg"]),
        }

    def truth_dict(self) -> dict[str, float]:
        s = self.s
        return {"x": s.x, "y": s.y, "z": s.z, "yaw": s.yaw, "pitch": -s.tilt_fwd, "roll": s.tilt_right,
                "vx": s.vx, "vy": s.vy, "vz": s.vz, "yaw_rate": s.yaw_rate, "cam_tilt": s.cam_tilt,
                "motors_on": 1.0 if s.motors_on else 0.0}

    def status(self) -> dict:
        s = self.s
        bat = self.p.get("battery", {})
        left = float(bat.get("flight_time_s", 480)) - s.flight_time_s
        return {
            "phase": s.phase, "motors_on": s.motors_on, "link_lost": s.link_lost,
            "headless": s.headless, "x": round(s.x, 3), "y": round(s.y, 3), "z": round(s.z, 3),
            "yaw_deg": round(math.degrees(s.yaw), 2), "yaw_rate_dps": round(math.degrees(s.yaw_rate), 2),
            "vx": round(s.vx, 3), "vy": round(s.vy, 3), "vz": round(s.vz, 3),
            "cam_tilt_deg": round(math.degrees(s.cam_tilt), 2),
            "tilt_fwd_deg": round(math.degrees(s.tilt_fwd), 2),
            "battery_left_s": round(left, 1), "battery_low": left <= float(bat.get("low_warn_s", 60)),
            "battery_empty": s.battery_empty, "landing_reason": s.landing_reason,
            "sticks": dict(self.sticks), "collisions": s.collisions,
            "events": {k: getattr(self.events, k) for k in ("takeoff", "land", "estop", "calibrate", "flip",
                                                             "tilt_presses", "ignored_takeoff",
                                                             "unknown_flag_packets", "link_losses")},
            "recent_events": list(self.events.last_events)[-10:],
        }


class RcFaultInjector:
    """Injects RC packet loss, burst loss (Gilbert-Elliott), and order-preserving jitter.

    admit(t) returns the time at which the packet should be delivered, or None if it is lost.
    """

    def __init__(self, seed: int = 1) -> None:
        self.rng = random.Random(seed)
        self.loss = 0.0              # independent loss probability per packet
        self.burst_rate = 0.0        # probability per packet of entering a burst
        self.burst_len = 5           # mean burst length in packets
        self.jitter_ms = 0.0         # extra delay uniform in [0, jitter_ms]
        self._in_burst = False
        self._last_release = 0.0
        self.dropped = 0
        self.delivered = 0

    def configure(self, loss: Optional[float] = None, burst_rate: Optional[float] = None,
                  burst_len: Optional[int] = None, jitter_ms: Optional[float] = None) -> None:
        if loss is not None:
            self.loss = max(0.0, min(1.0, float(loss)))
        if burst_rate is not None:
            self.burst_rate = max(0.0, min(1.0, float(burst_rate)))
        if burst_len is not None:
            self.burst_len = max(1, int(burst_len))
        if jitter_ms is not None:
            self.jitter_ms = max(0.0, float(jitter_ms))

    def admit(self, t: float) -> Optional[float]:
        if self._in_burst:
            if self.rng.random() < 1.0 / self.burst_len:
                self._in_burst = False
        elif self.burst_rate > 0 and self.rng.random() < self.burst_rate:
            self._in_burst = True
        if self._in_burst or (self.loss > 0 and self.rng.random() < self.loss):
            self.dropped += 1
            return None
        release = t + (self.rng.uniform(0.0, self.jitter_ms) / 1000.0 if self.jitter_ms > 0 else 0.0)
        release = max(release, self._last_release)        # never reorder
        self._last_release = release
        self.delivered += 1
        return release


class RcLinkStats:
    """Inter-arrival statistics of RC control packets, with the PLAN's p99 > 60 ms alarm."""

    def __init__(self, window: int = 500, alarm_ms: float = 60.0) -> None:
        self.arrivals: deque = deque(maxlen=window + 1)
        self.alarm_ms = alarm_ms
        self.count = 0
        self.keepalives = 0
        self.errors: dict[str, int] = {}
        self.last_keepalive_t: Optional[float] = None

    def packet(self, t: float) -> None:
        self.arrivals.append(t)
        self.count += 1

    def keepalive(self, t: float) -> None:
        self.keepalives += 1
        self.last_keepalive_t = t

    def error(self, reason: str) -> None:
        self.errors[reason] = self.errors.get(reason, 0) + 1

    def intervals_ms(self) -> list[float]:
        a = list(self.arrivals)
        return [(a[i + 1] - a[i]) * 1000.0 for i in range(len(a) - 1)]

    @staticmethod
    def _pct(sorted_vals: list[float], q: float) -> Optional[float]:
        if not sorted_vals:
            return None
        idx = min(len(sorted_vals) - 1, max(0, int(math.ceil(q * len(sorted_vals))) - 1))
        return sorted_vals[idx]

    def summary(self) -> dict:
        iv = sorted(self.intervals_ms())
        p50, p99 = self._pct(iv, 0.50), self._pct(iv, 0.99)
        return {
            "packets": self.count, "keepalives": self.keepalives, "errors": dict(self.errors),
            "interarrival_p50_ms": None if p50 is None else round(p50, 2),
            "interarrival_p99_ms": None if p99 is None else round(p99, 2),
            "interarrival_max_ms": None if not iv else round(iv[-1], 2),
            "samples": len(iv),
            "p99_alarm": bool(p99 is not None and len(iv) >= 50 and p99 > self.alarm_ms),
        }
