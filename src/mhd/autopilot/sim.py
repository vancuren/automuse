"""Minimal in-process closed-loop sim: the real Governor (with the autopilot inside) flying a
kinematic X69 that follows a walking subject, observed through a delayed oracle perception.

    $MHD_PY -m mhd.autopilot.sim                      # table at total delays 0.6, 1.0, 1.5 s
    $MHD_PY -m mhd.autopilot.sim --scenario step --delays 1.0 --ff

Plant (100 Hz): wire counts -> firmware deadband -> first-order lags -> yaw rate and forward speed.
Camera (15 fps): pinhole with the autopilot's HFOV at hover height; boxes are clipped to the frame
like a detector's would be. Observations reach the governor `obs_delay` after capture, where
    total delay theta = obs_delay + half the camera period + half the RC period + yaw lag,
the PLAN 4.4 decomposition. The autopilot is told theta (as it will be after Session A/B).
Scenarios: `step` (stationary subject at +20 deg) and `s1` (one walker, ~0.06 rad/s bearing rate,
walking right, pausing, walking back). Metrics use the TRUE bearing from the camera axis.
"""
from __future__ import annotations

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from mhd.contracts import Box, Intent, Mission, SubjectObs

from mhd.governor.config import (AutopilotConfig, GovernorConfig, load_autopilot_config,
                                 load_governor_config, replace)
from mhd.governor.core import Governor
from mhd.governor.wire import CENTER


@dataclass
class SimParams:
    total_delay_s: float = 1.0
    scenario: str = "s1"                 # "s1" | "step" | "fast"
    duration_s: float = 45.0             # after the subject is first locked
    camera_fps: float = 15.0
    rc_hz: float = 25.0
    k_true_scale: float = 1.0            # true yaw rate per count = autopilot's K x this (robustness)
    yaw_tau_s: float = 0.15              # first-order lag on yaw rate (counts in the "yaw lag")
    fwd_tau_s: float = 0.4
    firmware_deadband_counts: int = 0
    cam_height_m: float = 1.2
    subject_w_m: float = 0.45
    subject_h_m: float = 1.7
    start_distance_m: float = 3.2
    noise_px: float = 2.0
    detector_period_s: float = 1.0
    feedforward: bool = False
    seed: int = 1


@dataclass
class SimResult:
    params: SimParams
    bearing_rms_rad: float = math.nan
    bearing_max_rad: float = math.nan
    in_frame_frac: float = math.nan
    overshoot_pct: float = math.nan
    settle_s: float = math.nan
    followed_s: float = 0.0
    final_state: str = ""
    lock_time_s: float = math.nan
    min_distance_m: float = math.nan
    events: list = field(default_factory=list)

    def row(self) -> str:
        p = self.params
        return (f"{p.scenario:5s} theta={p.total_delay_s:.1f}s ff={'on ' if p.feedforward else 'off'} "
                f"K_true={p.k_true_scale:.2f}x | bearing RMS {self.bearing_rms_rad:.3f} rad, "
                f"max {self.bearing_max_rad:.3f}, in-frame {self.in_frame_frac * 100:.0f}%, "
                f"overshoot {self.overshoot_pct:.1f}%, settle {self.settle_s:.1f}s "
                f"(7*theta={7 * p.total_delay_s:.1f}s), min dist {self.min_distance_m:.2f} m, "
                f"final {self.final_state}")


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _subject_path(p: SimParams, t: float) -> tuple[float, float]:
    """Subject position (x east, y north) at time t since the subject started moving."""
    d0 = p.start_distance_m
    if p.scenario == "step":
        ang = math.radians(20.0)
        return d0 * math.sin(ang), d0 * math.cos(ang)
    v = 0.06 * d0 if p.scenario == "s1" else 0.3 * d0     # lateral speed for ~0.06 / 0.3 rad/s
    # walk right 8 s, pause 3 s, walk left 16 s, pause 3 s, walk right 8 s, repeat
    cycle = [(8.0, +1), (3.0, 0), (16.0, -1), (3.0, 0), (8.0, +1)]
    total = sum(c[0] for c in cycle)
    tt = t % total
    x = 0.0
    for dur, sgn in cycle:
        seg = min(tt, dur)
        x += sgn * v * seg
        tt -= seg
        if tt <= 0:
            break
    return x, d0


def run(p: SimParams, gov_cfg: Optional[GovernorConfig] = None,
        ap_cfg: Optional[AutopilotConfig] = None) -> SimResult:
    rng = random.Random(p.seed)
    gcfg = gov_cfg or load_governor_config()
    acfg = ap_cfg or load_autopilot_config()
    acfg = replace(acfg, **{"delay.theta_s": p.total_delay_s, "yaw.feedforward.enabled": p.feedforward})
    gcfg = replace(gcfg, **{"wire.deadband_counts.yaw": 0}) if p.firmware_deadband_counts == 0 else gcfg
    clock = {"t": 0.0}
    # No flight-timer file and no tick log: the sim writes nothing to disk.
    gov = Governor(gcfg, acfg, "twin", clock=lambda: clock["t"], persist_timer=False, tick_log=False)

    W, H = acfg.camera.frame_w, acfg.camera.frame_h
    f = (W / 2) / math.tan(math.radians(acfg.camera.hfov_deg) / 2)
    k_true = acfg.plant.yaw_rate_per_count * p.k_true_scale
    v_per_count = acfg.plant.forward_speed_per_count
    obs_delay_ms = 1000.0 * (p.total_delay_s - 0.5 / p.camera_fps - 0.5 / p.rc_hz - p.yaw_tau_s)
    obs_delay_ms = max(0.0, obs_delay_ms)

    # drone state
    x = y = psi = 0.0
    yaw_rate = fwd_speed = 0.0
    wire = {"throttle": CENTER, "yaw": CENTER, "pitch": CENTER, "roll": CENTER}
    pending: deque = deque()            # (deliver_ms, obs)
    dt_ms = 10.0
    next_rc = 0.0
    next_cam = 0.0
    fid = 0
    phase = "ground"
    mission_at = None
    subject_t0 = None
    lock_t = None
    samples = []                        # (t_since_lock, true bearing, in_frame, distance)

    t_end_guard = 120_000.0
    while clock["t"] < t_end_guard:
        t = clock["t"]
        # --- scenario control
        if phase == "ground" and t >= 500.0:
            gov.command("takeoff")
            phase = "takeoff"
        if phase == "takeoff" and gov.state.value == "AIRBORNE":
            gov.set_mission(Mission(mission_id="sim", intent=Intent(
                action="follow", subject_phrase="person in blue shirt", category="person", color="blue",
                distance="medium", reason_code="none"), transcript="sim", model="sim", latency_ms=0.0))
            mission_at = t
            phase = "mission"
        if phase == "mission" and gov.autopilot.mode.value == "FOLLOW":
            lock_t = t
            subject_t0 = t
            phase = "follow"
        if phase == "follow" and t - lock_t >= p.duration_s * 1000.0:
            break
        if phase == "mission" and mission_at is not None and t - mission_at > 30_000.0:
            break                       # never locked

        # --- subject and true geometry
        st = 0.0 if subject_t0 is None else (t - subject_t0) / 1000.0
        sx, sy = _subject_path(p, st)
        dx, dy = sx - x, sy - y
        dist = math.hypot(dx, dy)
        beta = _wrap(math.atan2(dx, dy) - psi)
        d_fwd = dist * math.cos(beta)
        in_frame = d_fwd > 0.2 and abs(math.tan(beta) * f) < W / 2
        if phase == "follow":
            samples.append(((t - lock_t) / 1000.0, beta, in_frame, dist))

        # --- camera frame -> delayed observation
        if t >= next_cam:
            next_cam += 1000.0 / p.camera_fps
            fid += 1
            gov.note_frame(t, t + obs_delay_ms * 0.6)     # video part of the delay (not critical)
            obs = None
            if in_frame:
                cx = W / 2 + f * math.tan(beta) + rng.gauss(0, p.noise_px)
                bw = f * p.subject_w_m / d_fwd * (1 + rng.gauss(0, 0.02))
                top = H / 2 - f * (p.subject_h_m - p.cam_height_m) / d_fwd
                bot = H / 2 + f * p.cam_height_m / d_fwd
                x1, x2 = max(0.0, cx - bw / 2), min(float(W), cx + bw / 2)
                y1, y2 = max(0.0, top), min(float(H), bot)
                if x2 - x1 > 2 and y2 - y1 > 2:
                    conf_age = (t % (p.detector_period_s * 1000.0))
                    obs = SubjectObs(mission_id="sim", frame_id=fid, capture_ms=t, state="confirmed",
                                     track_id=1, box=Box(x1=x1, y1=y1, x2=x2, y2=y2), confidence=0.9,
                                     tracker_score=0.9, confirm_age_ms=conf_age, source="oracle",
                                     detector_latency_ms=400.0)
            if obs is None:
                obs = SubjectObs(mission_id="sim", frame_id=fid, capture_ms=t, state="searching",
                                 source="oracle", confirm_age_ms=1e9)
            pending.append((t + obs_delay_ms, obs))
        while pending and pending[0][0] <= t:
            _, o = pending.popleft()
            gov.submit_obs(o, receive_ms=t)

        # --- RC tick
        if t >= next_rc:
            next_rc += 1000.0 / p.rc_hz
            gov.heartbeat(t)
            out = gov.tick(t)
            wire = out.wire

        # --- plant (100 Hz)
        def eff(axis):
            off = (wire[axis] - CENTER) * gcfg.wire.sign.get(axis)
            d = p.firmware_deadband_counts
            return math.copysign(max(0, abs(off) - d), off) if off else 0.0
        airborne = gov.state.value in ("AIRBORNE", "TAKING_OFF")
        yaw_cmd = k_true * eff("yaw") if airborne else 0.0
        fwd_cmd = v_per_count * eff("pitch") if airborne else 0.0
        a_y = dt_ms / 1000.0 / p.yaw_tau_s
        a_f = dt_ms / 1000.0 / p.fwd_tau_s
        yaw_rate += (yaw_cmd - yaw_rate) * min(1.0, a_y)
        fwd_speed += (fwd_cmd - fwd_speed) * min(1.0, a_f)
        psi = _wrap(psi + yaw_rate * dt_ms / 1000.0)
        x += fwd_speed * math.sin(psi) * dt_ms / 1000.0
        y += fwd_speed * math.cos(psi) * dt_ms / 1000.0
        clock["t"] = t + dt_ms

    res = SimResult(params=p, final_state=gov.autopilot.mode.value)
    res.events = [e for e in gov.drain_events() if e.get("kind") in ("state", "safety")]
    gov.close()
    if lock_t is None or not samples:
        return res
    res.lock_time_s = (lock_t - (mission_at or 0.0)) / 1000.0
    res.followed_s = samples[-1][0]
    errs = [b for (_, b, _, _) in samples]
    res.bearing_rms_rad = math.sqrt(sum(b * b for b in errs) / len(errs))
    res.bearing_max_rad = max(abs(b) for b in errs)
    res.in_frame_frac = sum(1 for s in samples if s[2]) / len(samples)
    res.min_distance_m = min(s[3] for s in samples)
    if p.scenario == "step":
        b0 = errs[0]
        sgn = 1.0 if b0 > 0 else -1.0
        res.overshoot_pct = max(0.0, max(-sgn * b for b in errs)) / abs(b0) * 100.0
        band = max(0.05 * abs(b0), acfg.yaw.bearing_deadband_rad + 0.01)
        settle = 0.0
        for ts, b, _, _ in samples:
            if abs(b) > band:
                settle = ts
        res.settle_s = settle
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delays", type=float, nargs="+", default=[0.6, 1.0, 1.5])
    ap.add_argument("--scenario", nargs="+", default=["step", "s1"])
    ap.add_argument("--ff", action="store_true", help="enable yaw feed-forward")
    ap.add_argument("--k-scale", type=float, nargs="+", default=[1.0], help="true K / assumed K")
    args = ap.parse_args(argv)
    for sc in args.scenario:
        for d in args.delays:
            for k in args.k_scale:
                print(run(SimParams(total_delay_s=d, scenario=sc, feedforward=args.ff, k_true_scale=k)).row())


if __name__ == "__main__":
    main()
