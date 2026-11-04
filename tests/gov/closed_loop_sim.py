"""Closed-loop X69 follow simulation with a governor in REAL mode, for the reversed-sign tests.

Adapted from the 2026-11-04 safety review's probe (logs/review2/rev_sign_sim.py): the same plant and
camera model as mhd.autopilot.sim (100 Hz kinematic plant, 15 fps pinhole camera, first-order lags,
observations delayed so the total theta is 1.0 s), but the PLANT's pitch and/or yaw sign can be
flipped relative to the config, the plant can be slower or faster per count than the model, and it
can have a firmware deadband the config does or does not compensate. The subject follows a
scripted path straight ahead of the drone:
  'stat:D'                 stationary at D m
  'walk:D0:V:STOP'         walks toward the drone from D0 at V m/s and stops at STOP m
  'away:D0:V:END'          walks away
  'cycle:D0:V:NEAR:FAR'    walks between NEAR and FAR
run() returns a dict: lock, travel_toward_m (toward the subject), max_away_m, min_dist_m,
ww_t_s / ww_reason (first wrong-way latch, seconds after FOLLOW), budget_t_s, final_ap, ...
"""
from __future__ import annotations

import math
import os
import random
from collections import deque

from mhd.contracts import Box, Intent, Mission, SubjectObs
from mhd.governor.config import load_autopilot_config, load_governor_config, replace
from mhd.governor.core import Governor
from mhd.governor.wire import CENTER


def subject_fn(spec: str):
    """'stat:D' stationary at D m; 'walk:D0:V:STOP' walks toward the drone from D0 at V m/s to STOP m;
    'away:D0:V:END' walks away; 'cycle:D0:V:NEAR:FAR' walks between NEAR and FAR (relative to start)."""
    parts = spec.split(":")
    kind = parts[0]
    if kind == "stat":
        d = float(parts[1])
        return lambda t: (0.0, d)
    if kind == "walk":
        d0, v, stop = map(float, parts[1:4])
        return lambda t: (0.0, max(stop, d0 - v * t))
    if kind == "away":
        d0, v, end = map(float, parts[1:4])
        return lambda t: (0.0, min(end, d0 + v * t))
    if kind == "cycle":
        d0, v, near, far = map(float, parts[1:5])
        # toward to NEAR, pause 4 s, away to FAR, pause 4 s, repeat
        legs = [(abs(d0 - near) / v, d0, near), (4.0, near, near), (abs(far - near) / v, near, far), (4.0, far, far),
                (abs(far - near) / v, far, near), (4.0, near, near)]
        def f(t):
            tt = t
            first = True
            while True:
                for i, (dur, a, b) in enumerate(legs):
                    if not first and i == 0:
                        continue
                    if tt <= dur:
                        return (0.0, a + (b - a) * (tt / dur if dur > 0 else 1.0))
                    tt -= dur
                first = False
        return f
    raise ValueError(spec)


def run(spec, *, cfg_dir, log_dir, pitch_sign=-1.0, yaw_sign=1.0, k_scale=1.0, speed_scale=1.0, plant_db=0,
        gov_over=None, ap_over=None, mode="real", dur_s=40.0, seed=1, fps=15.0, theta=1.0,
        distance="medium", noise_px=2.0, start_angle_deg=0.0, trace=False, gait_amp=0.0, gait_hz=1.8,
        truth_log=None):
    rng = random.Random(seed)
    gcfg = load_governor_config(os.path.join(cfg_dir, "governor.yaml"))
    acfg = load_autopilot_config(os.path.join(cfg_dir, "autopilot.yaml"))
    if gov_over:
        gcfg = replace(gcfg, **gov_over)
    if ap_over:
        acfg = replace(acfg, **ap_over)
    clock = {"t": 0.0}
    gov = Governor(gcfg, acfg, mode, clock=lambda: clock["t"], persist_timer=False, tick_log=False,
                   log_dir=log_dir)
    W, H = acfg.camera.frame_w, acfg.camera.frame_h
    f = (W / 2) / math.tan(math.radians(acfg.camera.hfov_deg) / 2)
    k_true = acfg.plant.yaw_rate_per_count * k_scale * yaw_sign
    v_true = acfg.plant.forward_speed_per_count * speed_scale * pitch_sign
    yaw_tau, fwd_tau, rc_hz = 0.15, 0.4, 25.0
    obs_delay_ms = max(0.0, 1000.0 * (theta - 0.5 / fps - 0.5 / rc_hz - yaw_tau))
    path = subject_fn(spec)
    ang0 = math.radians(start_angle_deg)

    x = y = psi = 0.0
    yaw_rate = fwd_speed = 0.0
    wire = {"throttle": CENTER, "yaw": CENTER, "pitch": CENTER, "roll": CENTER}
    pending: deque = deque()
    dt_ms = 10.0
    next_rc = next_cam = 0.0
    fid = 0
    phase = "ground"
    lock_t = None
    y_lock = None
    ww_t = ww_y = ww_reason = None
    budget_t = budget_y = None
    lockout_t = None
    probes = 0
    in_probe = False
    first_standoff_t = None
    min_dist = math.inf
    min_dist_t = None
    max_away = 0.0
    events_seen = []
    tr = []
    path_moved = 0.0
    last_xy = (0.0, 0.0)
    while clock["t"] < 120_000.0:
        t = clock["t"]
        if phase == "ground" and t >= 500.0:
            if mode == "real":
                assert gov.command("arm", arm_phrase="ARM").ok
            assert gov.command("takeoff").ok, "takeoff refused"
            phase = "takeoff"
        if phase == "takeoff" and gov.state.value == "AIRBORNE":
            gov.set_mission(Mission(mission_id="sim", intent=Intent(
                action="follow", subject_phrase="person in blue shirt", category="person", color="blue",
                distance=distance, reason_code="none"), transcript="sim", model="sim", latency_ms=0.0))
            phase = "mission"
            mission_t = t
        if phase == "mission" and gov.autopilot.mode.value == "FOLLOW":
            lock_t, y_lock = t, y
            phase = "follow"
        if phase == "follow" and t - lock_t >= dur_s * 1000.0:
            break
        if phase == "mission" and t - mission_t > 30_000.0:
            break
        st = 0.0 if lock_t is None else (t - lock_t) / 1000.0
        sx0, sy0 = path(st)
        sx = sx0 * math.cos(ang0) + sy0 * math.sin(ang0)
        sy = -sx0 * math.sin(ang0) + sy0 * math.cos(ang0)
        dx, dy = sx - x, sy - y
        dist = math.hypot(dx, dy)
        beta = (math.atan2(dx, dy) - psi + math.pi) % (2 * math.pi) - math.pi
        d_fwd = dist * math.cos(beta)
        in_frame = d_fwd > 0.2 and abs(math.tan(beta) * f) < W / 2
        if phase == "follow":
            if dist < min_dist:
                min_dist, min_dist_t = dist, st
            max_away = max(max_away, y_lock - y)
            path_moved += math.hypot(x - last_xy[0], y - last_xy[1])
        last_xy = (x, y)
        if t >= next_cam:
            next_cam += 1000.0 / fps
            fid += 1
            gov.note_frame(t, t + obs_delay_ms * 0.6)
            obs = None
            if truth_log is not None:
                truth_log.append((t, dist, d_fwd, beta))
            if in_frame:
                cx = W / 2 + f * math.tan(beta) + rng.gauss(0, noise_px)
                bw = f * 0.45 / d_fwd * (1 + rng.gauss(0, 0.02)) * (1 + gait_amp * math.sin(2 * math.pi * gait_hz * t / 1000.0 + 0.7))
                top = H / 2 - f * (1.7 - 1.2) / d_fwd
                bot = H / 2 + f * 1.2 / d_fwd
                x1, x2 = max(0.0, cx - bw / 2), min(float(W), cx + bw / 2)
                y1, y2 = max(0.0, top), min(float(H), bot)
                if x2 - x1 > 2 and y2 - y1 > 2:
                    obs = SubjectObs(mission_id="sim", frame_id=fid, capture_ms=t, state="confirmed",
                                     track_id=1, box=Box(x1=x1, y1=y1, x2=x2, y2=y2), confidence=0.9,
                                     tracker_score=0.9, confirm_age_ms=(t % 1000.0), source="oracle",
                                     detector_latency_ms=400.0)
            if obs is None:
                obs = SubjectObs(mission_id="sim", frame_id=fid, capture_ms=t, state="searching",
                                 source="oracle", confirm_age_ms=1e9)
            pending.append((t + obs_delay_ms, obs))
        while pending and pending[0][0] <= t:
            _, o = pending.popleft()
            gov.submit_obs(o, receive_ms=t)
        if t >= next_rc:
            next_rc += 1000.0 / rc_hz
            gov.heartbeat(t)
            out = gov.tick(t)
            wire = out.wire
            if phase == "follow":
                fl = out.safety_flags
                if ww_t is None and "wrong_way" in fl:
                    ww_t, ww_y = st, y
                    ww_reason = next((x_ for x_ in fl if x_.startswith("wrong_way:")), "wrong_way")
                if first_standoff_t is None and gov.autopilot.dbg_gate == "standoff":
                    first_standoff_t = st
                if budget_t is None and gov.autopilot.dbg_gate == "backward_budget":
                    budget_t, budget_y = st, y
                if lockout_t is None and "back_lockout" in fl:
                    lockout_t = st
                if "pitch_probe" in fl and not in_probe:
                    probes += 1
                in_probe = "pitch_probe" in fl
                if trace and int(t) % 500 == 0:
                    tr.append((round(st, 2), round(dist, 3), wire["pitch"] - CENTER, gov.autopilot.mode.value,
                               gov.autopilot.dbg_gate, round(gov.autopilot.backward_m, 3), [q for q in fl]))

        def eff(axis):
            off = (wire[axis] - CENTER) * gcfg.wire.sign.get(axis)
            return math.copysign(max(0, abs(off) - plant_db), off) if off else 0.0
        airborne = gov.state.value in ("AIRBORNE", "TAKING_OFF")
        yaw_cmd = k_true * eff("yaw") if airborne else 0.0
        fwd_cmd = v_true * eff("pitch") if airborne else 0.0
        yaw_rate += (yaw_cmd - yaw_rate) * min(1.0, dt_ms / 1000.0 / yaw_tau)
        fwd_speed += (fwd_cmd - fwd_speed) * min(1.0, dt_ms / 1000.0 / fwd_tau)
        psi = (psi + yaw_rate * dt_ms / 1000.0 + math.pi) % (2 * math.pi) - math.pi
        x += fwd_speed * math.sin(psi) * dt_ms / 1000.0
        y += fwd_speed * math.cos(psi) * dt_ms / 1000.0
        clock["t"] = t + dt_ms

    evs = [e for e in gov.drain_events() if e.get("kind") in ("state", "safety")]
    gov.close()
    st_end = (clock["t"] - lock_t) / 1000.0 if lock_t is not None else None
    sx0, sy0 = path(st_end or 0.0)
    res = {
        "spec": spec, "pitch_sign": pitch_sign, "yaw_sign": yaw_sign, "k_scale": k_scale,
        "speed_scale": speed_scale, "plant_db": plant_db, "gov_over": gov_over, "ap_over": ap_over,
        "cfg": str(cfg_dir), "lock": lock_t is not None, "lock_ms": lock_t,
        "travel_toward_m": None if y_lock is None else round(max(0.0, y - y_lock), 3),
        "max_away_m": round(max_away, 3), "path_m": round(path_moved, 3),
        "final_disp_m": None if y_lock is None else round(y - y_lock, 3),
        "min_dist_m": round(min_dist, 3), "min_dist_t_s": None if min_dist_t is None else round(min_dist_t, 2),
        "ww_t_s": None if ww_t is None else round(ww_t, 2), "ww_reason": ww_reason,
        "ww_disp_m": None if ww_y is None else round(ww_y - y_lock, 3),
        "budget_t_s": None if budget_t is None else round(budget_t, 2),
        "budget_disp_m": None if budget_y is None else round(budget_y - y_lock, 3),
        "standoff_t_s": None if first_standoff_t is None else round(first_standoff_t, 2),
        "lockout_t_s": None if lockout_t is None else round(lockout_t, 2), "probes": probes,
        "final_ap": gov.autopilot.mode.value, "psi_final_deg": round(math.degrees(psi), 1),
        "safety_events": [e.get("text") for e in evs if e.get("kind") == "safety"][:6],
        "state_events": [f'{e.get("from")}->{e.get("to")}:{e.get("reason")}@{round((e["ts_ms"] - (lock_t or 0)) / 1000.0, 1)}'
                         for e in evs if e.get("kind") == "state"][:12],
    }
    if trace:
        res["trace"] = tr
    return res
