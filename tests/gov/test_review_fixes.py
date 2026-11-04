"""Regression tests for the adversarial review of ws/gov (docs/ws/gov.md, "Review fixes").

One section per finding, in the review's order. The live Ctrl-C test (finding 2) is in
test_backend_live.py because it needs the real launcher, uvicorn and an RTSP stream.
"""
from __future__ import annotations

import json
import math
import signal
import threading
import time

import pytest
from fastapi import WebSocket

from conftest import ROOT, TICK_MS, make_mission, make_obs

from mhd.contracts import FlightState, Mission, SubjectObs
from mhd.governor import ConfigError, Governor, GovernorError
from mhd.governor.config import load_autopilot_config, load_governor_config, replace


NEUTRAL = {"throttle": 128, "yaw": 128, "pitch": 128, "roll": 128}


# =============================================================================================
# 1 (major). Latency that creeps up, or rises and stays high, must stay visible in real mode.

def _real_capture_run(rig, latency_fn, seconds, frame_ms=66.7, perception_ms=120.0):
    """Real-mode capture estimate (make_capture_clock('real') + estimate) feeding a following
    real-mode governor, while the true camera-to-backend latency follows latency_fn(t_s)."""
    from mhd.backend.capture import estimate, make_capture_clock
    r = rig(mode="real")
    r.frames, r.obs_fn = False, None
    clock = make_capture_clock("real")
    st = {"next": r.t, "pending": [], "fid": 0, "t0": None}

    def feed():
        while st["next"] <= r.t:
            c = st["next"]
            st["next"] += frame_ms
            t_rel = 0.0 if st["t0"] is None else max(0.0, (c - st["t0"]) / 1000.0)
            st["pending"].append((c, c + 1000.0 * latency_fn(t_rel)))
        keep = []
        for c, recv in st["pending"]:
            if recv <= r.t:
                cap, _, _ = estimate(None, c - 5000.0, recv, clock, "real")   # pts: camera clock, any origin
                r.g.note_frame(cap, recv)
                st["fid"] += 1
                r.g.submit_obs(make_obs(st["fid"], cap, cx=900 + (st["fid"] % 2), w=100, h=300),
                               receive_ms=recv + perception_ms)
            else:
                keep.append((c, recv))
        st["pending"] = keep

    def step():
        r.g.heartbeat(r.t)
        feed()
        out = r.g.tick(r.t)
        r.t += TICK_MS
        return out

    for _ in range(100):
        step()
    assert r.g.command("arm", arm_phrase="ARM").ok and r.g.command("takeoff").ok
    for _ in range(110):
        step()
    r.g.set_mission(make_mission())
    for _ in range(60):
        step()
    assert step().source == "autopilot"
    st["t0"] = r.t
    outs = [(r.t - st["t0"], step()) for _ in range(int(seconds * 25))]
    return r, outs


def test_latency_creep_is_seen_and_lands(rig):
    # Review probe p07: true latency 0.5 s -> 3.5 s over 60 s. It used to stay invisible.
    r, outs = _real_capture_run(rig, lambda t: 0.5 + 0.05 * t, 60.0)
    t_stale = next(t for t, o in outs if "video_stale" in o.safety_flags) / 1000.0
    t_land = next(t for t, o in outs if o.flight_state == FlightState.landing) / 1000.0
    # 0.7 s of extra lag after ~14 s and 2.0 s after ~40 s (minus at most 2 ms/s of drift allowance).
    assert 12.0 <= t_stale <= 18.0 and 38.0 <= t_land <= 46.0, (t_stale, t_land)
    assert all(o.source != "autopilot" for t, o in outs if t / 1000.0 > t_stale + 0.1)


def test_lasting_latency_step_keeps_the_autopilot_neutral(rig):
    # Review probe p07: +1.5 s for good. The autopilot used to resume after ~10 s on 2 s old video.
    r, outs = _real_capture_run(rig, lambda t: 0.5 + (1.5 if t >= 5.0 else 0.0), 40.0)
    after = [(t, o) for t, o in outs if t > 7000.0]
    assert after and all(o.source != "autopilot" and "video_stale" in o.safety_flags for _, o in after)
    lag = r.g.snapshot().latency_ms["frame_lag"]
    assert lag >= 1100.0, lag                 # 1.5 s minus at most the 300 ms rise cap


def test_slow_camera_clock_drift_is_absorbed():
    # 100 ppm (0.1 ms/s) of camera clock drift over 7 minutes must not look like latency.
    from mhd.backend.capture import CaptureClock
    clock = CaptureClock(drift_ms_per_s=2.0, max_rise_ms=300.0)
    worst = 0.0
    t = 0.0
    while t < 420_000.0:
        pts = 5000.0 + t * (1.0 - 1e-4)          # camera clock runs 100 ppm slow
        recv = t + 400.0 + (37.0 * ((int(t) // 67) % 3))   # 400 ms latency, jitter 0..74 ms
        cap = clock.update(pts, recv)
        worst = max(worst, (recv - cap) - 400.0)
        t += 66.7
    assert worst <= 80.0, worst                # only the jitter, never the drift


def test_governor_floors_never_rise():
    # Video and observation floors are session minimums now (a sliding minimum re-baselined).
    from mhd.governor.stats import SessionMin
    m = SessionMin()
    m.add(0.0, 500.0)
    m.add(60_000.0, 3500.0)
    assert m.value(120_000.0) == 500.0


def test_observation_latency_creep_goes_stale(rig):
    r = rig()
    r.following(cx=900)
    for i in range(r.ticks_for(30.0)):
        r.obs_latency = 150.0 + 60.0 * (i * TICK_MS / 1000.0)    # perception slows by 60 ms/s
        out = r.tick()
        if "obs_stale" in out.safety_flags:
            break
    assert "obs_stale" in out.safety_flags and out.source != "autopilot"
    assert r.obs_latency - 150.0 < 600.0       # caught within ~0.45 s of extra latency


def test_capture_env_values_are_validated(monkeypatch):
    from mhd.backend.capture import make_capture_clock
    for name, bad in (("MHD_CAPTURE_OFFSET_MS", "nan"), ("MHD_CAPTURE_DRIFT_MS_PER_S", "500"),
                      ("MHD_CAPTURE_MAX_RISE_MS", "-1")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError):
            make_capture_clock("real")
        monkeypatch.delenv(name)
    assert make_capture_clock("real").drift_per_ms == pytest.approx(0.002)
    assert make_capture_clock("twin").drift_per_ms is None


# =============================================================================================
# 2 (major). Ctrl-C must land at once even with /mjpeg open (live test in test_backend_live.py).

def test_begin_shutdown_lands_now_and_runs_callbacks(rig):
    from mhd.backend.hub import Hub
    r = rig()
    r.airborne()
    hub = Hub()
    hub.gov = r.g
    called = []
    hub.on_shutdown(lambda: called.append(1))
    hub.begin_shutdown("test")
    hub.begin_shutdown("again")                     # idempotent
    out = r.tick()
    assert out.land and out.wire == NEUTRAL and out.source == "shutdown" and called == [1]


def test_signal_hook_begins_shutdown_then_chains(rig):
    from mhd.backend.hub import Hub
    r = rig()
    r.airborne()
    hub = Hub()
    hub.gov = r.g
    seen = []
    old_int, old_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGINT, lambda s, f: seen.append(("prev", s)))
        assert hub.install_signal_hooks(min_land_s=0.2)
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)                # first Ctrl-C: landing starts, uvicorn told
        assert hub.shutting_down and r.tick().land and seen == [("prev", signal.SIGINT)]
        t0 = time.monotonic()
        handler(signal.SIGINT, None)                # force quit: waits out the minimum land time
        assert time.monotonic() - t0 >= 0.1 and len(seen) == 2
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def test_signal_hook_outside_uvicorn_lands_then_terminates(tmp_path):
    """With no uvicorn handler (SIG_DFL), SIGTERM still begins the landing, waits min_land_s, then
    terminates the process the default way."""
    import subprocess
    import sys
    code = (
        "import os, signal, sys, time\n"
        "from mhd.backend.hub import Hub\n"
        "class G:\n"
        "    def begin_shutdown(self):\n"
        "        print('landing', flush=True)\n"
        "h = Hub(); h.gov = G()\n"
        "signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
        "h.install_signal_hooks(min_land_s=0.3)\n"
        "t = time.monotonic(); os.kill(os.getpid(), signal.SIGTERM); time.sleep(5)\n"
        "print('survived', flush=True)\n")
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert p.returncode == -signal.SIGTERM and "landing" in p.stdout and "survived" not in p.stdout
    assert 0.3 <= time.monotonic() - t0 < 5.0


def test_long_rc_gaps_are_flagged(rig):
    # Review verdict: CPU-bound threads stretched RC gaps to 112 ms p99 and nothing noticed.
    r = rig()
    r.airborne()
    assert "rc_slow" not in r.tick().safety_flags
    r.t += 120.0                                      # one 160 ms gap between packets
    out = r.tick()
    assert "rc_slow" in out.safety_flags and r.g.snapshot().latency_ms["rc_gap_max"] >= 150.0
    r.run(2.2)
    assert "rc_slow" not in r.tick().safety_flags     # clears once the last ~2 s are normal


def test_signal_hooks_need_the_main_thread():

    from mhd.backend.hub import Hub
    res = []
    th = threading.Thread(target=lambda: res.append(Hub().install_signal_hooks()))
    th.start()
    th.join()
    assert res == [False]


# =============================================================================================
# 3 (major). A late follow Mission must not undo a later stop, E-stop or wrong-way hold.

def _mission(mid: str, latency_ms: float) -> Mission:
    m = make_mission(mid)
    return m.model_copy(update={"latency_ms": latency_ms})


def test_late_mission_does_not_override_a_later_pause(rig):
    r = rig(mode="real")
    r.following(cx=900)
    assert r.g.command("pause", source="agent").ok           # voice "stop", fast keyword path
    r.run(3.0)
    # The LLM's Mission for "follow ..." (said before "stop") arrives 3 s later: latency 4 s.
    res = r.g.set_mission(_mission("m2", 4000.0))
    assert res["ok"] and res["paused"] and res["reason"] == "paused_after_utterance"
    r.steady_obs(cx=900, mission_id="m2")
    outs = [r.tick() for _ in range(r.ticks_for(3.0))]
    assert all(o.source != "autopilot" and o.wire == NEUTRAL for o in outs)
    assert "autopilot_paused" in outs[-1].safety_flags
    assert r.g.command("resume").ok                            # the operator decides
    r.run(1.0)
    assert r.tick().source == "autopilot"


def test_late_mission_after_a_resume_pauses_again(rig):
    r = rig()
    r.following(cx=900)
    assert r.g.command("pause", source="agent").ok
    r.run(1.0)
    assert r.g.command("resume").ok                           # the operator resumes mission m1...
    r.run(1.0)
    assert r.tick().source == "autopilot"
    res = r.g.set_mission(_mission("m2", 4000.0))             # ...then a Mission said before the stop
    assert res["paused"] and res["reason"] == "paused_after_utterance"
    r.steady_obs(cx=900, mission_id="m2")
    outs = [r.tick() for _ in range(r.ticks_for(2.0))]
    assert all(o.source != "autopilot" for o in outs)


def test_mission_spoken_after_the_pause_releases_it(rig):

    r = rig()
    r.following(cx=900)
    assert r.g.command("pause", source="agent").ok
    r.run(6.0)                                                 # "follow the person in red", later
    res = r.g.set_mission(_mission("m3", 1500.0))
    assert res["ok"] and not res["paused"]
    r.steady_obs(cx=900, mission_id="m3")
    r.run(1.6)
    assert r.tick().source == "autopilot"


def test_estop_drops_the_mission_and_refuses_late_missions(rig):
    r = rig(mode="real")
    r.following(cx=900)
    assert r.g.command("estop").ok
    assert r.g.mission is None and r.g.autopilot.mode.value == "IDLE"
    assert r.g.set_mission(_mission("m4", 4000.0)) == {"ok": False, "action": "follow", "reason": "estopped"}
    # Review probe p04: reset, ARM, takeoff -> the drone used to start following by itself.
    assert r.g.command("estop_reset").ok and r.g.command("confirm_touchdown").ok
    r.airborne()
    outs = [r.tick() for _ in range(r.ticks_for(5.0))]
    assert all(o.source != "autopilot" for o in outs) and r.g.mission is None


def test_mission_set_on_the_ground_waits_for_resume_after_takeoff(rig):
    r = rig()
    r.tick(5)
    r.g.set_mission(make_mission())
    r.steady_obs(cx=900)
    r.run(2.0)
    r.airborne()                                               # takeoff pauses the autopilot
    outs = [r.tick() for _ in range(r.ticks_for(2.0))]
    assert all(o.source != "autopilot" for o in outs)
    assert r.g.command("resume").ok
    r.tick()
    assert r.tick().source == "autopilot"


def test_new_mission_never_clears_a_wrong_way_hold(rig):
    r = rig()
    r.following(cx=900)
    r.g._wrong_way_hold, r.g._wrong_way_reason = True, "yaw"   # as if the detector had fired
    r.run(4.0)
    res = r.g.set_mission(_mission("m5", 500.0))
    assert res["ok"] and res["paused"] and res["reason"] == "wrong_way_hold"
    r.steady_obs(cx=900, mission_id="m5")
    r.run(2.0)
    out = r.tick()
    assert out.source != "autopilot" and "wrong_way" in out.safety_flags


def test_mission_with_unusable_latency_counts_as_slow(rig):
    r = rig()
    r.following(cx=900)
    r.g.command("pause")
    r.run(10.0)
    assert r.g.set_mission(_mission("m6", float("nan")))["paused"]     # NaN: treated as 30 s
    r.run(25.0)                                                         # 35 s after the pause
    assert r.g.set_mission(_mission("m7", -5.0))["paused"] is False     # negative: also 30 s
    assert r.g._spoken_at(_mission("m8", 1e12), 100_000.0) == 100_000.0 - 30_000.0 - 3_000.0


# =============================================================================================
# 4 (major). No config can switch a safety check off, in any mode.

@pytest.mark.parametrize("yaml_text", [
    "heartbeat: {neutral_after_s: 1.0e9, land_after_s: 2.0e9}",
    "heartbeat: {neutral_after_s: .nan}",
    "video: {neutral_lag_s: .nan, land_lag_s: .nan}",
    "observation: {stale_min_s: .inf}",
    "observation: {stale_min_s: 5.0}",
    "slew: {autopilot_per_s: -1.0}",
    "flight_timer: {land_s: 100000}",
    "landing: {reassert_per_s: .nan}",
    "human: {stale_s: 30}",
    "rc_rate_hz: 100",
])
def test_config_refuses_unsafe_or_non_finite_values(tmp_path, yaml_text):
    from mhd.governor.config import load_governor_config as load
    p = tmp_path / "g.yaml"
    p.write_text(yaml_text + "\n")
    with pytest.raises(ConfigError):
        load(p)


def test_real_mode_refuses_takeoff_without_heartbeat_requirement(tmp_path):
    cfg = replace(load_governor_config(), **{"takeoff.require_heartbeat": False})
    with pytest.raises(GovernorError):
        Governor(cfg, load_autopilot_config(), "real", log_dir=tmp_path, tick_log=False)
    Governor(cfg, load_autopilot_config(), "twin", log_dir=tmp_path, tick_log=False).close()


def test_governor_validates_configs_built_in_code(tmp_path):
    cfg = replace(load_governor_config(), **{"video.land_lag_s": math.nan})
    with pytest.raises(ConfigError):
        Governor(cfg, load_autopilot_config(), "twin", log_dir=tmp_path, tick_log=False)


@pytest.mark.parametrize("rate,mode,ok", [("nan", "twin", False), ("-25", "twin", False), ("30", "real", False),
                                          ("25", "real", True), ("30", "twin", True), ("abc", "twin", False)])
def test_control_rate_is_validated(rate, mode, ok):
    from mhd.backend.modes import ModeError, validate_env
    env = {"FLIGHT_MODE": mode, "DRONE_IP": "127.0.0.1" if mode == "twin" else "172.16.11.1", "CONTROL_RATE": rate}
    if ok:
        validate_env(env)
    else:
        with pytest.raises(ModeError):
            validate_env(env)


def test_paced_controller_refuses_bad_rates():
    from mhd.governor.turbodrone_glue import validate_control_rate
    for bad in (math.nan, -25.0, 0.0, 1000.0, "x"):
        with pytest.raises(ValueError):
            validate_control_rate(bad)
    assert validate_control_rate("25") == 25.0


# =============================================================================================
# 5 (minor). Non-finite or out-of-frame observations and frames are refused.

def _obs_with(**fields) -> SubjectObs:
    base = json.loads(make_obs(1, 9_000.0).model_dump_json())
    base.update(fields)
    return SubjectObs.model_validate(base)


@pytest.mark.parametrize("fields,why", [
    ({"capture_ms": math.nan}, "capture_ms_not_finite"),
    ({"capture_ms": math.inf}, "capture_ms_not_finite"),
    ({"confirm_age_ms": math.nan}, "confirm_age_ms_invalid"),
    ({"tracker_score": math.nan}, "tracker_score_invalid"),
    ({"tracker_score": 7.0}, "tracker_score_invalid"),
    ({"detector_latency_ms": math.inf}, "detector_latency_invalid"),
    ({"box": {"x1": 4900, "y1": 100, "x2": 5100, "y2": 500}}, "box_outside_frame"),
    ({"box": {"x1": -1e6, "y1": 100, "x2": 100, "y2": 500}}, "box_outside_frame"),
    ({"box": {"x1": 1e308, "y1": 100, "x2": 1e308, "y2": 500}}, "box_outside_frame"),
    ({"box": {"x1": math.nan, "y1": 100, "x2": 300, "y2": 500}}, "box_not_finite"),
])
def test_invalid_observations_are_refused(rig, fields, why):
    r = rig()
    r.tick()
    assert not r.g.submit_obs(_obs_with(**fields), receive_ms=10_000.0)
    assert r.g.obs_rejected == {why: 1}


def test_nan_capture_obs_cannot_keep_the_autopilot_alive(rig):
    # Review probe p01: one NaN observation, then perception dies -> forward + yaw for 30 s.
    r = rig()
    r.following(cx=900)
    r.g.submit_obs(_obs_with(capture_ms=math.nan, frame_id=999_999), receive_ms=r.t)
    r.obs_fn = None
    outs = [r.tick() for _ in range(r.ticks_for(1.0))]
    first_neutral = next(i for i, o in enumerate(outs) if o.wire == NEUTRAL) * TICK_MS / 1000.0
    assert first_neutral <= 0.5


def test_frames_with_bad_capture_times_are_refused(rig):
    r = rig()
    assert not r.g.note_frame(math.nan, 10_000.0)
    assert not r.g.note_frame(20_000.0, 10_000.0)            # future capture: frozen video would look fresh
    assert not r.g.note_frame(9_000.0, math.inf)
    assert r.g.note_frame(9_900.0, 10_000.0) and r.g.frames_rejected == 3


def test_hub_json_refuses_nan_and_infinity():
    from mhd.backend.hub import loads_strict
    for text in ('{"capture_ms": NaN}', '{"x": Infinity}', '{"x": -Infinity}'):
        with pytest.raises(ValueError):
            loads_strict(text)
    assert loads_strict('{"x": 1.5}') == {"x": 1.5}


# =============================================================================================
# 6 (minor). Pitch moving the wrong way, and 13 (minor) a frozen box, latch a hover.

def test_pitch_wrong_way_hovers(rig):
    r = rig()
    r.following(cx=640, w=100, h=300)       # small box: the autopilot flies forward
    r.run(0.5)
    assert r.tick().wire["pitch"] > 128
    r.obs_fn = _Plant(r, d0=3.75, u=0.0, sign=-1.0)    # ... but the pitch sign is reversed
    outs = [r.tick() for _ in range(r.ticks_for(12.0))]
    flagged = lambda o: "pitch_probe" in o.safety_flags or "wrong_way:pitch" in o.safety_flags  # noqa: E731
    held = next((i for i, o in enumerate(outs) if flagged(o)), None)
    # The condition fires within a few seconds and latches the hover (the pitch probe is off by
    # default; when on, it holds pitch at once and latches about 3.5 s later).
    assert held is not None and held * TICK_MS / 1000.0 <= 6.0
    latched = next((i for i, o in enumerate(outs) if "wrong_way:pitch" in o.safety_flags), None)
    assert latched is not None and (latched - held) * TICK_MS / 1000.0 <= 4.5
    assert all(o.wire["pitch"] == 128 for o in outs[held:])


def test_pitch_check_quiet_when_approach_works(rig):
    r = rig()
    r.following(cx=640, w=100, h=300)
    state = {"w": 100.0}

    def fn(t):                               # forward flight makes the box grow: consistent
        state["w"] = min(130.0, state["w"] * 1.002)
        r._fid += 1
        return make_obs(r._fid, t - r.obs_latency, cx=640 + (r._fid % 2), w=state["w"], h=state["w"] * 3)
    r.obs_fn = fn
    outs = [r.tick() for _ in range(r.ticks_for(5.0))]
    assert not any("wrong_way" in o.safety_flags for o in outs)


def test_pitch_check_quiet_under_a_gait_swing(rig):
    # Integration finding: a walker's box width swings +/-20 % with its gait. With raw end-point
    # widths that tripped the pitch check while the drone was correctly flying forward.
    import math as _m
    r = rig()
    r.following(cx=640, w=100, h=300)

    def fn(t):                                    # slow approach plus a 1 Hz gait swing
        r._fid += 1
        base = 100.0 * (1.0 + 0.01 * (t / 1000.0))
        w = base * (1.0 + 0.2 * _m.sin(2 * _m.pi * t / 1000.0))
        return make_obs(r._fid, t - r.obs_latency, cx=640 + (r._fid % 2), w=w, h=300)
    r.obs_fn = fn
    outs = [r.tick() for _ in range(r.ticks_for(8.0))]
    assert not any("wrong_way" in o.safety_flags for o in outs)


def test_pitch_check_quiet_when_a_walker_recedes_below_the_cap(rig):
    # Integration finding: a subject walking away makes the box shrink while the drone correctly
    # flies forward below its cap (P control lags). That must not latch a wrong-way hover.
    r = rig()
    r.following(cx=640, w=100, h=300)         # box below the reference: flies forward, below the cap
    state = {"w": 100.0}

    def fn(t):
        state["w"] = max(92.0, state["w"] * 0.9995)    # slow shrink (subject outpaces a little)
        r._fid += 1
        return make_obs(r._fid, t - r.obs_latency, cx=640 + (r._fid % 2), w=state["w"], h=state["w"] * 3)
    r.obs_fn = fn
    outs = [r.tick() for _ in range(r.ticks_for(8.0))]
    assert any(o.wire["pitch"] > 128 for o in outs)          # it does fly forward
    assert not any("wrong_way" in o.safety_flags for o in outs)


def _approaching_subject(r):
    state = {"w": 200.0}

    def fn(t):
        state["w"] = min(330.0, state["w"] * 1.002)   # the subject keeps approaching
        r._fid += 1
        return make_obs(r._fid, t - r.obs_latency, cx=640 + (r._fid % 2), w=state["w"], h=500)
    return fn


def test_box_growing_while_backing_locks_out_backing_without_a_hover(rig):
    # A person walking up to a correctly backing drone, or a reversed pitch sign flying it toward
    # them: either way the box grows while it backs up. Review (2026-11-04): check it, and stop all
    # autopilot backing for the rest of the mission instead of latching a hover.
    r = rig(ap_overrides={"forward.backward_budget_m": 0.5})     # backing is off by default
    r.following(cx=640, w=200, h=500)         # closer than the reference: the autopilot backs up
    r.obs_fn = _approaching_subject(r)
    outs = [r.tick() for _ in range(r.ticks_for(8.0))]
    assert any(o.wire["pitch"] < 128 for o in outs)          # it did back up at first
    lock = next((i for i, o in enumerate(outs) if "back_lockout" in o.safety_flags), None)
    assert lock is not None
    assert all(o.wire["pitch"] >= 128 for o in outs[lock:])  # and never again after the lockout
    assert not any("wrong_way" in o.safety_flags for o in outs)
    assert any(e.get("kind") == "safety" and "backing up stopped" in e.get("text", "") for e in r.g.drain_events())
    r.g.set_mission(make_mission(mid="m2"))                  # a new mission may back up again
    assert "back_lockout" not in r.tick().safety_flags


def test_backward_check_can_be_switched_off(rig):
    r = rig(gov_overrides={"wrong_way.pitch_check_backward": False},
            ap_overrides={"forward.backward_budget_m": 0.5})
    r.following(cx=640, w=200, h=500)
    r.obs_fn = _approaching_subject(r)
    outs = [r.tick() for _ in range(r.ticks_for(6.0))]
    assert not any("back_lockout" in o.safety_flags or "wrong_way" in o.safety_flags for o in outs)
    assert any(o.wire["pitch"] < 128 for o in outs)
    assert r.g.autopilot.dbg_gate == "backward_budget"       # only the 0.5 m budget stopped it


class _Plant:
    """Distance to the subject driven by the governor's own applied pitch (1.524 m/s per unit, the
    modeled 0.012 m/s per count): distance' = u - sign * v. Box width = 375 / distance."""

    def __init__(self, r, d0: float, u: float, sign: float = 1.0):
        self.r, self.d, self.u, self.sign = r, d0, u, sign

    def __call__(self, t):
        v = self.r.g._applied["pitch"] * 0.012 * 127.0
        self.d = max(1.0, self.d + (self.u - self.sign * v) * TICK_MS / 1000.0)
        self.r._fid += 1
        w = 375.0 / self.d
        return make_obs(self.r._fid, t - self.r.obs_latency, cx=640 + (self.r._fid % 2), w=w, h=min(600.0, 3 * w))


def _probe_starts(outs):
    return sum(1 for a, b in zip(outs, outs[1:]) if "pitch_probe" in b.safety_flags and "pitch_probe" not in a.safety_flags)


PROBE_ON = {"wrong_way.pitch_probe_window_s": 1.5}                  # the probe is off by default since review 3


NO_FAR_GATE = {"forward.gate.min_box_w_frac": 0.0}   # these walkers outrun the 64 px gate


def test_pitch_probe_clears_when_the_subject_outpaces_a_correct_drone(rig):
    r = rig(gov_overrides=PROBE_ON, ap_overrides=NO_FAR_GATE)
    r.following(cx=640, w=110, h=330)
    plant = _Plant(r, d0=375.0 / 110, u=0.6)          # walks away faster than the 0.38 m/s cap
    r.obs_fn = plant
    outs = [r.tick() for _ in range(r.ticks_for(7.0))]
    assert _probe_starts(outs) == 1
    assert not any("wrong_way" in o.safety_flags for o in outs)
    assert "pitch_probe" not in outs[-1].safety_flags and outs[-1].wire["pitch"] > 128   # following again
    assert any(e.get("kind") == "safety" and "pitch check passed" in e.get("text", "") for e in r.g.drain_events())


def test_pitch_probe_latches_when_holding_still_stops_the_shrink(rig):
    r = rig(gov_overrides=PROBE_ON)
    r.following(cx=640, w=110, h=330)
    r.obs_fn = _Plant(r, d0=375.0 / 110, u=0.0, sign=-1.0)   # stationary subject, reversed pitch sign
    outs = [r.tick() for _ in range(r.ticks_for(12.0))]
    assert _probe_starts(outs) == 1
    latched = next((i for i, o in enumerate(outs) if "wrong_way:pitch" in o.safety_flags), None)
    assert latched is not None
    start = next(i for i, o in enumerate(outs) if "pitch_probe" in o.safety_flags)
    assert all(o.wire["pitch"] == 128 for o in outs[start + 1:])


def test_pitch_probe_clears_are_limited_per_mission(rig):
    r = rig(gov_overrides=PROBE_ON, ap_overrides=NO_FAR_GATE)
    r.following(cx=640, w=110, h=330)
    r.obs_fn = _Plant(r, d0=375.0 / 110, u=1.2)      # far faster than the drone can follow
    outs = [r.tick() for _ in range(r.ticks_for(40.0))]
    max_clears = r.g.cfg.wrong_way.pitch_probe_max_clears
    assert _probe_starts(outs) == max_clears
    assert "wrong_way:pitch" in outs[-1].safety_flags


def test_default_config_never_backs_up_toward_an_approaching_subject(rig):
    r = rig()
    r.following(cx=640, w=200, h=500)
    r.obs_fn = _approaching_subject(r)
    outs = [r.tick() for _ in range(r.ticks_for(6.0))]
    assert all(o.wire["pitch"] >= 128 for o in outs)
    assert r.g.cfg.wrong_way.pitch_probe_window_s == 0.0


def test_twin_only_configs_are_refused_in_real_mode(tmp_path):
    twin_gov = load_governor_config(ROOT / "config" / "twin" / "governor.yaml")
    twin_ap = load_autopilot_config(ROOT / "config" / "twin" / "autopilot.yaml")
    gov, ap = load_governor_config(), load_autopilot_config()
    assert twin_gov.twin_only and twin_ap.twin_only and not gov.twin_only and not ap.twin_only
    for g, a in ((twin_gov, ap), (gov, twin_ap), (twin_gov, twin_ap)):
        with pytest.raises(GovernorError, match="twin-only"):
            Governor(g, a, "real", log_dir=tmp_path, tick_log=False, persist_timer=False)
    Governor(twin_gov, twin_ap, "twin", log_dir=tmp_path, tick_log=False, persist_timer=False).close()
    Governor(gov, ap, "real", log_dir=tmp_path, tick_log=False, persist_timer=False).close()


def test_backward_budget_is_per_flight(rig):
    import dataclasses
    r = rig()
    r.following(cx=640, w=128, h=380)
    r.g._ap = dataclasses.replace(r.g._ap, backward_m=0.4)
    r.g.set_mission(make_mission(mid="m2"))          # a new mission keeps what this flight used
    assert r.g.autopilot.backward_m == pytest.approx(0.4)
    assert r.g.command("resume").ok
    assert r.g.autopilot.backward_m == pytest.approx(0.4)
    assert r.g.command("land").ok
    r.run(1.0)
    assert r.g.command("confirm_touchdown").ok
    r.tick(2)
    assert r.g.command("takeoff").ok                 # a new flight starts with the whole budget
    assert r.g.autopilot.backward_m == 0.0


def test_frozen_box_while_turning_hovers(rig):
    # Review probe: a stuck tracker at cx=1000 used to spin the drone at ~14 counts forever.
    r = rig()
    r.airborne()
    r.g.set_mission(make_mission())
    r.steady_obs(cx=1000, jitter_px=0.0)
    outs = [r.tick() for _ in range(r.ticks_for(6.0))]
    fired = next((i for i, o in enumerate(outs) if "wrong_way:frozen_box" in o.safety_flags), None)
    assert fired is not None
    assert all(o.wire["yaw"] == 128 for o in outs[fired + 1:])


def test_frozen_box_quiet_when_not_turning(rig):
    r = rig()
    r.airborne()
    r.g.set_mission(make_mission())
    r.steady_obs(cx=640, jitter_px=0.0)       # centered: no yaw, so an unchanged box is fine
    outs = [r.tick() for _ in range(r.ticks_for(6.0))]
    assert not any("wrong_way" in o.safety_flags for o in outs)


# =============================================================================================
# 7 (minor). The state machine refuses commands out of context.

def test_confirm_touchdown_only_while_landing(rig):
    r = rig()
    r.airborne()
    assert r.g.command("confirm_touchdown").reason == "not_landing"
    r.hb = False
    r.run(5.2)
    assert r.g.state == FlightState.landing             # the heartbeat landing still fires
    assert r.g.command("confirm_touchdown").ok


def test_takeoff_cannot_be_confirmed_away(rig):
    r = rig()
    r.tick(2)
    assert r.g.command("takeoff").ok
    assert r.g.command("confirm_touchdown").reason == "not_landing"
    assert r.g.command("land").ok and r.g.command("confirm_touchdown").ok    # lost 0x01: LAND, then Landed


def test_resume_after_land_on_ground_stays_on_ground(rig):
    # Review probe p03: LAND on the ground, then resume -> AIRBORNE on the ground, ARM bypassed.
    r = rig(mode="real")
    r.tick(3)
    assert not r.g.armed
    assert r.g.command("land").ok and r.g.command("resume").ok
    assert r.g.state == FlightState.on_ground
    r.g.submit_human({"throttle": 1.0, "yaw": 1.0}, "abs", now=r.t)
    assert r.tick().wire == NEUTRAL
    assert r.g.command("takeoff").reason == "not_armed"


def test_resume_during_takeoff_landing_restarts_the_hold(rig):
    r = rig()
    r.tick(2)
    assert r.g.command("takeoff").ok
    r.run(1.0)
    assert r.g.command("land").ok and r.g.command("resume").ok
    assert r.g.state == FlightState.taking_off
    outs = [r.tick() for _ in range(r.ticks_for(3.8))]
    assert all(not o.takeoff and o.wire == NEUTRAL for o in outs)      # no second 0x01, neutral hold
    r.run(0.5)
    assert r.g.state == FlightState.airborne


# =============================================================================================
# 8 (minor). HTTP and other origins cannot command the drone.

def test_post_command_whitelist_and_agent_role(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from mhd.backend.hub import Hub, install_routes
    hub = Hub()
    hub.gov = Governor(load_governor_config(), load_autopilot_config(), "twin", log_dir=tmp_path, tick_log=False)
    app = FastAPI()
    install_routes(app, hub)
    c = TestClient(app)
    try:
        for cmd in ("takeoff", "arm", "resume", "confirm_touchdown", "disarm", "tilt_up", "pack_swap"):
            assert c.post("/command", json={"type": "cmd", "command": cmd}).json()["reason"] == "not_allowed_over_http"
        assert c.post("/command", json={"type": "cmd", "command": "pause"}).json()["ok"]
        assert c.post("/command?role=agent", json={"type": "cmd", "command": "estop"}).json()["reason"] == \
            "not_allowed_from_agent"
        assert c.post("/command?role=agent", json={"type": "cmd", "command": "land"}).json()["ok"]
        assert c.post("/command?role=pilot", json={"type": "cmd", "command": "land"}).status_code == 422
        evil = {"Origin": "http://evil.example"}
        assert c.post("/command", json={"type": "cmd", "command": "estop"}, headers=evil).status_code == 403
        assert c.post("/mission", json=json.loads(make_mission().model_dump_json()), headers=evil).status_code == 403
        ok = {"Origin": "http://localhost:5173"}
        assert c.post("/command", json={"type": "cmd", "command": "estop"}, headers=ok).json()["ok"]
    finally:
        hub.gov.close()


def test_websockets_refuse_foreign_origins(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from mhd.backend.hub import Hub, install_routes
    hub = Hub()
    hub.gov = Governor(load_governor_config(), load_autopilot_config(), "twin", log_dir=tmp_path, tick_log=False)
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_ui(ws: WebSocket):
        await hub.serve_ui(ws)
    install_routes(app, hub)
    c = TestClient(app)
    try:
        for path in ("/ws", "/ws/subject", "/ws/agent?role=agent"):
            with pytest.raises(WebSocketDisconnect):
                with c.websocket_connect(path, headers={"origin": "http://evil.example"}) as ws:
                    ws.receive_text()
        with c.websocket_connect("/ws", headers={"origin": "http://127.0.0.1:5173"}) as ws:
            ws.send_text(json.dumps({"type": "cmd", "command": "estop"}))
            time.sleep(0.1)
        assert hub.gov.state == FlightState.estopped and hub.counters["origin_refused"] == 3
    finally:
        hub.gov.close()


def test_estop_clears_the_find_request(tmp_path):
    from mhd.backend.hub import Hub
    from mhd.contracts import OperatorCommand
    hub = Hub()
    hub.gov = Governor(load_governor_config(), load_autopilot_config(), "twin", log_dir=tmp_path, tick_log=False)
    try:
        hub.apply_mission(make_mission("m1"))
        assert hub.find_request is not None
        hub.command(OperatorCommand(command="estop"), source="ui")
        assert hub.find_request is None
        assert hub.apply_mission(make_mission("m2"))["ok"] is False and hub.find_request is None
    finally:
        hub.gov.close()


# =============================================================================================
# 9 (minor). Legacy UI axes in real mode, and zeros from another socket.

def _real_hub(tmp_path):
    from mhd.backend.hub import Hub
    hub = Hub()
    t = {"now": 10_000.0}
    hub.gov = Governor(load_governor_config(), load_autopilot_config(), "real", clock=lambda: t["now"],
                       log_dir=tmp_path, tick_log=False)
    return hub, t


def test_real_mode_refuses_legacy_axes_and_counts_only_ui_heartbeats(tmp_path):
    hub, t = _real_hub(tmp_path)
    g = hub.gov
    try:
        legacy = {"type": "axes", "mode": "inc", "throttle": 1.0, "yaw": 0, "pitch": 0, "roll": 0,
                  "camera_tilt_direction": 0}
        assert hub.handle_ui_message(legacy, conn=1)["reason"] == "axes_not_operator_input"
        assert g._human_last_ms is None and g._hb_last_ms is None          # no input, no heartbeat
        assert hub.handle_ui_message({"type": "axes", "mode": "mouse", "yaw": 0.5}, conn=1) is not None
        hub.handle_ui_message({"type": "axes", "mode": "abs", "yaw": 0.5}, conn=1)  # OperatorInput
        assert g._human_last_ms == t["now"] and g._hb_last_ms is None
        hub.handle_ui_message({"type": "hb", "client": "sysid"}, conn=2)
        assert g._hb_last_ms is None
        hub.handle_ui_message({"type": "hb", "client": "ui"}, conn=3)
        assert g._hb_last_ms == t["now"]
        tel = json.loads(hub.telemetry_json())
        assert "legacy_axes_refused" in tel["safety_flags"]
    finally:
        g.close()


def test_zeros_from_another_socket_do_not_release_the_override(rig):
    r = rig()
    r.following(cx=900)
    yaws = []
    for _ in range(50):
        r.g.submit_human({"yaw": -0.5}, "abs", now=r.t, owner=1)       # the operator holds left
        r.g.submit_human({"yaw": 0.0}, "abs", now=r.t, owner=2)        # idle legacy tab: zeros
        out = r.tick()
        yaws.append((out.source, out.wire["yaw"]))
    assert all(s == "human" and y == 128 - 64 for s, y in yaws)
    r.g.submit_human({"yaw": 0.0}, "abs", now=r.t, owner=1)            # the owner lets go: immediate
    assert r.tick().source == "autopilot"
    r.g.submit_human({"yaw": 0.5}, "abs", now=r.t, owner=1)
    r.g.ui_disconnected(owner=2)                                       # another socket closing: no effect
    assert r.tick().source == "human"
    r.g.ui_disconnected(owner=1)
    assert r.tick().source == "autopilot"


# =============================================================================================
# 10 (minor). Real mode only flies on the drone's own video.

@pytest.mark.parametrize("url,ok", [("rtsp://127.0.0.1:8554/live/ch00_1", False), ("rtsp://localhost/x", False),
                                    ("rtsp://172.16.11.1:554/live/ch00_1", True), ("", True)])
def test_real_mode_rtsp_url_must_be_the_drone(url, ok):
    from mhd.backend.modes import ModeError, validate_env
    env = {"FLIGHT_MODE": "real", "DRONE_IP": "172.16.11.1", "X69_LG_RTSP_URL": url}
    if ok:
        assert validate_env(env).flight_mode == "real"
    else:
        with pytest.raises(ModeError, match="172.16.11.1"):
            validate_env(env)


# =============================================================================================
# 11 (minor). The D1 keepalive survives a transient OSError; send errors are visible.

def test_keepalive_survives_oserror_and_errors_are_counted():
    import errno
    import socket
    from mhd.governor.turbodrone_glue import GovernedX69LgRcAdapter
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 23518))                       # stands in for the drone; never read
    ad = GovernedX69LgRcAdapter("127.0.0.1", 23518, local_port=0)

    real = ad.sock
    fails = {"n": 1}

    class Flaky:
        def sendto(self, data, addr):
            if fails["n"] > 0:
                fails["n"] -= 1
                raise OSError(errno.ENOBUFS, "No buffer space available")
            return real.sendto(data, addr)

        def __getattr__(self, name):
            return getattr(real, name)
    ad.sock = Flaky()
    try:
        time.sleep(3.3)                                  # keepalive at 1 Hz: first send fails
        s = ad.link_stats()
        assert s["keepalive_errors"] == 1 and s["keepalive_sent"] >= 2 and s["keepalive_alive"]
        fails["n"] = 1
        ad.send_control_packet(b"\x00" * 28)             # RC send error: counted, not silent
        ad.send_control_packet(b"\x00" * 28)
        s = ad.link_stats()
        assert s["rc_errors"] == 1 and s["rc_sent"] == 1
    finally:
        ad.sock = real
        ad.stop()
        rx.close()


# =============================================================================================
# 12 (minor). While LANDING, a failing tick still asserts land at once.

def test_fallback_lands_immediately_when_already_landing(rig):
    from mhd.governor.turbodrone_glue import install_governor
    r = rig()
    r.airborne()
    model = install_governor(r.g)
    assert r.g.command("land").ok

    def boom(now):
        raise RuntimeError("tick exploded")
    r.g._tick_locked = boom
    lands = 0
    for _ in range(10):                                   # 0.4 s at 25 Hz
        model.update(0.04, {})
        lands += int(model.land_flag)
        assert model.yaw == 128.0 and not model.stop_flag
    assert lands == 2                                     # every 5th packet: 5 Hz, from the start


# =============================================================================================
# 14 (minor). sysid aborts when the UI dies or the governor stops flying its sticks.

def test_sysid_abort_reasons():
    from mhd.governor.sysid import Step, _abort_reason
    st = Step("yaw", 0.1, 1.0, 1.0)
    ok = {"flight_state": "AIRBORNE", "flight_mode": "real", "active_source": "human",
          "latency_ms": {"hb_age": 200.0}, "sticks_wire": {"yaw": 141}}
    assert _abort_reason(ok, st, "yaw", 0.5) is None and st.wire_count == 13
    assert "heartbeat" in _abort_reason(dict(ok, latency_ms={"hb_age": 1500.0}), st, "yaw", 0.5)
    assert "heartbeat" in _abort_reason(dict(ok, latency_ms={}), st, "yaw", 0.5)
    assert _abort_reason(dict(ok, active_source="autopilot"), st, "yaw", 0.1) is None    # grace
    assert "not flying our sticks" in _abort_reason(dict(ok, active_source="autopilot"), st, "yaw", 0.5)
    assert "LANDING" in _abort_reason(dict(ok, flight_state="LANDING"), st, "yaw", 0.5)
    twin = dict(ok, flight_mode="twin", latency_ms={})
    assert _abort_reason(twin, st, "yaw", 0.5) is None


def test_no_spoken_lines_while_landing_and_the_mission_ends_at_touchdown(rig):
    # Demo findings: after a voice land the autopilot said "I lost the subject. Looking around."
    # while landing, then cycled FIND/CONFIRM on the ground.
    r = rig()
    r.following(cx=640, w=128, h=380)
    r.g.drain_events()
    assert r.g.command("land").ok
    r.obs_fn = None                                   # the subject is gone from the view
    r.run(6.0)
    evs = r.g.drain_events()
    assert any(e.get("kind") == "state" and e.get("to") == "LOST" for e in evs)   # it still notices
    assert not any(e.get("kind") == "announce" for e in evs)                     # but says nothing
    assert r.g.command("confirm_touchdown").ok
    assert r.g.autopilot.mode.value == "IDLE"
    r.steady_obs(cx=640, w=128, h=380, score=0.5)
    r.run(3.0)
    assert not any(e.get("kind") == "state" for e in r.g.drain_events())       # no FIND/CONFIRM flapping

