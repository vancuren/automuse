"""One test (or a few) per governor rule in PLAN.md 4.5 and the gov workstream brief."""
from __future__ import annotations

import json
import math

import pytest

from conftest import make_mission, make_obs
from mhd.contracts import FlightState
from mhd.governor import ConfigError, Governor, GovernorError
from mhd.governor.config import (
    governor_config_from_dict, autopilot_config_from_dict, load_autopilot_config, load_governor_config, replace,
)
from mhd.governor.wire import axis_offset, round_half_away, to_wire

NEUTRAL = {"throttle": 128, "yaw": 128, "pitch": 128, "roll": 128}


# --- configuration ---------------------------------------------------------------------------

def test_repo_configs_load_with_plan_caps():
    g, a = load_governor_config(), load_autopilot_config()
    assert (g.caps.autopilot.yaw, g.caps.autopilot.pitch, g.caps.autopilot.roll, g.caps.autopilot.throttle) == (38, 32, 0, 0)
    assert g.wire.deadband_counts.yaw == 0 and g.takeoff.hold_s == 4.0 and g.landing.reassert_per_s == 5
    assert (g.flight_timer.warn_s, g.flight_timer.land_s) == (300, 390)
    assert g.twin_headless_bypass is False
    assert a.yaw.loop_gain == 0.4 and a.yaw.feedforward.enabled is False


def test_config_rejects_unknown_keys_and_bad_types():
    with pytest.raises(ConfigError):
        governor_config_from_dict({"caps": {"autopilot": {"yaw": 38, "yawn": 1}}})
    with pytest.raises(ConfigError):
        governor_config_from_dict({"human": {"stale_s": "quarter"}})
    with pytest.raises(ConfigError):
        governor_config_from_dict({"wire": {"sign": {"yaw": 2}}})
    with pytest.raises(ConfigError):
        autopilot_config_from_dict({"yaw": {"loop_gain": 0.8}})   # > 0.7 needs feed-forward


def test_headless_bypass_impossible_in_real_mode(tmp_path):
    cfg = replace(load_governor_config(), twin_headless_bypass=True)
    with pytest.raises(GovernorError):
        Governor(cfg, load_autopilot_config(), "real", log_dir=tmp_path, tick_log=False)
    with pytest.raises(GovernorError):
        Governor(load_governor_config(), load_autopilot_config(), "sim", log_dir=tmp_path, tick_log=False)


# --- wire conversion -------------------------------------------------------------------------

def test_symmetric_rounding_and_caps_in_counts():
    wire, caps = load_governor_config().wire, load_governor_config().caps.autopilot
    assert round_half_away(0.5) == 1 and round_half_away(-0.5) == -1 and round_half_away(1.49) == 1
    # +0.3 and -0.3 are mirror images (upstream gave +38 / -39)
    hc = load_governor_config().caps.human
    assert to_wire({"yaw": 0.3}, wire, hc, False)["yaw"] == 128 + 38
    assert to_wire({"yaw": -0.3}, wire, hc, False)["yaw"] == 128 - 38
    # caps are exact counts
    w = to_wire({"yaw": 1.0, "pitch": -1.0, "roll": 1.0, "throttle": 1.0}, wire, caps, True)
    assert w == {"yaw": 128 + 38, "pitch": 128 - 32, "roll": 128, "throttle": 128}
    assert to_wire({"yaw": float("nan")}, wire, caps, True)["yaw"] == 128


def test_sign_and_deadband_compensation():
    cfg = governor_config_from_dict({"wire": {"sign": {"yaw": -1}, "deadband_counts": {"yaw": 6}}})
    cap = 38
    assert axis_offset(0.1, "yaw", cfg.wire, cap, compensate=True) == -(6 + 13)    # sign, then d + |u|*127
    assert axis_offset(0.005, "yaw", cfg.wire, cap, compensate=True) == 0          # below threshold: center
    assert axis_offset(0.1, "yaw", cfg.wire, cap, compensate=False) == -13         # humans: no compensation
    assert axis_offset(1.0, "yaw", cfg.wire, cap, compensate=True) == -38          # cap after compensation


# --- flight state machine --------------------------------------------------------------------

def test_on_ground_sends_only_neutral(rig):
    r = rig()
    r.g.submit_human({"yaw": 1.0, "throttle": 1.0}, "abs", now=r.t)
    out = r.tick()
    assert out.flight_state == FlightState.on_ground and out.wire == NEUTRAL


def test_takeoff_flag_count_and_neutral_hold(rig):
    r = rig(gov_overrides={"takeoff.packets": 3})
    r.tick(2)
    assert r.g.command("takeoff").ok
    flags = []
    for _ in range(r.ticks_for(3.9)):
        r.g.submit_human({"pitch": 1.0}, "abs", now=r.t)      # ignored while TAKING_OFF
        out = r.tick()
        flags.append(out.takeoff)
        assert out.wire == NEUTRAL and out.flight_state == FlightState.taking_off
    assert flags[:4] == [True, True, True, False] and sum(flags) == 3
    r.run(0.2)
    assert r.g.state == FlightState.airborne


def test_takeoff_needs_arm_in_real_mode(rig):
    r = rig(mode="real")
    r.tick(2)
    assert r.g.command("takeoff").reason == "not_armed"
    assert r.g.command("arm", arm_phrase="arm").reason == "wrong_arm_phrase"
    assert r.g.command("arm", arm_phrase=" ARM ").ok
    assert r.g.command("takeoff").ok


def test_twin_is_auto_armed(rig):
    r = rig()
    assert r.g.armed
    r.tick(2)
    assert r.g.command("takeoff").ok


def test_takeoff_refused_without_heartbeat_or_video(rig):
    r = rig()
    r.hb = False
    r.run(2.0)
    assert r.g.command("takeoff").reason == "no_heartbeat"
    r2 = rig()
    r2.frames = False
    r2.tick(3)
    assert r2.g.command("takeoff").reason == "no_video"


def test_estop_is_latched_until_reset(rig):
    r = rig(mode="real")
    r.airborne()
    assert r.g.command("estop", source="ui").ok
    for _ in range(50):
        r.g.submit_human({"yaw": 1.0}, "abs", now=r.t)
        out = r.tick()
        assert out.stop and out.wire == NEUTRAL and not out.land and not out.takeoff
    for cmd in ("takeoff", "land", "resume", "confirm_touchdown"):
        assert not r.g.command(cmd).ok
        assert r.tick().stop
    assert r.g.command("estop_reset").ok
    assert not r.tick().stop and r.g.state == FlightState.on_ground
    assert r.g.command("takeoff").reason == "not_armed"     # real mode: ARM again after an E-stop


def test_land_reasserted_five_per_second_until_touchdown(rig):
    r = rig()
    r.airborne()
    assert r.g.command("land").ok
    lands = []
    for _ in range(r.ticks_for(2.0)):
        r.g.submit_human({"yaw": 1.0}, "abs", now=r.t)       # human sticks never cancel a landing
        out = r.tick()
        assert out.wire == NEUTRAL and out.flight_state == FlightState.landing
        lands.append(out.land)
    assert sum(lands) == 10                                    # 5 per second, evenly spaced
    idx = [i for i, v in enumerate(lands) if v]
    assert all(b - a == 5 for a, b in zip(idx, idx[1:]))
    assert r.g.command("confirm_touchdown").ok and r.g.state == FlightState.on_ground
    assert not any(o.land for o in [r.tick() for _ in range(10)])


def test_land_works_from_the_ground_too(rig):
    r = rig()
    r.tick(2)
    assert r.g.command("land").ok and r.tick().land


# --- priority and human input ----------------------------------------------------------------

def test_human_overrides_autopilot_then_autopilot_resumes(rig):
    r = rig()
    r.following(cx=900)                          # subject right of center: autopilot yaws right
    assert r.tick().source == "autopilot"
    r.g.submit_human({"yaw": -0.5}, "abs", now=r.t)
    out = r.tick()
    assert out.source == "human" and out.wire["yaw"] == 128 - 64
    r.run(0.3)                                   # no more human messages: stale after 0.25 s
    assert r.tick().source == "autopilot"


def test_all_zero_axes_are_not_input_and_release_immediately(rig):
    r = rig()
    r.following(cx=900)
    for _ in range(30):                          # an idle UI streaming zeros cannot block the autopilot
        assert r.g.submit_human({"yaw": 0.0, "pitch": 0.0}, "abs", now=r.t) is False
        assert r.tick().source == "autopilot"
    r.g.submit_human({"pitch": 0.5}, "abs", now=r.t)
    assert r.tick().source == "human"
    r.g.submit_human({"pitch": 0.0}, "abs", now=r.t)    # release: immediate, not after 0.25 s
    assert r.tick().source == "autopilot"


def test_human_input_stale_after_quarter_second(rig):
    r = rig()
    r.airborne()
    r.g.submit_human({"yaw": 0.4}, "abs", now=r.t)
    seen = [r.tick().wire["yaw"] for _ in range(10)]
    assert seen[0] == 128 + 51 and seen[6] == 128 + 51     # 0.24 s: still fresh
    assert seen[7] == 128 and seen[-1] == 128              # 0.28 s: zero


def test_gamepad_drift_is_not_input(rig):
    r = rig()
    r.airborne()
    assert r.g.submit_human({"yaw": 0.03, "pitch": -0.04}, "abs", now=r.t) is False
    assert r.tick().wire == NEUTRAL


def test_inc_mode_is_a_capped_ramp(rig):
    r = rig()
    r.airborne()
    yaws = []
    for _ in range(25):
        r.g.submit_human({"yaw": 1.0}, "inc", now=r.t)
        yaws.append(r.tick().wire["yaw"] - 128)
    assert yaws[0] == 5 and yaws[1] == 10                  # 1.0/s ramp = 5.08 counts per tick
    assert max(yaws) == 64 and yaws[-1] == 64             # capped at the human cap
    r.g.submit_human({"yaw": -1.0}, "inc", now=r.t)
    assert r.tick().wire["yaw"] - 128 == -5               # reversal goes through zero, no boost
    r.g.submit_human({"yaw": 0.0}, "inc", now=r.t)
    assert r.tick().wire["yaw"] == 128                    # release is immediate


# --- heartbeat, video, observation staleness -------------------------------------------------

def test_heartbeat_missing_neutral_then_land(rig):
    r = rig()
    r.following(cx=900)
    r.hb = False
    outs = [r.tick() for _ in range(r.ticks_for(6.0))]
    t_neutral = next(i for i, o in enumerate(outs) if o.wire == NEUTRAL) * 0.04
    t_land = next(i for i, o in enumerate(outs) if o.flight_state == FlightState.landing) * 0.04
    assert 1.4 <= t_neutral <= 1.6 and 4.9 <= t_land <= 5.1
    assert "heartbeat_missing" in outs[40].safety_flags


def test_headless_bypass_skips_heartbeat_in_twin(rig):
    r = rig(gov_overrides={"twin_headless_bypass": True})
    r.hb = False
    r.tick(2)
    assert r.g.command("takeoff").ok
    r.run(10.0)
    assert r.g.state == FlightState.airborne


def test_video_lag_neutralizes_autopilot_then_lands(rig):
    r = rig()
    r.following(cx=900)
    r.frames = False
    outs = [r.tick() for _ in range(r.ticks_for(2.5))]
    i_neutral = next(i for i, o in enumerate(outs) if o.source != "autopilot")
    i_land = next(i for i, o in enumerate(outs) if o.flight_state == FlightState.landing)
    assert 0.5 <= i_neutral * 0.04 <= 0.7 and 1.8 <= i_land * 0.04 <= 2.0


def test_video_stale_keeps_human_control(rig):
    r = rig()
    r.airborne()
    r.frames = False
    r.run(1.0)
    r.g.submit_human({"yaw": 0.5}, "abs", now=r.t)
    out = r.tick()
    assert "video_stale" in out.safety_flags and out.source == "human"


def test_constant_pipeline_latency_is_not_staleness(rig):
    r = rig()
    r.frame_latency = 1200.0          # e.g. twin with 1.2 s injected camera latency
    r.obs_latency = 1300.0
    r.following(cx=900)
    r.run(3.0)
    out = r.tick()
    assert out.source == "autopilot" and "video_stale" not in out.safety_flags


def test_killing_perception_gives_neutral_within_half_a_second(rig):
    r = rig()
    r.following(cx=1000)
    assert r.tick().wire["yaw"] > 128
    r.obs_fn = None                    # perception dies
    outs = [r.tick() for _ in range(25)]
    i = next(i for i, o in enumerate(outs) if o.wire == NEUTRAL)
    assert (i + 1) * 0.04 <= 0.5
    assert "obs_stale" in outs[i].safety_flags


def test_burst_of_old_observations_is_stale(rig):
    """A stall followed by a backlog burst: the late observations must not re-enable motion."""
    r = rig()
    r.following(cx=1000)
    last_capture = r.g.latest_obs.capture_ms
    r.obs_fn = None
    r.run(0.6)                                   # stall: nothing arrives
    assert r.tick().wire == NEUTRAL
    backlog = {"c": last_capture}

    def burst(t):                                # 3 old frames per tick, captures continue in order
        for _ in range(2):
            backlog["c"] += 40.0
            r._fid += 1
            r.g.submit_obs(make_obs(r._fid, backlog["c"], cx=1000), receive_ms=t)
        backlog["c"] += 40.0
        r._fid += 1
        return make_obs(r._fid, backlog["c"], cx=1000)
    r.obs_fn = burst
    outs = [r.tick() for _ in range(2)]
    assert all(o.wire == NEUTRAL and "obs_stale" in o.safety_flags for o in outs)


def test_safety_neutral_bypasses_slew(rig):
    r = rig(ap_overrides={"delay.theta_s": 0.3})   # high gain so the yaw cap binds
    r.following(cx=1200)               # far right: yaw at the cap
    r.run(1.0)
    assert r.tick().wire["yaw"] == 128 + 38
    r.obs_fn = None                    # perception stops: stale after 0.4 s
    yaws = [r.tick().wire["yaw"] for _ in range(15)]
    assert set(yaws) == {128 + 38, 128}            # straight from the cap to center, no ramp
    first = yaws.index(128)
    assert all(y == 128 for y in yaws[first:])


def test_autopilot_slew_limit(rig):
    r = rig(ap_overrides={"delay.theta_s": 0.3})   # high gain so the yaw cap binds
    r.airborne()
    r.g.set_mission(make_mission())
    r.steady_obs(cx=1200)
    yaws = [r.tick().wire["yaw"] - 128 for _ in range(r.ticks_for(3.0))]
    steps = [b - a for a, b in zip(yaws, yaws[1:])]
    assert max(steps) <= 6             # 1.0/s at 25 Hz = 5.08 counts per tick
    assert max(yaws) == 38


# --- flight timer ------------------------------------------------------------------------------

def test_flight_timer_warns_then_lands_and_persists(rig, tmp_path):
    r = rig(gov_overrides={"flight_timer.warn_s": 5.0, "flight_timer.land_s": 8.0})
    r.airborne()                                    # 4.2 s of flight already
    r.run(1.0)
    assert "flight_timer_warn" in r.tick().safety_flags
    r.run(3.0)
    assert r.g.state == FlightState.landing
    r.g.flight_timer.save_if_dirty()
    data = json.loads((tmp_path / "flight_timer.json").read_text())
    assert data["packs"][data["current_pack"]]["flight_s"] >= 8.0
    assert not r.g.command("pack_swap", pack_id="2").ok          # not while flying
    r.g.command("confirm_touchdown")
    assert r.g.command("pack_swap", pack_id="2").ok and r.g.flight_timer.flight_s() == 0.0


def test_resume_cancels_timer_landing_once(rig):
    r = rig(gov_overrides={"flight_timer.warn_s": 3.0, "flight_timer.land_s": 5.0})
    r.airborne()
    r.run(1.0)
    assert r.g.state == FlightState.landing
    assert r.g.command("resume").ok and r.g.state == FlightState.airborne
    r.run(2.0)
    assert r.g.state == FlightState.airborne       # timer landing fires once per flight


def test_resume_cannot_beat_a_persisting_safety_cause(rig):
    r = rig()
    r.airborne()
    r.hb = False
    r.run(5.2)
    assert r.g.state == FlightState.landing
    assert r.g.command("resume", source="http").ok
    r.tick()
    assert r.g.state == FlightState.landing        # heartbeat still missing: lands again


def test_resume_after_landing_keeps_autopilot_paused(rig):
    r = rig()
    r.following(cx=900)
    r.g.command("land")
    r.tick()
    assert r.g.command("resume").ok and r.g.state == FlightState.airborne
    assert r.tick().source == "none"               # first resume: hover
    assert r.g.command("resume").ok
    r.tick()
    assert r.tick().source == "autopilot"          # second resume: follow again


# --- wrong-way detector -----------------------------------------------------------------------

def test_wrong_way_detector_hovers_and_alerts(rig):
    r = rig()
    r.following(cx=900)
    # The drone yaws right (positive) but the subject drifts further right: a sign error.
    state = {"cx": 900.0}

    def fn(t):
        state["cx"] = min(1250.0, state["cx"] + 6.0)
        r._fid += 1
        return make_obs(r._fid, t - r.obs_latency, cx=state["cx"])
    r.obs_fn = fn
    fired = None
    for i in range(r.ticks_for(4.0)):
        out = r.tick()
        if "wrong_way" in out.safety_flags:
            fired = i
            break
    assert fired is not None
    assert r.tick().source == "none" and r.tick().wire == NEUTRAL
    assert any(e.get("text", "").startswith("wrong-way") for e in r.g.drain_events())
    r.steady_obs(cx=700)
    assert r.g.command("resume").ok
    r.tick(3)
    assert r.tick().source == "autopilot"


def test_wrong_way_quiet_when_tracking_works(rig):
    r = rig()
    r.following(cx=1000)
    # Subject bearing shrinks as we turn toward it: consistent with the commanded yaw.
    state = {"cx": 1000.0}

    def fn(t):
        state["cx"] = max(640.0, state["cx"] - 4.0)
        r._fid += 1
        return make_obs(r._fid, t - r.obs_latency, cx=state["cx"])
    r.obs_fn = fn
    outs = [r.tick() for _ in range(r.ticks_for(4.0))]
    assert not any("wrong_way" in o.safety_flags for o in outs)


# --- commands, missions, sources ----------------------------------------------------------------

def test_agent_can_never_estop_or_take_off(rig):
    r = rig()
    r.tick(2)
    for cmd in ("estop", "takeoff", "estop_reset", "resume", "arm"):
        assert r.g.command(cmd, source="agent").reason == "not_allowed_from_agent"
    assert r.g.state == FlightState.on_ground
    r.airborne()
    assert r.g.command("pause", source="agent").ok
    assert r.g.command("land", source="agent").ok and r.g.state == FlightState.landing


def test_missions_follow_hover_land(rig):
    r = rig()
    r.airborne()
    assert r.g.set_mission(make_mission(action="follow"))["find"]
    assert r.g.autopilot.mode.value == "FIND"
    r.g.set_mission(make_mission(action="hover"))
    assert "autopilot_paused" in r.tick().safety_flags
    r.g.set_mission(make_mission(action="land"))
    assert r.g.state == FlightState.landing


def test_obs_from_another_mission_is_ignored(rig):
    r = rig()
    r.airborne()
    r.g.set_mission(make_mission("m2"))
    assert not r.g.submit_obs(make_obs(1, r.t - 100, mission_id="m1"), receive_ms=r.t)
    assert r.g.submit_obs(make_obs(2, r.t - 100, mission_id="m2"), receive_ms=r.t)
    assert not r.g.submit_obs(make_obs(3, r.t + 500, mission_id="m2"), receive_ms=r.t)   # future capture


def test_tilt_is_human_only_and_times_out(rig):
    r = rig(gov_overrides={"tilt.max_hold_s": 1.0})
    r.airborne()
    r.g.command("tilt_up")
    assert r.tick().tilt_state == 2
    r.run(1.1)
    assert r.tick().tilt_state == 0
    r.g.command("tilt_down")
    assert r.tick().tilt_state == 1
    r.g.command("tilt_stop")
    assert r.tick().tilt_state == 0
    r.g.command("tilt_up")
    r.g.ui_disconnected()
    assert r.tick().tilt_state == 0


def test_autopilot_exception_gives_neutral_and_flag(rig, monkeypatch):
    r = rig()
    r.following(cx=1000)
    import mhd.governor.core as core

    def boom(*a, **k):
        raise RuntimeError("synthetic autopilot failure")
    monkeypatch.setattr(core, "autopilot_step", boom)
    out = r.tick()
    assert out.wire == NEUTRAL and "autopilot_fault" in out.safety_flags


def test_shutdown_sends_neutral_plus_land(rig):
    r = rig()
    r.following(cx=1000)
    r.g.begin_shutdown()
    outs = [r.tick() for _ in range(5)]
    assert all(o.land and o.wire == NEUTRAL for o in outs)


def test_tick_under_one_millisecond_while_following(rig):
    r = rig()
    r.following(cx=900)
    r.run(10.0)
    stats = r.g.tick_stats()
    assert stats["p99_us"] < 1000.0, stats


def test_throttle_ramp_landing_fallback(rig):
    r = rig(gov_overrides={"landing.throttle_ramp": True, "landing.throttle_down": 0.35,
                           "landing.throttle_ramp_s": 1.0})
    r.airborne()
    r.g.command("land")
    outs = [r.tick() for _ in range(r.ticks_for(2.0))]
    thr = [o.wire["throttle"] - 128 for o in outs]
    assert thr[0] == 0 and thr[12] == -round(0.35 * 127 * 12 * 0.04) and min(thr) == -44
    assert all(b <= a for a, b in zip(thr, thr[1:]))            # monotonic ramp down
    assert all(o.wire["yaw"] == 128 and o.wire["pitch"] == 128 for o in outs)
    assert sum(o.land for o in outs) == 10                      # flag still re-asserted 5/s
    r.g.command("confirm_touchdown")
    assert r.tick().wire["throttle"] == 128


def test_throttle_ramp_is_off_by_default(rig):
    r = rig()
    r.airborne()
    r.g.command("land")
    assert all(r.tick().wire["throttle"] == 128 for _ in range(50))
