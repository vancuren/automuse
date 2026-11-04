"""X69 twin dynamics: flags, lags, deadband, tilt, headless, link loss, battery, fault injection."""
from __future__ import annotations

import copy
import math

import pytest

from mhd.twin import protocol as P
from mhd.twin.dynamics import (FALLING, FLYING, LANDED, LANDING, RcFaultInjector, RcLinkStats, X69Dynamics,
                               load_params)

RC_DT = 0.04   # 25 Hz like turbodrone's FlightController


def make(**overrides) -> X69Dynamics:
    params = copy.deepcopy(load_params())
    for dotted, value in overrides.items():
        node = params
        keys = dotted.split("__")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    d = X69Dynamics(params, seed=3)
    d.drift_scale = 0.0          # deterministic unless a test turns drift on
    return d


def fly(d: X69Dynamics, seconds: float, flags1: int = 0, flags2: int = 0, first_only: bool = True, **sticks):
    """Send RC packets at 25 Hz for `seconds` (flags in the first packet only, like one-shot flags)."""
    n = int(round(seconds / RC_DT))
    for i in range(n):
        f1 = flags1 if (i == 0 or not first_only) else 0
        f2 = flags2 if (i == 0 or not first_only) else 0
        pkt = P.decode_datagram(P.encode_control(flags1=f1, flags2=f2, **sticks))
        d.on_packet(pkt, d.t)
        d.advance_to(d.t + RC_DT)


def takeoff(d: X69Dynamics) -> None:
    fly(d, 4.0, flags1=P.FLAG_TAKEOFF)
    assert d.s.phase == FLYING


def test_takeoff_climbs_to_hover_height_and_holds():
    d = make()
    fly(d, 0.5)
    assert d.s.phase == LANDED and d.s.z == 0.0
    takeoff(d)
    hover = d.p["altitude"]["hover_height_m"]
    assert abs(d.s.z - hover) < 0.02 and d.s.motors_on
    fly(d, 3.0)                          # neutral sticks: altitude hold
    assert abs(d.s.z - hover) < 0.02
    # repeated takeoff flags while flying are no-ops
    fly(d, 0.5, flags1=P.FLAG_TAKEOFF, first_only=False)
    assert d.events.takeoff == 1


def test_yaw_rate_per_count_and_deadband():
    d = make()
    takeoff(d)
    db = d.p["sticks"]["deadband_counts"]["yaw"]
    fly(d, 1.0, yaw=128 + db)            # inside the deadband: no rotation
    assert abs(d.s.yaw_rate) < 1e-6
    fly(d, 2.0, yaw=255)                 # full right
    expected = d.p["yaw"]["rate_per_count_dps"] * (127 - db)
    assert math.degrees(d.s.yaw_rate) == pytest.approx(expected, rel=0.02)
    yaw0 = d.s.yaw
    fly(d, 1.0, yaw=255)
    assert math.degrees((d.s.yaw - yaw0) % (2 * math.pi)) == pytest.approx(expected, rel=0.05)
    fly(d, 2.0, yaw=0)                   # full left turns the other way
    assert d.s.yaw_rate < 0


def test_first_order_lag_on_yaw():
    d = make()
    takeoff(d)
    tau = d.p["yaw"]["tau_s"]
    fly(d, 0.04, yaw=255)
    fly(d, tau - 0.04, yaw=255)
    expected = d.p["yaw"]["rate_per_count_dps"] * (127 - d.p["sticks"]["deadband_counts"]["yaw"])
    frac = math.degrees(d.s.yaw_rate) / expected
    assert 0.55 < frac < 0.72          # about 1 - 1/e after one time constant


def test_forward_flight_pitches_camera_down_then_levels():
    d = make()
    takeoff(d)
    pitch0 = d.camera_pose()["pitch"]
    fly(d, 0.3, pitch=200)
    assert d.s.tilt_fwd > math.radians(1.0)          # nose down while accelerating
    assert d.camera_pose()["pitch"] < pitch0
    fly(d, 4.0, pitch=200)
    v_expected = d.p["pitch"]["speed_per_count_mps"] * (72 - d.p["sticks"]["deadband_counts"]["pitch"])
    assert d.s.vy == pytest.approx(v_expected, rel=0.1)  # yaw 0 = facing north = +y
    assert abs(d.s.vx) < 1e-6
    assert abs(d.s.tilt_fwd) < math.radians(2.0)     # near level at constant speed (drag only)
    fly(d, 3.0)                                      # release: brakes to a stop
    assert abs(d.s.vy) < 0.05


def test_land_descends_and_disarms_and_land_on_ground_is_noop():
    d = make()
    fly(d, 0.2, flags1=P.FLAG_LAND)
    assert d.s.phase == LANDED and d.events.land == 0
    takeoff(d)
    fly(d, 0.2, flags1=P.FLAG_LAND)
    assert d.s.phase == LANDING
    fly(d, 4.0)
    assert d.s.phase == LANDED and not d.s.motors_on and d.s.z == 0.0


def test_estop_cuts_motors_and_falls():
    d = make()
    takeoff(d)
    fly(d, 0.04, flags1=P.FLAG_ESTOP)
    assert not d.s.motors_on and d.s.phase in (FALLING, LANDED)
    fly(d, 1.0)
    assert d.s.phase == LANDED and d.s.z == 0.0
    # sticks do nothing after an E-stop; a new takeoff is needed
    fly(d, 1.0, throttle=255, yaw=255)
    assert d.s.z == 0.0 and d.s.yaw_rate == 0.0
    # E-stop in the same packet as takeoff wins
    fly(d, 1.0, flags1=P.FLAG_ESTOP | P.FLAG_TAKEOFF)
    assert d.s.phase == LANDED


def test_link_loss_lands_after_timeout():
    d = make()
    takeoff(d)
    timeout = d.p["link"]["loss_timeout_s"]
    d.advance_to(d.t + timeout * 0.8)              # short gap: fine
    assert not d.s.link_lost and d.s.phase == FLYING
    d.advance_to(d.t + timeout * 0.5)              # now over the timeout
    assert d.s.link_lost and d.s.phase == LANDING and d.s.landing_reason == "link_loss"
    d.advance_to(d.t + 4.0)
    assert d.s.phase == LANDED


def test_link_loss_hover_mode_keeps_altitude():
    d = make(link__loss_action="hover")
    takeoff(d)
    fly(d, 1.0, yaw=255)
    d.advance_to(d.t + 3.0)
    assert d.s.link_lost and d.s.phase == FLYING and abs(d.s.yaw_rate) < 0.01


def test_tilt_moves_one_step_per_press():
    d = make()
    step = math.radians(d.p["camera"]["tilt_step_deg"])
    t0 = d.s.cam_tilt
    fly(d, 0.4, flags2=P.FLAG2_TILT_DOWN, first_only=False)      # held for 10 packets
    assert d.s.cam_tilt == pytest.approx(t0 - step)
    fly(d, 0.1)                                                  # release
    fly(d, 0.1, flags2=P.FLAG2_TILT_DOWN, first_only=False)      # second press
    assert d.s.cam_tilt == pytest.approx(t0 - 2 * step)
    fly(d, 0.1, flags2=P.FLAG2_TILT_UP)
    assert d.s.cam_tilt == pytest.approx(t0 - step)
    for _ in range(30):                                          # clamps at the servo limit
        fly(d, 0.04); fly(d, 0.04, flags2=P.FLAG2_TILT_DOWN)
    assert d.s.cam_tilt == pytest.approx(math.radians(d.p["camera"]["tilt_min_deg"]))


def test_tilt_while_held_mode():
    d = make(camera__tilt_mode="while_held")
    t0 = d.s.cam_tilt
    fly(d, 1.0, flags2=P.FLAG2_TILT_DOWN, first_only=False)
    rate = math.radians(d.p["camera"]["tilt_rate_dps"])
    assert d.s.cam_tilt == pytest.approx(t0 - rate * 1.0, abs=math.radians(1.5))


def test_headless_uses_takeoff_heading():
    d = make()
    d.s.headless = True
    takeoff(d)
    fly(d, 0.75, yaw=255)            # turn about 80-90 degrees right
    assert math.degrees(d.s.yaw) > 60
    fly(d, 4.0, pitch=200)           # forward = takeoff heading (north, +y), not the nose
    assert d.s.vy > 0.5 and abs(d.s.vx) < 0.1


def test_battery_empty_lands_and_refuses_takeoff():
    d = make(battery__flight_time_s=6.0)
    takeoff(d)
    fly(d, 3.0)
    assert d.s.phase == LANDING and d.s.landing_reason == "battery"
    fly(d, 4.0)
    assert d.s.phase == LANDED
    fly(d, 1.0, flags1=P.FLAG_TAKEOFF)
    assert d.s.phase == LANDED and d.events.ignored_takeoff == 1


def test_drift_is_a_bounded_random_walk_and_worse_on_low_texture():
    def run(texture):
        d = make()
        d.drift_scale = 1.0
        d.texture = texture
        takeoff(d)
        x0, y0 = d.s.x, d.s.y
        fly(d, 30.0)
        return math.hypot(d.s.x - x0, d.s.y - y0)
    good = sum(run(1.0) for _ in range(3)) / 3
    poor = sum(run(0.0) for _ in range(3)) / 3
    assert good < 3.0          # meters over 30 s of hover
    assert poor > good


def test_arena_bounds_stop_the_drone():
    d = make()
    d.bounds = (-1.0, -1.0, 1.0, 1.5)
    takeoff(d)
    fly(d, 6.0, pitch=255)
    assert d.s.y == pytest.approx(1.5) and d.s.collisions > 0


def test_fault_injector_loss_burst_and_jitter_order():
    inj = RcFaultInjector(seed=5)
    inj.configure(loss=0.3)
    out = [inj.admit(i * RC_DT) for i in range(2000)]
    lost = sum(o is None for o in out) / len(out)
    assert 0.25 < lost < 0.35
    inj = RcFaultInjector(seed=5)
    inj.configure(burst_rate=0.02, burst_len=10)
    out = [inj.admit(i * RC_DT) for i in range(5000)]
    runs, cur = [], 0
    for o in out:
        if o is None:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    assert runs and sum(runs) / len(runs) > 4          # losses come in bursts
    inj = RcFaultInjector(seed=5)
    inj.configure(jitter_ms=100)
    rel = [inj.admit(i * RC_DT) for i in range(500)]
    assert all(b >= a for a, b in zip(rel, rel[1:]))   # never reordered
    assert max(r - i * RC_DT for i, r in enumerate(rel)) <= 0.1 + 1e-9 + 0.1


def test_link_stats_p99_alarm():
    st = RcLinkStats(window=500, alarm_ms=60)
    t = 0.0
    for _ in range(300):
        t += 0.04
        st.packet(t)
    s = st.summary()
    assert s["interarrival_p50_ms"] == pytest.approx(40, abs=0.5) and not s["p99_alarm"]
    for i in range(300):
        t += 0.04 if i % 20 else 0.12        # 5% of gaps are 120 ms
        st.packet(t)
    s = st.summary()
    assert s["interarrival_p99_ms"] >= 100 and s["p99_alarm"]
