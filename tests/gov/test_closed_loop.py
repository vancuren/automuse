"""Closed loop: the real Governor + autopilot flying the in-process kinematic sim (mhd.autopilot.sim).

Reports bearing-error RMS and step overshoot at total delays of 0.6, 1.0 and 1.5 s and asserts the
PLAN 4.4 acceptance where the default (P-only) controller is expected to meet it.
"""
from __future__ import annotations

import pytest

from mhd.autopilot.sim import SimParams, run
from mhd.governor.config import load_governor_config, replace

DELAYS = (0.6, 1.0, 1.5)


@pytest.mark.parametrize("delay", DELAYS)
def test_yaw_step_overshoot_and_settling(delay, record_property):
    r = run(SimParams(total_delay_s=delay, scenario="step", duration_s=20))
    print("\n" + r.row())
    record_property("step", r.row())
    assert r.final_state == "FOLLOW"
    assert r.overshoot_pct <= 10.0                        # PLAN: overshoot <= 10%
    assert r.settle_s <= 7 * delay                        # PLAN: settling within 7 theta


@pytest.mark.parametrize("delay", DELAYS)
def test_s1_walk_bearing_rms(delay, record_property):
    r = run(SimParams(total_delay_s=delay, scenario="s1"))
    print("\n" + r.row())
    record_property("s1", r.row())
    assert r.final_state == "FOLLOW" and r.in_frame_frac == 1.0
    assert r.min_distance_m > 2.0                         # never closes in on the subject
    if delay <= 1.0:
        assert r.bearing_rms_rad <= 0.15                  # PLAN: <= 0.15 rad at 0.06 rad/s
    else:
        # P-only lag on a ramp is ~2.5*omega*theta = 0.225 rad at 1.5 s: reported, not asserted.
        assert r.bearing_rms_rad <= 0.25


@pytest.mark.parametrize("k_scale", (0.7, 1.3))
@pytest.mark.parametrize("delay", DELAYS)
def test_feedforward_robust_to_30pct_gain_error(delay, k_scale):
    """PLAN: feed-forward stays off until a run with +/-30% error in K shows no ringing."""
    step = run(SimParams(total_delay_s=delay, scenario="step", duration_s=20, feedforward=True,
                         k_true_scale=k_scale))
    walk = run(SimParams(total_delay_s=delay, scenario="s1", feedforward=True, k_true_scale=k_scale))
    print("\n" + step.row() + "\n" + walk.row())
    assert step.overshoot_pct <= 10.0
    assert walk.bearing_rms_rad <= 0.15


@pytest.mark.parametrize("delay", DELAYS)
def test_reversed_yaw_sign_is_caught_by_wrong_way_detector(delay):
    """A sign error on the yaw axis makes P control run away; the governor must stop it."""
    r = run(SimParams(total_delay_s=delay, scenario="step", duration_s=20, k_true_scale=-1.0))
    assert any(e.get("text", "").startswith("wrong-way") for e in r.events)
    assert r.bearing_max_rad < 0.8                        # stopped near the frame edge
    off = run(SimParams(total_delay_s=delay, scenario="step", duration_s=20, k_true_scale=-1.0),
              gov_cfg=replace(load_governor_config(), **{"wrong_way.enabled": False}))
    assert off.bearing_max_rad > r.bearing_max_rad + 0.5  # without it the drone keeps turning away
