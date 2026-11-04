"""Closed loop with a REAL-mode governor and the base configs (config/*.yaml): the plant's pitch sign
is reversed relative to the config, the drone is 30 % slower or faster per count than modeled, and it
has a firmware deadband the config compensates (Session B) or does not. Reviews 2 and 3
(2026-11-04, logs/review2/, logs/review3/): the forward check must stop a reversed sign within
about 1.5 m (2 m with an unmeasured deadband), the 64 px gate keeps far subjects out of the check's
blind zone, and with autopilot backing off the drone never moves toward the person. With a correct
sign, a stationary subject never trips anything. See closed_loop_sim.py for the model.
"""
from __future__ import annotations

import itertools

import pytest

from closed_loop_sim import run
from conftest import ROOT

CFG = ROOT / "config"
SPEEDS = (0.7, 1.0, 1.3)
DEADBANDS = ((0, 0), (6, 6), (10, 10))      # (plant deadband, compensated by the config)


def sim(spec, tmp_path, *, sign=-1.0, speed=1.0, db=(0, 0), dur=40.0, gait=0.0):
    over = {"wire.deadband_counts.pitch": db[1]} if db[1] else None
    return run(spec, cfg_dir=CFG, log_dir=tmp_path, pitch_sign=sign, speed_scale=speed, plant_db=db[0],
               gov_over=over, dur_s=dur, gait_amp=gait)


@pytest.mark.parametrize("spec", ["stat:3.5", "stat:4.0", "stat:5.0"])
def test_reversed_pitch_forward_runaway_latches_within_1_5_m(spec, tmp_path):
    for speed, db in itertools.product(SPEEDS, DEADBANDS):
        r = sim(spec, tmp_path, speed=speed, db=db)
        assert r["lock"]
        assert r["ww_reason"] == "wrong_way:pitch", (spec, speed, db, r)
        assert r["max_away_m"] <= 1.5, (spec, speed, db, r["max_away_m"])
        assert r["final_ap"] == "FOLLOW" and r["travel_toward_m"] == 0.0


def test_reversed_pitch_with_an_uncompensated_deadband_stays_bounded(tmp_path):
    # Slowest case: a 6-count firmware deadband nobody measured. The drift can be too slow for any
    # 2 s window; the wrong-way latch or the 64 px forward gate stops it within 2 m.
    for speed in SPEEDS:
        r = sim("stat:3.5", tmp_path, speed=speed, db=(6, 0), dur=150.0)
        assert r["max_away_m"] <= 2.0, (speed, r["max_away_m"])


@pytest.mark.parametrize("spec", ["stat:6.5", "stat:9.0"])
def test_reversed_pitch_never_flies_forward_toward_a_far_subject(spec, tmp_path):
    # Beyond the 64 px gate (a person farther than about 5.9 m) the autopilot only turns, so the
    # check's blind zone for far subjects (review 3: 11-23 m runaways) is never entered.
    for speed in (0.5, 1.0, 1.5):
        r = sim(spec, tmp_path, speed=speed, dur=40.0)
        assert r["max_away_m"] <= 0.1 and r["travel_toward_m"] <= 0.05, (spec, speed, r)


@pytest.mark.parametrize("spec", ["stat:2.0", "stat:2.5", "walk:4.0:0.5:1.5"])
def test_reversed_pitch_never_moves_the_drone_toward_the_person(spec, tmp_path):
    # The autopilot never backs up (forward.backward_budget_m 0): backing is its only way toward
    # the person under a reversed sign. Even 1.5x faster than modeled and with any deadband.
    for speed, db in itertools.product(SPEEDS + (1.5,), DEADBANDS + ((6, 0),)):
        r = sim(spec, tmp_path, speed=speed, db=db)
        assert r["travel_toward_m"] <= 0.02, (spec, speed, db, r["travel_toward_m"])
        assert r["min_dist_m"] >= 1.45, (spec, speed, db, r["min_dist_m"])


@pytest.mark.parametrize("spec", ["stat:2.5", "stat:3.5"])
def test_correct_sign_stationary_subject_trips_nothing(spec, tmp_path):
    for speed, db, gait in itertools.product(SPEEDS, ((0, 0), (6, 6)), (0.0, 0.2)):
        r = sim(spec, tmp_path, sign=1.0, speed=speed, db=db, gait=gait, dur=30.0)
        assert r["ww_t_s"] is None and r["lockout_t_s"] is None and r["probes"] == 0, (spec, speed, db, gait, r)


def test_correct_sign_slow_walker_is_followed_without_a_hover(tmp_path):
    for speed, db, gait in itertools.product(SPEEDS, ((0, 0), (6, 6)), (0.0, 0.1, 0.2)):
        r = sim("away:2.9:0.2:8", tmp_path, sign=1.0, speed=speed, db=db, gait=gait, dur=40.0)
        assert r["ww_t_s"] is None, (speed, db, gait, r)
        assert r["max_away_m"] <= 0.15 and r["final_disp_m"] > 3.0      # it followed (moved forward)


def test_reversed_yaw_latches(tmp_path):
    for k in (0.7, 1.0, 1.3):
        r = run("stat:3.0", cfg_dir=CFG, log_dir=tmp_path, pitch_sign=1.0, yaw_sign=-1.0, k_scale=k,
                dur_s=30.0, start_angle_deg=15.0)
        assert r["ww_reason"] == "wrong_way:yaw", (k, r)
