"""Yaw wrong-way check against a person walking sideways (flight 2026-11-04 17:42).

On the real drone the yaw check latched 1.3 s into the first FOLLOW and stayed latched for three
flights: a person who moves sideways faster than the drone turns looks like a reversed yaw sign.
closed_loop_sim reproduces it. wrong_way.yaw_enabled: false lets a config switch the yaw check off
(the pitch and frozen-box checks stay); a reversed yaw sign then only turns the drone in place.
"""
from __future__ import annotations

from closed_loop_sim import run
from conftest import ROOT

CFG = ROOT / "config"
OFF = {"wrong_way.yaw_enabled": False}


def test_sideways_walker_trips_the_yaw_check_with_a_correct_sign(tmp_path):
    # Documents the false positive: correct signs, a person walking sideways at 1 m/s, 2.5 m away.
    r = run("side:2.5:1.0:2.5", cfg_dir=CFG, log_dir=tmp_path, pitch_sign=1.0, path_from="mission", dur_s=20.0)
    assert r["lock"] and r["ww_reason"] == "wrong_way:yaw" and r["ww_t_s"] < 5.0, r


def test_yaw_check_off_follows_the_sideways_walker(tmp_path):
    for k in (0.6, 1.0, 1.3):
        for vx in (0.5, 1.0):
            r = run(f"side:2.5:{vx}:2.5", cfg_dir=CFG, log_dir=tmp_path, pitch_sign=1.0, k_scale=k,
                    path_from="mission", dur_s=20.0, gov_over=OFF)
            assert r["lock"] and r["ww_t_s"] is None, (k, vx, r)


def test_yaw_check_off_reversed_yaw_never_moves_toward_the_subject(tmp_path):
    # Without the yaw check a reversed yaw sign turns the drone away until the subject is lost.
    # It must not travel: forward needs |bearing| < 0.4 rad.
    for k in (0.6, 1.0, 1.3):
        r = run("stat:3.0", cfg_dir=CFG, log_dir=tmp_path, pitch_sign=1.0, yaw_sign=-1.0, k_scale=k,
                dur_s=30.0, start_angle_deg=15.0, gov_over=OFF)
        assert r["ww_t_s"] is None and r["travel_toward_m"] <= 0.05 and r["path_m"] <= 0.05, (k, r)


def test_yaw_check_stays_on_by_default_and_still_catches_a_reversed_yaw(tmp_path):
    r = run("stat:3.0", cfg_dir=CFG, log_dir=tmp_path, pitch_sign=1.0, yaw_sign=-1.0, dur_s=30.0,
            start_angle_deg=15.0)
    assert r["ww_reason"] == "wrong_way:yaw", r
