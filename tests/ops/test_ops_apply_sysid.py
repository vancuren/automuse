"""scripts/apply_sysid.py on synthetic sysid results (the format mhd.governor.sysid.run writes)."""
from __future__ import annotations

import importlib.util
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import pytest

from mhd.governor.config import load_autopilot_config, load_governor_config
from mhd.governor.sysid import build_schedule, fit_steps

TREE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("apply_sysid", TREE / "scripts" / "apply_sysid.py")
apply_sysid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply_sysid)


def synth(axis: str, amps, k: float, db: float, lag_s: float, native: int, hold_s: float, hfov_used=75.0,
          seed=1) -> dict:
    """Simulate a sysid run: yaw -> bearing of a fixed marker, pitch -> range to it, at 15 fps."""
    rng = random.Random(seed)
    steps = build_schedule(axis, list(amps), hold_s, 2.0)
    t = 1000.0
    for s in steps:
        s.t_start_ms, s.t_end_ms = t, t + hold_s * 1000
        s.wire_count = int(round(s.value * 127))
        t = s.t_end_ms + s.rest_s * 1000
    t_end = t
    samples, bearing, rng_m = [], 0.0, 2.5
    dt = 1000 / 15
    tt = 0.0
    while tt < t_end + 500:
        rate = 0.0
        for s in steps:
            if s.t_start_ms + lag_s * 1000 <= tt < s.t_end_ms + lag_s * 1000:
                c = s.wire_count
                rate = native * math.copysign(k * max(0.0, abs(c) - db), c)
        if axis == "yaw":
            bearing -= rate * dt / 1000          # clockwise turn: a fixed marker moves left
        else:
            rng_m -= rate * dt / 1000            # forward: the range shrinks
        # what the run measured with its own HFOV (here the true one unless stated)
        samples.append((tt, bearing + rng.gauss(0, 0.002), rng_m + rng.gauss(0, 0.004)))
        tt += dt
    fit = fit_steps(steps, samples, axis)
    return {"axis": axis, "amps": list(amps), "hold_s": hold_s, "rest_s": 2.0, "hfov_deg": hfov_used,
            "marker_size_m": 0.15, "aborted": None, "schedule": [asdict(s) for s in steps],
            "samples": samples, "fit": fit}


@pytest.fixture
def runs(tmp_path):
    y = synth("yaw", (0.10, 0.20, 0.35), k=0.010, db=5, lag_s=0.25, native=+1, hold_s=1.0)
    p = synth("pitch", (0.10, 0.20, 0.25), k=0.012, db=6, lag_s=0.35, native=-1, hold_s=1.5, seed=2)
    (tmp_path / "sysid-yaw.json").write_text(json.dumps(y))
    (tmp_path / "sysid-pitch.json").write_text(json.dumps(p))
    return tmp_path


def args(runs, *extra):
    return ["--yaw", str(runs / "sysid-yaw.json"), "--pitch", str(runs / "sysid-pitch.json"),
            "--out-dir", str(runs / "real"), "--base", str(TREE / "config"),
            "--sysid-capture-offset-ms", "500", *extra]


def test_dry_run_writes_nothing(runs, capsys):
    assert apply_sysid.main(args(runs, "--hfov-deg", "75", "--latency-ms", "500")) == 0
    out = capsys.readouterr().out
    assert "dry run: nothing written" in out and "+  yaw_rate_per_count: 0.009" in out
    assert not (runs / "real").exists()


def test_write_applies_only_the_measured_values(runs, capsys):
    assert apply_sysid.main(args(runs, "--hfov-deg", "78.4", "--latency-ms", "500", "--write")) == 0
    out = capsys.readouterr().out
    assert "PITCH AXIS SIGN FLIPPED: 1 -> -1" in out
    ap = load_autopilot_config(runs / "real" / "autopilot.yaml")
    gv = load_governor_config(runs / "real" / "governor.yaml")
    # The run used HFOV 75; the samples are re-fitted at 78.4, which scales K by about
    # tan(39.2 deg) / tan(37.5 deg) = 1.064 near the image center.
    assert ap.plant.yaw_rate_per_count == pytest.approx(0.010 * 1.064, rel=0.08)
    assert ap.plant.forward_speed_per_count == pytest.approx(0.012 / 1.064, rel=0.10)
    assert ap.camera.hfov_deg == 78.4
    assert 0.2 <= ap.plant.yaw_lag_s <= 0.45
    assert ap.delay.theta_s == pytest.approx(0.5 + 0.06 + 0.02 + 0.02 + ap.plant.yaw_lag_s, abs=0.002)
    assert gv.wire.sign.get("yaw") == 1 and gv.wire.sign.get("pitch") == -1
    assert abs(gv.wire.deadband_counts.get("yaw") - 5) <= 1
    assert abs(gv.wire.deadband_counts.get("pitch") - 6) <= 1
    # Everything else is the default config, comments included.
    base_ap = load_autopilot_config(TREE / "config" / "autopilot.yaml")
    base_gv = load_governor_config(TREE / "config" / "governor.yaml")
    assert ap.yaw == base_ap.yaw and ap.forward == base_ap.forward and ap.lifecycle == base_ap.lifecycle
    assert gv.caps == base_gv.caps and gv.landing == base_gv.landing and gv.heartbeat == base_gv.heartbeat
    text = (runs / "real" / "autopilot.yaml").read_text()
    assert "# K: yaw rate (rad/s) per wire count" in text and "[measured" in text
    assert text.startswith("# REAL-DRONE calibration of config/autopilot.yaml")


def test_stage_switches_are_incremental(runs, capsys):
    assert apply_sysid.main(args(runs, "--hfov-deg", "78.4", "--latency-ms", "500", "--write")) == 0
    k1 = load_autopilot_config(runs / "real" / "autopilot.yaml").plant.yaw_rate_per_count
    # C1: yaw only at 60 % gain, starting from the file just written (default base = out dir)
    assert apply_sysid.main(["--out-dir", str(runs / "real"), "--yaw-only", "on", "--yaw-gain-scale", "0.6",
                             "--feedforward", "on", "--throttle-ramp", "on", "--write"]) == 0
    ap = load_autopilot_config(runs / "real" / "autopilot.yaml")
    gv = load_governor_config(runs / "real" / "governor.yaml")
    assert ap.plant.yaw_rate_per_count == k1 and ap.camera.hfov_deg == 78.4      # kept
    assert ap.yaw.loop_gain == pytest.approx(0.24) and ap.yaw.feedforward.enabled is True
    assert gv.caps.autopilot.get("pitch") == 0 and gv.landing.throttle_ramp is True
    # C2: forward back on at full gain
    assert apply_sysid.main(["--out-dir", str(runs / "real"), "--yaw-only", "off", "--yaw-gain-scale", "1.0",
                             "--write"]) == 0
    gv = load_governor_config(runs / "real" / "governor.yaml")
    assert gv.caps.autopilot.get("pitch") == 32
    assert load_autopilot_config(runs / "real" / "autopilot.yaml").yaw.loop_gain == pytest.approx(0.4)


def test_out_of_range_and_protected_paths_are_refused(runs, tmp_path, capsys):
    fast = synth("yaw", (0.10, 0.20, 0.35), k=0.1, db=5, lag_s=0.25, native=+1, hold_s=1.0)
    (tmp_path / "fast.json").write_text(json.dumps(fast))
    assert apply_sysid.main(["--yaw", str(tmp_path / "fast.json"), "--out-dir", str(tmp_path / "r2"),
                             "--base", str(TREE / "config"), "--write"]) == 1
    assert "OUT OF RANGE" in capsys.readouterr().out and not (tmp_path / "r2").exists()
    assert apply_sysid.main(["--hfov-deg", "150", "--out-dir", str(tmp_path / "r2"), "--write"]) == 1
    assert not (tmp_path / "r2").exists()
    with pytest.raises(SystemExit):
        apply_sysid.main(["--hfov-deg", "80", "--out-dir", str(TREE / "config")])
    with pytest.raises(SystemExit):
        apply_sysid.main(["--hfov-deg", "80", "--base", str(TREE / "config" / "twin"),
                          "--out-dir", str(tmp_path / "r3")])
    aborted = json.loads((runs / "sysid-yaw.json").read_text())
    aborted["aborted"] = "flight state LANDING"
    (tmp_path / "ab.json").write_text(json.dumps(aborted))
    with pytest.raises(SystemExit):
        apply_sysid.main(["--yaw", str(tmp_path / "ab.json"), "--out-dir", str(tmp_path / "r4")])


def test_wall_measurement_and_yaml_helpers():
    text = "a:\n  b: 1.0   # keep me\n  c:\n    d: 2\nb: 3\n"
    t2, old, new = apply_sysid.set_value(text, "a.c.d", 5, "set now")
    assert "    d: 5   # [set now]" in t2 and old == "2"
    t3, _, _ = apply_sysid.set_value(t2, "a.b", 0.25, "measured x")
    assert "  b: 0.25  # keep me [measured x]" in t3
    assert apply_sysid.get_value(t3, "b") == "3"
    assert apply_sysid.fmt(1.0) == "1.0" and apply_sysid.fmt(0.012345678) == "0.012346"
