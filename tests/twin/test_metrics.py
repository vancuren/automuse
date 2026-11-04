"""metrics.py on synthetic logs built from the real scenario projection (S2: blue and red crossing)."""
from __future__ import annotations

import json
import math

import pytest

from mhd.contracts import Box, ObsSource, SubjectObs, SubjectState, Truth
from mhd.twin import metrics
from mhd.twin.dynamics import X69Dynamics
from mhd.twin.scenarios import load_scenario


def make_truth(scn="s2", seconds=20.0, fps=25.0):
    sc = load_scenario(scn)
    d = X69Dynamics()
    d.s.z = 1.0
    pose = d.camera_pose()
    out = []
    for i in range(int(seconds * fps)):
        t = i / fps
        tr = Truth(render_id=i + 1, render_ms=1000.0 + t * 1000.0, drone=d.truth_dict(),
                   actors=sc.truth_actors(t, pose))
        out.append(json.loads(tr.model_dump_json()))
    return out


def obs_for(tr, actor_id, state="confirmed", track_id=1, shift_px=0.0):
    a = next(a for a in tr["actors"] if a["actor_id"] == actor_id)
    b = a["box"]
    box = Box(x1=b["x1"] + shift_px, y1=b["y1"], x2=b["x2"] + shift_px, y2=b["y2"]) if b else None
    o = SubjectObs(frame_id=tr["render_id"], capture_ms=tr["render_ms"], render_id=tr["render_id"],
                   state=SubjectState(state) if box else SubjectState.lost, box=box, track_id=track_id,
                   source=ObsSource.oracle)
    return json.loads(o.model_dump_json())


def test_perfect_oracle_on_s2():
    truth = make_truth()
    obs = [obs_for(tr, "walker_blue") for tr in truth]
    r = metrics.compute(truth, obs, subject="person in blue shirt")["report"]
    assert r["time_to_find_s"] == pytest.approx(0.0, abs=0.05)
    assert r["lock_rate"] == pytest.approx(1.0)
    assert r["id_switches"] == 0 and r["track_id_switches"] == 0
    assert r["perception_bearing_err_rms_rad"] == pytest.approx(0.0, abs=1e-6)
    # the drone does not turn here, so the control bearing error is just the walker's bearing
    assert 0.1 < r["bearing_rms_rad"] < 0.6


def test_late_find_dropouts_shift_and_identity_switch():
    truth = make_truth()
    obs = []
    for tr in truth:
        t = (tr["render_ms"] - 1000.0) / 1000.0
        if t < 2.0:
            obs.append(json.loads(SubjectObs(frame_id=tr["render_id"], capture_ms=tr["render_ms"],
                                             render_id=tr["render_id"], state=SubjectState.searching,
                                             source=ObsSource.oracle).model_dump_json()))
        elif 9.0 <= t < 11.0:        # during the crossing the tracker jumps to the red shirt
            obs.append(obs_for(tr, "walker_red", track_id=2))
        elif 14.0 <= t < 15.0:       # dropout: no observation at all
            continue
        else:
            obs.append(obs_for(tr, "walker_blue", track_id=1 if t < 9 else 3, shift_px=10.0))
    r = metrics.compute(truth, obs, subject="walker_blue")["report"]
    assert r["time_to_find_s"] == pytest.approx(2.0, abs=0.05)
    assert r["id_switches"] == 2                       # blue -> red -> blue
    assert r["track_id_switches"] == 2
    assert 0.75 < r["lock_rate"] < 0.95
    f = 640 / math.tan(math.radians(75) / 2)
    assert r["perception_bearing_err_rms_rad"] == pytest.approx(10 / f, rel=0.2)


def test_cli_with_mixed_log(tmp_path):
    truth = make_truth(seconds=6.0)
    obs = [obs_for(tr, "walker_blue") for tr in truth]
    mixed = tmp_path / "mixed.jsonl"
    with open(mixed, "w") as f:
        for tr, o in zip(truth, obs):
            f.write(json.dumps(tr) + "\n")
            f.write(json.dumps(o) + "\n")
    out = tmp_path / "report.json"
    csv_path = tmp_path / "rows.csv"
    metrics.main(["--mixed", str(mixed), "--subject", "person in blue shirt", "--out", str(out),
                  "--csv", str(csv_path)])
    rep = json.loads(out.read_text())
    assert rep["lock_rate"] == pytest.approx(1.0)
    assert csv_path.read_text().count("\n") == len(truth) + 1
