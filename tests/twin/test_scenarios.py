"""Scenarios: loading, speed cap, paths, camera projection conventions, occlusion, skeleton copy."""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import yaml

from mhd.twin import scenarios as S
from mhd.twin.dynamics import X69Dynamics

TREE = Path(__file__).resolve().parents[2]


def cam(x=0.0, y=0.0, z=1.0, yaw=0.0, pitch=0.0, roll=0.0, hfov=75.0):
    return {"x": x, "y": y, "z": z, "yaw": yaw, "pitch": pitch, "roll": roll, "hfov_deg": hfov}


def test_all_scenarios_load_and_respect_the_walk_speed_cap():
    names = S.list_scenarios()
    assert {"s1", "s2", "s3"} <= set(names)
    for n in names:
        sc = S.load_scenario(n)
        assert sc.actors, n
        for a in sc.actors:
            assert a.speed_mps <= S.MAX_WALK_SPEED + 1e-9, (n, a.id)
    labels = {n: {a.label for a in S.load_scenario(n).actors} for n in ("s1", "s2", "s3")}
    assert "person in blue shirt" in labels["s1"]
    assert {"person in blue shirt", "person in red shirt"} <= labels["s2"]
    assert "green car" in labels["s3"]


def test_fast_actor_rejected_unless_allowed(tmp_path):
    raw = yaml.safe_load((S.SCENARIO_DIR / "s1.yaml").read_text())
    raw["actors"][0]["speed_mps"] = 0.6
    p = tmp_path / "fast.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        S.load_scenario(p)
    raw["actors"][0]["allow_fast"] = True
    p.write_text(yaml.safe_dump(raw))
    assert S.load_scenario(p).actors[0].speed_mps == 0.6


def test_pingpong_and_loop_paths():
    a = S.Actor(id="a", kind="person", label="x", path=[(0, 0), (3, 0)], speed_mps=0.3, mode="pingpong")
    assert a.state(5.0)["x"] == pytest.approx(1.5)
    assert a.state(5.0)["heading"] == pytest.approx(math.pi / 2)          # walking east
    assert a.state(15.0)["x"] == pytest.approx(1.5)
    assert a.state(15.0)["heading"] == pytest.approx(-math.pi / 2)        # walking back west
    assert a.state(20.0)["x"] == pytest.approx(0.0, abs=1e-9)
    assert a.state(15.0)["dist"] == pytest.approx(4.5)
    car = S.load_scenario("s3").actors[0]
    p0, p1 = car.state(1.0), car.state(1.0 + car._cycle)
    assert (p0["x"], p0["y"]) == pytest.approx((p1["x"], p1["y"]), abs=1e-6)  # closed lap


def test_projection_center_size_and_yaw_sign():
    a = S.Actor(id="p", kind="person", label="person", path=[(0.0, 4.0)], speed_mps=0.0, height_m=1.74)
    sc = S.Scenario(name="t", description="", world={}, drone={}, actors=[a], raw={})
    tr = sc.truth_actors(0.0, cam(z=0.87))
    box = tr[0].box
    f = 640 / math.tan(math.radians(37.5))
    assert box.cx == pytest.approx(640, abs=4)
    assert box.h == pytest.approx(1.74 * f / 4.0, rel=0.08)
    assert tr[0].bearing_rad == pytest.approx(0.0, abs=1e-4) and tr[0].distance_m == pytest.approx(4.0)
    # +yaw (nose right) puts a straight-ahead subject left of center, with a negative bearing
    tr = sc.truth_actors(0.0, cam(z=0.87, yaw=math.radians(10)))
    assert tr[0].box.cx < 640 and tr[0].bearing_rad == pytest.approx(-math.radians(10), abs=1e-4)
    # looking up (+pitch) moves the subject down in the image
    up = sc.truth_actors(0.0, cam(z=0.87, pitch=math.radians(10)))[0].box
    assert up.cy > box.cy
    # behind the camera: no box
    assert sc.truth_actors(0.0, cam(yaw=math.pi))[0].box is None


def test_person_height_matches_and_box_follows_the_stride():
    pts, bob = S.person_local_points(0.0, 0.0)
    assert max(p[1] for p in pts) == pytest.approx(1.74, abs=0.01)
    standing = max(p[2] for p in pts) - min(p[2] for p in pts)
    mid = S.PERSON_SKELETON["stride"] / 4
    pts, _ = S.person_local_points(mid, 0.3)
    striding = max(p[2] for p in pts) - min(p[2] for p in pts)
    assert striding > standing + 0.3


def test_occlusion_at_the_s2_crossing():
    sc = S.load_scenario("s2")
    d = X69Dynamics()
    d.s.z = 1.0
    pose = d.camera_pose()
    # find the moment the red walker passes in front of the blue one
    worst = min((sc.truth_actors(t / 10, pose) for t in range(60, 160)),
                key=lambda tr: next(a for a in tr if a.actor_id == "walker_blue").visible_frac)
    blue = next(a for a in worst if a.actor_id == "walker_blue")
    red = next(a for a in worst if a.actor_id == "walker_red")
    assert blue.visible_frac < 0.5 and red.visible_frac > 0.9


def test_walls_occlude():
    a = S.Actor(id="p", kind="person", label="person", path=[(0.0, 6.0)], speed_mps=0.0)
    world = {"walls": [{"x1": -3, "y1": 3, "x2": 3, "y2": 3, "height": 2.5}]}
    sc = S.Scenario(name="t", description="", world=world, drone={}, actors=[a], raw={})
    assert sc.truth_actors(0.0, cam())[0].visible_frac == 0.0


def test_js_skeleton_matches_python():
    js = (TREE / "twin" / "web" / "js" / "world.js").read_text()
    block = re.search(r"export const SKEL = \{(.*?)\};", js, re.S).group(1)
    pairs = dict((k, float(v)) for k, v in re.findall(r"(\w+):\s*(-?[0-9.]+)", block))
    assert pairs == pytest.approx({k: float(v) for k, v in S.PERSON_SKELETON.items()})
