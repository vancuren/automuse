"""Twin scenarios: world layout, actors that walk or drive along paths, and ground truth.

YAML files live in config/twin/scenarios/<name>.yaml. World frame is ENU (x east, y north, z up),
the same as dynamics.py. Headings are clockwise from north, like the drone's yaw.

Ground truth (`truth_actors`) projects each actor into the drone camera with the same pinhole
model the Three.js renderer uses (twin/web/js/common.js):
    forward f0 = (sin yaw, cos yaw, 0), right r0 = (cos yaw, -sin yaw, 0), up u0 = (0, 0, 1)
    pitch (+ up) rotates about the right axis, roll (+ right side down) about the forward axis
    u = 640 + F * x_c / z_c,  v = 360 - F * y_c / z_c,  F = 640 / tan(HFOV / 2)
Actor boxes come from sample points on the actor's body envelope; visibility is the fraction of
those points that are in the image and not hidden by walls, trees, or a closer actor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from mhd.contracts import FRAME_H, FRAME_W, Box, TruthActor

SCENARIO_DIR = Path(__file__).resolve().parents[3] / "config" / "twin" / "scenarios"
MAX_WALK_SPEED = 0.3     # m/s: the subject is asked to walk at 0.3 m/s or slower (PLAN 4.4)

# Person envelope: (height fraction, half width, half depth) in meters for a 1.74 m person.
# The renderer builds its people inside this envelope (arms at the sides, stride at the feet).
PERSON_PROFILE = [
    (0.00, 0.13, 0.22), (0.27, 0.15, 0.17), (0.52, 0.19, 0.13), (0.72, 0.24, 0.13),
    (0.82, 0.25, 0.12), (0.88, 0.10, 0.11), (0.95, 0.10, 0.11), (1.00, 0.08, 0.09),
]
PERSON_REF_HEIGHT = 1.74


@dataclass
class Actor:
    id: str
    kind: str                         # person | car
    label: str                        # e.g. "person in blue shirt", "green car"
    path: list[tuple[float, float]]
    speed_mps: float = MAX_WALK_SPEED
    mode: str = "pingpong"            # pingpong | loop | once
    pause_s: float = 0.0              # pause at each waypoint
    start_s: float = 0.0              # stands still until then
    colors: dict = field(default_factory=dict)
    height_m: float = PERSON_REF_HEIGHT     # person height
    size_m: tuple[float, float, float] = (0.45, 0.24, 0.17)   # car length, width, height
    heading_deg: Optional[float] = None     # for static actors

    def __post_init__(self) -> None:
        self._build_timeline()

    def _build_timeline(self) -> None:
        pts = [tuple(map(float, p)) for p in self.path]
        if self.mode == "pingpong" and len(pts) > 1:
            route = pts + pts[-2::-1]
        elif self.mode == "loop" and len(pts) > 1:
            route = pts + [pts[0]]
        else:
            route = pts
        # timeline entries: (t0, t1, p0, p1, d0) for legs, pauses are legs with p0 == p1
        self._legs = []
        t, dist = 0.0, 0.0
        for i in range(len(route) - 1):
            p0, p1 = route[i], route[i + 1]
            seg = math.dist(p0, p1)
            if seg > 1e-9 and self.speed_mps > 0:
                dur = seg / self.speed_mps
                self._legs.append((t, t + dur, p0, p1, dist))
                t += dur
                dist += seg
            if self.pause_s > 0:
                self._legs.append((t, t + self.pause_s, p1, p1, dist))
                t += self.pause_s
        self._cycle = t
        self._cycle_dist = dist
        self._route = route

    def state(self, t: float) -> dict:
        """Position, heading (rad, clockwise from north), speed, and distance walked at time t."""
        p_first = self._route[0]
        heading0 = math.radians(self.heading_deg) if self.heading_deg is not None else self._leg_heading(0)
        if not self._legs or t <= self.start_s:
            return {"x": p_first[0], "y": p_first[1], "heading": heading0, "speed": 0.0, "dist": 0.0}
        tt = t - self.start_s
        laps = 0
        if self.mode in ("pingpong", "loop") and self._cycle > 0:
            laps = int(tt // self._cycle)
            tt = tt - laps * self._cycle
        elif tt >= self._cycle:
            last = self._legs[-1]
            return {"x": last[3][0], "y": last[3][1], "heading": self._heading_of(len(self._legs) - 1),
                    "speed": 0.0, "dist": self._cycle_dist}
        for i, (t0, t1, p0, p1, d0) in enumerate(self._legs):
            if t0 <= tt < t1 or i == len(self._legs) - 1:
                a = 0.0 if t1 <= t0 else max(0.0, min(1.0, (tt - t0) / (t1 - t0)))
                x = p0[0] + (p1[0] - p0[0]) * a
                y = p0[1] + (p1[1] - p0[1]) * a
                moving = p0 != p1
                seg = math.dist(p0, p1)
                return {"x": x, "y": y, "heading": self._heading_of(i),
                        "speed": self.speed_mps if moving else 0.0,
                        "dist": laps * self._cycle_dist + d0 + seg * a}
        raise AssertionError("unreachable")

    def _leg_heading(self, i: int) -> float:
        if len(self._route) < 2:
            return 0.0
        p0, p1 = self._route[i], self._route[i + 1]
        return math.atan2(p1[0] - p0[0], p1[1] - p0[1])

    def _heading_of(self, leg_index: int) -> float:
        # pauses keep the heading of the leg before them
        for j in range(leg_index, -1, -1):
            t0, t1, p0, p1, _ = self._legs[j]
            if p0 != p1:
                return math.atan2(p1[0] - p0[0], p1[1] - p0[1])
        if self.heading_deg is not None:
            return math.radians(self.heading_deg)
        return self._leg_heading(0)

    # -- geometry -------------------------------------------------------------------------
    def sample_points(self, st: dict) -> list[tuple[float, float, float]]:
        """Points on the actor's envelope in world coordinates, used for the box and visibility."""
        x, y, hd = st["x"], st["y"], st["heading"]
        fwd = (math.sin(hd), math.cos(hd))
        right = (math.cos(hd), -math.sin(hd))
        pts = []
        if self.kind == "person":
            s = self.height_m / PERSON_REF_HEIGHT
            for frac, hw, hdp in PERSON_PROFILE:
                z = frac * self.height_m
                for k in range(8):
                    a = 2 * math.pi * k / 8
                    lx, ly = hw * s * math.cos(a), hdp * s * math.sin(a)
                    pts.append((x + right[0] * lx + fwd[0] * ly, y + right[1] * lx + fwd[1] * ly, z))
                pts.append((x, y, z))
        else:
            length, width, height = self.size_m
            for lx in (-width / 2, 0.0, width / 2):
                for ly in (-length / 2, -length / 4, 0.0, length / 4, length / 2):
                    for z in (0.0, height * 0.5, height):
                        pts.append((x + right[0] * lx + fwd[0] * ly, y + right[1] * lx + fwd[1] * ly, z))
        return pts

    def center_height(self) -> float:
        return self.height_m / 2 if self.kind == "person" else self.size_m[2] / 2


class Camera:
    """Pinhole camera matching the renderer (see module docstring)."""

    def __init__(self, pose: dict, width: int = FRAME_W, height: int = FRAME_H) -> None:
        self.w, self.h = width, height
        self.pos = (pose["x"], pose["y"], pose["z"])
        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
        f0 = (math.sin(yaw), math.cos(yaw), 0.0)
        r0 = (math.cos(yaw), -math.sin(yaw), 0.0)
        u0 = (0.0, 0.0, 1.0)
        cp, sp = math.cos(pitch), math.sin(pitch)
        f1 = tuple(f0[i] * cp + u0[i] * sp for i in range(3))
        u1 = tuple(u0[i] * cp - f0[i] * sp for i in range(3))
        cr, sr = math.cos(roll), math.sin(roll)
        self.r = tuple(r0[i] * cr - u1[i] * sr for i in range(3))
        self.u = tuple(u1[i] * cr + r0[i] * sr for i in range(3))
        self.f = f1
        self.yaw = yaw
        self.focal = (width / 2) / math.tan(math.radians(pose["hfov_deg"]) / 2)

    def to_cam(self, p: tuple[float, float, float]) -> tuple[float, float, float]:
        d = (p[0] - self.pos[0], p[1] - self.pos[1], p[2] - self.pos[2])
        return (sum(d[i] * self.r[i] for i in range(3)), sum(d[i] * self.u[i] for i in range(3)),
                sum(d[i] * self.f[i] for i in range(3)))

    def project(self, p: tuple[float, float, float]) -> Optional[tuple[float, float, float]]:
        xc, yc, zc = self.to_cam(p)
        if zc <= 0.05:
            return None
        return (self.w / 2 + self.focal * xc / zc, self.h / 2 - self.focal * yc / zc, zc)

    def bearing(self, x: float, y: float) -> float:
        """Horizontal bearing of a world point from the camera heading (+ = right)."""
        dx, dy = x - self.pos[0], y - self.pos[1]
        fwd = (math.sin(self.yaw), math.cos(self.yaw))
        right = (math.cos(self.yaw), -math.sin(self.yaw))
        return math.atan2(dx * right[0] + dy * right[1], dx * fwd[0] + dy * fwd[1])


def _seg_intersect(p, q, a, b) -> Optional[float]:
    """Parameter s in [0, 1] along p->q where it crosses segment a-b (2D), or None."""
    rx, ry = q[0] - p[0], q[1] - p[1]
    sx, sy = b[0] - a[0], b[1] - a[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-12:
        return None
    t = ((a[0] - p[0]) * sy - (a[1] - p[1]) * sx) / den
    u = ((a[0] - p[0]) * ry - (a[1] - p[1]) * rx) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return None


@dataclass
class Scenario:
    name: str
    description: str
    world: dict
    drone: dict
    actors: list[Actor]
    raw: dict

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        b = self.world.get("bounds", [-10, -10, 10, 10])
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    @property
    def texture(self) -> float:
        return float(self.world.get("ground", {}).get("texture", 1.0))

    def actor_states(self, t: float) -> list[dict]:
        out = []
        for a in self.actors:
            st = a.state(t)
            out.append({"id": a.id, "x": round(st["x"], 4), "y": round(st["y"], 4),
                        "heading": round(st["heading"], 4), "speed": st["speed"], "dist": round(st["dist"], 4)})
        return out

    def scene_json(self) -> dict:
        """Static scene description for the renderer."""
        return {
            "name": self.name, "description": self.description, "world": self.world,
            "actors": [{"id": a.id, "kind": a.kind, "label": a.label, "colors": a.colors,
                        "height_m": a.height_m, "size_m": list(a.size_m)} for a in self.actors],
        }

    # -- ground truth ---------------------------------------------------------------------
    def _occluded_by_world(self, cam: Camera, p: tuple[float, float, float]) -> bool:
        c = cam.pos
        for wall in self.world.get("walls", []):
            s = _seg_intersect((c[0], c[1]), (p[0], p[1]), (wall["x1"], wall["y1"]), (wall["x2"], wall["y2"]))
            if s is not None and s < 0.999:
                z = c[2] + s * (p[2] - c[2])
                if 0.0 <= z <= float(wall.get("height", 2.5)):
                    return True
        for tree in self.world.get("trees", []):
            if self._ray_hits_tree(c, p, tree):
                return True
        return False

    @staticmethod
    def _ray_hits_tree(c, p, tree) -> bool:
        tx, ty, h = float(tree["x"]), float(tree["y"]), float(tree.get("height", 4.0))
        r = float(tree.get("radius", 1.2))
        trunk_r = max(0.08, 0.1 * r)
        crown_z = h - r
        dx, dy, dz = p[0] - c[0], p[1] - c[1], p[2] - c[2]
        L2 = dx * dx + dy * dy + dz * dz
        if L2 < 1e-9:
            return False
        # crown: sphere (slightly shrunk, foliage has gaps)
        s = ((tx - c[0]) * dx + (ty - c[1]) * dy + (crown_z - c[2]) * dz) / L2
        if 0.0 < s < 0.999:
            qx, qy, qz = c[0] + s * dx, c[1] + s * dy, c[2] + s * dz
            if (qx - tx) ** 2 + (qy - ty) ** 2 + (qz - crown_z) ** 2 < (0.8 * r) ** 2:
                return True
        # trunk: vertical cylinder below the crown
        L2h = dx * dx + dy * dy
        if L2h > 1e-9:
            s = ((tx - c[0]) * dx + (ty - c[1]) * dy) / L2h
            if 0.0 < s < 0.999:
                qx, qy, qz = c[0] + s * dx, c[1] + s * dy, c[2] + s * dz
                if (qx - tx) ** 2 + (qy - ty) ** 2 < trunk_r ** 2 and 0.0 <= qz <= crown_z:
                    return True
        return False

    def truth_actors(self, t: float, cam_pose: dict) -> list[TruthActor]:
        cam = Camera(cam_pose)
        states = [a.state(t) for a in self.actors]
        projected = []      # per actor: (points, projections, unclipped box, depth of center)
        for a, st in zip(self.actors, states):
            pts = a.sample_points(st)
            proj = [cam.project(p) for p in pts]
            front = [q for q in proj if q is not None]
            ub = None
            if front:
                ub = (min(q[0] for q in front), min(q[1] for q in front),
                      max(q[0] for q in front), max(q[1] for q in front))
            center = cam.to_cam((st["x"], st["y"], a.center_height()))
            projected.append((pts, proj, ub, center[2]))

        out = []
        for i, (a, st) in enumerate(zip(self.actors, states)):
            pts, proj, ub, depth = projected[i]
            visible = 0
            for p, q in zip(pts, proj):
                if q is None or not (0 <= q[0] < cam.w and 0 <= q[1] < cam.h):
                    continue
                if self._occluded_by_world(cam, p):
                    continue
                hidden = False
                for j, (_, _, ub2, depth2) in enumerate(projected):
                    if j == i or ub2 is None or depth2 >= q[2] or depth2 <= 0.05:
                        continue
                    # shrink the occluder's box: its envelope is not a solid rectangle
                    mx, my = 0.15 * (ub2[2] - ub2[0]), 0.08 * (ub2[3] - ub2[1])
                    if ub2[0] + mx <= q[0] <= ub2[2] - mx and ub2[1] + my <= q[1] <= ub2[3] - my:
                        hidden = True
                        break
                if not hidden:
                    visible += 1
            box = None
            if ub is not None:
                x1, y1 = max(0.0, ub[0]), max(0.0, ub[1])
                x2, y2 = min(float(cam.w), ub[2]), min(float(cam.h), ub[3])
                if x2 > x1 + 0.5 and y2 > y1 + 0.5 and depth > 0.05:
                    box = Box(x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1))
            dist = math.hypot(st["x"] - cam.pos[0], st["y"] - cam.pos[1])
            out.append(TruthActor(actor_id=a.id, label=a.label, box=box,
                                  visible_frac=round(visible / len(pts), 3) if box is not None else 0.0,
                                  distance_m=round(dist, 3), bearing_rad=round(cam.bearing(st["x"], st["y"]), 4)))
        return out


def _actor_from_dict(d: dict) -> Actor:
    speed = float(d.get("speed_mps", MAX_WALK_SPEED))
    if speed > MAX_WALK_SPEED + 1e-9 and not d.get("allow_fast", False):
        raise ValueError(f"actor {d.get('id')}: speed {speed} m/s exceeds {MAX_WALK_SPEED} m/s "
                         "(set allow_fast: true to override)")
    path = [tuple(p) for p in d["path"]]
    return Actor(id=d["id"], kind=d.get("kind", "person"), label=d["label"], path=path, speed_mps=speed,
                 mode=d.get("mode", "pingpong"), pause_s=float(d.get("pause_s", 0.0)),
                 start_s=float(d.get("start_s", 0.0)), colors=dict(d.get("colors", {})),
                 height_m=float(d.get("height_m", PERSON_REF_HEIGHT)),
                 size_m=tuple(d.get("size_m", (0.45, 0.24, 0.17))),
                 heading_deg=d.get("heading_deg"))


def _expand_track_path(d: dict, world: dict) -> dict:
    """`path: track` follows the world's oval track center line."""
    if d.get("path") != "track":
        return d
    tr = world["track"]
    n = int(tr.get("samples", 64))
    cx, cy, rx, ry = float(tr["cx"]), float(tr["cy"]), float(tr["rx"]), float(tr["ry"])
    direction = -1.0 if d.get("clockwise", False) else 1.0
    pts = [(round(cx + rx * math.cos(direction * 2 * math.pi * k / n), 4),
            round(cy + ry * math.sin(direction * 2 * math.pi * k / n), 4)) for k in range(n)]
    d = dict(d)
    d["path"] = pts
    d["mode"] = "loop"
    return d


def load_scenario(name_or_path: str | Path) -> Scenario:
    path = Path(name_or_path)
    if not path.suffix:
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    world = raw.get("world", {})
    actors = [_actor_from_dict(_expand_track_path(a, world)) for a in raw.get("actors", [])]
    return Scenario(name=raw.get("name", path.stem), description=raw.get("description", ""), world=world,
                    drone=raw.get("drone", {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}), actors=actors, raw=raw)


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))
