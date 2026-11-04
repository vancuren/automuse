"""Shared helpers for governor tests: a governor on a manual clock, plus obs and mission makers."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "turbodrone" / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mhd.contracts import Box, Intent, Mission, SubjectObs  # noqa: E402
from mhd.governor import Governor  # noqa: E402
from mhd.governor.config import load_autopilot_config, load_governor_config, replace  # noqa: E402

TICK_MS = 40.0


def make_mission(mid: str = "m1", action: str = "follow", distance: str = "medium") -> Mission:
    return Mission(mission_id=mid, intent=Intent(action=action, subject_phrase="person in blue shirt",
                                                  category="person", color="blue", distance=distance,
                                                  reason_code="none"),
                   transcript="follow the person in the blue shirt", model="test", latency_ms=1.0)


def make_obs(frame_id: int, capture_ms: float, *, cx: float = 640.0, w: float = 180.0, h: float = 400.0,
             cy: float = 360.0, state: str = "confirmed", score: float = 0.9, confirm_age_ms: float = 200.0,
             mission_id: Optional[str] = "m1", track_id: int = 1, det_ms: Optional[float] = None) -> SubjectObs:
    box = Box(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2) if w > 0 else None
    return SubjectObs(mission_id=mission_id, frame_id=frame_id, capture_ms=capture_ms, state=state,
                      track_id=track_id, box=box, confidence=0.9, tracker_score=score,
                      confirm_age_ms=confirm_age_ms, source="oracle", detector_latency_ms=det_ms)


class Rig:
    """A Governor driven tick by tick on a fake clock (40 ms per tick)."""

    def __init__(self, tmp_path: Path, mode: str = "twin", gov_overrides: Optional[dict] = None,
                 ap_overrides: Optional[dict] = None):
        cfg = replace(load_governor_config(), **(gov_overrides or {}))
        ap = replace(load_autopilot_config(), **(ap_overrides or {}))
        self.t = 10_000.0
        self.g = Governor(cfg, ap, mode, clock=lambda: self.t, log_dir=tmp_path, tick_log=False)
        self.frame_latency = 100.0
        self.hb = True
        self.frames = True
        self.obs_fn: Optional[Callable[[float], Optional[SubjectObs]]] = None
        self.obs_latency = 150.0
        self._fid = 0
        self.outs = []

    def tick(self, n: int = 1):
        out = None
        for _ in range(n):
            if self.hb:
                self.g.heartbeat(self.t)
            if self.frames:
                self.g.note_frame(self.t - self.frame_latency, self.t)
            if self.obs_fn is not None:
                o = self.obs_fn(self.t)
                if o is not None:
                    self.g.submit_obs(o, receive_ms=self.t)
            out = self.g.tick(self.t)
            self.outs.append(out)
            self.t += TICK_MS
        return out

    def ticks_for(self, seconds: float) -> int:
        return int(round(seconds * 1000.0 / TICK_MS))

    def run(self, seconds: float):
        return self.tick(self.ticks_for(seconds))

    def airborne(self):
        if self.g.flight_mode == "real" and not self.g.armed:
            assert self.g.command("arm", arm_phrase="ARM").ok
        self.tick(2)
        r = self.g.command("takeoff")
        assert r.ok, r
        self.run(4.2)
        assert self.g.state.value == "AIRBORNE"

    def steady_obs(self, **kw) -> None:
        """Perception sends a confirmed observation every tick, captured obs_latency ago."""
        def fn(t):
            self._fid += 1
            return make_obs(self._fid, t - self.obs_latency, **kw)
        self.obs_fn = fn

    def following(self, **kw):
        self.airborne()
        self.g.set_mission(make_mission())
        self.steady_obs(**kw)
        self.run(1.6)
        assert self.g.autopilot.mode.value == "FOLLOW", self.g.autopilot.mode

    def close(self):
        self.g.close()


@pytest.fixture
def rig(tmp_path):
    rigs = []

    def factory(**kw) -> Rig:
        r = Rig(tmp_path, **kw)
        rigs.append(r)
        return r

    yield factory
    for r in rigs:
        r.close()
