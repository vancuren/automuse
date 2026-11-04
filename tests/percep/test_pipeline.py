"""PerceptionCore end to end on synthetic frames, with detections supplied by the test."""
from __future__ import annotations

import pytest

from mhd.contracts import Candidates, FindRequest, FrameMeta, Release, Select, SubjectState
from mhd_ml.perception.pipeline import PerceptionConfig, PerceptionCore
from mhd_ml.perception.trackers import make_tracker
from mhd_ml.perception.types import Detection, iou
from synth import BLUE, RED, Person, background, mask_for, render

FPS = 20.0


def _core(**kw):
    cfg = PerceptionConfig(mask_counter="off", **kw)
    return PerceptionCore(cfg, "sam_mlx", lambda: make_tracker("vit"))


def _meta(i, t_ms, render_id=None):
    return FrameMeta(frame_id=i, capture_ms=t_ms, receive_ms=t_ms + 30, width=1280, height=720,
                     render_id=render_id, flight_mode="twin")


class Scene:
    def __init__(self, people, seed=7):
        self.people = people
        self.bg = background(seed)

    def frame(self, i):
        return render(self.people, i / FPS, self.bg)


def run(core, scene, n, detect_every=None, det_fn=None, start=0, latency_frames=0, obs_log=None):
    """Feed n frames. When the core asks for a detection, compute it on that frame (det_fn) and
    deliver it latency_frames later (the detector is slow; this exercises re-seeding)."""
    pending = []
    out = []
    for i in range(start, start + n):
        img, boxes = scene.frame(i)
        t = i * 1000.0 / FPS
        msgs = core.on_frame(_meta(i, t, render_id=1000 + i), img)
        out.extend(msgs)
        job = core.detection_due(t)
        if job is not None:
            pending.append((i + latency_frames, job, det_fn(job, boxes)))
        for item in [p for p in pending if p[0] <= i]:
            pending.remove(item)
            out.extend(core.on_detection(item[1], item[2], latency_ms=123.0, now_ms=t))
        if obs_log is not None:
            obs_log.extend(msgs)
    return out


def truth_dets(boxes, scores=None, masks=True):
    scores = scores or [0.9] * len(boxes)
    return [Detection(box=b, score=s, mask=mask_for(b) if masks else None) for b, s in zip(boxes, scores)]


def test_find_confirm_and_follow_one_person():
    core = _core()
    scene = Scene([Person(200, 180, vx=150, shirt=BLUE)])
    core.on_find(FindRequest(mission_id="m1", subject_phrase="person in blue shirt", category="person", color="blue"))
    msgs = run(core, scene, 60, det_fn=lambda job, boxes: truth_dets(boxes), latency_frames=6)
    obs = [m for m in msgs if m.type == "subject_obs"]
    assert len(obs) == 60                                     # one per frame, camera rate
    assert obs[0].state == SubjectState.searching and obs[0].box is None
    states = [o.state for o in obs]
    assert SubjectState.candidate in states and states[-1] == SubjectState.confirmed
    last = obs[-1]
    _, (box,) = scene.frame(59)
    assert iou((last.box.x1, last.box.y1, last.box.x2, last.box.y2), box) > 0.6
    assert last.render_id == 1059 and last.mission_id == "m1" and last.track_id == 1
    assert last.tracker_score > 0.3
    # The detector is 6 frames (300 ms) behind; confirmation age is measured from its frame.
    assert 250 <= last.confirm_age_ms <= 1300
    assert any(o.source.value == "sam_mlx" and o.detector_latency_ms == 123.0 for o in obs)
    assert core.stats["reseed_ok"] >= 1 and core.stats["tracks_started"] == 1


def test_lookalikes_tie_sends_candidates_then_select():
    core = _core()
    scene = Scene([Person(200, 200, shirt=BLUE), Person(800, 200, shirt=BLUE)])
    core.on_find(FindRequest(mission_id="m2", subject_phrase="person in blue shirt", category="person", color="blue"))
    msgs = run(core, scene, 12, det_fn=lambda job, boxes: truth_dets(boxes))
    cands = [m for m in msgs if isinstance(m, Candidates)]
    assert len(cands) == 1 and len(cands[0].items) == 2
    items = cands[0].items
    assert [c.index for c in items] == [1, 2] and items[0].box.x1 < items[1].box.x1   # left to right
    obs = [m for m in msgs if m.type == "subject_obs"]
    assert obs[-1].state == SubjectState.candidate and obs[-1].box is None
    core.on_select(Select(mission_id="m2", index=2))
    msgs = run(core, scene, 5, start=12, det_fn=lambda job, boxes: truth_dets(boxes))
    last = [m for m in msgs if m.type == "subject_obs"][-1]
    assert last.state == SubjectState.confirmed and last.box.x1 > 600              # the right one


def test_color_breaks_the_tie_blue_vs_red():
    core = _core()
    scene = Scene([Person(200, 200, shirt=BLUE), Person(800, 200, shirt=RED)])
    core.on_find(FindRequest(mission_id="m3", subject_phrase="person in red shirt", category="person", color="red"))
    # The detector scores both alike (as SAM may for "person in red shirt" with two people).
    msgs = run(core, scene, 30, det_fn=lambda job, boxes: truth_dets(boxes, [0.8, 0.8]))
    assert not any(isinstance(m, Candidates) for m in msgs)
    last = [m for m in msgs if m.type == "subject_obs"][-1]
    assert last.state == SubjectState.confirmed and last.box.x1 > 600


def test_subject_leaves_frame_reports_lost():
    core = _core()
    p = Person(500, 200, shirt=BLUE)
    scene = Scene([p])
    core.on_find(FindRequest(mission_id="m4", subject_phrase="person", category="person"))
    run(core, scene, 30, det_fn=lambda job, boxes: truth_dets(boxes))
    p.x = 5000                                           # gone
    msgs = run(core, scene, 10, start=30, det_fn=lambda job, boxes: [])
    obs = [m for m in msgs if m.type == "subject_obs"]
    assert obs[-1].state == SubjectState.lost and obs[-1].box is None
    assert obs[-1].track_id == 1 and obs[-1].tracker_score == 0.0
    assert obs[-1].confirm_age_ms > 400


def test_detector_cadence_2hz_finding_1hz_following():
    core = _core()
    scene = Scene([Person(500, 200, shirt=BLUE)])
    core.on_find(FindRequest(mission_id="m5", subject_phrase="person", category="person"))
    times = []

    def det(job, boxes):
        times.append(job.submit_ms)
        return truth_dets(boxes)

    run(core, scene, 100, det_fn=det)                    # 5 s
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert gaps[0] == pytest.approx(500, abs=60)         # finding / candidate: 2 Hz
    assert gaps[-1] == pytest.approx(1000, abs=60)       # following: 1 Hz


def test_release_stops_observations_and_ignores_late_results():
    core = _core()
    scene = Scene([Person(500, 200, shirt=BLUE)])
    core.on_find(FindRequest(mission_id="m6", subject_phrase="person", category="person"))
    img, boxes = scene.frame(0)
    core.on_frame(_meta(0, 0.0), img)
    job = core.detection_due(0.0)
    assert job is not None
    core.on_release(Release(mission_id="m6"))
    assert core.on_frame(_meta(1, 50.0), img) == []
    core.on_find(FindRequest(mission_id="m7", subject_phrase="person", category="person"))
    core.on_frame(_meta(2, 100.0), img)
    assert core.on_detection(job, truth_dets(boxes)) == []       # belongs to m6
    assert core.track is None


def test_reseed_rejected_when_replay_disagrees():
    core = _core(max_reject_streak=2)
    scene = Scene([Person(200, 200, vx=0, shirt=BLUE)])
    core.on_find(FindRequest(mission_id="m8", subject_phrase="person", category="person"))
    run(core, scene, 20, det_fn=lambda job, boxes: truth_dets(boxes))
    assert core.track is not None and core.track.n_confirms >= 2
    started = core.stats["tracks_started"]
    # A detection that overlaps the track (passes the association gate) but sits 0.4 box sizes
    # away: the replay ends far from the live track, so the re-seed is rejected.
    def shifted(job, boxes):
        x1, y1, x2, y2 = boxes[0]
        dx = 0.45 * ((x2 - x1) * (y2 - y1)) ** 0.5
        return [Detection(box=(x1 + dx, y1, x2 + dx, y2), score=0.9)]
    run(core, scene, 25, start=20, det_fn=shifted)
    assert core.stats["reseed_rejected"] >= 1
    assert core.stats["tracks_started"] > started             # restarted after the streak


def test_low_scores_and_counter_detections_ignored():
    core = PerceptionCore(PerceptionConfig(mask_counter="on"), "sam_api", lambda: make_tracker("kcf"))
    scene = Scene([Person(500, 200, shirt=BLUE)])
    core.on_find(FindRequest(mission_id="m9", subject_phrase="person", category="person"))
    msgs = run(core, scene, 10, det_fn=lambda job, boxes: [
        Detection(box=boxes[0], score=0.2),                         # below min_score 0.4
        Detection(box=(1130.0, 580.0, 1270.0, 710.0), score=0.95)])  # the counter block
    obs = [m for m in msgs if m.type == "subject_obs"]
    assert all(o.state == SubjectState.searching for o in obs)
    assert obs[-1].source.value == "sam_api"
