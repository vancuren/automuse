"""Regression tests for the issues found in the code review (00:45 ET)."""
from __future__ import annotations

import threading
import time

import httpx
import numpy as np
import pytest

from mhd.contracts import Candidates, FindRequest, Select, SubjectState
from mhd_ml.perception.detectors.sam_api import SamApiDetector
from mhd_ml.perception.pipeline import PerceptionConfig, PerceptionCore
from mhd_ml.perception.spend import SpendMeter, spent_so_far
from mhd_ml.perception.trackers import SamMlxTracker, make_tracker
from synth import BLUE, RED, Person, background, render
from test_pipeline import FPS, Scene, _core, _meta, run, truth_dets


def test_refused_api_calls_are_not_billed(ledger):
    """Two abandoned requests block the pool; the third call is refused before sending, unbilled."""
    release = threading.Event()

    def handler(request):
        release.wait(5)
        return httpx.Response(200, json={"output": []})

    meter = SpendMeter("t", 0.0025, 8.0, path=ledger, flush_every=1)
    det = SamApiDetector(api_key="k", transport=httpx.MockTransport(handler), meter=meter,
                         timeout_s=0.2, api_base="http://sam.test/v1")
    for i in range(2):
        det._cache.clear()
        det.detect(np.full((720, 1280, 3), i, np.uint8), "person")
    assert spent_so_far(ledger) == pytest.approx(0.005)
    det.detect(np.full((720, 1280, 3), 99, np.uint8), "person")
    assert "not sent" in det.last_error
    assert spent_so_far(ledger) == pytest.approx(0.005)          # still two calls
    release.set()
    det.close()


def test_spend_cap_is_shared_by_concurrent_meters(ledger):
    a = SpendMeter("a", 1.0, cap_usd=10.0, path=ledger, flush_every=1)
    b = SpendMeter("b", 1.0, cap_usd=10.0, path=ledger, flush_every=1)
    for _ in range(9):
        a.charge(1)
    assert b.can_spend(1) and not b.can_spend(2)                 # b sees a's spend
    b.charge(1)
    assert not a.can_spend(1)


def test_result_for_an_older_phrase_is_ignored():
    core = _core()
    scene = Scene([Person(200, 200, shirt=RED), Person(800, 200, shirt=BLUE)])
    core.on_find(FindRequest(mission_id="g", subject_phrase="person in red shirt", category="person", color="red"))
    img, boxes = scene.frame(0)
    core.on_frame(_meta(0, 0.0), img)
    job = core.detection_due(0.0)
    # Same mission id, new phrase, while the red detection is in flight.
    core.on_find(FindRequest(mission_id="g", subject_phrase="person in blue shirt", category="person", color="blue"))
    core.on_frame(_meta(1, 50.0), img)
    core.on_detection(job, truth_dets([boxes[0]]), now_ms=60.0)  # the red person's box
    assert core.track is None


class _BrokenTracker:
    def init(self, img, box):
        pass

    def update(self, img):
        return None, 0.0


def test_select_falls_back_to_reidentification_when_the_start_fails():
    core = _core()
    scene = Scene([Person(200, 200, shirt=BLUE), Person(800, 200, shirt=BLUE)])
    core.on_find(FindRequest(mission_id="sf", subject_phrase="person in blue shirt", category="person", color="blue"))
    msgs = run(core, scene, 4, det_fn=lambda job, boxes: truth_dets(boxes))
    assert any(isinstance(m, Candidates) for m in msgs)
    run(core, scene, 10, start=4, det_fn=lambda job, boxes: [])  # 0.5 s pass, the frame is still buffered
    core.make_tracker = lambda: _BrokenTracker()                # the replay will lose the subject
    core.on_select(Select(mission_id="sf", index=1))
    assert core.track is None and core.selected is not None     # queued, not dropped
    core.make_tracker = lambda: make_tracker("vit")
    msgs = run(core, scene, 10, start=14, det_fn=lambda job, boxes: truth_dets(boxes))
    last = [m for m in msgs if m.type == "subject_obs"][-1]
    assert last.state == SubjectState.confirmed and last.box.x2 < 640  # the left one


def test_duplicate_frame_id_is_skipped_not_a_restart():
    core = _core()
    scene = Scene([Person(500, 200, shirt=BLUE)])
    core.on_find(FindRequest(mission_id="d", subject_phrase="person", category="person"))
    run(core, scene, 30, det_fn=lambda job, boxes: truth_dets(boxes))
    img, _ = scene.frame(29)
    assert core.on_frame(_meta(29, 29 * 1000.0 / FPS), img) == []
    assert not core.track.lost and len(core.ring) > 1


class _FakeEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.resolution = 672

    def set_resolution(self, r):
        self.resolution = r

    def preprocess(self, img):
        return img

    def backbone(self, px):
        raise RuntimeError("MLX failure")


def test_sam_tracker_restores_detector_resolution_on_errors():
    eng = _FakeEngine()
    tr = SamMlxTracker(eng, "person")
    with pytest.raises(RuntimeError):
        tr.init(np.zeros((720, 1280, 3), np.uint8), (10.0, 10.0, 100.0, 100.0))
    assert eng.resolution == 672


class _PhraseTracker:
    phrase = "object"
    seen = []

    def init(self, img, box):
        _PhraseTracker.seen.append(self.phrase)

    def update(self, img):
        return (500.0, 200.0, 620.0, 530.0), 0.9


def test_trackers_get_the_mission_phrase():
    core = PerceptionCore(PerceptionConfig(mask_counter="off"), "sam_mlx", lambda: _PhraseTracker())
    scene = Scene([Person(500, 200, shirt=BLUE)])
    core.on_find(FindRequest(mission_id="p", subject_phrase="person in blue shirt", category="person", color="blue"))
    run(core, scene, 5, det_fn=lambda job, boxes: truth_dets(boxes))
    assert _PhraseTracker.seen and set(_PhraseTracker.seen) == {"person in blue shirt"}
