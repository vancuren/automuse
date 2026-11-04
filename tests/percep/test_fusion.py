"""Fusion unit tests: Kalman timing, color score, association, re-seeding, counter block."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mhd_ml.perception.fusion import (BoxKalman, Camera, FrameRing, associate, box_offset,
                                      color_fraction, color_score, counter_roi, hue_hist,
                                      mask_counter, normalize_color, overlaps_counter, reseed,
                                      size_prior, work_image)
from mhd_ml.perception.trackers import make_tracker
from mhd_ml.perception.types import Detection, iou
from synth import BLUE, GREEN, RED, Person, background, mask_for, render


# ----- Kalman filter keyed on capture time -------------------------------------------------
def _box_at_bearing(cam, beta, width=100.0, cy=360.0, h=250.0):
    cx = cam.cx(beta)
    return (cx - width / 2, cy - h / 2, cx + width / 2, cy + h / 2)


def test_kalman_velocity_from_irregular_capture_times():
    cam = Camera(80)
    rate = 0.2                                     # rad/s
    rng = np.random.default_rng(1)
    t, times = 0.0, []
    while t < 2000:
        times.append(t)
        t += rng.choice([33, 40, 66, 120, 400])    # jittered frames and a stall
    kf = BoxKalman(cam, _box_at_bearing(cam, -0.2), 0.0)
    for t in times[1:]:
        kf.update(t, _box_at_bearing(cam, -0.2 + rate * t / 1000.0), "tracker", 0.9)
    assert kf.bearing_rate == pytest.approx(rate, abs=0.03)
    # Prediction half a second ahead lands where the subject will be.
    t_end = times[-1]
    pred = kf.predicted_box(t_end + 500)
    assert cam.bearing((pred[0] + pred[2]) / 2) == pytest.approx(-0.2 + rate * (t_end + 500) / 1000, abs=0.02)


def test_kalman_arrival_order_does_not_matter_only_capture_time():
    """Late frames delivered in a burst carry their real capture times: same estimate as smooth."""
    cam = Camera(80)
    truth = lambda t: _box_at_bearing(cam, 0.1 * t / 1000.0)
    a, b = BoxKalman(cam, truth(0), 0.0), BoxKalman(cam, truth(0), 0.0)
    for t in range(40, 1600, 40):
        a.update(float(t), truth(t))
    # b: the same capture times, but processed in a burst after a "network stall" (the filter
    # never sees arrival times), so it must match a exactly.
    for t in range(40, 1600, 40):
        b.update(float(t), truth(t))
    assert a.bearing == pytest.approx(b.bearing) and a.bearing_rate == pytest.approx(b.bearing_rate)
    assert a.bearing_rate == pytest.approx(0.1, abs=0.02)


def test_kalman_refuses_measurements_older_than_the_filter():
    cam = Camera(80)
    kf = BoxKalman(cam, _box_at_bearing(cam, 0.0), 1000.0)
    kf.update(1100.0, _box_at_bearing(cam, 0.05))
    before = kf.x.copy()
    assert kf.update(1050.0, _box_at_bearing(cam, 0.3)) is None
    assert np.allclose(kf.x, before)


def test_kalman_stall_widens_uncertainty():
    cam = Camera(80)
    kf = BoxKalman(cam, _box_at_bearing(cam, 0.0), 0.0)
    for t in range(40, 1000, 40):
        kf.update(float(t), _box_at_bearing(cam, 0.0))
    p_before = kf.P[0, 0]
    _, P = kf.predict_state(1000.0 + 1500.0)
    assert P[0, 0] > 20 * p_before
    # After the stall a new measurement is trusted almost fully.
    kf.update(2500.0, _box_at_bearing(cam, 0.25))
    assert kf.bearing == pytest.approx(0.25, abs=0.02)


def test_kalman_size_and_box_roundtrip():
    cam = Camera(80)
    box = (600.0, 200.0, 700.0, 500.0)
    kf = BoxKalman(cam, box, 0.0)
    assert kf.box() == pytest.approx(box, abs=1e-6)
    for t in range(40, 1000, 40):                  # subject getting closer: width 100 -> 150
        w = 100 + 50 * t / 1000
        kf.update(float(t), (650 - w / 2, 350 - 1.5 * w, 650 + w / 2, 350 + 1.5 * w))
    b = kf.box()
    assert b[2] - b[0] == pytest.approx(148, abs=6)
    assert kf.x[3] > 0.2                           # d ln(w) / dt > 0: approaching


def test_bearing_convention():
    cam = Camera(90)
    assert cam.f == pytest.approx(640.0)
    assert cam.bearing(640) == 0.0
    assert cam.bearing(1280) == pytest.approx(math.pi / 4)
    assert cam.cx(cam.bearing(900)) == pytest.approx(900)


# ----- color score -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def two_people():
    bg = background(3)
    people = [Person(300, 200, shirt=BLUE), Person(800, 200, shirt=RED)]
    img, boxes = render(people, 0.0, bg)
    return work_image(img, False), boxes


def test_color_score_blue_vs_red(two_people):
    work, (blue_box, red_box) = two_people
    fb = color_fraction(work, blue_box, None, "blue")
    fr = color_fraction(work, red_box, None, "blue")
    assert fb > 0.6 and fr < 0.05
    assert color_fraction(work, red_box, None, "red") > 0.6
    assert color_score(fb) == pytest.approx(1.0) and color_score(fr) < 0.25
    assert color_fraction(work, blue_box, None, "navy") == pytest.approx(fb)
    assert color_fraction(work, blue_box, None, "none") is None and color_score(None) == 1.0


def test_color_uses_upper_body_and_mask(two_people):
    bg = background(4)
    p = Person(500, 150, shirt=RED, trousers=BLUE)       # red shirt, blue trousers
    img, (box,) = render([p], 0.0, bg)
    work = work_image(img, False)
    assert color_fraction(work, box, None, "blue") < 0.05
    assert color_fraction(work, box, None, "red") > 0.6
    # A box that is too wide still scores the shirt when the mask says where the person is.
    wide = (box[0] - 150, box[1], box[2] + 150, box[3])
    assert color_fraction(work, wide, mask_for(box), "red") > color_fraction(work, wide, None, "red")


def test_hue_hist_separates_colors(two_people):
    work, (blue_box, red_box) = two_people
    hb, hr = hue_hist(work, blue_box, None), hue_hist(work, red_box, None)
    assert hb is not None and hb.sum() == pytest.approx(1.0)
    assert np.minimum(hb, hb).sum() == pytest.approx(1.0)
    assert np.minimum(hb, hr).sum() < 0.2


def test_normalize_color():
    assert normalize_color("Navy") == "blue" and normalize_color("grey") == "gray"
    assert normalize_color("plaid") is None and normalize_color(None) is None


def test_size_prior():
    assert size_prior((0, 0, 10, 10)) == pytest.approx(0.3)
    assert size_prior((500, 200, 700, 600)) == 1.0
    assert size_prior((0, 0, 1280, 720)) < 1.0


# ----- counter block ------------------------------------------------------------------------
def test_counter_block_masked_and_ignored():
    x1, y1, x2, y2 = counter_roi()
    assert x2 == 1280 and y2 == 720 and x2 - x1 >= 160 - 1e-6
    img = np.zeros((720, 1280, 3), np.uint8)
    mask_counter(img)
    assert img[int(y1) + 5, int(x1) + 5].tolist() == [128, 128, 128]
    assert img[int(y1) - 5, int(x1) - 5].tolist() == [0, 0, 0]
    small = np.zeros((360, 640, 3), np.uint8)
    mask_counter(small)
    assert small[359, 639].tolist() == [128, 128, 128]
    assert overlaps_counter((x1 + 10, y1 + 10, x2, y2)) and not overlaps_counter((0, 0, 200, 200))


# ----- association -------------------------------------------------------------------------
def test_associate_by_iou_and_color(two_people):
    work, (blue_box, red_box) = two_people
    track_box = (blue_box[0] + 20, blue_box[1] + 10, blue_box[2] + 20, blue_box[3] + 10)
    track_hist = hue_hist(work, blue_box, None)
    dets = [Detection(box=red_box, score=0.95), Detection(box=blue_box, score=0.8)]
    hists = [hue_hist(work, d.box, None) for d in dets]
    assert associate(track_box, track_hist, dets, hists) == 1
    # Color breaks a near-tie in overlap: a red box at the same place loses to a blue one nearby.
    shifted_blue = (blue_box[0] + 50, blue_box[1], blue_box[2] + 50, blue_box[3])
    dets2 = [Detection(box=track_box, score=0.9), Detection(box=shifted_blue, score=0.9)]
    hists2 = [hue_hist(work, red_box, None), hue_hist(work, blue_box, None)]
    assert associate(track_box, track_hist, dets2, hists2) == 1
    # Nothing near the track: no association.
    far = [Detection(box=(1000, 100, 1100, 400), score=0.9)]
    assert associate(track_box, track_hist, far, [None]) is None


# ----- ring buffer and re-seeding ------------------------------------------------------------
def _moving_sequence(n=20, fps=20.0, vx=240.0, shirt=BLUE):
    bg = background(5)
    p = Person(300, 200, vx=vx, shirt=shirt)
    ring = FrameRing(span_ms=2000)
    truth = {}
    for i in range(n):
        t = i / fps
        img, (box,) = render([p], t, bg)
        ring.add(i, t * 1000.0, work_image(img, False))
        truth[i] = box
    return ring, truth


def test_ring_buffer_keeps_two_seconds():
    ring = FrameRing(span_ms=2000)
    img = np.zeros((360, 640, 3), np.uint8)
    for i in range(100):
        ring.add(i, i * 50.0, img)
    assert ring.get(0) is None and ring.get(99) is not None
    assert ring.latest().capture_ms - min(f.capture_ms for f in ring.after(-1)) <= 2000
    assert [f.frame_id for f in ring.after(97)] == [98, 99]


@pytest.mark.parametrize("kind", ["vit", "kcf", "csrt"])
def test_reseed_replays_detection_to_the_present(kind):
    ring, truth = _moving_sequence()
    det_frame = 5                                  # the detector saw frame 5; now is frame 19
    rs = reseed(lambda: make_tracker(kind), ring, det_frame, truth[det_frame], max_steps=30)
    assert rs is not None and rs.frame_id == 19 and rs.steps == 14
    assert box_offset(rs.box, truth[19]) < 0.15
    assert iou(rs.box, truth[19]) > 0.6


def test_reseed_subsamples_long_replays():
    ring, truth = _moving_sequence(n=40, fps=40.0, vx=200.0)
    rs = reseed(lambda: make_tracker("vit"), ring, 2, truth[2], max_steps=8)
    assert rs is not None and rs.steps <= 8 and rs.frame_id == 39
    assert box_offset(rs.box, truth[39]) < 0.3


def test_reseed_missing_frame_returns_none():
    ring, truth = _moving_sequence()
    assert reseed(lambda: make_tracker("kcf"), ring, 999, truth[0]) is None


def test_box_offset_threshold_semantics():
    a = (0.0, 0.0, 100.0, 100.0)
    assert box_offset((20.0, 0.0, 120.0, 100.0), a) == pytest.approx(0.2)
    assert box_offset((40.0, 0.0, 140.0, 100.0), a) > 0.3


def test_counter_roi_env_override(monkeypatch):
    counter_roi.cache_clear()
    try:
        monkeypatch.setenv("MHD_COUNTER_ROI", "1008,640,1280,720")
        assert counter_roi() == (1008.0, 640.0, 1280.0, 720.0)
        counter_roi.cache_clear()
        monkeypatch.setenv("MHD_COUNTER_ROI", "garbage")
        assert counter_roi()[2:] == (1280.0, 720.0)
    finally:
        monkeypatch.delenv("MHD_COUNTER_ROI", raising=False)
        counter_roi.cache_clear()
