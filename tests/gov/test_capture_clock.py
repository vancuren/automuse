"""CaptureClock and the per-frame capture estimate (mhd.backend.capture)."""
from __future__ import annotations

import numpy as np

from mhd.backend.capture import CaptureClock, estimate


def test_min_offset_gives_capture_time_and_late_frames_look_old():
    c = CaptureClock(window_ms=10_000)
    # pts on the camera clock; receive on ours = pts + 5000 + jitter (min jitter 20 ms)
    for i, jit in enumerate([40, 20, 35, 60, 25]):
        cap = c.update(i * 66.7, 5000 + i * 66.7 + jit)
    assert abs(cap - (5000 + 4 * 66.7 + 20)) < 1e-6        # pts + min(receive - pts)
    # a frame delivered 2 s late keeps its pts: it looks 2 s old
    late = c.update(5 * 66.7, 5000 + 5 * 66.7 + 20 + 2000)
    assert (5000 + 5 * 66.7 + 20 + 2000) - late == 2000.0


def test_window_forgets_old_minimum_and_pts_restart_resets():
    c = CaptureClock(window_ms=1000)
    c.update(0.0, 100.0)                                   # offset 100
    for i in range(1, 30):
        c.update(i * 100.0, i * 100.0 + 300.0)             # offset 300 from now on
    assert c.update(3000.0, 3300.0) == 3000.0 + 300.0      # the old 100 ms minimum expired
    c2 = CaptureClock()
    c2.update(50_000.0, 60_000.0)
    assert c2.update(10.0, 70_000.0) == 10.0 + 69_990.0    # pts went backwards: new session


def test_offset_and_pts_hold_and_receive_fallback():
    img = np.zeros((4, 4, 3), np.uint8)
    c = CaptureClock(offset_ms=450.0)
    cap, rid, src = estimate(img, 1000.0, 2000.0, c, "real")
    assert (cap, rid, src) == (1000.0 + 1000.0 - 450.0, None, "pts")
    cap2, _, src2 = estimate(img, 0.0, 2500.0, c, "real")  # pts missing after a read timeout
    assert (cap2, src2) == (cap, "pts_hold")
    fresh = CaptureClock()
    assert estimate(img, -1.0, 3000.0, fresh, "real") == (3000.0, None, "receive")


def test_twin_counter_wins_when_it_decodes():
    img = np.zeros((4, 4, 3), np.uint8)
    c = CaptureClock()
    assert estimate(img, 500.0, 900.0, c, "twin", counter=lambda im: (42, 777.0)) == (777.0, 42, "counter")
    assert estimate(img, 500.0, 900.0, c, "twin", counter=lambda im: None)[2] == "pts"
    assert estimate(img, 500.0, 900.0, c, "real", counter=lambda im: (42, 777.0))[2] == "pts"
