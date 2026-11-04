"""Counter block: exact round trips, robustness to JPEG/resize/HEVC, CRC rejection, render-time unwrap."""
from __future__ import annotations

import os
import subprocess

import cv2
import numpy as np
import pytest

from mhd.twin import counter


def scene(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)
    return cv2.resize(base, (1280, 720), interpolation=cv2.INTER_CUBIC)


def test_roi_is_the_bottom_right_corner():
    x1, y1, x2, y2 = counter.COUNTER_ROI
    assert (x2, y2) == (1280, 720) and x1 % 16 == 0 and y1 % 16 == 0
    assert (x2 - x1) * (y2 - y1) < 0.03 * 1280 * 720


def test_round_trip_exact():
    ref = 974_000_000.0
    for rid, ms in ((1, ref - 120.0), (123456, ref - 3.0), (2 ** 20 - 1, ref - 1999.0)):
        img = counter.stamp(scene(rid % 7), rid, ms)
        assert counter.decode(img, ref_ms=ref) == (rid, float(round(ms)))


def test_render_ms_unwraps_across_the_modulo_boundary():
    m = counter.MS_MOD
    ms = 5 * m + 10.0                    # just after a wrap
    img = counter.stamp(scene(), 7, ms)
    assert counter.decode(img, ref_ms=ms + 400.0) == (7, ms)
    ms2 = 5 * m - 10.0                   # just before a wrap, decoded just after it
    img = counter.stamp(scene(), 8, ms2)
    assert counter.decode(img, ref_ms=5 * m + 300.0) == (8, ms2)


def test_survives_jpeg_and_downscale():
    img = counter.stamp(scene(3), 4242, 1_000_000.0)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert counter.decode(dec, ref_ms=1_000_100.0) == (4242, 1_000_000.0)
    small = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    assert counter.decode(small, ref_ms=1_000_100.0) == (4242, 1_000_000.0)


def test_crc_rejects_corruption_and_blank_frames():
    img = counter.stamp(scene(), 99, 5000.0)
    bits = counter.encode_bits(99, 5000.0)
    for i in (0, 10, 30, 63):
        flipped = list(bits)
        flipped[i] ^= 1
        assert counter.decode_bits(flipped, ref_ms=5000.0) is None
    assert counter.decode(np.zeros((720, 1280, 3), np.uint8)) is None
    assert counter.decode(np.full((720, 1280, 3), 128, np.uint8)) is None
    counter.mask_roi(img)
    assert counter.decode(img) is None


def test_hevc_videotoolbox_3mbps_decode_rate(tmp_path):
    """The real encoder path (file instead of RTSP): >= 99% of frames must decode."""
    path = tmp_path / "counter.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "1280x720", "-r", "25", "-i", "pipe:0", "-c:v", "hevc_videotoolbox", "-b:v", "3M", "-bf", "0",
           "-g", "25", "-realtime", "1", "-tag:v", "hvc1", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    big = cv2.resize(np.random.default_rng(1).integers(0, 255, (90, 160, 3), dtype=np.uint8), (2560, 1440),
                     interpolation=cv2.INTER_CUBIC)
    rng = np.random.default_rng(2)
    expect = []
    n = 250
    for i in range(n):
        dx, dy = int(600 + 500 * np.sin(i / 17)), int(300 + 250 * np.cos(i / 23))
        img = big[dy:dy + 720, dx:dx + 1280].copy()
        cv2.circle(img, (int(640 + 560 * np.sin(i / 7)), 650), 90, (255, 255, 255), -1)  # bright blob by the block
        img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
        counter.stamp(img, i + 1, 2_000_000.0 + 40.0 * i)
        expect.append((i + 1, 2_000_000.0 + 40.0 * i))
        p.stdin.write(img.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    ok_count = total = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if counter.decode(f, ref_ms=2_000_000.0) == expect[total]:
            ok_count += 1
        total += 1
    assert total == n
    assert ok_count / total >= 0.99, f"{ok_count}/{total}"
