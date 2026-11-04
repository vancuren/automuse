"""Acceptance through the real video path: Chrome/WebGL render -> JPEG -> noise -> counter ->
latency buffer -> hevc_videotoolbox 3 Mb/s -> RTSP (mediamtx) -> cv2.VideoCapture (CAP_FFMPEG, TCP).

Measures total latency with our own reader, independent of the twin's internal probe.
Needs Google Chrome with a GPU (the renderer refuses SwiftShader).
"""
from __future__ import annotations

import json
import os
import statistics
import time

import cv2
import numpy as np
import pytest

from mhd.contracts import now_ms
from mhd.twin import counter
from mhd.twin.video import PROBE_CAPTURE_OPTIONS, open_rtsp_capture

from procs import RTSP_PORT, start_renderer, start_twin, twin_control, twin_status, wait_until

URL = f"rtsp://127.0.0.1:{RTSP_PORT}/live/ch00_1"
pytestmark = pytest.mark.slow
RESULTS = {}


@pytest.fixture(scope="module")
def stack():
    twin = start_twin("s1", rc_port=23523)
    rend = start_renderer()
    try:
        wait_until(lambda: twin_status()["renderer"]["camera_connected"], 30, what="renderer connected")
        wait_until(lambda: twin_status()["fps"] > 10, 20, what="frames flowing")
    except Exception:
        print(rend.tail(), twin.tail())
        rend.stop(); twin.stop()
        raise
    yield twin, rend
    out = os.path.join(os.environ.get("MHD_LOGS", "logs"), "tests", "video_path_results.json")
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2)
    rend.stop()
    twin.stop()


def read_frames(n: int, cap=None, timeout_s: float = 60.0):
    """Returns list of (decode_ms, frame_or_None, decoded (rid, render_ms) or None)."""
    own = cap is None
    cap = cap or open_rtsp_capture(URL)
    assert cap.isOpened(), "RTSP did not open"
    out = []
    t_end = time.monotonic() + timeout_s
    while len(out) < n and time.monotonic() < t_end:
        ok, f = cap.read()
        if not ok:
            continue
        t = now_ms()
        out.append((t, f, counter.decode(f, ref_ms=t)))
    if own:
        cap.release()
    return out


def test_webgl_is_gpu(stack):
    gl = twin_status()["renderer"]["webgl"] or ""
    RESULTS["webgl"] = gl
    assert gl and "swiftshader" not in gl.lower()


def test_opencv_reads_rtsp_with_backend_settings(stack):
    """Exactly the upstream X69 adapter's capture options (and its default decoder threading)."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = PROBE_CAPTURE_OPTIONS
    cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    assert cap.isOpened()
    frames = read_frames(120, cap=cap)
    cap.release()
    assert len(frames) == 120
    assert frames[-1][1].shape == (720, 1280, 3)
    lat = [t - d[1] for t, _, d in frames[40:] if d]
    RESULTS["latency_default_threads_ms_p50"] = round(statistics.median(lat), 1)


def test_counter_decodes_on_99_percent(stack):
    frames = read_frames(500)
    first = next(i for i, (_, _, d) in enumerate(frames) if d)
    tail = frames[first:]
    ok = sum(1 for _, _, d in tail if d)
    rate = ok / len(tail)
    RESULTS["counter_decode"] = {"frames": len(tail), "decoded": ok, "rate": round(rate, 4)}
    ids = [d[0] for _, _, d in tail if d]
    assert all(b > a for a, b in zip(ids, ids[1:])), "render ids must increase"
    assert rate >= 0.99


@pytest.mark.parametrize("target", [350, 600])
def test_total_latency_matches_setpoint(stack, target):
    twin_control(latency_ms=target)
    time.sleep(5.0)                     # let the downstream estimate settle
    frames = read_frames(250)
    lat = np.array([t - d[1] for t, _, d in frames[25:] if d])
    p05, p50, p95 = (float(np.percentile(lat, q)) for q in (5, 50, 95))
    st = twin_status()
    RESULTS[f"latency_{target}"] = {"p05": round(p05, 1), "p50": round(p50, 1), "p95": round(p95, 1),
                                    "n": int(lat.size), "baseline_ms_p50": st["latency"]["baseline_ms_p50"],
                                    "fps": st["fps"]}
    assert abs(p50 - target) <= 50, RESULTS[f"latency_{target}"]
    assert target - 50 <= p05 and p95 <= target + 50, RESULTS[f"latency_{target}"]


def test_stall_then_burst(stack):
    """A 1.5 s stall: nothing arrives, then the held frames arrive back to back, then it recovers.

    VideoToolbox keeps the last few frames until new input arrives, so the reader's gap starts at
    the moment of the stall request, not after the frames already in flight.
    """
    twin_control(latency_ms=350)
    time.sleep(2.0)
    cap = open_rtsp_capture(URL)
    read_frames(30, cap=cap)
    t_stall = now_ms()
    twin_control(stall={"duration_ms": 1500, "mode": "burst"})
    frames = read_frames(100, cap=cap)
    ts = [t_stall] + [t for t, _, _ in frames]
    gaps = np.diff(ts)
    i = int(np.argmax(gaps))
    first_after = ts[i + 1]
    burst = sum(1 for t in ts[i + 1:] if t - first_after < 500)
    lat_after = [t - d[1] for t, _, d in frames[i:i + 3] if d]
    read_frames(150, cap=cap)             # keep reading (a paused reader builds its own backlog)
    rec = [t - d[1] for t, _, d in read_frames(100, cap=cap)[20:] if d]
    cap.release()
    RESULTS["stall"] = {"max_gap_ms": round(float(gaps[i]), 1), "frames_within_500ms_after": burst,
                        "first_latencies_after_ms": [round(x) for x in lat_after],
                        "recovered_latency_p50_ms": round(statistics.median(rec), 1)}
    assert gaps[i] >= 1200, RESULTS["stall"]
    assert burst >= 15, RESULTS["stall"]                    # held frames arrive back to back
    assert lat_after and lat_after[0] >= 1500, RESULTS["stall"]
    assert abs(statistics.median(rec) - 350) <= 50, RESULTS["stall"]
    twin_control(stall={})


@pytest.mark.parametrize("scenario", ["s1", "s2", "s3"])
def test_scenario_renders_at_15fps_or_more(stack, scenario):
    twin_control(scenario=scenario, latency_ms=0)
    time.sleep(4.0)
    samples = []
    for _ in range(4):
        samples.append(twin_status()["fps"])
        time.sleep(0.8)
    st = twin_status()
    RESULTS[f"fps_{scenario}"] = {"min": min(samples), "samples": samples,
                                  "renderer_stats": {k: v for k, v in (st["renderer"] or {}).items() if k != "clients"}}
    assert min(samples) >= 15, samples
