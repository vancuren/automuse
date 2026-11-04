"""Session B step-identification tool (mhd.governor.sysid)."""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from mhd.governor.sysid import Step, aruco_measure, build_schedule, fit_steps, run

ROOT = Path(__file__).resolve().parents[2]


def _marker_image(cx: float, side: int, w: int = 1280, h: int = 720) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = cv2.aruco.generateImageMarker(d, 7, side)
    img = np.full((h, w, 3), 255, np.uint8)
    x0, y0 = int(round(cx - side / 2)), (h - side) // 2
    img[y0:y0 + side, x0:x0 + side] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return img


def test_schedule_flies_each_amplitude_both_ways_and_refuses_big_steps():
    s = build_schedule("yaw", [0.1, 0.2], 2.0, 3.0)
    assert [x.value for x in s] == [0.1, -0.1, 0.2, -0.2]
    with pytest.raises(ValueError):
        build_schedule("yaw", [0.6], 2.0, 3.0)
    with pytest.raises(ValueError):
        build_schedule("flip", [0.1], 2.0, 3.0)


def test_aruco_bearing_and_range_from_a_synthetic_frame():
    hfov = 75.0
    f = 640 / math.tan(math.radians(hfov) / 2)
    img = _marker_image(cx=900, side=120)
    bearing, rng = aruco_measure(img, hfov, marker_size_m=0.15)
    assert bearing == pytest.approx(math.atan((900 - 640) / f), abs=0.01)
    assert rng == pytest.approx(f * 0.15 / 120, rel=0.05)
    assert aruco_measure(np.full((720, 1280, 3), 255, np.uint8), hfov, 0.15) is None


def test_fit_recovers_gain_deadband_and_lag():
    k_true, db_true, lag_true = 0.012, 6.0, 0.25
    steps, samples, t = [], [], 0.0
    bearing = 0.0
    for counts in (13, -13, 25, -25, 44, -44):
        st = Step("yaw", counts / 127.0, 2.0, 1.0, t_start_ms=t, t_end_ms=t + 2000.0, wire_count=counts)
        steps.append(st)
        for i in range(int(3.0 * 15)):               # 15 fps through hold + rest
            tt = t + i * 1000.0 / 15
            moving = st.t_start_ms + lag_true * 1000.0 <= tt <= st.t_end_ms
            rate = k_true * math.copysign(max(0.0, abs(counts) - db_true), counts) if moving else 0.0
            bearing -= rate / 15.0                   # turning right moves the marker left
            samples.append((tt, bearing + np.random.default_rng(i).normal(0, 0.001), 2.0))
        t += 3000.0
    fit = fit_steps(steps, samples, "yaw")
    assert fit["k_per_count"] == pytest.approx(k_true, rel=0.1)
    assert fit["deadband_counts"] == pytest.approx(db_true, abs=1.5)
    assert fit["lag_s_median"] == pytest.approx(lag_true, abs=0.15)


def _pitch_steps(moves_with_stick: bool):
    """Synthetic pitch steps: forward speed = -d(range)/dt; a correct axis closes on the marker for
    a positive stick."""
    k, db = 0.012, 6.0
    steps, samples, t, rng_m = [], [], 0.0, 3.0
    for counts in (13, -13, 25, -25, 32, -32):
        st = Step("pitch", counts / 127.0, 1.5, 1.0, t_start_ms=t, t_end_ms=t + 1500.0, wire_count=counts)
        steps.append(st)
        for i in range(int(2.5 * 15)):
            tt = t + i * 1000.0 / 15
            moving = st.t_start_ms + 200.0 <= tt <= st.t_end_ms
            v = k * math.copysign(max(0.0, abs(counts) - db), counts) if moving else 0.0
            rng_m -= (v if moves_with_stick else -v) / 15.0
            samples.append((tt, 0.0, rng_m + np.random.default_rng(i).normal(0, 0.002)))
        t += 2500.0
    return steps, samples


def test_fit_reports_the_direction_and_flags_a_reversed_axis():
    from mhd.governor.sysid import direction_message
    ok = fit_steps(*_pitch_steps(True), "pitch")
    rev = fit_steps(*_pitch_steps(False), "pitch")
    # the K / deadband fit uses absolute values, so it cannot tell the two apart ...
    assert ok["k_per_count"] == pytest.approx(rev["k_per_count"], rel=0.05)
    # ... the per-step signed rates can
    assert ok["direction"] == "OK" and rev["direction"] == "REVERSED"
    assert all(r["moved_commanded_way"] is False for r in rev["steps"] if r["moved_commanded_way"] is not None)
    assert "REVERSED" in direction_message("pitch", rev) and "Do not fly follow" in direction_message("pitch", rev)
    assert "OK" in direction_message("pitch", ok)


@pytest.fixture
def marker_stream(tmp_path):
    from test_rtsp_integration import _mediamtx, _port_open
    mtx, ff = _mediamtx(), shutil.which("ffmpeg")
    if not mtx.exists() or not ff or _port_open(8105):
        pytest.skip("mediamtx/ffmpeg unavailable or port 8105 busy")
    png = tmp_path / "marker.png"
    cv2.imwrite(str(png), _marker_image(cx=640, side=160))
    log = open(tmp_path / "rtsp.log", "w")
    procs = [subprocess.Popen([str(mtx), str(Path(__file__).with_name("mediamtx_gov.yml"))],
                              stdout=log, stderr=log, cwd=tmp_path)]
    try:
        for _ in range(50):
            if _port_open(8105):
                break
            time.sleep(0.1)
        procs.append(subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-re", "-loop", "1", "-i", str(png),
                                       "-r", "15", "-c:v", "hevc_videotoolbox", "-b:v", "2M", "-g", "15", "-bf", "0",
                                       "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp",
                                       "rtsp://127.0.0.1:8105/live/ch00_1"], stdout=log, stderr=log))
        time.sleep(2.0)
        yield
    finally:
        for p in reversed(procs):
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        log.close()


def test_live_sysid_plumbing(marker_stream, tmp_path, monkeypatch):
    """The tool refuses on the ground, then flies its schedule through the real backend."""
    from test_wire import Receiver
    rx = Receiver(23512)
    rx.start()
    log = open(tmp_path / "backend.log", "w")
    proc = subprocess.Popen([str(ROOT / "scripts" / "run_backend.sh"), "--mode", "twin", "--port", "8100",
                             "--control-port", "23512", "--local-control-port", "23513",
                             "--rtsp-url", "rtsp://127.0.0.1:8105/live/ch00_1", "--headless",
                             "--logs", str(tmp_path / "backend-logs")],
                            stdout=log, stderr=log, env=dict(os.environ, LOG_LEVEL="WARNING"))
    monkeypatch.setenv("MHD_LOGS", str(tmp_path))
    try:
        base = "http://127.0.0.1:8100"
        for _ in range(100):
            try:
                if httpx.get(base + "/state", timeout=1.0).json().get("frame_age_ms") is not None:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.2)
        with pytest.raises(SystemExit):
            run(base, "yaw", [0.1], 0.5, 0.3, 0.15)           # ON_GROUND: refused
        # Takeoff goes over /ws (POST /command refuses it since the review fixes).
        assert not httpx.post(base + "/command", json={"type": "cmd", "command": "takeoff"}).json()["ok"]
        import websockets.sync.client as wsc
        with wsc.connect("ws://127.0.0.1:8100/ws") as ws:
            ws.send(json.dumps({"type": "cmd", "command": "takeoff"}))
            time.sleep(0.2)
        assert httpx.get(base + "/state").json()["flight_state"] == "TAKING_OFF"
        time.sleep(4.4)

        res = run(base, "yaw", [0.10, 0.20], 0.6, 0.4, 0.15)
        assert res["aborted"] is None
        counts = [s["wire_count"] for s in res["schedule"]]
        assert counts == [13, -13, 25, -25]
        assert len(res["samples"]) > 20                         # marker seen on real decoded frames
        assert all(abs(b) < 0.02 for _, b, _ in res["samples"])  # static marker at the center
        wire_yaws = {p["yaw"] - 128 for _, p in rx.packets}
        assert {13, -13, 25, -25} <= wire_yaws and max(abs(v) for v in wire_yaws) <= 25
        assert list(Path(tmp_path).glob("sysid-yaw-*.json"))
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        rx.close()
        log.close()
