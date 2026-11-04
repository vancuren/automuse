"""Live end-to-end: scripts/run_backend.sh -> uvicorn (port 8100) with real RTSP video from mediamtx
(127.0.0.1:8105) and a UDP listener standing in for the drone (127.0.0.1:23512).

Proves the launcher, FLIGHT_MODE=twin, the governor on the real RC path, /ws/frames with real
capture-time estimates, takeoff, land re-assert, and neutral+land on graceful shutdown (SIGINT).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from mhd.contracts import decode_frame, now_ms
from test_rtsp_integration import rtsp_stream  # noqa: F401  (fixture: mediamtx + HEVC publisher)
from test_wire import Receiver

ROOT = Path(__file__).resolve().parents[2]
HTTP = "http://127.0.0.1:8100"


@pytest.fixture
def live_backend(rtsp_stream, tmp_path):  # noqa: F811
    rx = Receiver(23512)
    rx.start()
    log = open(tmp_path / "backend.log", "w")
    env = dict(os.environ, LOG_LEVEL="WARNING")
    proc = subprocess.Popen([str(ROOT / "scripts" / "run_backend.sh"), "--mode", "twin", "--port", "8100",
                             "--control-port", "23512", "--local-control-port", "23513",
                             "--rtsp-url", "rtsp://127.0.0.1:8105/live/ch00_1", "--headless"],
                            stdout=log, stderr=log, env=env)
    try:
        ok = False
        for _ in range(100):
            try:
                st = httpx.get(HTTP + "/state", timeout=1.0).json()
                if st.get("frame_age_ms") is not None:
                    ok = True
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.2)
        if not ok:
            pytest.fail("backend did not come up with video; see " + str(tmp_path / "backend.log"))
        yield proc, rx
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        rx.close()
        log.close()


def test_live_backend_end_to_end(live_backend, record_property):
    import websockets.sync.client as wsc
    proc, rx = live_backend
    c = httpx.Client(base_url=HTTP, timeout=3.0)

    # Real frames over the pull protocol, with capture <= receive.
    with wsc.connect("ws://127.0.0.1:8100/ws/frames") as ws:
        ws.send(json.dumps({"type": "next", "after_frame_id": -1}))
        meta, jpeg = decode_frame(ws.recv(timeout=3))
        assert (meta.width, meta.height) == (1280, 720) and jpeg[:2] == b"\xff\xd8"
        assert meta.capture_ms <= meta.receive_ms + 1.0 and meta.receive_ms <= now_ms()
        record_property("frame_receive_minus_capture_ms", round(meta.receive_ms - meta.capture_ms, 1))

    # Idle: neutral packets at ~25 Hz.
    t0 = now_ms()
    time.sleep(1.0)
    idle = rx.since(t0)
    assert len(idle) >= 22 and all(p["f1"] == 0 and p["yaw"] == 128 for _, p in idle)

    # Takeoff (headless twin: no UI heartbeat needed; video is fresh).
    t0 = now_ms()
    r = c.post("/command", json={"type": "cmd", "command": "takeoff"}).json()
    assert r["ok"], r
    time.sleep(4.4)
    st = c.get("/state").json()
    assert st["flight_state"] == "AIRBORNE" and "headless_bypass" in st["safety_flags"]
    assert sum(1 for _, p in rx.since(t0) if p["f1"] & 0x01) == 1

    # Land: 5 land packets per second on the wire.
    t0 = now_ms()
    assert c.post("/command", json={"type": "cmd", "command": "land"}).json()["ok"]
    time.sleep(2.0)
    n_land = sum(1 for _, p in rx.since(t0 + 30) if p["f1"] & 0x80)
    assert 9 <= n_land <= 11, n_land
    assert c.post("/command", json={"type": "cmd", "command": "confirm_touchdown"}).json()["ok"]

    # Graceful shutdown: neutral sticks plus land for 0.5 s.
    t0 = now_ms()
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=10)
    tail = rx.since(t0)
    lands = [p for _, p in tail if p["f1"] & 0x80]
    record_property("shutdown_land_packets", len(lands))
    print(f"\nlive: receive-capture {meta.receive_ms - meta.capture_ms:.1f} ms; idle packets/s {len(idle)}; "
          f"land/2 s {n_land}; shutdown land packets {len(lands)}")
    assert len(lands) >= 8 and all(p["yaw"] == 128 and p["pitch"] == 128 for p in lands)
    assert proc.returncode is not None
