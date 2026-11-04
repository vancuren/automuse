"""Acceptance: the unchanged upstream turbodrone backend flies the twin over the wire.

The backend runs exactly as for the real drone (DRONE_TYPE=x69_lg, X69 RC adapter, RTSP video
adapter), pointed at 127.0.0.1. We drive it like the stock UI's keyboard: {"type":"axes",
"mode":"incremental", ...} at 30 Hz over its /ws, plus takeoff / land / estop messages.
No Chrome here: the twin streams its placeholder video so the backend's RTSP client has a stream.
"""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
import websockets

from procs import BACKEND_PORT, start_backend, start_twin, twin_status, wait_until

RC_PORT = 23522
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def stack(upstream):
    twin = start_twin("s1", rc_port=RC_PORT, extra=["--no-probe"])
    try:
        backend = start_backend(rc_port=RC_PORT)
    except Exception:
        twin.stop()
        raise
    yield twin, backend
    backend.stop()
    twin.stop()


async def _drive(messages_by_phase):
    """messages_by_phase: list of (seconds, message or None, repeat_hz)."""
    async with websockets.connect(f"ws://127.0.0.1:{BACKEND_PORT}/ws") as ws:
        for seconds, msg, hz in messages_by_phase:
            t_end = time.monotonic() + seconds
            if msg is None:
                await asyncio.sleep(seconds)
                continue
            if not hz:
                await ws.send(json.dumps(msg))
                await asyncio.sleep(seconds)
                continue
            while time.monotonic() < t_end:
                await ws.send(json.dumps(msg))
                await asyncio.sleep(1.0 / hz)


def drive(*phases):
    asyncio.run(_drive(list(phases)))


def keys(yaw=0, pitch=0, roll=0, throttle=0):
    return {"type": "axes", "mode": "incremental", "throttle": throttle, "yaw": yaw, "pitch": pitch, "roll": roll}


def drone():
    return twin_status()["drone"]


def test_backend_streams_rc_and_reads_rtsp(stack):
    st = wait_until(lambda: (lambda s: s if s["rc"]["packets"] > 50 else None)(twin_status()), 10,
                    what="RC packets from the backend")
    rc = st["rc"]
    assert not rc["errors"]
    assert rc["keepalives"] >= 1
    # upstream sleeps 40 ms after each tick, so it runs a little under 25 Hz
    assert 38 <= rc["interarrival_p50_ms"] <= 60
    print(f"\nupstream backend RC inter-arrival: p50 {rc['interarrival_p50_ms']} ms, "
          f"p99 {rc['interarrival_p99_ms']} ms, max {rc['interarrival_max_ms']} ms, alarm {rc['p99_alarm']}")
    # the backend's MJPEG endpoint serves frames that came through the twin's RTSP stream
    with httpx.stream("GET", f"http://127.0.0.1:{BACKEND_PORT}/mjpeg", timeout=10) as r:
        got = b""
        for chunk in r.iter_bytes():
            got += chunk
            if got.count(b"\xff\xd9") >= 2:
                break
    assert b"Content-Type: image/jpeg" in got


def test_keyboard_axes_takeoff_yaw_pitch_land(stack):
    drive((0.3, {"type": "takeoff"}, 0))
    wait_until(lambda: drone()["phase"] == "flying", 8, what="flying after takeoff")
    yaw0 = drone()["yaw_deg"]
    drive((2.0, keys(yaw=1), 30), (1.5, keys(), 30))        # hold the right-yaw key, then release
    d = drone()
    turned = (d["yaw_deg"] - yaw0) % 360
    assert 30 < turned < 330 and turned < 300, f"expected a clockwise turn, got {turned}"
    assert abs(d["yaw_rate_dps"]) < 15                    # stick decayed back to center after release
    y0, x0 = d["y"], d["x"]
    drive((1.5, keys(pitch=1), 30), (1.5, keys(), 30))       # forward
    d = drone()
    moved = ((d["x"] - x0) ** 2 + (d["y"] - y0) ** 2) ** 0.5
    assert moved > 0.3, f"expected forward motion, moved {moved:.2f} m"
    drive((0.3, {"type": "land"}, 0))
    wait_until(lambda: drone()["phase"] == "landed", 6, what="landed")
    assert not drone()["motors_on"]


def test_estop_cuts_motors_in_the_air(stack):
    drive((0.3, {"type": "takeoff"}, 0))
    wait_until(lambda: drone()["phase"] == "flying", 8, what="flying")
    n0 = drone()["events"]["estop"]
    drive((0.3, {"type": "estop"}, 0))
    wait_until(lambda: not drone()["motors_on"], 1.0, what="motors cut")
    wait_until(lambda: drone()["phase"] == "landed", 2, what="fell to the ground")
    assert drone()["events"]["estop"] == n0 + 1           # one-shot upstream: exactly one packet carried it


def test_camera_tilt_from_ui_message(stack):
    t0 = drone()["cam_tilt_deg"]
    drive((0.5, {"type": "set_camera_tilt", "camera_tilt_direction": -1}, 0),
          (0.2, {"type": "set_camera_tilt", "camera_tilt_direction": 0}, 0))
    assert drone()["cam_tilt_deg"] < t0
