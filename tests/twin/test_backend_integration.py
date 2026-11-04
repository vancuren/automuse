"""Acceptance: the governed turbodrone backend flies the twin over the wire.

The backend runs exactly as for the real drone (DRONE_TYPE=x69_lg, X69 RC adapter, RTSP video
adapter, safety governor), pointed at 127.0.0.1. We drive it like the stock UI's keyboard
({"type":"axes","mode":"incremental"} at 30 Hz) and like the new UI ({"type":"axes","mode":"abs"},
sent only while nonzero, and {"type":"cmd"} commands) over its /ws.
No Chrome here: the twin streams its placeholder video so the backend's RTSP client has a stream.
The governor deliberately changes upstream behavior, and these tests assert the governed behavior:
held keys ramp to a capped stick, E-stop is latched into every packet, landing ends with an
operator touchdown confirmation, and camera tilt arrives as an explicit command.
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


def abs_axes(yaw=0.0, pitch=0.0, roll=0.0, throttle=0.0):
    return {"type": "axes", "mode": "abs", "throttle": throttle, "yaw": yaw, "pitch": pitch, "roll": roll}


def cmd(name):
    return {"type": "cmd", "command": name}


def drone():
    return twin_status()["drone"]


def gov():
    return httpx.get(f"http://127.0.0.1:{BACKEND_PORT}/state", timeout=5).json()


def takeoff_and_wait_airborne():
    drive((0.3, {"type": "takeoff"}, 0))
    wait_until(lambda: drone()["phase"] == "flying", 8, what="twin flying after takeoff")
    # the governor holds neutral sticks for its takeoff period before it accepts any stick input
    wait_until(lambda: gov()["flight_state"] == "AIRBORNE", 8, what="governor AIRBORNE")


def land_and_confirm():
    drive((0.3, {"type": "land"}, 0))
    wait_until(lambda: drone()["phase"] == "landed", 8, what="landed")
    assert not drone()["motors_on"]
    drive((0.3, cmd("confirm_touchdown"), 0))
    wait_until(lambda: gov()["flight_state"] == "ON_GROUND", 3, what="governor ON_GROUND")


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
    takeoff_and_wait_airborne()
    # Legacy keyboard (incremental): the governor emulates it as a capped ramp, so the turn is
    # deliberately gentler than upstream's ramp to full deflection.
    yaw0 = drone()["yaw_deg"]
    drive((2.0, keys(yaw=1), 30), (1.5, None, 0))   # hold right-yaw, then release (stale after 0.25 s)
    d = drone()
    turned = (d["yaw_deg"] - yaw0) % 360
    print(f"\nlegacy incremental yaw, 2 s: turned {turned:.1f} deg")
    # twin yaw rate per count is a placeholder until Session B; assert direction, not size
    assert 5 < turned < 300, f"expected a clockwise turn, got {turned}"
    assert abs(d["yaw_rate_dps"]) < 15              # sticks back to neutral after release
    # New UI (absolute, nonzero only): half right stick for 2 s.
    yaw1 = d["yaw_deg"]
    drive((1.0, abs_axes(yaw=0.5), 30), (1.5, None, 0))
    turned2 = (drone()["yaw_deg"] - yaw1) % 360
    print(f"abs yaw 0.5, 1 s: turned {turned2:.1f} deg")
    assert 5 < turned2 < 300, f"expected a clockwise turn, got {turned2}"
    assert abs(drone()["yaw_rate_dps"]) < 15
    d = drone()
    y0, x0 = d["y"], d["x"]
    drive((2.0, abs_axes(pitch=0.5), 30), (1.5, None, 0))   # forward
    d = drone()
    moved = ((d["x"] - x0) ** 2 + (d["y"] - y0) ** 2) ** 0.5
    print(f"abs pitch 0.5, 2 s: moved {moved:.2f} m")
    assert moved > 0.1, f"expected forward motion, moved {moved:.2f} m"
    land_and_confirm()


def test_estop_latches_and_cuts_motors_in_the_air(stack):
    takeoff_and_wait_airborne()
    n0 = drone()["events"]["estop"]
    drive((0.3, {"type": "estop"}, 0))
    wait_until(lambda: not drone()["motors_on"], 1.0, what="motors cut")
    wait_until(lambda: drone()["phase"] == "landed", 2, what="fell to the ground")
    time.sleep(0.5)
    n1 = drone()["events"]["estop"]
    assert n1 >= n0 + 5, f"E-stop must be latched into every packet; saw {n1 - n0} packets"
    assert gov()["flight_state"] == "ESTOPPED"
    # Reset goes to LANDING (never straight to flight), then the operator confirms touchdown.
    drive((0.3, cmd("estop_reset"), 0))
    wait_until(lambda: gov()["flight_state"] in ("LANDING", "ON_GROUND"), 3, what="reset to LANDING")
    if gov()["flight_state"] == "LANDING":
        drive((0.3, cmd("confirm_touchdown"), 0))
    wait_until(lambda: gov()["flight_state"] == "ON_GROUND", 3, what="governor ON_GROUND")


def test_camera_tilt_from_ui_command(stack):
    assert gov()["flight_state"] != "ESTOPPED"
    t0 = drone()["cam_tilt_deg"]
    drive((0.6, cmd("tilt_down"), 0), (0.2, cmd("tilt_stop"), 0))
    t1 = drone()["cam_tilt_deg"]
    print(f"\ntilt: {t0:.1f} -> {t1:.1f} deg")
    assert t1 < t0
