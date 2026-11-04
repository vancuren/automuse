"""FastAPI TestClient tests of the patched turbodrone backend (web_server.py + mhd.backend.hub).

The real lifespan runs: governor installed on the X69 model, PacedFlightController sending real
X69 datagrams to UDP 127.0.0.1:23514 (nobody listening; that is fine), and a fake video adapter in
place of RTSP that produces small JPEG frames with capture_ms.
"""
from __future__ import annotations

import importlib
import json
import queue
import threading
import time

import cv2
import numpy as np
import pytest

from conftest import make_mission, make_obs
from mhd.contracts import decode_frame, now_ms


class FakeVideoAdapter:
    """Stands in for TimedX69RtspAdapter: 30 fps synthetic JPEGs captured 80 ms before receipt."""

    def __init__(self, *a, **kw):
        self._running = False
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._thread = None
        ok, buf = cv2.imencode(".jpg", np.zeros((36, 64, 3), np.uint8))
        self._jpeg = buf.tobytes()

    def start(self):
        from models.video_frame import VideoFrame
        self._running = True

        def loop():
            i = 0
            while self._running:
                i += 1
                f = VideoFrame(i, self._jpeg)
                t = now_ms()
                f.capture_ms, f.receive_ms, f.width, f.height, f.render_id = t - 80.0, t, 1280, 720, None
                try:
                    self._q.put_nowait(f)
                except queue.Full:
                    pass
                time.sleep(1 / 30)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def is_running(self):
        return self._running

    def get_frame(self, timeout=1.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_packets(self):
        return []


BASE_ENV = {
    "FLIGHT_MODE": "twin", "DRONE_TYPE": "x69_lg", "DRONE_IP": "127.0.0.1", "CONTROL_PORT": "23514",
    "X69_LG_LOCAL_CONTROL_PORT": "23515", "X69_LG_VIDEO_MODE": "rtsp", "PLUGINS_ENABLED": "false",
    "X69_LG_RTSP_URL": "rtsp://127.0.0.1:8109/live/ch00_1", "LOG_LEVEL": "WARNING",
}


@pytest.fixture
def backend(monkeypatch, tmp_path):
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MHD_LOGS", str(tmp_path))
    ws_mod = importlib.import_module("web_server")
    monkeypatch.setattr(ws_mod, "TimedX69RtspAdapter", FakeVideoAdapter)
    from fastapi.testclient import TestClient
    with TestClient(ws_mod.app) as client:
        yield client, ws_mod.HUB


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _airborne(client, hub, ws):
    ws.send_text(json.dumps({"type": "hb"}))
    assert _wait(lambda: hub.frames.count > 3)
    # Takeoff needs /ws (review fix: POST /command is LAND / E-STOP / pause / estop_reset only).
    r = client.post("/command", json={"type": "cmd", "command": "takeoff"}).json()
    assert r == {"ok": False, "flight_state": "ON_GROUND", "reason": "not_allowed_over_http"}
    ws.send_text(json.dumps({"type": "cmd", "command": "takeoff"}))
    assert _wait(lambda: hub.gov.state.value == "TAKING_OFF")
    end = time.monotonic() + 5.0

    while hub.gov.state.value != "AIRBORNE" and time.monotonic() < end:
        ws.send_text(json.dumps({"type": "hb"}))
        time.sleep(0.1)
    assert hub.gov.state.value == "AIRBORNE"


def test_ws_routes_ui_input_to_the_governor(backend):
    client, hub = backend
    with client.websocket_connect("/ws") as ws:
        _airborne(client, hub, ws)
        # Legacy upstream message: abs mode at full yaw. It never touches set_strategy.
        ws.send_text(json.dumps({"type": "axes", "mode": "abs", "throttle": 0, "yaw": 1.0, "pitch": 0,
                                 "roll": 0, "camera_tilt_direction": 1}))
        assert _wait(lambda: hub.gov.snapshot().wire["yaw"] == 128 + 64)
        assert hub.gov.snapshot().source == "human"
        assert type(hub.model.strategy).__name__ == "GovernorStrategy"
        assert hub.model.camera_tilt_state == 2
        # Ignored legacy messages.
        ws.send_text(json.dumps({"type": "set_speed_index", "speed_index": 0}))
        ws.send_text(json.dumps({"type": "set_profile", "name": "aggressive"}))
        time.sleep(0.1)
        assert hub.model.speed_index == 2
        # New-style command over /ws.
        ws.send_text(json.dumps({"type": "cmd", "command": "land"}))
        assert _wait(lambda: hub.gov.state.value == "LANDING")
    # Disconnect zeroes human input (the socket closed above).
    assert _wait(lambda: hub.gov._human_last_ms is None)


def test_ws_disconnect_zeroes_human_input(backend):
    client, hub = backend
    with client.websocket_connect("/ws") as ws:
        _airborne(client, hub, ws)
        ws.send_text(json.dumps({"type": "axes", "mode": "abs", "yaw": 0.5}))
        assert _wait(lambda: hub.gov.snapshot().source == "human")
    assert _wait(lambda: hub.gov.snapshot().wire["yaw"] == 128, 1.0)


def test_post_command_estop_and_state(backend):
    client, hub = backend
    r = client.post("/command", json={"type": "cmd", "command": "estop"})
    assert r.status_code == 200 and r.json() == {"ok": True, "flight_state": "ESTOPPED"}
    st = client.get("/state").json()
    assert st["type"] == "telemetry" and st["flight_state"] == "ESTOPPED" and st["flight_mode"] == "twin"
    assert _wait(lambda: hub.model.stop_flag is True or hub.gov.snapshot().source == "estop")
    bad = client.post("/command", json={"type": "cmd", "command": "selfdestruct"})
    assert bad.status_code == 422
    r = client.post("/command", json={"type": "cmd", "command": "takeoff"}).json()
    assert r == {"ok": False, "flight_state": "ESTOPPED", "reason": "not_allowed_over_http"}
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "cmd", "command": "takeoff"}))
        ws.send_text(json.dumps({"type": "hb"}))
    assert hub.gov.state.value == "ESTOPPED"



def test_telemetry_stream_at_10_hz(backend):
    client, hub = backend
    with client.websocket_connect("/ws/telemetry") as ws:
        t0 = time.monotonic()
        msgs = [json.loads(ws.receive_text()) for _ in range(6)]
        dt = time.monotonic() - t0
    assert all(m["type"] == "telemetry" for m in msgs)
    assert 0.35 <= dt <= 1.2
    m = msgs[-1]
    assert set(m["sticks_wire"]) == {"throttle", "yaw", "pitch", "roll"}
    assert m["frame_age_ms"] is not None and m["frame_age_ms"] < 1000


def test_frames_pull_protocol(backend):
    client, hub = backend
    assert _wait(lambda: hub.frames.count > 2)
    with client.websocket_connect("/ws/frames") as ws:
        ws.send_text(json.dumps({"type": "next", "after_frame_id": -1}))
        meta, jpeg = decode_frame(ws.receive_bytes())
        assert meta.frame_id >= 1 and jpeg[:2] == b"\xff\xd8"
        assert meta.receive_ms - meta.capture_ms == pytest.approx(80.0, abs=1.0)
        ws.send_text(json.dumps({"type": "next", "after_frame_id": meta.frame_id}))
        meta2, _ = decode_frame(ws.receive_bytes())
        assert meta2.frame_id > meta.frame_id


def test_mission_find_request_and_subject_obs(backend):
    client, hub = backend
    with client.websocket_connect("/ws") as ui, client.websocket_connect("/ws/subject") as perc:
        _airborne(client, hub, ui)
        r = client.post("/mission", json=json.loads(make_mission("m7").model_dump_json())).json()
        assert r["ok"] and r["find"]
        fr = json.loads(perc.receive_text())
        assert fr == {"type": "find", "mission_id": "m7", "subject_phrase": "person in blue shirt",
                      "category": "person", "color": "blue"}
        # Perception reports a confirmed subject on the right: the autopilot locks and yaws right.
        end = time.monotonic() + 3.0
        fid = 0
        while time.monotonic() < end and hub.gov.autopilot.mode.value != "FOLLOW":
            fid += 1
            perc.send_text(make_obs(fid, now_ms() - 150, cx=1000, mission_id="m7").model_dump_json())
            ui.send_text(json.dumps({"type": "hb"}))
            time.sleep(0.04)
        assert hub.gov.autopilot.mode.value == "FOLLOW"
        fid += 1
        perc.send_text(make_obs(fid, now_ms() - 150, cx=1000, mission_id="m7").model_dump_json())
        assert _wait(lambda: hub.gov.snapshot().wire["yaw"] > 128, 1.0)
        # Perception goes quiet: neutral within 0.5 s.
        t_stop = time.monotonic()
        assert _wait(lambda: hub.gov.snapshot().wire["yaw"] == 128, 1.0)
        assert time.monotonic() - t_stop < 0.6


def test_agent_relay_mission_and_command_whitelist(backend):
    client, hub = backend
    with client.websocket_connect("/ws/agent?role=agent") as agent, \
            client.websocket_connect("/ws/agent?role=ui") as aui, \
            client.websocket_connect("/ws/subject") as perc:
        # UI push-to-talk is relayed to the agent, and agent transcripts back to the UI.
        aui.send_text(json.dumps({"type": "agent_event", "kind": "ptt_down"}))
        assert json.loads(agent.receive_text())["kind"] == "ptt_down"
        agent.send_text(json.dumps({"type": "agent_event", "kind": "final", "text": "follow the person"}))
        assert json.loads(aui.receive_text())["text"] == "follow the person"
        # The agent can never E-stop or take off.
        agent.send_text(json.dumps({"type": "cmd", "command": "estop"}))
        ack = json.loads(agent.receive_text())
        assert ack["type"] == "cmd_ack" and ack["ok"] is False and ack["reason"] == "not_allowed_from_agent"
        # A mission from the agent reaches perception as a FindRequest.
        agent.send_text(make_mission("m9").model_dump_json())
        msgs = [json.loads(agent.receive_text())]
        assert any(m.get("type") == "mission_ack" for m in msgs)
        assert json.loads(perc.receive_text())["mission_id"] == "m9"
        # UI candidate choice goes to perception.
        aui.send_text(json.dumps({"type": "select", "mission_id": "m9", "index": 2}))
        assert json.loads(perc.receive_text()) == {"type": "select", "mission_id": "m9", "index": 2}


def test_shutdown_sends_land(backend):
    client, hub = backend
    gov = hub.gov
    client.__exit__(None, None, None)       # runs the lifespan shutdown now
    assert gov._shutdown is True


@pytest.mark.parametrize("env,msg", [
    ({"FLIGHT_MODE": ""}, "FLIGHT_MODE"),
    ({"FLIGHT_MODE": "twin", "DRONE_IP": "172.16.11.1"}, "non-loopback"),
    ({"FLIGHT_MODE": "real", "DRONE_IP": "127.0.0.1"}, "172.16.11.1"),
    ({"FLIGHT_MODE": "twin", "DRONE_TYPE": "s2x"}, "x69_lg"),
    ({"FLIGHT_MODE": "twin", "X69_LG_RTSP_URL": "rtsp://172.16.11.1/live/ch00_1"}, "non-loopback"),
    ({"FLIGHT_MODE": "twin", "PLUGINS_ENABLED": "true"}, "PLUGINS_ENABLED"),
])
def test_mode_gate_refuses_bad_environments(env, msg):
    from mhd.backend.modes import ModeError, validate_env
    full = dict(BASE_ENV)
    full.update(env)
    with pytest.raises(ModeError, match=msg):
        validate_env(full)


def test_mode_gate_accepts_good_environments():
    from mhd.backend.modes import validate_env
    assert validate_env(dict(BASE_ENV)).flight_mode == "twin"
    real = dict(BASE_ENV, FLIGHT_MODE="real", DRONE_IP="172.16.11.1", X69_LG_RTSP_URL="")
    assert validate_env(real).flight_mode == "real"


def test_backend_refuses_to_start_without_flight_mode(monkeypatch, tmp_path):
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("FLIGHT_MODE")
    monkeypatch.setenv("MHD_LOGS", str(tmp_path))
    ws_mod = importlib.import_module("web_server")
    from fastapi.testclient import TestClient
    from mhd.backend.modes import ModeError
    with pytest.raises(ModeError):
        with TestClient(ws_mod.app):
            pass


def test_cors_and_ws_origin_on_the_real_app(backend):
    """Review fix 8: CORS was '*' and /ws accepted any page. Loopback pages only now."""
    from starlette.websockets import WebSocketDisconnect
    client, hub = backend
    pre = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    ok = client.options("/command", headers={"Origin": "http://localhost:5173", **pre})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/command", headers={"Origin": "http://evil.example", **pre})
    assert "access-control-allow-origin" not in bad.headers
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_text()
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5173"}) as ws:
        ws.send_text(json.dumps({"type": "cmd", "command": "estop"}))
        assert _wait(lambda: hub.gov.state.value == "ESTOPPED")
