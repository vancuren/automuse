"""The mock backend itself: it must speak the frozen contracts and judge UI messages correctly,
or the browser tests would prove nothing."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mhd.contracts import FRAME_H, FRAME_W, AgentEvent, Candidates, Intent, Telemetry

MOCK_PATH = Path(__file__).resolve().parents[2] / "turbodrone/frontend/mock/mock_backend.py"
spec = importlib.util.spec_from_file_location("mock_backend", MOCK_PATH)
mock = importlib.util.module_from_spec(spec)
sys.modules["mock_backend"] = mock  # dataclasses look the module up while it loads
spec.loader.exec_module(mock)


@pytest.fixture
def client(tmp_path):
    sim = mock.Sim(scenario="manual")
    app = mock.build_app(sim, mock.MessageLog(tmp_path / "log.jsonl"))
    with TestClient(app) as c:
        c.sim = sim
        c.log_path = tmp_path / "log.jsonl"
        yield c


def log_lines(c) -> list[dict]:
    return [json.loads(line) for line in c.log_path.read_text().splitlines()]


def test_state_and_telemetry_stream_validate_against_the_contract(client):
    Telemetry.model_validate(client.get("/state").json())
    with client.websocket_connect("/ws/telemetry") as ws:
        for _ in range(3):
            Telemetry.model_validate_json(ws.receive_text())


def test_control_socket_judges_messages_by_the_contract(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hb", "client": "ui"}))
        ws.send_text(json.dumps({"type": "axes", "throttle": 0, "yaw": 0.5, "pitch": 0, "roll": 0, "mode": "inc"}))
        ws.send_text(json.dumps({"type": "axes", "throttle": 0, "yaw": 0, "pitch": 0, "roll": 0, "mode": "inc"}))
        ws.send_text(json.dumps({"type": "axes", "yaw": 1, "mode": "inc", "camera_tilt_direction": 0}))
        ws.send_text(json.dumps({"type": "axes", "yaw": 1, "mode": "mouse"}))
        ws.send_text(json.dumps({"type": "set_speed_index", "speed_index": 2}))
        ws.send_text(json.dumps({"type": "cmd", "command": "estop"}))
        ws.send_text("not json")
        ws.send_text(json.dumps({"type": "hb", "client": "ui"}))  # sync point
    entries = log_lines(client)
    assert [e["valid"] for e in entries] == [True, True, False, False, False, False, True, False, True]
    assert "all-zero" in entries[2]["error"]
    assert client.sim.flight_state.value == "ESTOPPED"


def test_http_command_fallback(client):
    r = client.post("/command", json={"type": "cmd", "command": "estop"})
    assert r.json() == {"ok": True, "flight_state": "ESTOPPED", "note": "estop latched"}
    assert client.post("/command", json={"type": "cmd", "command": "explode"}).status_code == 422
    assert [e["channel"] for e in log_lines(client)] == ["http", "http"]


def test_real_mode_takeoff_needs_the_exact_arm_phrase(client):
    client.post("/mock/control", json={"flight_mode": "real"})
    assert client.post("/command", json={"type": "cmd", "command": "takeoff"}).json()["ok"] is False
    assert client.post("/command", json={"type": "cmd", "command": "arm", "arm_phrase": "arm"}).json()["ok"] is False
    assert client.post("/command", json={"type": "cmd", "command": "arm", "arm_phrase": "ARM"}).json()["ok"] is True
    r = client.post("/command", json={"type": "cmd", "command": "takeoff"}).json()
    assert r["ok"] is True and r["flight_state"] == "TAKING_OFF"


def test_estop_latches_until_reset(client):
    client.post("/command", json={"type": "cmd", "command": "estop"})
    for cmd in ("takeoff", "land", "confirm_touchdown"):
        client.post("/command", json={"type": "cmd", "command": cmd})
        assert client.sim.flight_state.value == "ESTOPPED", cmd
    client.post("/command", json={"type": "cmd", "command": "estop_reset"})
    assert client.sim.flight_state.value == "ON_GROUND"


def test_socket_down_mode_refuses_the_control_socket(client):
    client.post("/mock/control", json={"ws_mode": "down"})
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws"):
            pass
    client.post("/mock/control", json={"ws_mode": "normal"})
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hb", "client": "ui"}))


def test_push_to_talk_produces_transcript_mission_and_reply(client):
    with client.websocket_connect("/ws/agent?role=ui") as ws:
        ws.send_text(json.dumps({"type": "agent_event", "kind": "ptt_down"}))
        ws.send_text(json.dumps({"type": "agent_event", "kind": "ptt_up"}))
        kinds, data = [], None
        while "reply" not in kinds:
            ev = AgentEvent.model_validate_json(ws.receive_text())
            kinds.append(ev.kind)
            if ev.kind == "intent":
                data = ev.data  # the real agent's shape: intent, model, latency_ms, reason, transcript
    assert "final" in kinds and kinds.index("final") < kinds.index("intent") < kinds.index("reply")
    intent = Intent.model_validate(data["intent"])
    assert intent.subject_phrase == "person in blue shirt"
    assert data["model"] in ("glimmer-30b+lora", "muse-spark-1.3") and data["latency_ms"] > 0
    assert client.sim.mission.intent == intent and client.sim.mission.model == data["model"]


def test_candidates_message_validates(client):
    Candidates.model_validate(client.post("/mock/candidates").json())


def test_subject_mask_stays_inside_its_box_and_the_frame():
    for i in range(200):
        t = i * 0.37
        for box in (mock.subject_box(t), mock.lookalike_box(t)):
            assert 0 <= box.x1 < box.x2 <= FRAME_W and 0 <= box.y1 < box.y2 <= FRAME_H
            for x, y in mock.person_polygon(box):
                assert box.x1 - 0.5 <= x <= box.x2 + 0.5 and box.y1 - 0.5 <= y <= box.y2 + 0.5


def test_test_pattern_is_a_1280x720_jpeg():
    jpeg = mock.render_jpeg(mock._background(), 1.0, 1)
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (FRAME_H, FRAME_W, 3)


def test_follow_sticks_steer_toward_the_subject(client):
    sim = client.sim
    sim.set_flight(mock.FS.airborne)
    sim.default_mission()
    sim.set_ap(mock.AP.follow, step=mock.LOOP_FROM)
    tm = sim.telemetry()
    assert tm.active_source == "autopilot" and tm.subject.state.value == "confirmed"
    # +yaw when the subject is right of center (conventions.md)
    assert math.copysign(1, tm.sticks_by_source["autopilot"]["yaw"] or 1) == math.copysign(1, tm.subject.box.cx - 640 or 1)
