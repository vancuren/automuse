"""Flight recorder (src/mhd/backend/recorder.py): flights are saved for the UI's replay, read-only routes."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from mhd.backend import recorder as rec_mod
from mhd.backend.recorder import FlightRecorder, list_flights, read_frame
from test_backend_api import _airborne, _wait, backend  # noqa: F401  (fixture)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _tlm(state):
    return json.dumps({"type": "telemetry", "flight_state": state})


def _wait_saved(root, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        flights = list_flights(root)
        if flights and flights[0].get("complete"):
            return flights
        time.sleep(0.02)
    return list_flights(root)


def test_a_flight_is_saved_with_preroll_commands_and_frames(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(rec_mod.time, "monotonic", clock)
    r = FlightRecorder(tmp_path, flight_mode="twin", fps=10)
    for _ in range(100):                                  # 10 s on the ground: only the pre-roll is kept
        r.sample(_tlm("ON_GROUND"), "ON_GROUND", None)
        clock.t += 0.1
    assert not r.recording and list_flights(tmp_path) == []
    r.note("cmd", {"command": "takeoff", "source": "ui", "ok": True, "reason": None})
    for i in range(30):
        state = "TAKING_OFF" if i < 10 else "AIRBORNE"
        r.sample(_tlm(state), state, (SimpleNamespace(frame_id=i), b"jpeg%03d" % i))
        if i == 15:
            r.note("mission", {"mission_id": "m-1", "transcript": "follow the blue shirt", "model": "glimmer",
                               "intent": {"action": "follow", "subject_phrase": "person in blue shirt"}, "ok": True})
            r.note("agent", '{"type":"agent_event","kind":"final","text":"follow the blue shirt"}')
        clock.t += 0.1
    assert r.recording
    r.note("cmd", {"command": "land", "source": "ui", "ok": True, "reason": None})
    for _ in range(40):                                   # 4 s on the ground: the tail, then it closes
        r.sample(_tlm("ON_GROUND"), "ON_GROUND", None)
        clock.t += 0.1
    assert not r.recording

    flights = _wait_saved(tmp_path)
    assert len(flights) == 1 and flights[0]["complete"] and flights[0]["flight_mode"] == "twin"
    meta = flights[0]
    assert meta["frames"] == 30 and 9000 < meta["duration_ms"] < 11500
    assert [c.get("command") or c.get("transcript") for c in meta["commands"]] == \
        ["takeoff", "follow the blue shirt", "land"]
    events = [json.loads(line) for line in (tmp_path / meta["id"] / "events.jsonl").read_text().splitlines()]
    kinds = {e["k"] for e in events}
    assert kinds == {"tlm", "cmd", "mission", "agent", "frame"}
    assert events[0]["t"] == 0.0 and events[0]["d"]["flight_state"] == "ON_GROUND"      # the pre-roll
    takeoff = next(e for e in events if e["k"] == "cmd")
    assert takeoff["d"]["command"] == "takeoff" and 3000 < takeoff["t"] < 4200
    assert all(a["t"] <= b["t"] for a, b in zip(events, events[1:]))
    assert read_frame(tmp_path, meta["id"], 0) == b"jpeg000"
    assert read_frame(tmp_path, meta["id"], 29) == b"jpeg029"
    assert read_frame(tmp_path, meta["id"], 30) is None
    assert read_frame(tmp_path, "../" + meta["id"], 0) is None


def test_recorder_never_raises_and_disabled_writes_nothing(tmp_path):
    r = FlightRecorder(tmp_path / "x", enabled=False)
    r.sample(_tlm("AIRBORNE"), "AIRBORNE", None)
    r.note("cmd", {"command": "land"})
    assert not (tmp_path / "x").exists()
    r2 = FlightRecorder(tmp_path / "y")
    r2.note("cmd", object())                               # not JSON: swallowed
    r2.sample("{}", "AIRBORNE", ("not a frame",))          # bad frame tuple: swallowed
    r2.close()


def test_backend_records_a_flight_and_serves_it(backend, monkeypatch, tmp_path):  # noqa: F811
    client, hub = backend
    assert hub.recorder is None, "recording is opt-in (MHD_RECORD=1, set by scripts/run_stack.sh)"
    assert client.get("/flights").json() == {"flights": []}
    assert client.get("/flights/nope/events").status_code == 404


@pytest.fixture
def recording_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MHD_RECORD", "1")
    monkeypatch.setenv("MHD_FLIGHTS_DIR", str(tmp_path / "flights"))
    monkeypatch.setattr(rec_mod, "TAIL_S", 0.3)


def test_backend_with_recording_on(recording_env, backend):  # noqa: F811
    client, hub = backend
    assert hub.recorder is not None
    with client.websocket_connect("/ws") as ws:
        _airborne(client, hub, ws)
        assert _wait(lambda: hub.recorder.recording)
        time.sleep(0.5)
        ws.send_text(json.dumps({"type": "cmd", "command": "estop"}))
        assert _wait(lambda: hub.gov.state.value == "ESTOPPED")
        assert _wait(lambda: not hub.recorder.recording, timeout=3.0)
    flights = []
    assert _wait(lambda: (flights.clear() or flights.extend(client.get("/flights").json()["flights"]) or True)
                 and flights and flights[0]["complete"], timeout=3.0)
    f = flights[0]
    assert f["frames"] >= 2 and [c["command"] for c in f["commands"]][:1] == ["takeoff"]
    assert "estop" in [c["command"] for c in f["commands"]]
    lines = client.get(f"/flights/{f['id']}/events").text.splitlines()
    kinds = {json.loads(line)["k"] for line in lines}
    assert {"tlm", "cmd", "frame", "agent"} <= kinds       # agent: the governor's flight-state events
    jpg = client.get(f"/flights/{f['id']}/frame/0")
    assert jpg.status_code == 200 and jpg.headers["content-type"] == "image/jpeg" and jpg.content[:2] == b"\xff\xd8"
    assert client.get(f"/flights/{f['id']}/frame/99999").status_code == 404
